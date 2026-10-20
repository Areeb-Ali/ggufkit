"""Regenerate the synthetic example images used in the README.

    uv run python scripts/make_examples.py

Both images come from small GGUF files built here from seeded random data, not from a real
model. They illustrate what the commands show; the README says so next to each image.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter, quants

from ggufkit.cli import main

ASSETS = Path(__file__).resolve().parent.parent / "assets"
Q = GGMLQuantizationType


def write(path: Path, tensors: dict[str, tuple[np.ndarray, GGMLQuantizationType]]) -> Path:
    writer = GGUFWriter(path, "llama")
    writer.add_string("general.name", "synthetic example")
    for name, (arr, qtype) in tensors.items():
        if qtype == Q.F32:
            writer.add_tensor(name, arr.astype(np.float32))
        elif qtype in (Q.Q4_0, Q.Q8_0):
            writer.add_tensor(name, quants.quantize(arr, qtype), raw_dtype=qtype)
        else:  # K-quants: gguf-py can only decode them, so fill blocks with seeded bytes
            shape = quants.quant_shape_to_byte_shape(arr.shape, qtype)
            raw = np.random.default_rng(len(name)).integers(0, 256, shape, dtype=np.uint8)
            writer.add_tensor(name, raw, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def main_examples() -> int:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        # One input channel with 8x larger weights, as real attention projections have.
        w = (rng.standard_normal((256, 512)) * 0.02).astype(np.float32)
        w[:, 100] *= 8
        src = write(d / "src.gguf", {"blk.0.attn_q.weight": (w, Q.F32)})
        q4 = write(d / "q4.gguf", {"blk.0.attn_q.weight": (w, Q.Q4_0)})
        rc = main(
            [
                "viz",
                "error",
                str(src),
                str(q4),
                "blk.0.attn_q.weight",
                "-o",
                str(ASSETS / "example-error.png"),
                "--force",
                "-q",
            ]
        )
        if rc:
            return rc

        # A Q4_K_M-style mixture: Q4_K body, Q6_K for some attn_v / ffn_down blocks.
        t: dict[str, tuple[np.ndarray, GGMLQuantizationType]] = {
            "token_embd.weight": (np.zeros((16, 256), np.float32), Q.Q6_K),
            "output_norm.weight": (np.ones(256, np.float32), Q.F32),
        }
        for b in range(12):
            t[f"blk.{b}.attn_norm.weight"] = (np.ones(256, np.float32), Q.F32)
            for role in (
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ):
                hi = (role == "attn_v" and b in (0, 11)) or (role == "ffn_down" and b in (0, 1, 11))
                t[f"blk.{b}.{role}.weight"] = (
                    np.zeros((8, 256), np.float32),
                    Q.Q6_K if hi else Q.Q4_K,
                )
            t[f"blk.{b}.ffn_norm.weight"] = (np.ones(256, np.float32), Q.F32)
        mix = write(d / "synthetic-Q4_K_M.gguf", t)
        return main(
            [
                "viz",
                "layers",
                str(mix),
                "-o",
                str(ASSETS / "example-layers.png"),
                "--force",
                "--title",
                "synthetic Q4_K_M-style mix",
            ]
        )


if __name__ == "__main__":
    sys.exit(main_examples())
