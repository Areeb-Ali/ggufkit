"""Regenerate the synthetic example image used in the README.

    uv run python scripts/make_examples.py

The image comes from small GGUF files built here from seeded random data, not from a real
model. It illustrates a mechanism; the README says so next to it. The real-model images are
made by scripts/make_readme_assets.py.
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

        return 0


if __name__ == "__main__":
    sys.exit(main_examples())
