"""Regression guard for the ggufkit-fp/1 byte format.

The fixture uses deterministic data and only raw-byte or element tensors, so the golden file
depends on nothing but the specification. If this test fails, either the spec changed (bump the
spec id and regenerate with ``pytest --update-golden``) or something broke.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gguf import GGMLQuantizationType as Q
from gguf import GGUFValueType as V

from tests.conftest import det_floats, fingerprint_of, write_gguf

GOLDEN = Path(__file__).parent / "golden" / "tiny.json"
GOLDEN_WEIGHTS = Path(__file__).parent / "golden" / "weights.json"


def golden_model(tmp_path: Path, *, quantized_by_gguf: bool = True) -> Path:
    kv = {
        "general.name": "golden",
        "general.file_type": 7,
        "general.flag": False,
        "general.count": (1 << 40, V.UINT64),
        "general.ratio": (0.125, V.FLOAT32),
        "general.ratio64": (0.0625, V.FLOAT64),
        "general.small": (-3, V.INT8),
        "tokenizer.ggml.tokens": ["<s>", "a", "b", "日本"],
        "tokenizer.ggml.scores": [0.0, -1.5, 2.25, 3.0],
        "tokenizer.ggml.token_type": [1, 2, 3, 4],
    }
    tensors = {
        "token_embd.weight": (det_floats((4, 32), 21), Q.F16),
        "blk.0.attn_q.weight": (det_floats((4, 64), 22), Q.Q8_0),
        "blk.0.ffn_down.weight": (det_floats((2, 256), 23), Q.Q4_K),
        "blk.0.ffn_up.weight": (det_floats((2, 256), 24), Q.IQ2_XXS),
        "output_norm.weight": (det_floats((32,), 25), Q.F32),
    }
    if not quantized_by_gguf:
        del tensors["blk.0.attn_q.weight"]
    return write_gguf(tmp_path / "golden.gguf", kv=kv, tensors=tensors)[0]


def test_golden_fingerprint(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    fp = fingerprint_of(golden_model(tmp_path)).to_dict()
    # Q8_0 is produced by gguf.quants.quantize, whose rounding is not part of our spec.
    del fp["tensors"]["blk.0.attn_q.weight"]
    fp["weights"] = None
    fp["overall"] = None
    if request.config.getoption("--update-golden"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert fp == expected


def test_golden_weights_and_overall(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """Pins ``weights`` and ``overall`` on a model whose tensor bytes are fully deterministic.

    Without the Q8_0 tensor every byte comes from the test itself, so a change to how
    ``weights`` binds names to data, or to ``overall``, cannot slip through.
    """
    fp = fingerprint_of(golden_model(tmp_path, quantized_by_gguf=False)).to_dict()
    pinned = {k: fp[k] for k in ("spec", "kv", "tensor_info", "weights", "overall")}
    if request.config.getoption("--update-golden"):
        GOLDEN_WEIGHTS.write_text(
            json.dumps(pinned, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    assert pinned == json.loads(GOLDEN_WEIGHTS.read_text(encoding="utf-8"))
