"""Synthetic GGUF fixtures: tiny models built in-process, no downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFEndian, GGUFValueType, GGUFWriter, quants

from ggufkit.fingerprint import Fingerprint, fingerprint
from ggufkit.model import open_model

Q = GGMLQuantizationType
V = GGUFValueType

#: Types the installed gguf package can quantize from float32. Everything else gets raw bytes.
QUANTIZABLE = frozenset({Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0, Q.BF16, Q.MXFP4})

ELEMENT_DTYPES = {
    Q.F32: np.float32,
    Q.F16: np.float16,
    Q.F64: np.float64,
    Q.I8: np.int8,
    Q.I16: np.int16,
    Q.I32: np.int32,
    Q.I64: np.int64,
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite tests/golden/*.json from the current implementation",
    )


def det_floats(shape: tuple[int, ...], seed: int = 1) -> np.ndarray:
    """Deterministic pseudo-random floats in [-0.5, 0.5), independent of numpy's RNG streams."""
    n = int(np.prod(shape))
    idx = np.arange(n, dtype=np.uint64)
    vals = (idx * np.uint64(7919) + np.uint64(seed)) % np.uint64(10007)
    return (vals.astype(np.float32) / np.float32(10007) - np.float32(0.5)).reshape(shape)


def det_bytes(shape: tuple[int, ...], seed: int = 1) -> np.ndarray:
    n = int(np.prod(shape))
    idx = np.arange(n, dtype=np.uint64)
    vals = (idx * np.uint64(2654435761) + np.uint64(seed)) % np.uint64(1 << 32)
    return (vals >> np.uint64(24)).astype(np.uint8).reshape(shape)


def _infer(value: Any) -> tuple[Any, GGUFValueType, GGUFValueType | None]:
    if isinstance(value, tuple):
        if len(value) == 2:
            return value[0], value[1], None
        return value[0], value[1], value[2]
    if isinstance(value, bool):
        return value, V.BOOL, None
    if isinstance(value, int):
        return value, (V.UINT32 if value >= 0 else V.INT32), None
    if isinstance(value, float):
        return value, V.FLOAT32, None
    if isinstance(value, str):
        return value, V.STRING, None
    if isinstance(value, list):
        if not value:
            msg = "empty arrays cannot be written by GGUFWriter"
            raise ValueError(msg)
        first = value[0]
        if isinstance(first, str):
            return value, V.ARRAY, V.STRING
        if isinstance(first, bool):
            return value, V.ARRAY, V.BOOL
        if isinstance(first, int):
            return value, V.ARRAY, V.INT32
        if isinstance(first, float):
            return value, V.ARRAY, V.FLOAT32
    msg = f"cannot infer GGUF type for {value!r}"
    raise TypeError(msg)


def add_tensor_fixture(
    writer: GGUFWriter,
    name: str,
    arr: np.ndarray,
    qtype: GGMLQuantizationType,
    *,
    seed: int = 1,
) -> None:
    if qtype in ELEMENT_DTYPES:
        writer.add_tensor(name, np.ascontiguousarray(arr.astype(ELEMENT_DTYPES[qtype])))
    elif qtype in QUANTIZABLE:
        writer.add_tensor(name, quants.quantize(arr.astype(np.float32), qtype), raw_dtype=qtype)
    else:
        byte_shape = quants.quant_shape_to_byte_shape(arr.shape, qtype)
        writer.add_tensor(name, det_bytes(byte_shape, seed), raw_dtype=qtype)


def write_gguf(
    path: Path,
    *,
    arch: str = "llama",
    kv: dict[str, Any] | None = None,
    tensors: dict[str, tuple[np.ndarray, GGMLQuantizationType]] | None = None,
    split_max_tensors: int = 0,
    endianess: GGUFEndian = GGUFEndian.LITTLE,
) -> list[Path]:
    """Write a GGUF model and return its file(s), in shard order."""
    writer = GGUFWriter(
        path,
        arch,
        use_temp_file=False,
        endianess=endianess,
        split_max_tensors=split_max_tensors,
    )
    for key, value in (kv or {}).items():
        val, vtype, sub = _infer(value)
        writer.add_key_value(key, val, vtype, sub_type=sub)
    for i, (name, (arr, qtype)) in enumerate((tensors or {}).items()):
        add_tensor_fixture(writer, name, arr, qtype, seed=i + 1)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return writer.format_shard_names(Path(path))


def tiny_kv() -> dict[str, Any]:
    return {
        "general.name": "tiny",
        "general.file_type": 1,
        "general.description": "",
        "general.flag": True,
        "general.count": (7, V.UINT64),
        "general.ratio": (0.25, V.FLOAT64),
        "llama.block_count": 1,
        "tokenizer.ggml.tokens": ["<s>", "a", "b", "héllo"],
        "tokenizer.ggml.scores": [0.0, -1.5, 2.25, 3.0],
        "tokenizer.ggml.token_type": [1, 2, 3, 4],
    }


def tiny_tensors() -> dict[str, tuple[np.ndarray, GGMLQuantizationType]]:
    return {
        "token_embd.weight": (det_floats((4, 64), 11), Q.F16),
        "blk.0.attn_q.weight": (det_floats((64, 64), 12), Q.Q8_0),
        "blk.0.attn_k.weight": (det_floats((32, 64), 13), Q.Q4_0),
        "blk.0.ffn_down.weight": (det_floats((8, 256), 14), Q.Q4_K),
        "blk.0.attn_norm.weight": (det_floats((64,), 15), Q.F32),
        "output_norm.weight": (det_floats((64,), 16), Q.F32),
    }


def fingerprint_of(path: Path, **kwargs: Any) -> Fingerprint:
    with open_model(path) as model:
        return fingerprint(model, per_kv=True, **kwargs)


@pytest.fixture
def tiny_gguf(tmp_path: Path) -> Path:
    return write_gguf(tmp_path / "tiny.gguf", kv=tiny_kv(), tensors=tiny_tensors())[0]
