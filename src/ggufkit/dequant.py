"""Turn any tensor ggufkit can read into float32, reusing llama.cpp's own dequantizers."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
from gguf import GGMLQuantizationType, GGUFEndian, quants
from gguf.constants import GGML_QUANT_SIZES

from ggufkit.canon import UnsupportedModelError, is_element_type

if TYPE_CHECKING:
    from gguf.gguf_reader import ReaderTensor

ELEMENT_TYPES = frozenset(
    {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.F64,
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
    }
)


@cache
def can_dequantize(qtype: GGMLQuantizationType) -> bool:
    """Whether the installed ``gguf`` package knows how to decode ``qtype``."""
    if qtype in ELEMENT_TYPES:
        return True
    if qtype not in GGML_QUANT_SIZES:
        return False
    _block, type_size = GGML_QUANT_SIZES[qtype]
    try:
        quants.dequantize(np.zeros((1, type_size), dtype=np.uint8), qtype)
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 - zero blocks are not always valid, but the path exists
        return True
    return True


def to_f32(
    tensor: ReaderTensor,
    file_endianess: GGUFEndian = GGUFEndian.LITTLE,
    rows: slice | None = None,
) -> np.ndarray:
    """Decode ``tensor`` (or a slice of its leading axis) to a float32 array.

    The result has numpy shape, which is the reverse of the GGUF ``ne`` order: a
    ``[n_embd, n_vocab]`` GGUF tensor comes back as ``(n_vocab, n_embd)``. Slicing by rows
    keeps memory bounded on multi-gigabyte tensors: quantization blocks never span rows.
    """
    data = tensor.data if rows is None else tensor.data[rows]
    if is_element_type(tensor):
        return data.astype(np.float32, copy=False)
    if file_endianess == GGUFEndian.BIG:
        msg = (
            f"Tensor {tensor.name!r} is {tensor.tensor_type.name} in a big-endian file; "
            "convert the file to little-endian first (gguf_convert_endian)"
        )
        raise UnsupportedModelError(msg)
    if not can_dequantize(tensor.tensor_type):
        msg = f"The installed gguf package cannot dequantize {tensor.tensor_type.name}"
        raise UnsupportedModelError(msg)
    return quants.dequantize(data, tensor.tensor_type)
