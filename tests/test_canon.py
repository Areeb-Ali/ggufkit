from __future__ import annotations

import struct
from pathlib import Path

from gguf import GGMLQuantizationType as Q
from gguf import GGUFValueType as V

from ggufkit import canon
from ggufkit.model import open_model
from tests.conftest import det_floats, write_gguf


def test_scalar_and_string_entries_encode_as_documented(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "m.gguf",
        kv={"x.count": (300, V.UINT32), "x.name": "héllo", "x.flag": True},
    )[0]
    with open_model(path) as model:
        fields = {f.name: f for f in model.metadata_fields()}
        assert canon.encode_kv(fields["x.count"]) == (
            b"kv\0"
            + struct.pack("<Q", 7)
            + b"x.count"
            + struct.pack("<I", int(V.UINT32))
            + struct.pack("<I", 300)
        )
        name = "héllo".encode()
        assert canon.encode_kv(fields["x.name"]) == (
            b"kv\0"
            + struct.pack("<Q", 6)
            + b"x.name"
            + struct.pack("<I", int(V.STRING))
            + struct.pack("<Q", len(name))
            + name
        )
        assert canon.encode_kv(fields["x.flag"]).endswith(struct.pack("<I", int(V.BOOL)) + b"\x01")


def test_array_entries_keep_item_type_and_count(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "m.gguf", kv={"x.list": [3, 1, 2], "x.words": ["a", "bc"]})[0]
    with open_model(path) as model:
        fields = {f.name: f for f in model.metadata_fields()}
        ints = canon.encode_kv(fields["x.list"])
        assert ints.endswith(
            struct.pack("<I", int(V.ARRAY))
            + struct.pack("<I", int(V.INT32))
            + struct.pack("<Q", 3)
            + struct.pack("<iii", 3, 1, 2)
        )
        words = canon.encode_kv(fields["x.words"])
        assert words.endswith(
            struct.pack("<I", int(V.STRING))
            + struct.pack("<Q", 2)
            + struct.pack("<Q", 1)
            + b"a"
            + struct.pack("<Q", 2)
            + b"bc"
        )


def test_tensor_info_encoding_uses_gguf_dimension_order(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "m.gguf",
        tensors={"t.weight": (det_floats((3, 8)), Q.F32)},
    )[0]
    with open_model(path) as model:
        t = model.tensor("t.weight")
        assert [int(d) for d in t.shape] == [8, 3]  # ne[0] is the fastest-varying dimension
        assert canon.encode_tensor_info(t) == (
            b"ti\0"
            + struct.pack("<Q", 8)
            + b"t.weight"
            + struct.pack("<I", 2)
            + struct.pack("<QQ", 8, 3)
            + struct.pack("<I", int(Q.F32))
        )


def test_chunked_hash_matches_one_shot_hash(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "m.gguf", tensors={"t.weight": (det_floats((64, 64)), Q.F32)})[0]
    with open_model(path) as model:
        t = model.tensor("t.weight")
        whole = canon.tensor_data_digest(t, model.endianess)
        chunked = canon.tensor_data_digest(t, model.endianess, chunk_size=1000)
        assert whole == chunked


def test_sorted_by_name_is_bytewise() -> None:
    assert canon.sorted_by_name(["b", "a", "é", "Z"]) == ["Z", "a", "b", "é"]
