from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from gguf import GGMLQuantizationType as Q
from gguf import GGUFEndian

from ggufkit.canon import SPEC_ID, UnsupportedModelError
from ggufkit.fingerprint import Fingerprint
from ggufkit.model import open_model
from tests.conftest import det_floats, fingerprint_of, tiny_kv, tiny_tensors, write_gguf


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flip_first_byte_of_tensor(path: Path, name: str) -> None:
    with open_model(path) as model:
        offset = model.tensor(name).data_offset
    with path.open("r+b") as f:
        f.seek(offset)
        b = f.read(1)
        f.seek(offset)
        f.write(bytes([b[0] ^ 0xFF]))


def test_fingerprint_shape(tiny_gguf: Path) -> None:
    fp = fingerprint_of(tiny_gguf)
    assert fp.spec == SPEC_ID
    assert fp.n_tensors == len(tiny_tensors())
    assert fp.n_kv == len(tiny_kv()) + 1  # general.architecture is added by the writer
    assert fp.tensors is not None
    assert set(fp.tensors) == set(tiny_tensors())
    for digest in (fp.kv, fp.tensor_info, fp.weights, fp.overall):
        assert digest is not None
        assert len(digest) == 64
    assert fp.kvs is not None
    assert "general.architecture" in fp.kvs
    assert "GGUF.version" not in fp.kvs


def test_metadata_order_does_not_matter(tmp_path: Path) -> None:
    kv = tiny_kv()
    tensors = tiny_tensors()
    a = write_gguf(tmp_path / "a.gguf", kv=kv, tensors=tensors)[0]
    b = write_gguf(
        tmp_path / "b.gguf",
        kv=dict(reversed(list(kv.items()))),
        tensors=dict(reversed(list(tensors.items()))),
    )[0]
    assert file_sha256(a) != file_sha256(b)
    fa, fb = fingerprint_of(a), fingerprint_of(b)
    assert fa.kv == fb.kv
    assert fa.tensor_info == fb.tensor_info
    assert fa.weights == fb.weights
    assert fa.overall == fb.overall


def test_metadata_edit_changes_only_kv(tmp_path: Path) -> None:
    kv = tiny_kv()
    a = write_gguf(tmp_path / "a.gguf", kv=kv, tensors=tiny_tensors())[0]
    kv["general.name"] = "renamed"
    b = write_gguf(tmp_path / "b.gguf", kv=kv, tensors=tiny_tensors())[0]
    fa, fb = fingerprint_of(a), fingerprint_of(b)
    assert fa.kv != fb.kv
    assert fa.kvs is not None
    assert fb.kvs is not None
    changed = {k for k in fa.kvs if fa.kvs[k] != fb.kvs[k]}
    assert changed == {"general.name"}
    assert fa.tensor_info == fb.tensor_info
    assert fa.weights == fb.weights
    assert fa.overall != fb.overall


def test_tensor_rename_keeps_the_tensor_data_digest(tmp_path: Path) -> None:
    """``weights`` binds each name to its bytes, so a rename changes it, but the renamed
    tensor's own data digest is unchanged, which is how ``verify`` recognizes a rename."""
    tensors = tiny_tensors()
    a = write_gguf(tmp_path / "a.gguf", kv=tiny_kv(), tensors=tensors)[0]
    tensors["blk.0.attn_q.weight.renamed"] = tensors.pop("blk.0.attn_q.weight")
    b = write_gguf(tmp_path / "b.gguf", kv=tiny_kv(), tensors=tensors)[0]
    fa, fb = fingerprint_of(a), fingerprint_of(b)
    assert fa.kv == fb.kv
    assert fa.tensor_info != fb.tensor_info
    assert fa.weights != fb.weights
    assert fa.tensors is not None
    assert fb.tensors is not None
    assert fa.tensors["blk.0.attn_q.weight"].data == fb.tensors["blk.0.attn_q.weight.renamed"].data


def test_weight_change_is_attributed_to_the_tensor(tiny_gguf: Path) -> None:
    before = fingerprint_of(tiny_gguf)
    flip_first_byte_of_tensor(tiny_gguf, "blk.0.attn_k.weight")
    after = fingerprint_of(tiny_gguf)
    assert before.kv == after.kv
    assert before.tensor_info == after.tensor_info
    assert before.weights != after.weights
    assert before.tensors is not None
    assert after.tensors is not None
    changed = {n for n in before.tensors if before.tensors[n].data != after.tensors[n].data}
    assert changed == {"blk.0.attn_k.weight"}


def test_sharded_model_fingerprints_like_the_single_file(tmp_path: Path) -> None:
    single = write_gguf(tmp_path / "one.gguf", kv=tiny_kv(), tensors=tiny_tensors())[0]
    shards = write_gguf(
        tmp_path / "split.gguf",
        kv=tiny_kv(),
        tensors=tiny_tensors(),
        split_max_tensors=2,
    )
    assert len(shards) == 3
    assert shards[0].name == "split-00001-of-00003.gguf"
    with open_model(shards[1]) as model:
        assert model.n_shards == 3
        assert len(model.tensors) == len(tiny_tensors())
        assert not any(
            name.startswith("split.") for name in (f.name for f in model.metadata_fields())
        )
    fs, fo = fingerprint_of(single), fingerprint_of(shards[2])
    assert fs.kv == fo.kv
    assert fs.tensor_info == fo.tensor_info
    assert fs.weights == fo.weights
    assert fs.overall == fo.overall


def test_big_endian_element_tensors_match_little_endian(tmp_path: Path) -> None:
    tensors = {
        "a.weight": (det_floats((4, 8), 1), Q.F32),
        "b.weight": (det_floats((4, 8), 2), Q.F16),
        "c.weight": (det_floats((16,), 3), Q.F32),
    }
    le = write_gguf(tmp_path / "le.gguf", kv=tiny_kv(), tensors=tensors)[0]
    be = write_gguf(tmp_path / "be.gguf", kv=tiny_kv(), tensors=tensors, endianess=GGUFEndian.BIG)[
        0
    ]
    with open_model(be) as model:
        assert model.endianess == GGUFEndian.BIG
    assert file_sha256(le) != file_sha256(be)
    assert fingerprint_of(le).overall == fingerprint_of(be).overall


def test_big_endian_quantized_tensors_are_refused(tmp_path: Path) -> None:
    be = write_gguf(
        tmp_path / "be.gguf",
        kv=tiny_kv(),
        tensors={"q.weight": (det_floats((4, 64), 1), Q.Q8_0)},
        endianess=GGUFEndian.BIG,
    )[0]
    with pytest.raises(UnsupportedModelError, match="big-endian"):
        fingerprint_of(be)
    assert fingerprint_of(be, data=False).kv


def test_no_data_skips_weights(tiny_gguf: Path) -> None:
    fp = fingerprint_of(tiny_gguf, data=False)
    assert fp.weights is None
    assert fp.overall is None
    assert fp.tensors is not None
    assert all(t.data is None for t in fp.tensors.values())
    assert fp.kv == fingerprint_of(tiny_gguf).kv


def test_jobs_do_not_change_the_result(tiny_gguf: Path) -> None:
    assert (
        fingerprint_of(tiny_gguf, jobs=1).to_dict() == fingerprint_of(tiny_gguf, jobs=4).to_dict()
    )


def test_round_trips_through_dict(tiny_gguf: Path) -> None:
    fp = fingerprint_of(tiny_gguf)
    assert Fingerprint.from_dict(fp.to_dict()) == fp
