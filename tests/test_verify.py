from __future__ import annotations

import json
from pathlib import Path

import pytest

from ggufkit.cli import main
from ggufkit.fingerprint import Fingerprint
from ggufkit.manifest import build_manifest, save_manifest
from ggufkit.model import open_model
from ggufkit.verify import (
    IDENTICAL,
    METADATA_AND_TENSOR_INFO,
    METADATA_ONLY,
    TENSOR_INFO_ONLY,
    WEIGHTS_CHANGED,
    compare,
    verify_against_manifest,
)
from tests.conftest import fingerprint_of, tiny_kv, tiny_tensors, write_gguf
from tests.test_fingerprint import flip_first_byte_of_tensor


def variant(tmp_path: Path, name: str, *, kv=None, tensors=None) -> Path:
    return write_gguf(
        tmp_path / f"{name}.gguf", kv=kv or tiny_kv(), tensors=tensors or tiny_tensors()
    )[0]


def test_identical(tmp_path: Path) -> None:
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b")
    r = compare(fingerprint_of(a), fingerprint_of(b))
    assert r.verdict == IDENTICAL
    assert r.ok()
    assert r.key_changes == []
    assert r.tensor_changes == []


def test_metadata_only(tmp_path: Path) -> None:
    kv = tiny_kv()
    kv["general.name"] = "other"
    kv["general.extra"] = "new"
    del kv["general.flag"]
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b", kv=kv)
    r = compare(fingerprint_of(a), fingerprint_of(b))
    assert r.verdict == METADATA_ONLY
    assert not r.ok()
    assert r.ok(allow_metadata=True)
    assert [(c.key, c.status) for c in r.key_changes or []] == [
        ("general.extra", "added"),
        ("general.flag", "removed"),
        ("general.name", "changed"),
    ]
    ignored = compare(
        fingerprint_of(a),
        fingerprint_of(b),
        ignore_keys=["general.name", "general.extra", "general.flag"],
    )
    assert ignored.verdict == IDENTICAL
    ignored_glob = compare(fingerprint_of(a), fingerprint_of(b), ignore_keys=["general.*"])
    assert ignored_glob.verdict == IDENTICAL


def test_rename_is_tensor_info_only(tmp_path: Path) -> None:
    tensors = tiny_tensors()
    tensors["blk.0.attn_q.renamed"] = tensors.pop("blk.0.attn_q.weight")
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b", tensors=tensors)
    r = compare(fingerprint_of(a), fingerprint_of(b))
    assert r.verdict == TENSOR_INFO_ONLY
    assert [(c.name, c.kind, c.detail) for c in r.tensor_changes or []] == [
        ("blk.0.attn_q.weight", "renamed", "-> blk.0.attn_q.renamed")
    ]
    assert r.weights_same is False
    assert any("new names" in n for n in r.notes)


def test_swapped_tensor_data_is_weights_changed(tmp_path: Path) -> None:
    """Two tensors with the same shape and type trade their data: every name, shape and type
    is unchanged, and so is the multiset of tensor bytes, but the model is not the same."""
    tensors = tiny_tensors()
    tensors["blk.0.attn_norm.weight"], tensors["output_norm.weight"] = (
        tensors["output_norm.weight"],
        tensors["blk.0.attn_norm.weight"],
    )
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b", tensors=tensors)
    fa, fb = fingerprint_of(a), fingerprint_of(b)
    assert fa.kv == fb.kv
    assert fa.tensor_info == fb.tensor_info
    assert fa.weights != fb.weights
    assert fa.overall != fb.overall
    r = compare(fa, fb)
    assert r.verdict == WEIGHTS_CHANGED
    assert not r.ok()
    assert {(c.name, c.kind) for c in r.tensor_changes or []} == {
        ("blk.0.attn_norm.weight", "data"),
        ("output_norm.weight", "data"),
    }


def test_added_and_removed_tensors(tmp_path: Path) -> None:
    tensors = tiny_tensors()
    del tensors["output_norm.weight"]
    tensors["extra.weight"] = tiny_tensors()["blk.0.attn_k.weight"]
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b", tensors=tensors)
    r = compare(fingerprint_of(a), fingerprint_of(b))
    # The multiset of weight bytes changed: one tensor's bytes left, a duplicate arrived.
    assert r.verdict == WEIGHTS_CHANGED
    kinds = {c.name: c.kind for c in r.tensor_changes or []}
    assert kinds == {"output_norm.weight": "removed", "extra.weight": "added"}


def test_weights_changed_names_the_tensor(tmp_path: Path) -> None:
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b")
    flip_first_byte_of_tensor(b, "blk.0.attn_k.weight")
    r = compare(fingerprint_of(a), fingerprint_of(b))
    assert r.verdict == WEIGHTS_CHANGED
    assert [(c.name, c.kind) for c in r.tensor_changes or []] == [("blk.0.attn_k.weight", "data")]


def test_retyped_tensor(tmp_path: Path) -> None:
    from gguf import GGMLQuantizationType as Q

    tensors = tiny_tensors()
    arr, _ = tensors["blk.0.attn_k.weight"]
    tensors["blk.0.attn_k.weight"] = (arr, Q.Q8_0)
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b", tensors=tensors)
    r = compare(fingerprint_of(a), fingerprint_of(b))
    assert r.verdict == WEIGHTS_CHANGED
    assert [(c.name, c.kind, c.detail) for c in r.tensor_changes or []] == [
        ("blk.0.attn_k.weight", "retyped", "Q4_0 -> Q8_0")
    ]


def test_metadata_and_tensor_info(tmp_path: Path) -> None:
    kv = tiny_kv()
    kv["general.name"] = "other"
    tensors = tiny_tensors()
    tensors["x"] = tensors.pop("output_norm.weight")
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b", kv=kv, tensors=tensors)
    assert compare(fingerprint_of(a), fingerprint_of(b)).verdict == METADATA_AND_TENSOR_INFO


def test_no_data_compare_notes_it(tmp_path: Path) -> None:
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b")
    r = compare(fingerprint_of(a, data=False), fingerprint_of(b))
    assert r.verdict == IDENTICAL
    assert not r.weights_compared
    assert r.notes
    assert not r.ok()
    assert r.ok(allow_unverified_weights=True)


def test_ignore_keys_needs_per_kv(tmp_path: Path) -> None:
    a = variant(tmp_path, "a")
    with open_model(a) as model:
        from ggufkit.fingerprint import fingerprint

        plain = fingerprint(model, per_kv=False)
    with pytest.raises(ValueError, match="per-key"):
        compare(plain, plain, ignore_keys=["x"])


def test_spec_mismatch_is_an_error(tmp_path: Path) -> None:
    a = fingerprint_of(variant(tmp_path, "a"))
    other = Fingerprint.from_dict({**a.to_dict(), "spec": "something/9"})
    with pytest.raises(ValueError, match="specs differ"):
        compare(a, other)


def test_verify_against_manifest(tmp_path: Path) -> None:
    a = variant(tmp_path, "a")
    with open_model(a) as model:
        manifest = build_manifest(model, file_sha256=True)
        r = verify_against_manifest(model, manifest, check_files=True)
    assert r.verdict == IDENTICAL
    assert r.files == "match"
    assert r.ok()

    flip_first_byte_of_tensor(a, "token_embd.weight")
    with open_model(a) as model:
        r = verify_against_manifest(model, manifest, check_files=True)
    assert r.verdict == WEIGHTS_CHANGED
    assert r.files == "mismatch"
    assert any("sha256 differs" in n for n in r.notes)


def test_file_hashes_missing_from_manifest_are_not_a_pass(tmp_path: Path) -> None:
    a = variant(tmp_path, "a")
    with open_model(a) as model:
        manifest = build_manifest(model)
        r = verify_against_manifest(model, manifest, check_files=True)
    assert r.verdict == IDENTICAL
    assert r.files == "not checked"
    assert not r.ok()
    assert any("no file hash" in n for n in r.notes)


def test_verify_against_no_data_manifest_skips_weights(tmp_path: Path) -> None:
    a = variant(tmp_path, "a")
    with open_model(a) as model:
        manifest = build_manifest(model, data=False)
    flip_first_byte_of_tensor(a, "token_embd.weight")
    with open_model(a) as model:
        r = verify_against_manifest(model, manifest)
    assert r.verdict == IDENTICAL
    assert not r.weights_compared
    assert not r.ok(), "unchecked weights must not pass"
    assert r.ok(allow_unverified_weights=True)

    manifest_path = tmp_path / "a.manifest.json"
    save_manifest(manifest_path, manifest)
    assert main(["verify", str(a), "--manifest", str(manifest_path), "-q"]) == 1
    assert main(["verify", str(a), "--manifest", str(manifest_path), "--no-data", "-q"]) == 0


def test_verify_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    kv = tiny_kv()
    kv["general.name"] = "other"
    a, b = variant(tmp_path, "a"), variant(tmp_path, "b", kv=kv)
    assert main(["verify", str(a), "--against", str(a), "-q"]) == 0
    assert main(["verify", str(a), "--against", str(b)]) == 1
    out = capsys.readouterr().out
    assert "verdict      metadata-only" in out
    assert "~ general.name" in out
    assert main(["verify", str(a), "--against", str(b), "--allow-metadata", "-q"]) == 0
    assert main(["verify", str(a), "--against", str(b), "--ignore-keys", "general.name", "-q"]) == 0

    manifest_path = tmp_path / "a.manifest.json"
    with open_model(a) as model:
        save_manifest(manifest_path, build_manifest(model))
    assert main(["verify", str(a), "--manifest", str(manifest_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == IDENTICAL
    assert report["sections"] == {"kv": True, "tensor_info": True, "weights": True}
    assert main(["verify", str(b), "--manifest", str(manifest_path), "-q"]) == 1
    assert main(["verify", str(a), "--manifest", str(tmp_path / "nope.json"), "-q"]) == 2
    assert main(["verify", str(a), "--against", str(b), "--file-sha256", "-q"]) == 2
