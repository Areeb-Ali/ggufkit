from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ggufkit.model import ModelError, open_model, shard_paths
from tests.conftest import tiny_kv, tiny_tensors, write_gguf


def test_plain_file_is_its_own_shard_list(tmp_path: Path) -> None:
    p = tmp_path / "model.gguf"
    assert shard_paths(p) == [p]


def test_shard_names_are_expanded(tmp_path: Path) -> None:
    shards = write_gguf(
        tmp_path / "m.gguf", kv=tiny_kv(), tensors=tiny_tensors(), split_max_tensors=4
    )
    assert len(shards) == 2
    assert shard_paths(shards[1]) == shards


def test_missing_shard_is_an_error(tmp_path: Path) -> None:
    shards = write_gguf(
        tmp_path / "m.gguf", kv=tiny_kv(), tensors=tiny_tensors(), split_max_tensors=4
    )
    shards[1].unlink()
    with pytest.raises(ModelError, match="missing"):
        shard_paths(shards[0])


def test_declared_shard_count_must_match(tmp_path: Path) -> None:
    shards = write_gguf(
        tmp_path / "m.gguf", kv=tiny_kv(), tensors=tiny_tensors(), split_max_tensors=4
    )
    lonely = tmp_path / "lonely.gguf"
    shutil.copy(shards[0], lonely)
    with pytest.raises(ModelError, match="declares 2 shards"):
        open_model(lonely)


def test_model_properties(tiny_gguf: Path) -> None:
    with open_model(tiny_gguf) as model:
        assert model.architecture == "llama"
        assert model.version == 3
        assert model.alignment == 32
        assert model.n_shards == 1
        assert list(model.tensor_names()) == list(tiny_tensors())
        with pytest.raises(KeyError):
            model.tensor("nope")
