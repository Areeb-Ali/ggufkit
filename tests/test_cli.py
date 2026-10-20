from __future__ import annotations

import json
from pathlib import Path

import pytest

from ggufkit import __version__
from ggufkit.cli import main
from tests.conftest import fingerprint_of


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "fingerprint" in capsys.readouterr().out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_fingerprint_json_matches_library(
    tiny_gguf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["fingerprint", str(tiny_gguf), "--json", "--per-kv"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == fingerprint_of(tiny_gguf).to_dict()


def test_fingerprint_text_output(tiny_gguf: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fingerprint", str(tiny_gguf)]) == 0
    out = capsys.readouterr().out
    fp = fingerprint_of(tiny_gguf)
    assert fp.overall is not None
    assert fp.overall in out
    assert "TENSOR" in out
    assert "blk.0.attn_q.weight" in out


def test_fingerprint_no_data_and_output_file(
    tiny_gguf: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_file = tmp_path / "fp.json"
    assert (
        main(["fingerprint", str(tiny_gguf), "--no-data", "--no-per-tensor", "-o", str(out_file)])
        == 0
    )
    text = capsys.readouterr().out
    assert "weights" not in text
    saved = json.loads(out_file.read_text())
    assert saved["weights"] is None
    assert saved["overall"] is None
    assert "tensors" not in saved


def test_missing_file_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fingerprint", str(tmp_path / "nope.gguf")]) == 2
    assert "error:" in capsys.readouterr().err


def test_stitch_cli(tiny_gguf: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.gguf"
    assert (
        main(
            [
                "stitch",
                "--metadata",
                str(tiny_gguf),
                "--tensors",
                str(tiny_gguf),
                "-o",
                str(out),
                "-q",
            ]
        )
        == 0
    )
    assert fingerprint_of(out).overall == fingerprint_of(tiny_gguf).overall
    assert main(["stitch", "--metadata", str(tiny_gguf), "-o", str(out), "-q"]) == 2  # exists
    assert main(["stitch", "--metadata", str(tiny_gguf), "-o", str(out), "-q", "--force"]) == 0
