from __future__ import annotations

import builtins
import json
from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType as Q

from ggufkit.cli import main
from ggufkit.model import open_model
from ggufkit.viz.layers import bits_per_weight, build_layer_map, format_text, make_image, role_of
from tests.conftest import det_floats, tiny_kv, write_gguf


@pytest.fixture
def mixed(tmp_path: Path) -> Path:
    tensors = {"token_embd.weight": (det_floats((8, 64)), Q.F16)}
    for b in range(3):
        tensors[f"blk.{b}.attn_norm.weight"] = (det_floats((64,)), Q.F32)
        tensors[f"blk.{b}.attn_q.weight"] = (det_floats((64, 64)), Q.Q4_0)
        tensors[f"blk.{b}.attn_k.weight"] = (det_floats((32, 64)), Q.Q8_0 if b == 0 else Q.Q4_0)
        tensors[f"blk.{b}.ffn_down.weight"] = (det_floats((8, 256)), Q.Q6_K if b == 2 else Q.Q4_K)
    tensors["blk.1.ffn_up_exps.weight"] = (det_floats((2, 8, 32)), Q.F32)
    tensors["blk.0.custom_thing.weight"] = (det_floats((4, 32)), Q.F32)
    tensors["output_norm.weight"] = (det_floats((64,)), Q.F32)
    return write_gguf(tmp_path / "mixed.gguf", kv=tiny_kv(), tensors=tensors)[0]


def test_role_parsing() -> None:
    assert role_of("blk.12.attn_q.weight") == (12, "attn_q")
    assert role_of("blk.3.ffn_gate_exps.weight") == (3, "ffn_gate_exps")
    assert role_of("blk.0.attn_q.bias") == (0, "attn_q.bias")
    assert role_of("token_embd.weight") == (None, "token_embd")
    assert role_of("rope_freqs") == (None, "rope_freqs")


def test_bits_per_weight() -> None:
    assert bits_per_weight(Q.F32) == 32
    assert bits_per_weight(Q.F16) == 16
    assert bits_per_weight(Q.Q8_0) == pytest.approx(8.5)
    assert bits_per_weight(Q.Q4_K) == pytest.approx(4.5)
    assert bits_per_weight(Q.Q6_K) == pytest.approx(6.5625)
    assert bits_per_weight(Q.IQ2_XXS) == pytest.approx(2.0625)


def test_layer_map(mixed: Path) -> None:
    with open_model(mixed) as model:
        lm = build_layer_map(model)
    assert lm.blocks == [0, 1, 2]
    assert lm.roles == ["attn_norm", "attn_q", "attn_k", "ffn_down", "ffn_up_exps", "custom_thing"]
    assert lm.global_roles == ["token_embd", "output_norm"]
    assert lm.cells[(0, "attn_k")] == "Q8_0"
    assert lm.cells[(1, "attn_k")] == "Q4_0"
    assert lm.cells[(2, "ffn_down")] == "Q6_K"
    assert (0, "ffn_up_exps") not in lm.cells
    assert lm.type_order() == ["Q4_0", "Q4_K", "Q6_K", "Q8_0", "F16", "F32"]
    d = lm.to_dict()
    assert d["blocks"][1]["cells"]["ffn_up_exps"] == "F32"
    assert d["blocks"][0]["cells"]["ffn_up_exps"] is None
    assert d["global"] == {"token_embd": "F16", "output_norm": "F32"}
    assert d["types"]["Q4_0"]["tensors"] == 5


def test_text_table(mixed: Path) -> None:
    with open_model(mixed) as model:
        lines = format_text(build_layer_map(model))
    assert lines[0].split() == [
        "block",
        "attn_norm",
        "attn_q",
        "attn_k",
        "ffn_down",
        "ffn_up_exps",
        "custom_thing",
    ]
    assert lines[1].split()[:5] == ["0", "F32", "Q4_0", "Q8_0", "Q4_K"]
    assert any(line.startswith("global: token_embd=F16") for line in lines)
    assert any(line.startswith("Q4_0") and "bpw" in line for line in lines)


def test_image(mixed: Path) -> None:
    pytest.importorskip("PIL")
    with open_model(mixed) as model:
        lm = build_layer_map(model)
    img = make_image(lm, title="mixed.gguf")
    assert img.width > 6 * 18
    assert img.height > 3 * 12
    pixels = np.asarray(img).reshape(-1, 3)
    assert (
        len({tuple(p) for p in pixels[:: max(1, len(pixels) // 5000)]}) > 6
    )  # several colours drawn


def test_cli_text_json_and_png(
    mixed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["viz", "layers", str(mixed)]) == 0
    assert capsys.readouterr().out.startswith("block")
    assert main(["viz", "layers", str(mixed), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["roles"][1] == "attn_q"
    pytest.importorskip("PIL")
    out = tmp_path / "layers.png"
    assert main(["viz", "layers", str(mixed), "-o", str(out)]) == 0
    assert out.is_file()
    assert main(["viz", "layers", str(mixed), "-o", str(out)]) == 2  # exists
    assert main(["viz", "layers", str(mixed), "-o", str(out), "--force", "--cell", "10x6"]) == 0
    assert main(["viz", "layers", str(mixed), "--cell", "nonsense"]) == 2


def test_text_works_without_pillow(
    mixed: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = builtins.__import__

    def no_pil(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pil)
    assert main(["viz", "layers", str(mixed), "--text"]) == 0
    assert "attn_q" in capsys.readouterr().out


@pytest.fixture
def other_mix(tmp_path: Path) -> Path:
    tensors = {"token_embd.weight": (det_floats((8, 64)), Q.Q8_0)}
    for b in range(3):
        tensors[f"blk.{b}.attn_norm.weight"] = (det_floats((64,)), Q.F32)
        tensors[f"blk.{b}.attn_q.weight"] = (det_floats((64, 256)), Q.IQ4_XS)
    return write_gguf(tmp_path / "other.gguf", kv=tiny_kv(), tensors=tensors)[0]


def test_shared_colours_are_consistent(mixed: Path, other_mix: Path) -> None:
    from ggufkit.viz.layers import shared_colours

    with open_model(mixed) as a, open_model(other_mix) as b:
        la, lb = build_layer_map(a), build_layer_map(b)
    shared = shared_colours([la, lb])
    assert list(shared) == ["IQ4_XS", "Q4_0", "Q4_K", "Q6_K", "Q8_0", "F16", "F32"]
    assert len(set(shared.values())) == len(shared)
    alone = shared_colours([la])
    assert "IQ4_XS" not in alone
    assert alone["Q4_0"] != shared["Q4_0"]  # the union shifts the palette, which is the point


def test_comparison_image(mixed: Path, other_mix: Path) -> None:
    pytest.importorskip("PIL")
    from ggufkit.viz.layers import make_comparison

    with open_model(mixed) as a, open_model(other_mix) as b:
        la, lb = build_layer_map(a), build_layer_map(b)
    single = make_image(la, title="mixed")
    both = make_comparison([la, lb], ["mixed", "other"])
    assert both.width > single.width
    with pytest.raises(ValueError, match="shorter"):
        make_comparison([la, lb], ["only one title"])


def test_cli_with_two_models(
    mixed: Path, other_mix: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["viz", "layers", str(mixed), str(other_mix)]) == 0
    out = capsys.readouterr().out
    assert "== mixed.gguf" in out
    assert "== other.gguf" in out
    assert main(["viz", "layers", str(mixed), str(other_mix), "--json"]) == 0
    assert [m["model"] for m in json.loads(capsys.readouterr().out)] == ["mixed.gguf", "other.gguf"]
    assert main(["viz", "layers", str(mixed), str(other_mix), "--title", "only one"]) == 2
    pytest.importorskip("PIL")
    out_png = tmp_path / "compare.png"
    assert (
        main(
            [
                "viz",
                "layers",
                str(mixed),
                str(other_mix),
                "-o",
                str(out_png),
                "--title",
                "A",
                "--title",
                "B",
            ]
        )
        == 0
    )
    assert out_png.is_file()
