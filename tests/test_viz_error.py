from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType as Q

from ggufkit.cli import main
from ggufkit.model import open_model
from ggufkit.viz.errmap import error_map, make_image, pool, pool_factor, scale_top
from tests.conftest import det_floats, write_gguf

pytest.importorskip("PIL")


@pytest.fixture
def trio(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = det_floats((64, 128), 7)
    norm = det_floats((128,), 8)
    tensors = {"w": (data, Q.F32), "n": (norm, Q.F32)}
    src = write_gguf(tmp_path / "src.gguf", tensors=tensors)[0]
    q8 = write_gguf(tmp_path / "q8.gguf", tensors={"w": (data, Q.Q8_0), "n": (norm, Q.F32)})[0]
    q4 = write_gguf(tmp_path / "q4.gguf", tensors={"w": (data, Q.Q4_0), "n": (norm, Q.F32)})[0]
    return src, q8, q4


def em_of(a: Path, b: Path, name: str = "w", **kw):
    with open_model(a) as ma, open_model(b) as mb:
        return error_map(ma.tensor(name), mb.tensor(name), ma.endianess, mb.endianess, **kw)


def test_map_shape_and_stats(trio) -> None:
    src, q8, _ = trio
    em = em_of(src, q8)
    assert em.values.shape == (64, 128)
    assert em.pool_factor == 1
    assert em.source_shape == (64, 128)
    assert em.stats.a_type == "F32"
    assert em.stats.b_type == "Q8_0"
    assert 0 < (em.stats.rel_l2 or 0) < 0.05
    assert float(em.values.max()) == pytest.approx(em.stats.max_abs, rel=1e-5)


def test_identical_tensors_give_zero_map(trio) -> None:
    src, _, _ = trio
    em = em_of(src, src)
    assert not em.values.any()
    assert scale_top(em.values, 99.9) == 0.0
    img = make_image(em, 0.0, caption=False)
    assert img.size == (128, 64)


def test_q4_error_exceeds_q8(trio) -> None:
    src, q8, q4 = trio
    e8, e4 = em_of(src, q8), em_of(src, q4)
    assert float(e8.values.mean()) < float(e4.values.mean())
    shared = max(scale_top(e8.values, 99.9), scale_top(e4.values, 99.9))
    i8 = np.asarray(make_image(e8, shared, caption=False)).astype(float).mean()
    i4 = np.asarray(make_image(e4, shared, caption=False)).astype(float).mean()
    assert i8 < i4  # darker image for the smaller error on the same scale


def test_pooling_budget_and_max_keeps_outliers(tmp_path: Path) -> None:
    data = det_floats((64, 64), 3)
    spiked = data.copy()
    spiked[10, 10] += 5.0
    a = write_gguf(tmp_path / "a.gguf", tensors={"w": (data, Q.F32)})[0]
    b = write_gguf(tmp_path / "b.gguf", tensors={"w": (spiked, Q.F32)})[0]
    em_max = em_of(a, b, max_pixels=256, how="max")
    em_mean = em_of(a, b, max_pixels=256, how="mean")
    assert em_max.values.size <= 256
    assert em_max.pool_factor == pool_factor(64, 64, 256) == 4
    assert float(em_max.values.max()) == pytest.approx(5.0, rel=1e-4)
    assert float(em_mean.values.max()) == pytest.approx(5.0 / 16, rel=1e-3)


def test_chunking_matches_unchunked(trio) -> None:
    src, _, q4 = trio
    whole = em_of(src, q4, max_pixels=512)
    chunked = em_of(src, q4, max_pixels=512, chunk_bytes=2048)
    assert np.allclose(whole.values, chunked.values)
    assert whole.stats.rmse == pytest.approx(chunked.stats.rmse)


def test_signed_metric_and_pool_keep_sign() -> None:
    block = np.array([[1.0, -3.0], [0.5, 2.0]], dtype=np.float32)
    assert pool(block, 2, "max")[0, 0] == pytest.approx(-3.0)
    assert pool(block, 2, "mean")[0, 0] == pytest.approx(0.125)


def test_signed_and_rel_render(trio) -> None:
    src, _, q4 = trio
    signed = em_of(src, q4, metric="signed")
    assert float(signed.values.min()) < 0 < float(signed.values.max())
    rel = em_of(src, q4, metric="rel")
    assert float(rel.values.min()) >= 0
    for em in (signed, rel):
        img = make_image(em, scale_top(em.values, 99.0))
        assert img.width >= 128


def test_shape_mismatch_is_an_error(tmp_path: Path) -> None:
    a = write_gguf(tmp_path / "a.gguf", tensors={"w": (det_floats((4, 32)), Q.F32)})[0]
    b = write_gguf(tmp_path / "b.gguf", tensors={"w": (det_floats((8, 32)), Q.F32)})[0]
    with pytest.raises(ValueError, match="shape differs"):
        em_of(a, b)


def test_three_dimensional_tensor_is_flattened(tmp_path: Path) -> None:
    data = det_floats((2, 8, 32), 4)
    a = write_gguf(tmp_path / "a.gguf", tensors={"e": (data, Q.F32)})[0]
    b = write_gguf(tmp_path / "b.gguf", tensors={"e": (data * np.float32(1.1), Q.F32)})[0]
    em = em_of(a, b, name="e")
    assert em.values.shape == (16, 32)


def test_caption_adds_height(trio) -> None:
    src, q8, _ = trio
    em = em_of(src, q8)
    bare = make_image(em, 0.01, caption=False)
    captioned = make_image(em, 0.01)
    assert captioned.height > bare.height
    assert make_image(em, 0.01, caption=False, scale=2.0).size == (256, 128)


def test_cli(trio, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src, _, q4 = trio
    out = tmp_path / "imgs" / "err.png"
    assert main(["viz", "error", str(src), str(q4), "w", "-o", str(out)]) == 0
    assert out.is_file()
    assert "F32 -> Q4_0" in capsys.readouterr().out
    assert main(["viz", "error", str(src), str(q4), "w", "-o", str(out), "-q"]) == 2  # exists
    multi = tmp_path / "m.png"
    rc = main(
        [
            "viz",
            "error",
            str(src),
            str(q4),
            "*",
            "--match-glob",
            "--match-1d",
            "--shared-scale",
            "-o",
            str(multi),
            "-q",
        ]
    )
    assert rc == 0
    assert sorted(p.name for p in tmp_path.glob("*.m.png")) == ["n.m.png", "w.m.png"]
    rc = main(
        [
            "viz",
            "error",
            str(src),
            str(q4),
            "*",
            "--match-glob",
            "-o",
            str(tmp_path / "g.png"),
            "-q",
        ]
    )
    assert rc == 0
    # Only "w" matches once the 1-D "n" is skipped, so the exact -o path is used.
    assert (tmp_path / "g.png").is_file()
    assert not list(tmp_path.glob("*.g.png"))
    assert (
        main(["viz", "error", str(src), str(q4), "missing", "-o", str(tmp_path / "x.png"), "-q"])
        == 1
    )
    assert (
        main(
            [
                "viz",
                "error",
                str(src),
                str(q4),
                "w",
                "-o",
                str(tmp_path / "y.png"),
                "--percentile",
                "0",
            ]
        )
        == 2
    )
