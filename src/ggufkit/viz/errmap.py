"""Quantization-error heatmaps: where, inside a tensor, a quant departs from its source.

``ggufkit viz error SOURCE.gguf QUANT.gguf blk.0.attn_q.weight -o err.png`` decodes the two
tensors chunk by chunk, computes the per-weight error, pools it down to a pixel budget while it
goes (so a 4096 x 14336 tensor never needs a full-size error array), and renders it with a
perceptually uniform palette. Quantization error is blocky; ``--pool max`` keeps the worst
block of each pixel visible instead of averaging it away.
"""

from __future__ import annotations

import fnmatch
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ggufkit.diff import (
    NUMERIC_CHUNK_BYTES,
    ErrorAccumulator,
    NumericStats,
    comparable,
    iter_decoded_chunks,
)
from ggufkit.model import open_model
from ggufkit.viz import palette
from ggufkit.viz.render import output_path_for

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable

    from gguf import GGUFEndian
    from gguf.gguf_reader import ReaderTensor
    from PIL import Image

METRICS = ("abs", "rel", "signed")
POOLS = ("max", "mean")
DEFAULT_MAX_PIXELS = 4_000_000
REL_EPS = 1e-6


@dataclass
class ErrorMap:
    name: str
    values: np.ndarray  # 2-D, already pooled
    stats: NumericStats
    metric: str
    pool_factor: int
    source_shape: tuple[int, int]


def pool_factor(rows: int, cols: int, max_pixels: int) -> int:
    if max_pixels <= 0 or rows * cols <= max_pixels:
        return 1
    return int(np.ceil(np.sqrt(rows * cols / max_pixels)))


def pool(block: np.ndarray, factor: int, how: str) -> np.ndarray:
    """Pool a 2-D array by ``factor`` in both directions, trimming ragged edges.

    ``max`` keeps, per output cell, the input value with the largest magnitude (sign kept),
    so it works for signed errors too.
    """
    if factor == 1:
        return block
    rows, cols = block.shape
    r, c = (rows // factor) * factor, (cols // factor) * factor
    if r == 0 or c == 0:
        return np.zeros((0, max(cols // factor, 0)), dtype=block.dtype)
    cells = (
        block[:r, :c]
        .reshape(r // factor, factor, c // factor, factor)
        .transpose(0, 2, 1, 3)
        .reshape(r // factor, c // factor, factor * factor)
    )
    if how == "mean":
        return cells.mean(axis=-1)
    if how == "max":
        idx = np.abs(cells).argmax(axis=-1)
        return np.take_along_axis(cells, idx[..., None], axis=-1)[..., 0]
    msg = f"Unknown pooling {how!r}"
    raise ValueError(msg)


def _error(x: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
    d = y.astype(np.float32, copy=False) - x.astype(np.float32, copy=False)
    if metric == "signed":
        return d
    if metric == "abs":
        return np.abs(d)
    if metric == "rel":
        return np.abs(d) / (np.abs(x).astype(np.float32, copy=False) + np.float32(REL_EPS))
    msg = f"Unknown metric {metric!r}"
    raise ValueError(msg)


def error_map(
    ta: ReaderTensor,
    tb: ReaderTensor,
    endian_a: GGUFEndian,
    endian_b: GGUFEndian,
    *,
    metric: str = "abs",
    how: str = "max",
    max_pixels: int = DEFAULT_MAX_PIXELS,
    chunk_bytes: int = NUMERIC_CHUNK_BYTES,
) -> ErrorMap:
    """Compute the pooled error map of ``tb`` against ``ta`` and their error statistics."""
    why = comparable(ta, tb)
    if why is not None:
        raise ValueError(f"{ta.name}: {why}")
    cols = int(ta.shape[0]) if len(ta.shape) else 1
    rows = int(ta.n_elements) // max(cols, 1)
    factor = pool_factor(rows, cols, max_pixels)

    acc = ErrorAccumulator()
    pooled: list[np.ndarray] = []
    carry: np.ndarray | None = None
    for x, y in iter_decoded_chunks(ta, tb, endian_a, endian_b, chunk_bytes=chunk_bytes):
        acc.add(x, y)
        e = _error(x, y, metric)
        if carry is not None and carry.size:
            e = np.vstack([carry, e])
        usable = (e.shape[0] // factor) * factor
        if usable:
            pooled.append(pool(e[:usable], factor, how))
        carry = e[usable:]
    if carry is not None and carry.size and not pooled:
        # Fewer rows than one pooling step: keep what there is, pooled along columns only.
        pooled.append(pool(np.repeat(carry, factor, axis=0)[:factor], factor, how))
    values = np.vstack(pooled) if pooled else np.zeros((0, 0), dtype=np.float32)
    stats = acc.finish(
        NumericStats(ta.name, ta.tensor_type.name, tb.tensor_type.name, n=int(ta.n_elements))
    )
    return ErrorMap(
        ta.name, values.astype(np.float32, copy=False), stats, metric, factor, (rows, cols)
    )


def scale_top(values: np.ndarray, percentile: float) -> float:
    """Return the magnitude mapped to the top of the palette: a high percentile, not the max."""
    mags = np.abs(values[np.isfinite(values)])
    if mags.size == 0:
        return 0.0
    return float(np.percentile(mags, percentile))


def colourize(em: ErrorMap, top: float) -> np.ndarray:
    if em.metric == "signed":
        t = 0.5 if top <= 0 else (np.clip(em.values / top, -1.0, 1.0) + 1.0) / 2.0
        return palette.apply(palette.DIVERGING, np.broadcast_to(t, em.values.shape))
    t = np.zeros_like(em.values) if top <= 0 else np.clip(em.values / top, 0.0, 1.0)
    return palette.apply(palette.SEQUENTIAL, t)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4g}"


def caption_lines(em: ErrorMap, top: float) -> list[str]:
    s = em.stats
    rows, cols = em.source_shape
    pooled = f", {em.pool_factor}x{em.pool_factor} pooled" if em.pool_factor > 1 else ""
    return [
        f"{em.name}  {s.a_type} -> {s.b_type}  ({rows}x{cols}{pooled})",
        f"rmse {_fmt(s.rmse)}  rel_l2 {_fmt(s.rel_l2)}  cosine {_fmt(s.cosine)}  max {_fmt(s.max_abs)}",
        f"{em.metric} error, palette top {top:.3g}",
    ]


def make_image(
    em: ErrorMap,
    top: float,
    *,
    scale: float = 1.0,
    caption: bool = True,
) -> Image.Image:
    from PIL import Image, ImageDraw, ImageFont

    rgb = colourize(em, top)
    if rgb.size == 0:
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    img = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    if scale != 1.0:
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            resample=Image.Resampling.NEAREST,
        )
    if not caption:
        return img

    font = ImageFont.load_default()
    lines = caption_lines(em, top)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    text_w = max(int(probe.textlength(line, font=font)) for line in lines)
    line_h = 14
    bar_w, bar_h, pad = 256, 10, 6
    width = max(img.width, text_w + 2 * pad, bar_w + 2 * pad)
    height = img.height + pad + len(lines) * line_h + pad + bar_h + line_h + pad
    out = Image.new("RGB", (width, height), (255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    y = img.height + pad
    for line in lines:
        draw.text((pad, y), line, fill=(20, 20, 20), font=font)
        y += line_h
    y += pad // 2
    lut = palette.DIVERGING if em.metric == "signed" else palette.SEQUENTIAL
    bar = np.repeat(lut[np.linspace(0, len(lut) - 1, bar_w).astype(int)][None, :, :], bar_h, axis=0)
    out.paste(Image.fromarray(np.ascontiguousarray(bar), "RGB"), (pad, y))
    lo = f"{-top:.3g}" if em.metric == "signed" else "0"
    hi = f"{top:.3g}"
    draw.text((pad, y + bar_h + 1), lo, fill=(20, 20, 20), font=font)
    draw.text(
        (pad + bar_w - int(draw.textlength(hi, font=font)), y + bar_h + 1),
        hi,
        fill=(20, 20, 20),
        font=font,
    )
    return out


def save_png(img: Image.Image, target: Path) -> None:
    """Save as a 256-colour palette PNG.

    The heatmap uses at most 256 lookup-table colours, so it is stored exactly; only the
    anti-aliased caption text is approximated. Noisy error maps shrink about threefold.
    """
    from PIL import Image

    img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256).save(target, optimize=True)


def select(
    names: Iterable[str],
    patterns: list[str],
    *,
    glob: bool,
    regex: bool,
) -> list[str]:
    names = list(names)
    if glob:
        return [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in patterns)]
    if regex:
        compiled = [re.compile(p) for p in patterns]
        return [n for n in names if any(r.search(n) for r in compiled)]
    present = set(names)
    return [p for p in patterns if p in present]


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "error",
        help="render the per-weight error between two models' tensors as heatmaps",
        description=(
            "Decode a tensor from two GGUF files (typically a source and its quantization) and "
            "render the per-weight error as a heatmap, with a caption giving rmse, relative L2 "
            "and cosine similarity. Large tensors are pooled down to --max-pixels as they are "
            "decoded."
        ),
    )
    p.add_argument("a", type=Path, help="reference GGUF file (e.g. the BF16 source)")
    p.add_argument("b", type=Path, help="GGUF file to compare (e.g. the Q4_K_M quant)")
    p.add_argument("tensor", nargs="+", help="tensor name(s), or one pattern with --match-*")
    p.add_argument(
        "-o", "--output", type=Path, required=True, help="PNG to write (prefixed per tensor)"
    )
    match = p.add_mutually_exclusive_group()
    match.add_argument("--match-glob", action="store_true", help="treat the name as a glob")
    match.add_argument("--match-regex", action="store_true", help="treat the name as a regex")
    p.add_argument("--match-1d", action="store_true", help="also render 1-D tensors when matching")
    p.add_argument(
        "--metric", choices=METRICS, default="abs", help="abs, rel or signed (default: abs)"
    )
    p.add_argument(
        "--pool", choices=POOLS, default="max", help="pooling for large tensors (default: max)"
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        metavar="N",
        help="pool tensors larger than N pixels (default: %(default)s; 0 = never)",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="error percentile mapped to the top colour (default: %(default)s)",
    )
    p.add_argument(
        "--shared-scale",
        action="store_true",
        help="use one colour scale for every rendered tensor so images are comparable",
    )
    p.add_argument("--scale", type=float, default=1.0, help="resize factor (nearest-neighbour)")
    p.add_argument("--no-caption", action="store_true", help="omit the caption and colour bar")
    p.add_argument("--force", action="store_true", help="overwrite existing images")
    p.add_argument("-q", "--quiet", action="store_true", help="do not report progress")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if (args.match_glob or args.match_regex) and len(args.tensor) != 1:
        print("error: give exactly one pattern with --match-glob or --match-regex", file=sys.stderr)
        return 2
    if not 0 < args.percentile <= 100:
        print("error: --percentile must be in (0, 100]", file=sys.stderr)
        return 2
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("error: rendering needs Pillow: pip install 'ggufkit[viz]'", file=sys.stderr)
        return 2
    say: Callable[[str], None] = (lambda _m: None) if args.quiet else print

    maps: list[ErrorMap] = []
    with open_model(args.a) as ma, open_model(args.b) as mb:
        b_names = set(mb.tensor_names())
        names = [
            n
            for n in select(
                ma.tensor_names(), args.tensor, glob=args.match_glob, regex=args.match_regex
            )
            if n in b_names
        ]
        if not (args.match_glob or args.match_regex):
            for missing in [n for n in args.tensor if n not in names]:
                say(f"! Skipping {missing!r}: not in both models")
        for name in names:
            ta, tb = ma.tensor(name), mb.tensor(name)
            if len(ta.shape) == 1 and (args.match_glob or args.match_regex) and not args.match_1d:
                continue
            why = comparable(ta, tb)
            if why is not None:
                say(f"! Skipping {name!r}: {why}")
                continue
            maps.append(
                error_map(
                    ta,
                    tb,
                    ma.endianess,
                    mb.endianess,
                    metric=args.metric,
                    how=args.pool,
                    max_pixels=args.max_pixels,
                )
            )
    if not maps:
        print("error: no comparable tensors matched", file=sys.stderr)
        return 1

    shared = max(scale_top(m.values, args.percentile) for m in maps) if args.shared_scale else None
    single = len(maps) == 1
    for em in maps:
        target = output_path_for(args.output, em.name, single)
        if target.exists() and not args.force:
            msg = f"{target} already exists (use --force to overwrite)"
            raise FileExistsError(msg)
        top = shared if shared is not None else scale_top(em.values, args.percentile)
        img = make_image(em, top, scale=args.scale, caption=not args.no_caption)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_png(img, target)
        s = em.stats
        say(f"* {em.name}: {s.a_type} -> {s.b_type}, rel_l2 {_fmt(s.rel_l2)} -> {target}")
    return 0
