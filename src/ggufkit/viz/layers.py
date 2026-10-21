"""A map of which quantization type each tensor got, block by block.

Modern GGUF quants are mixtures: llama.cpp's ``Q4_K_M`` keeps some tensors at ``Q6_K``, and
importance-matrix "dynamic" quants vary the type per layer. ``ggufkit viz layers model.gguf``
draws that mixture as a grid, one row per transformer block and one column per tensor role,
coloured by bits per weight, so the recipe behind a file is visible at a glance.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gguf.constants import GGML_QUANT_SIZES

from ggufkit.model import open_model
from ggufkit.viz import palette

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

    from gguf import GGMLQuantizationType
    from PIL import Image

    from ggufkit.model import Model

BLOCK_RE = re.compile(r"^blk\.(?P<block>\d+)\.(?P<role>.+)$")

#: Usual llama.cpp tensor roles, in the order they appear in a transformer block.
CANONICAL_ROLES = (
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_q_norm",
    "attn_k_norm",
    "attn_output",
    "post_attention_norm",
    "ffn_norm",
    "ffn_gate_inp",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
    "post_ffw_norm",
)

GLOBAL_ROW = "global"


def bits_per_weight(qtype: GGMLQuantizationType) -> float:
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    return type_size * 8 / block_size


def role_of(name: str) -> tuple[int | None, str]:
    """Split ``blk.12.attn_q.weight`` into ``(12, "attn_q")``; non-block tensors get ``None``."""
    base = name.removesuffix(".weight")
    m = BLOCK_RE.match(base)
    if m is None:
        return None, base
    return int(m["block"]), m["role"]


@dataclass
class LayerMap:
    roles: list[str]
    blocks: list[int]
    cells: dict[tuple[int | None, str], str] = field(default_factory=dict)
    types: dict[str, float] = field(default_factory=dict)  # type name -> bits per weight
    counts: Counter[str] = field(default_factory=Counter)
    bytes_by_type: Counter[str] = field(default_factory=Counter)
    global_roles: list[str] = field(default_factory=list)

    def type_order(self) -> list[str]:
        return sorted(self.types, key=lambda t: (self.types[t], t))

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "blocks": [
                {"block": b, "cells": {r: self.cells.get((b, r)) for r in self.roles}}
                for b in self.blocks
            ],
            "global": {r: self.cells.get((None, r)) for r in self.global_roles},
            "types": {
                t: {
                    "bits_per_weight": self.types[t],
                    "tensors": self.counts[t],
                    "bytes": self.bytes_by_type[t],
                }
                for t in self.type_order()
            },
        }


def build_layer_map(model: Model) -> LayerMap:
    found_roles: list[str] = []
    global_roles: list[str] = []
    blocks: set[int] = set()
    lm = LayerMap(roles=[], blocks=[])
    for t in model.tensors:
        block, role = role_of(t.name)
        tname = t.tensor_type.name
        lm.cells[(block, role)] = tname
        lm.types[tname] = bits_per_weight(t.tensor_type)
        lm.counts[tname] += 1
        lm.bytes_by_type[tname] += int(t.n_bytes)
        if block is None:
            if role not in global_roles:
                global_roles.append(role)
        else:
            blocks.add(block)
            if role not in found_roles:
                found_roles.append(role)
    canonical = [r for r in CANONICAL_ROLES if r in found_roles]
    lm.roles = canonical + [r for r in found_roles if r not in canonical]
    lm.blocks = sorted(blocks)
    lm.global_roles = global_roles
    return lm


def format_text(lm: LayerMap) -> list[str]:
    if not lm.roles and not lm.global_roles:
        return ["(no tensors)"]
    width = max([len(t) for t in lm.types] + [3])
    col_w = {r: max(len(r), width) for r in lm.roles}
    lines: list[str] = []
    if lm.roles:
        lines.append(("block  " + "  ".join(r.ljust(col_w[r]) for r in lm.roles)).rstrip())
        for b in lm.blocks:
            cells = "  ".join((lm.cells.get((b, r)) or "-").ljust(col_w[r]) for r in lm.roles)
            lines.append(f"{b:<5}  {cells}".rstrip())
    if lm.global_roles:
        lines.append("")
        lines.append(
            f"{GLOBAL_ROW}: " + ", ".join(f"{r}={lm.cells[(None, r)]}" for r in lm.global_roles)
        )
    lines.append("")
    total = sum(lm.bytes_by_type.values()) or 1
    for t in lm.type_order():
        share = 100 * lm.bytes_by_type[t] / total
        lines.append(
            f"{t:<{width}}  {lm.types[t]:6.3f} bpw  {lm.counts[t]:5} tensors  {share:5.1f}% of bytes"
        )
    return lines


Colour = tuple[int, int, int]


def shared_colours(maps: Sequence[LayerMap]) -> dict[str, Colour]:
    """One colour per type across all ``maps``, ordered by bits per weight.

    Using the union of types keeps the same type the same colour in every panel of a
    comparison.
    """
    bpw: dict[str, float] = {}
    for lm in maps:
        bpw.update(lm.types)
    order = sorted(bpw, key=lambda t: (bpw[t], t))
    return dict(zip(order, palette.categorical(len(order)), strict=True))


def _font_and_width() -> tuple[Any, Callable[[str], int]]:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    return font, lambda s: int(probe.textlength(s, font=font))


_PAD, _LINE_H, _SWATCH = 8, 14, 14
_INK, _FAINT, _EMPTY = (20, 20, 20), (110, 110, 110), (225, 225, 225)


def _panel(
    lm: LayerMap,
    colours: dict[str, Colour],
    *,
    cell_w: int,
    cell_h: int,
    title: str | None,
) -> Image.Image:
    """Draw the block grid with its title and the non-block tensors, without a legend."""
    from PIL import Image, ImageDraw

    font, text_w = _font_and_width()
    pad, line_h, swatch = _PAD, _LINE_H, _SWATCH
    row_labels = [f"blk.{b}" for b in lm.blocks]
    label_w = max([text_w(s) for s in row_labels] + [10]) + pad
    header_h = (max([text_w(r) for r in lm.roles] + [0]) + pad) if lm.roles else 0
    grid_w = len(lm.roles) * cell_w
    grid_h = len(row_labels) * cell_h
    global_lines = [f"{r}  {lm.cells[(None, r)]}" for r in lm.global_roles]
    title_h = line_h + 2 if title else 0
    globals_h = line_h + len(global_lines) * line_h if global_lines else 0
    width = max(
        pad + label_w + grid_w + pad,
        pad + swatch + 6 + max([text_w(s) for s in global_lines] + [text_w(title or "")]) + pad,
    )
    height = pad + title_h + header_h + grid_h + pad + globals_h + pad
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = pad
    if title:
        draw.text((pad, y), title, fill=_INK, font=font)
        y += title_h
    gx, gy = pad + label_w, y + header_h

    for j, role in enumerate(lm.roles):
        tile = Image.new("RGB", (text_w(role) + 2, 12), (255, 255, 255))
        ImageDraw.Draw(tile).text((0, 0), role, fill=_INK, font=font)
        rotated = tile.rotate(90, expand=True)
        img.paste(
            rotated, (gx + j * cell_w + (cell_w - rotated.width) // 2, gy - rotated.height - 2)
        )

    for i, b in enumerate(lm.blocks):
        cy = gy + i * cell_h
        draw.text((pad, cy), row_labels[i], fill=_INK, font=font)
        for j, role in enumerate(lm.roles):
            cx = gx + j * cell_w
            tname = lm.cells.get((b, role))
            box = (cx, cy, cx + cell_w - 2, cy + cell_h - 2)
            if tname is None:
                draw.rectangle(box, outline=_EMPTY)
            else:
                draw.rectangle(box, fill=colours[tname])

    y = gy + grid_h + pad
    if global_lines:
        draw.text((pad, y), "non-block tensors", fill=_FAINT, font=font)
        y += line_h
        for role, line in zip(lm.global_roles, global_lines, strict=True):
            draw.rectangle((pad, y + 1, pad + swatch, y + 11), fill=colours[lm.cells[(None, role)]])
            draw.text((pad + swatch + 6, y), line, fill=_INK, font=font)
            y += line_h
    return img


def _legend(maps: Sequence[LayerMap], colours: dict[str, Colour]) -> Image.Image:
    """Types in bits order with bits per weight and tensor counts (one count per map)."""
    from PIL import Image, ImageDraw

    font, text_w = _font_and_width()
    bpw: dict[str, float] = {}
    for lm in maps:
        bpw.update(lm.types)
    order = sorted(bpw, key=lambda t: (bpw[t], t))
    width_t = max(len(t) for t in order) if order else 0
    lines = []
    for t in order:
        counts = " | ".join(str(lm.counts.get(t, 0)) for lm in maps)
        lines.append(f"{t:<{width_t}}  {bpw[t]:5.2f} bpw  ({counts} tensors)")
    heading = "types, fewest bits first"
    pad, line_h, swatch = _PAD, _LINE_H, _SWATCH
    width = pad + swatch + 6 + max([text_w(s) for s in [*lines, heading]]) + pad
    height = line_h + len(lines) * line_h + pad
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((pad, 0), heading, fill=_FAINT, font=font)
    y = line_h
    for t, line in zip(order, lines, strict=True):
        draw.rectangle((pad, y + 1, pad + swatch, y + 11), fill=colours[t])
        draw.text((pad + swatch + 6, y), line, fill=_INK, font=font)
        y += line_h
    return img


def _stack(top: Sequence[Image.Image], bottom: Image.Image) -> Image.Image:
    from PIL import Image

    width = max(sum(i.width for i in top), bottom.width)
    height = max(i.height for i in top) + bottom.height
    out = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in top:
        out.paste(panel, (x, 0))
        x += panel.width
    out.paste(bottom, (0, height - bottom.height))
    return out


def make_image(
    lm: LayerMap,
    *,
    cell_w: int = 18,
    cell_h: int = 12,
    title: str | None = None,
) -> Image.Image:
    """Draw one model's block grid, its non-block tensors and a legend underneath."""
    colours = shared_colours([lm])
    panel = _panel(lm, colours, cell_w=cell_w, cell_h=cell_h, title=title)
    return _stack([panel], _legend([lm], colours))


def make_comparison(
    maps: Sequence[LayerMap],
    titles: Sequence[str],
    *,
    cell_w: int = 18,
    cell_h: int = 12,
) -> Image.Image:
    """Draw several models side by side with one palette and one legend.

    Legend counts are given per panel, left to right.
    """
    colours = shared_colours(maps)
    panels = [
        _panel(lm, colours, cell_w=cell_w, cell_h=cell_h, title=t)
        for lm, t in zip(maps, titles, strict=True)
    ]
    return _stack(panels, _legend(maps, colours))


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "layers",
        help="map which quantization type each tensor got, block by block",
        description=(
            "Show the quantization mixture of a GGUF model as a grid: one row per block, one "
            "column per tensor role, coloured by bits per weight. Prints a text table by "
            "default; -o writes a PNG (needs Pillow); --json prints the grid as JSON. With "
            "several models, -o draws them side by side with one shared palette."
        ),
    )
    p.add_argument(
        "model", type=Path, nargs="+", help="GGUF file(s), or any shard of a split model"
    )
    p.add_argument("-o", "--output", type=Path, help="PNG to write")
    p.add_argument("--text", action="store_true", help="print the text table even with -o")
    p.add_argument("--json", action="store_true", help="print the grid as JSON")
    p.add_argument(
        "--cell", default="18x12", metavar="WxH", help="cell size in pixels (default: 18x12)"
    )
    p.add_argument(
        "--title",
        action="append",
        help="panel title, once per model in order (default: the file names)",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing image")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        cell_w, cell_h = (int(v) for v in str(args.cell).lower().split("x", 1))
    except ValueError:
        print("error: --cell expects WxH, for example 18x12", file=sys.stderr)
        return 2
    if args.title and len(args.title) != len(args.model):
        print("error: give one --title per model, or none", file=sys.stderr)
        return 2
    if args.output is not None and args.output.exists() and not args.force:
        msg = f"{args.output} already exists (use --force to overwrite)"
        raise FileExistsError(msg)
    maps: list[LayerMap] = []
    names: list[str] = []
    for path in args.model:
        with open_model(path) as model:
            maps.append(build_layer_map(model))
            names.append(model.path.name)
    titles = list(args.title) if args.title else [n.removesuffix(".gguf") for n in names]

    if args.json:
        if len(maps) == 1:
            json.dump(maps[0].to_dict(), sys.stdout, indent=2)
        else:
            json.dump(
                [{"model": n, **lm.to_dict()} for n, lm in zip(names, maps, strict=True)],
                sys.stdout,
                indent=2,
            )
        sys.stdout.write("\n")
    elif args.output is None or args.text:
        for i, (name, lm) in enumerate(zip(names, maps, strict=True)):
            if len(maps) > 1:
                print(("" if i == 0 else "\n") + f"== {name}")
            print("\n".join(format_text(lm)))
    if args.output is not None:
        try:
            import PIL  # noqa: F401
        except ImportError:
            print("error: rendering needs Pillow: pip install 'ggufkit[viz]'", file=sys.stderr)
            return 2
        if len(maps) == 1:
            img = make_image(maps[0], cell_w=cell_w, cell_h=cell_h, title=titles[0])
        else:
            img = make_comparison(maps, titles, cell_w=cell_w, cell_h=cell_h)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        img.save(args.output)
        if not args.json:
            print(f"-> {args.output}")
    return 0
