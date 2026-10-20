"""Render tensors as images, for any tensor type the ``gguf`` package can decode.

This is KerfuffleV2's ``gguf-tensor-to-image`` with the decoding handed to
``gguf.quants.dequantize`` (so Q4_K, IQ2_XXS, MXFP4 and friends all work, not just Q8_0),
a fix for the F32 path, an optional pixel budget for huge tensors, and PyTorch checkpoints
still supported when ``torch`` is installed.

Each pixel is one weight. Brightness follows the weight's magnitude, clipped at a few
standard deviations; hue says whether the weight sits well below the mean, well above it, or
in between. Attention projections in particular have a lot going on.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ggufkit.dequant import can_dequantize, to_f32
from ggufkit.model import open_model

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable

    from PIL import Image

# Clip values to at most this many standard deviations from the mean.
CFG_SD_CLIP_THRESHOLD = 7

# Standard deviations above the mean before a weight counts as "positive".
CFG_SDP_THRESHOLD = 1.25

# Standard deviations below the mean before a weight counts as "negative".
CFG_SDN_THRESHOLD = 1.25

# RGB scaling for negative, positive and in-between weights.
CFG_NEG_SCALE = (1.0, 0.4, 0.8)
CFG_POS_SCALE = (0.4, 0.8, 1.0)
CFG_MID_SCALE = (0.7, 0.7, 0.9)

MODES = ("devs-overall", "devs-rows", "devs-cols")


@dataclass(frozen=True)
class RenderOptions:
    mode: str = "devs-overall"
    adjust_1d_rows: int | None = None
    scale: float = 1.0
    max_pixels: int = 0


class TensorSource(Protocol):
    def tensor_names(self) -> Iterable[str]: ...
    def valid(self, key: str) -> tuple[bool, str]: ...
    def get_as_f32(self, key: str) -> np.ndarray: ...
    def get_type_name(self, key: str) -> str: ...
    def close(self) -> None: ...


class GGUFSource:
    def __init__(self, filename: Path | str) -> None:
        self.model = open_model(filename)
        self.tensors = OrderedDict((t.name, t) for t in self.model.tensors)

    def tensor_names(self) -> Iterable[str]:
        return self.tensors.keys()

    def valid(self, key: str) -> tuple[bool, str]:
        tensor = self.tensors.get(key)
        if tensor is None:
            return (False, "tensor not found")
        if not can_dequantize(tensor.tensor_type):
            return (False, f"cannot decode {tensor.tensor_type.name}")
        if len(tensor.shape) > 2:
            return (False, "more than two dimensions")
        return (True, "ok")

    def get_as_f32(self, key: str) -> np.ndarray:
        return to_f32(self.tensors[key], self.model.endianess)

    def get_type_name(self, key: str) -> str:
        return self.tensors[key].tensor_type.name

    def close(self) -> None:
        self.model.close()


class TorchSource:
    def __init__(self, filename: Path | str) -> None:
        try:
            import torch
        except ImportError as e:
            msg = "Loading PyTorch checkpoints requires torch: pip install torch"
            raise RuntimeError(msg) from e
        self.torch = torch
        obj = torch.load(filename, map_location="cpu", mmap=True, weights_only=True)
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            msg = "Expected a state dict (a mapping of tensor names to tensors)"
            raise TypeError(msg)
        self.tensors: OrderedDict[str, Any] = OrderedDict(
            (name, t.squeeze()) for name, t in obj.items() if hasattr(t, "dtype")
        )

    def tensor_names(self) -> Iterable[str]:
        return self.tensors.keys()

    def valid(self, key: str) -> tuple[bool, str]:
        tensor = self.tensors.get(key)
        if tensor is None:
            return (False, "tensor not found")
        if tensor.dtype not in (self.torch.float32, self.torch.float16, self.torch.bfloat16):
            return (False, f"unhandled dtype {tensor.dtype}")
        if len(tensor.shape) > 2:
            return (False, "more than two dimensions")
        return (True, "ok")

    def get_as_f32(self, key: str) -> np.ndarray:
        return self.tensors[key].to(dtype=self.torch.float32).numpy()

    def get_type_name(self, key: str) -> str:
        return str(self.tensors[key].dtype)

    def close(self) -> None:
        self.tensors.clear()


def open_source(filename: Path | str, model_type: str | None = None) -> TensorSource:
    name = str(filename).lower()
    if model_type == "gguf" or (model_type is None and name.endswith(".gguf")):
        return GGUFSource(filename)
    if model_type == "torch" or (model_type is None and name.endswith((".pth", ".pt", ".bin"))):
        return TorchSource(filename)
    msg = f"Cannot tell the model type of {filename}; pass --model-type"
    raise ValueError(msg)


def downsample(td: np.ndarray, max_pixels: int) -> np.ndarray:
    """Mean-pool a 2-D array until it has at most ``max_pixels`` cells."""
    if max_pixels <= 0 or td.size <= max_pixels:
        return td
    factor = int(np.ceil(np.sqrt(td.size / max_pixels)))
    rows, cols = td.shape
    rows_keep = (rows // factor) * factor
    cols_keep = (cols // factor) * factor
    if rows_keep == 0 or cols_keep == 0:
        return td
    trimmed = td[:rows_keep, :cols_keep]
    return trimmed.reshape(rows_keep // factor, factor, cols_keep // factor, factor).mean(
        axis=(1, 3)
    )


def make_image(td: np.ndarray, opts: RenderOptions) -> Image.Image:
    from PIL import Image

    td = np.nan_to_num(td.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if td.ndim == 1:
        if opts.adjust_1d_rows is not None:
            if td.shape[0] % opts.adjust_1d_rows != 0:
                msg = f"Tensor size {td.shape[0]} is not divisible by --adjust-1d-rows {opts.adjust_1d_rows}"
                raise ValueError(msg)
            td = td.reshape((opts.adjust_1d_rows, td.shape[0] // opts.adjust_1d_rows))
        else:
            td = td[None, :]
    elif td.ndim != 2:
        msg = f"Can only render 1-D or 2-D tensors, got shape {td.shape}"
        raise ValueError(msg)
    td = downsample(td, opts.max_pixels)

    if opts.mode == "devs-overall":
        sd = np.std(td)
        mean = np.mean(td)
    elif opts.mode == "devs-rows":
        sd = np.std(td, axis=1)[:, None]
        mean = np.mean(td, axis=1)[:, None]
    elif opts.mode == "devs-cols":
        sd = np.std(td, axis=0)
        mean = np.mean(td, axis=0)
    else:
        msg = f"Unknown mode {opts.mode!r}"
        raise ValueError(msg)
    sdp_max = mean + CFG_SD_CLIP_THRESHOLD * sd
    sdp_thresh = mean + CFG_SDP_THRESHOLD * sd
    sdn_thresh = mean - CFG_SDN_THRESHOLD * sd
    tda = np.minimum(np.abs(td), sdp_max).repeat(3, axis=-1).reshape((*td.shape, 3))
    tda[td <= sdn_thresh, ...] *= CFG_NEG_SCALE
    tda[td >= sdp_thresh, ...] *= CFG_POS_SCALE
    tda[np.logical_and(td > sdn_thresh, td < sdp_thresh), ...] *= CFG_MID_SCALE
    span = np.ptp(tda)
    tda = 255 * ((tda - np.min(tda)) / span) if span > 0 else np.zeros_like(tda)
    img = Image.fromarray(tda.astype(np.uint8, copy=False), "RGB")
    if opts.scale != 1.0:
        img = img.resize(
            (max(1, int(img.width * opts.scale)), max(1, int(img.height * opts.scale))),
            resample=Image.Resampling.LANCZOS,
        )
    return img


def select_names(
    source: TensorSource,
    patterns: list[str],
    *,
    glob: bool = False,
    regex: bool = False,
    log: Callable[[str], None] | None = None,
) -> list[str]:
    if glob:
        return [
            name
            for name in source.tensor_names()
            if any(fnmatch.fnmatchcase(name, pat) for pat in patterns)
        ]
    if regex:
        compiled = [re.compile(r) for r in patterns]
        return [name for name in source.tensor_names() if any(r.search(name) for r in compiled)]
    names: list[str] = []
    for name in patterns:
        ok, why = source.valid(name)
        if ok:
            names.append(name)
        elif log is not None:
            log(f"! Skipping {name!r}: {why}")
    return names


def output_path_for(output: Path, name: str, single: bool) -> Path:
    if single:
        return output
    safe = name.replace("/", "_").replace("\\", "_")
    return output.parent / f"{safe}.{output.name}"


def render(
    source: TensorSource,
    names: list[str],
    *,
    output: Path | None,
    opts: RenderOptions,
    show_with: str | None = None,
    match_1d: bool = True,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    say = log or (lambda _msg: None)
    written: list[Path] = []
    single = len(names) == 1
    for name in names:
        ok, why = source.valid(name)
        if not ok:
            say(f"! Skipping {name!r}: {why}")
            continue
        tensor = source.get_as_f32(name)
        if tensor.ndim == 1 and not match_1d:
            continue
        target = None if output is None else output_path_for(output, name, single)
        if target is not None and target.exists() and not force:
            msg = f"{target} already exists (use --force to overwrite)"
            raise FileExistsError(msg)
        say(f"* Rendering {name!r} ({source.get_type_name(name)}, shape {tuple(tensor.shape)})")
        img = make_image(tensor, opts)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            img.save(target)
            written.append(target)
            say(f"  -> {target}")
        if show_with:
            if target is not None:
                subprocess.call((show_with, str(target)))  # noqa: S603
            else:
                with tempfile.NamedTemporaryFile(suffix=".png") as fp:
                    img.save(fp, format="png")
                    fp.flush()
                    subprocess.call((show_with, fp.name))  # noqa: S603
    return written


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    import argparse

    p = sub.add_parser(
        "tensor",
        help="render one or more tensors as PNG images",
        description="Render tensors of a GGUF model (or a PyTorch checkpoint) as images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            modes:
              devs-overall  colour by distance from the mean, in standard deviations (default)
              devs-rows     same, but mean and deviation are computed per row
              devs-cols     same, but mean and deviation are computed per column

            examples:
              ggufkit viz tensor model.gguf output.weight -o out.png
              ggufkit viz tensor model.gguf --match-glob 'blk.0.*' -o imgs/t.png
            """
        ),
    )
    p.add_argument(
        "model", type=Path, help="GGUF file (or .pth/.pt checkpoint with torch installed)"
    )
    p.add_argument(
        "tensor",
        nargs="+",
        help="tensor name(s); a single pattern with --match-glob or --match-regex",
    )
    out = p.add_argument_group("output", "at least one of the following is required")
    out.add_argument(
        "-o",
        "--output",
        type=Path,
        help="PNG to write; with several tensors, each file is prefixed with the tensor name",
    )
    out.add_argument(
        "--show-with",
        metavar="APP",
        help="open each image with this application",
    )
    match = p.add_mutually_exclusive_group()
    match.add_argument("--match-glob", action="store_true", help="treat the name as a glob")
    match.add_argument("--match-regex", action="store_true", help="treat the name as a regex")
    p.add_argument("--match-1d", action="store_true", help="also render 1-D tensors when matching")
    p.add_argument(
        "--adjust-1d-rows",
        type=int,
        metavar="N",
        help="fold a 1-D tensor into N rows instead of one long row",
    )
    p.add_argument("--scale", type=float, default=1.0, help="resize factor (default: 1.0)")
    p.add_argument(
        "--max-pixels",
        type=int,
        default=0,
        metavar="N",
        help="mean-pool tensors larger than N pixels before rendering (default: no limit)",
    )
    p.add_argument("--mode", choices=MODES, default="devs-overall", help="colouring mode")
    p.add_argument("--model-type", choices=["gguf", "torch"], help="override type detection")
    p.add_argument("--force", action="store_true", help="overwrite existing images")
    p.add_argument("-q", "--quiet", action="store_true", help="do not report progress")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if not (args.show_with or args.output):
        print("error: at least one of --output or --show-with is required", file=sys.stderr)
        return 2
    if (args.match_regex or args.match_glob) and len(args.tensor) != 1:
        print("error: give exactly one pattern with --match-glob or --match-regex", file=sys.stderr)
        return 2
    try:
        import PIL  # noqa: F401
    except ImportError:
        print("error: rendering needs Pillow: pip install 'ggufkit[viz]'", file=sys.stderr)
        return 2

    log = None if args.quiet else print
    source = open_source(args.model, args.model_type)
    try:
        names = select_names(
            source,
            args.tensor,
            glob=args.match_glob,
            regex=args.match_regex,
            log=log,
        )
        if not names:
            print("error: no tensors matched", file=sys.stderr)
            return 1
        opts = RenderOptions(
            mode=args.mode,
            adjust_1d_rows=args.adjust_1d_rows,
            scale=args.scale,
            max_pixels=args.max_pixels,
        )
        render(
            source,
            names,
            output=args.output,
            opts=opts,
            show_with=args.show_with,
            match_1d=args.match_1d or not (args.match_glob or args.match_regex),
            force=args.force,
            log=log,
        )
    finally:
        source.close()
    return 0
