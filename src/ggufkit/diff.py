"""Show what differs between two GGUF models: metadata values, tensor descriptors, numbers.

``verify`` tells you *whether* two files differ and in which section. ``diff`` shows the
actual differences: which metadata values changed and to what, which tensors were added,
removed, retyped or rewritten, and, with ``--numeric``, how far the decoded weights are from
each other tensor by tensor. The numeric comparison is how you measure a quantization
against its source, or two quantizations against each other.
"""

from __future__ import annotations

import fnmatch
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from gguf import GGUFValueType

from ggufkit import canon
from ggufkit.dequant import can_dequantize, to_f32
from ggufkit.fingerprint import default_jobs, fingerprint
from ggufkit.model import open_model
from ggufkit.verify import comma_list

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

    from gguf import GGUFEndian
    from gguf.gguf_reader import ReaderField, ReaderTensor

    from ggufkit.fingerprint import Fingerprint
    from ggufkit.model import Model

#: Arrays up to this many items are shown inline; longer ones are summarized.
MAX_INLINE_ITEMS = 8

#: Bytes of decoded float32 (both sides together) held in memory per numeric chunk.
NUMERIC_CHUNK_BYTES = 64 << 20


def summarize_value(field: ReaderField) -> Any:
    """Return a printable metadata value, or a short description of a large array."""
    vtype = field.types[0]
    if vtype == GGUFValueType.ARRAY:
        n = len(field.data)
        if n <= MAX_INLINE_ITEMS and len(field.types) == 2:
            return field.contents()
        return f"<array {field.types[-1].name}[{n}] {canon.kv_digest(field).hex()[:12]}>"
    return field.contents()


@dataclass(frozen=True)
class KVEntry:
    key: str
    status: str  # added, removed, changed, same
    a: Any = None
    b: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "status": self.status, "a": self.a, "b": self.b}


def kv_diff(
    ma: Model,
    mb: Model,
    *,
    ignore: Sequence[str] = (),
    include_same: bool = False,
) -> list[KVEntry]:
    fa = {f.name: f for f in ma.metadata_fields()}
    fb = {f.name: f for f in mb.metadata_fields()}
    out: list[KVEntry] = []
    for key in canon.sorted_by_name(set(fa) | set(fb)):
        if any(fnmatch.fnmatchcase(key, pat) for pat in ignore):
            continue
        if key not in fa:
            out.append(KVEntry(key, "added", b=summarize_value(fb[key])))
        elif key not in fb:
            out.append(KVEntry(key, "removed", a=summarize_value(fa[key])))
        elif canon.kv_digest(fa[key]) != canon.kv_digest(fb[key]):
            out.append(KVEntry(key, "changed", summarize_value(fa[key]), summarize_value(fb[key])))
        elif include_same:
            out.append(KVEntry(key, "same", summarize_value(fa[key]), summarize_value(fb[key])))
    return out


@dataclass(frozen=True)
class TensorEntry:
    name: str
    status: str  # added, removed, changed, same
    a: dict[str, Any] | None = None
    b: dict[str, Any] | None = None
    what: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "a": self.a,
            "b": self.b,
            "what": list(self.what),
        }


def _desc(fp: Fingerprint, name: str) -> dict[str, Any]:
    t = (fp.tensors or {})[name]
    return {"type": t.type, "shape": list(t.shape), "data": t.data}


def tensor_diff(
    fa: Fingerprint,
    fb: Fingerprint,
    *,
    patterns: Sequence[str] = (),
    include_same: bool = False,
) -> list[TensorEntry]:
    ta = fa.tensors or {}
    tb = fb.tensors or {}
    out: list[TensorEntry] = []
    for name in canon.sorted_by_name(set(ta) | set(tb)):
        if patterns and not any(fnmatch.fnmatchcase(name, pat) for pat in patterns):
            continue
        if name not in ta:
            out.append(TensorEntry(name, "added", b=_desc(fb, name)))
        elif name not in tb:
            out.append(TensorEntry(name, "removed", a=_desc(fa, name)))
        else:
            x, y = ta[name], tb[name]
            what: list[str] = []
            if x.shape != y.shape:
                what.append("shape")
            if x.type != y.type:
                what.append("type")
            if x.data is not None and y.data is not None and x.data != y.data:
                what.append("data")
            if what:
                out.append(
                    TensorEntry(name, "changed", _desc(fa, name), _desc(fb, name), tuple(what))
                )
            elif include_same:
                out.append(TensorEntry(name, "same", _desc(fa, name), _desc(fb, name)))
    return out


@dataclass
class NumericStats:
    name: str
    a_type: str
    b_type: str
    n: int = 0
    mse: float | None = None
    rmse: float | None = None
    max_abs: float | None = None
    mean_abs: float | None = None
    rel_l2: float | None = None
    cosine: float | None = None
    skipped: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "a_type": self.a_type,
            "b_type": self.b_type,
            "n": self.n,
            "mse": self.mse,
            "rmse": self.rmse,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "rel_l2": self.rel_l2,
            "cosine": self.cosine,
            "skipped": self.skipped,
        }


def numeric_diff(
    ta: ReaderTensor,
    tb: ReaderTensor,
    endian_a: GGUFEndian,
    endian_b: GGUFEndian,
    *,
    chunk_bytes: int = NUMERIC_CHUNK_BYTES,
) -> NumericStats:
    """Error statistics between two tensors of the same shape, decoded to float32 in chunks."""
    stats = NumericStats(ta.name, ta.tensor_type.name, tb.tensor_type.name, n=int(ta.n_elements))
    if [int(d) for d in ta.shape] != [int(d) for d in tb.shape]:
        stats.skipped = "shape differs"
        return stats
    for t in (ta, tb):
        if not can_dequantize(t.tensor_type):
            stats.skipped = f"cannot decode {t.tensor_type.name}"
            return stats

    sum_sq_diff = sum_abs_diff = sum_sq_a = sum_sq_b = dot = 0.0
    max_abs = 0.0

    def accumulate(x: np.ndarray, y: np.ndarray) -> None:
        nonlocal sum_sq_diff, sum_abs_diff, sum_sq_a, sum_sq_b, dot, max_abs
        x64 = x.astype(np.float64, copy=False).ravel()
        y64 = y.astype(np.float64, copy=False).ravel()
        d = x64 - y64
        sum_sq_diff += float(np.dot(d, d))
        sum_abs_diff += float(np.abs(d).sum())
        max_abs = max(max_abs, float(np.abs(d).max(initial=0.0)))
        sum_sq_a += float(np.dot(x64, x64))
        sum_sq_b += float(np.dot(y64, y64))
        dot += float(np.dot(x64, y64))

    if ta.data.ndim <= 1:
        accumulate(to_f32(ta, endian_a), to_f32(tb, endian_b))
    else:
        rows = int(ta.data.shape[0])
        elems_per_row = int(ta.n_elements) // max(rows, 1)
        rows_per_chunk = max(1, chunk_bytes // max(elems_per_row * 4 * 2, 1))
        for start in range(0, rows, rows_per_chunk):
            sl = slice(start, min(rows, start + rows_per_chunk))
            accumulate(to_f32(ta, endian_a, rows=sl), to_f32(tb, endian_b, rows=sl))

    n = stats.n
    stats.mse = sum_sq_diff / n if n else 0.0
    stats.rmse = math.sqrt(stats.mse)
    stats.max_abs = max_abs
    stats.mean_abs = sum_abs_diff / n if n else 0.0
    stats.rel_l2 = math.sqrt(sum_sq_diff / sum_sq_a) if sum_sq_a > 0 else None
    denom = math.sqrt(sum_sq_a) * math.sqrt(sum_sq_b)
    stats.cosine = dot / denom if denom > 0 else None
    return stats


def _fmt(x: float | None, width: int = 10) -> str:
    if x is None:
        return f"{'n/a':>{width}}"
    return f"{x:>{width}.3e}" if (abs(x) < 1e-3 or abs(x) >= 1e4) and x != 0 else f"{x:>{width}.6f}"


def format_kv(entries: list[KVEntry], total: int) -> list[str]:
    counts = {s: sum(1 for e in entries if e.status == s) for s in ("changed", "added", "removed")}
    lines = [
        f"metadata: {counts['changed']} changed, {counts['added']} added, {counts['removed']} removed (of {total})"
    ]
    for e in entries:
        if e.status == "changed":
            lines.append(f"  ~ {e.key}: {e.a!r} -> {e.b!r}")
        elif e.status == "added":
            lines.append(f"  + {e.key}: {e.b!r}")
        elif e.status == "removed":
            lines.append(f"  - {e.key}: {e.a!r}")
        else:
            lines.append(f"  = {e.key}: {e.a!r}")
    return lines


def _shape(d: dict[str, Any] | None) -> str:
    return "?" if d is None else f"{d['type']} {'x'.join(map(str, d['shape']))}"


def format_tensors(entries: list[TensorEntry], total: int) -> list[str]:
    counts = {s: sum(1 for e in entries if e.status == s) for s in ("changed", "added", "removed")}
    lines = [
        f"tensors: {counts['changed']} changed, {counts['added']} added, {counts['removed']} removed (of {total})"
    ]
    for e in entries:
        if e.status == "changed":
            lines.append(f"  ~ {e.name}  {', '.join(e.what)}  [{_shape(e.a)} -> {_shape(e.b)}]")
        elif e.status == "added":
            lines.append(f"  + {e.name}  [{_shape(e.b)}]")
        elif e.status == "removed":
            lines.append(f"  - {e.name}  [{_shape(e.a)}]")
        else:
            lines.append(f"  = {e.name}  [{_shape(e.a)}]")
    return lines


def format_numeric(stats: list[NumericStats], top: int) -> list[str]:
    shown = stats if top <= 0 else stats[:top]
    header = f"numeric ({'all' if top <= 0 else f'top {len(shown)}'} by rel_l2):"
    lines = [
        header,
        f"  {'tensor':<36} {'a -> b':<16} {'rmse':>10} {'max_abs':>10} {'mean_abs':>10} {'rel_l2':>10} {'cosine':>10}",
    ]
    for s in shown:
        types = f"{s.a_type}->{s.b_type}"
        if s.skipped:
            lines.append(f"  {s.name:<36} {types:<16} skipped: {s.skipped}")
            continue
        lines.append(
            f"  {s.name:<36} {types:<16} {_fmt(s.rmse)} {_fmt(s.max_abs)} {_fmt(s.mean_abs)} {_fmt(s.rel_l2)} {_fmt(s.cosine)}"
        )
    return lines


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "diff",
        help="show metadata, tensor and (optionally) numeric differences between two models",
        description=(
            "List the metadata entries whose values differ, the tensors that were added, "
            "removed, retyped, reshaped or rewritten, and with --numeric the per-tensor error "
            "between the decoded weights (rmse, max abs, mean abs, relative L2, cosine). Exit "
            "status is 0 when nothing differs in the compared sections, 1 otherwise."
        ),
    )
    p.add_argument("a", type=Path, help="first GGUF file (the reference, e.g. the source)")
    p.add_argument("b", type=Path, help="second GGUF file (e.g. the quantization)")
    p.add_argument("--no-kv", action="store_true", help="skip the metadata comparison")
    p.add_argument("--no-tensors", action="store_true", help="skip the tensor comparison")
    p.add_argument("--numeric", action="store_true", help="decode and compare tensor values")
    p.add_argument(
        "--tensor",
        action="append",
        metavar="GLOB",
        help="only these tensors (glob, comma-separated or repeatable)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="numeric rows to print, worst first (0 = all; default: %(default)s)",
    )
    p.add_argument(
        "--ignore-keys",
        action="append",
        metavar="GLOB",
        help="metadata keys to ignore (glob, comma-separated or repeatable)",
    )
    p.add_argument("--same", action="store_true", help="also list unchanged entries")
    p.add_argument("--json", action="store_true", help="print the diff as JSON")
    p.add_argument("-j", "--jobs", type=int, default=default_jobs(), help="hashing threads")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    patterns = comma_list(args.tensor)
    ignore = comma_list(args.ignore_keys)
    jobs = max(1, args.jobs)
    compare_tensors = not args.no_tensors or args.numeric
    result: dict[str, Any] = {"a": str(args.a), "b": str(args.b)}
    lines: list[str] = []
    different = False

    with open_model(args.a) as ma, open_model(args.b) as mb:
        if not args.no_kv:
            entries = kv_diff(ma, mb, ignore=ignore, include_same=args.same)
            different |= any(e.status != "same" for e in entries)
            result["kv"] = [e.to_dict() for e in entries]
            total = len(
                {f.name for f in ma.metadata_fields()} | {f.name for f in mb.metadata_fields()}
            )
            lines.extend(format_kv(entries, total))

        if compare_tensors:
            fa = fingerprint(ma, data=True, per_tensor=True, jobs=jobs)
            fb = fingerprint(mb, data=True, per_tensor=True, jobs=jobs)
            tensors = tensor_diff(fa, fb, patterns=patterns, include_same=args.same)
            different |= any(e.status != "same" for e in tensors)
            if not args.no_tensors:
                result["tensors"] = [e.to_dict() for e in tensors]
                total = len(set(fa.tensors or {}) | set(fb.tensors or {}))
                lines.extend(format_tensors(tensors, total))

            if args.numeric:
                stats: list[NumericStats] = []
                shared = [
                    e.name
                    for e in tensor_diff(fa, fb, patterns=patterns, include_same=True)
                    if e.status in ("changed", "same")
                ]
                for name in shared:
                    ta, tb = ma.tensor(name), mb.tensor(name)
                    x, y = (fa.tensors or {})[name], (fb.tensors or {})[name]
                    if x.data is not None and x.data == y.data:
                        stats.append(
                            NumericStats(
                                name,
                                x.type,
                                y.type,
                                n=int(ta.n_elements),
                                mse=0.0,
                                rmse=0.0,
                                max_abs=0.0,
                                mean_abs=0.0,
                                rel_l2=0.0,
                                cosine=1.0,
                            )
                        )
                    else:
                        stats.append(numeric_diff(ta, tb, ma.endianess, mb.endianess))
                stats.sort(key=lambda s: (s.skipped is not None, -(s.rel_l2 or 0.0), s.name))
                result["numeric"] = [s.to_dict() for s in stats]
                lines.extend(format_numeric(stats, args.top))

    if args.json:
        result["different"] = different
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("\n".join(lines))
    return 1 if different else 0
