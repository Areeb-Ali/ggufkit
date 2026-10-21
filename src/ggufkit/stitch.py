"""Stitch the metadata of one GGUF file onto the tensors of another.

This is KerfuffleV2's ``gguf-frankenstein`` rewritten against the current ``GGUFWriter``
API. Give it a metadata source and, optionally, a tensor source, and it writes a new file with
the key/value metadata of the first and the tensor descriptors and data of the second. Without
a tensor source the result is a vocab-only model: just the metadata, which is a usable base
for training or for further stitching.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from gguf import GGUFEndian, GGUFValueType, GGUFWriter
from gguf.constants import Keys

from ggufkit._util import atomic_output, prepare_output, to_stderr
from ggufkit.canon import UnsupportedModelError
from ggufkit.model import Model, open_model

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable


def copy_metadata(writer: GGUFWriter, model: Model, *, skip: Iterable[str] = ()) -> list[str]:
    """Add every metadata entry of ``model`` to ``writer``; return the names copied.

    ``general.architecture`` is skipped because the writer adds it itself, and
    ``general.alignment`` because the writer must also pad the tensor data to it (see
    :func:`stitch`). Arrays nested more than one level deep are refused: the reader flattens
    them, so they cannot be copied faithfully. So are empty arrays, which ``GGUFWriter``
    cannot write.
    """
    skipped = set(skip) | {Keys.General.ARCHITECTURE, Keys.General.ALIGNMENT}
    copied: list[str] = []
    for f in model.metadata_fields():
        if f.name in skipped:
            continue
        vtype = f.types[0]
        try:
            if vtype == GGUFValueType.ARRAY:
                if not f.data:
                    msg = f"Cannot copy empty array metadata entry {f.name!r}: GGUFWriter cannot write empty arrays"
                    raise UnsupportedModelError(msg)
                if len(f.types) != 2:
                    msg = f"Cannot copy nested array metadata entry {f.name!r}"
                    raise UnsupportedModelError(msg)
                writer.add_key_value(f.name, f.contents(), vtype, sub_type=f.types[1])
            else:
                writer.add_key_value(f.name, f.contents(), vtype)
        except ValueError as e:
            msg = f"Cannot copy metadata entry {f.name!r}: {e}"
            raise UnsupportedModelError(msg) from e
        copied.append(f.name)
    return copied


def stitch(
    metadata: str | Path,
    tensors: str | Path | None,
    output: str | Path,
    *,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Write ``output`` with the metadata of ``metadata`` and the tensors of ``tensors``."""
    say = log or (lambda _msg: None)
    output = Path(output)
    inputs = {Path(metadata).resolve()}
    if tensors is not None:
        inputs.add(Path(tensors).resolve())
    if output.resolve() in inputs:
        msg = "The output file must differ from the input files"
        raise ValueError(msg)
    prepare_output(output, force=force)

    say(f"* Metadata source: {metadata}")
    with open_model(metadata) as md:
        arch = md.architecture
        if arch is None:
            msg = "The metadata source has no general.architecture entry"
            raise UnsupportedModelError(msg)
        td: Model | None = None
        try:
            if tensors is not None:
                say(f"* Tensor source: {tensors}")
                td = open_model(tensors)
                if td.endianess != GGUFEndian.LITTLE:
                    msg = "Big-endian tensor sources are not supported"
                    raise UnsupportedModelError(msg)
            else:
                say("* No tensor source given: writing a vocab-only model")

            with atomic_output(output) as tmp:
                writer = GGUFWriter(tmp, arch, use_temp_file=False)
                try:
                    _write(writer, md, td, say)
                finally:
                    writer.close()
        finally:
            if td is not None:
                td.close()
    say(f"* Wrote {output}")
    return output


def _write(writer: GGUFWriter, md: Model, td: Model | None, say: Callable[[str], None]) -> None:
    if Keys.General.ALIGNMENT in md.fields:
        # Sets the metadata entry and makes the writer pad the tensor data to match it.
        writer.add_custom_alignment(md.alignment)
    copied = copy_metadata(writer, md)
    say(f"* Copied {len(copied)} metadata entries")

    if td is not None:
        for t in td.tensors:
            writer.add_tensor_info(
                t.name,
                [int(v) for v in reversed(t.shape)],
                np.dtype(np.float32),
                int(t.n_bytes),
                raw_dtype=t.tensor_type,
            )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    if td is not None:
        n = len(td.tensors)
        for i, t in enumerate(td.tensors, 1):
            size = t.n_bytes / (1024 * 1024)
            say(f"  {i:4}/{n:<4} {t.tensor_type.name:<7} {size:9.2f} MiB  {t.name}")
            writer.write_tensor_data(t.data, tensor_endianess=GGUFEndian.LITTLE)


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "stitch",
        help="combine the metadata of one GGUF file with the tensors of another",
        description=(
            "Write a new GGUF file with the key/value metadata of --metadata and the tensors of "
            "--tensors. Omit --tensors to get a vocab-only model containing just the metadata."
        ),
    )
    p.add_argument("--metadata", required=True, type=Path, help="GGUF file to take metadata from")
    p.add_argument("--tensors", type=Path, help="GGUF file to take tensor data from")
    p.add_argument("-o", "--output", required=True, type=Path, help="GGUF file to write")
    p.add_argument("--force", action="store_true", help="overwrite the output if it exists")
    p.add_argument("-q", "--quiet", action="store_true", help="do not report progress")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    stitch(
        args.metadata,
        args.tensors,
        args.output,
        force=args.force,
        log=None if args.quiet else to_stderr,
    )
    return 0
