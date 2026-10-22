"""Small helpers shared by the commands: safe output files and argument checks."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def prepare_output(path: Path, *, force: bool) -> None:
    """Fail early, before any expensive work, if ``path`` cannot or must not be written."""
    if path.exists() and not force:
        msg = f"{path} already exists (use --force to overwrite)"
        raise FileExistsError(msg)
    parent = path.parent if str(path.parent) else Path()
    if not parent.is_dir():
        msg = f"{parent} is not a directory"
        raise FileNotFoundError(msg)
    if not os.access(parent, os.W_OK):
        msg = f"cannot write to {parent}"
        raise PermissionError(msg)


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Yield a temporary path next to ``path``; move it into place only if the block succeeds.

    An interrupted or failed write leaves ``path`` untouched instead of half-written.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        yield tmp
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    with atomic_output(path) as tmp:
        tmp.write_text(text, encoding="utf-8")


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        msg = f"must be at least 1, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return n


def non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        msg = f"must not be negative, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return n


def positive_float(value: str) -> float:
    x = float(value)
    if not x > 0:
        msg = f"must be greater than 0, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return x


def to_stderr(message: str) -> None:
    """Progress and warnings go to stderr, so stdout carries only results."""
    print(message, file=sys.stderr)
