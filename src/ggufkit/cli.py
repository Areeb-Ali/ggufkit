"""Command-line entry point: ``ggufkit <command> ...``."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from ggufkit import __version__
from ggufkit import fingerprint as fingerprint_cmd
from ggufkit import stitch as stitch_cmd
from ggufkit import viz as viz_cmd
from ggufkit.canon import UnsupportedModelError
from ggufkit.model import ModelError

if TYPE_CHECKING:
    from collections.abc import Sequence

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ggufkit",
        description="Fingerprint, verify and visualize GGUF models.",
    )
    parser.add_argument("--version", action="version", version=f"ggufkit {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    fingerprint_cmd.add_parser(sub)
    stitch_cmd.add_parser(sub)
    viz_cmd.add_parser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return EXIT_ERROR
    try:
        return int(func(args))
    except (
        ModelError,
        UnsupportedModelError,
        FileNotFoundError,
        FileExistsError,
        IsADirectoryError,
        PermissionError,
        ValueError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
