"""Visual tools: ``ggufkit viz <subcommand>``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ggufkit.viz import render

if TYPE_CHECKING:
    import argparse


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "viz",
        help="render tensors as images",
        description="Visual tools. Requires the 'viz' extra: pip install 'ggufkit[viz]'.",
    )
    vsub = p.add_subparsers(dest="viz_command", metavar="<subcommand>")
    render.add_parser(vsub)

    def show_help(_args: argparse.Namespace) -> int:
        p.print_help()
        return 2

    p.set_defaults(func=show_help)
