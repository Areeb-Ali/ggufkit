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
    p.set_defaults(func=lambda _args: (p.print_help(), 2)[1])
