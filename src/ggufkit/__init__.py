"""ggufkit: fingerprint, verify and visualize GGUF models."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ggufkit")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
