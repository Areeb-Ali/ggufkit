"""Colour look-up tables built in numpy, so rendering needs no plotting library."""

from __future__ import annotations

import numpy as np

# Control points sampled from matplotlib's "magma" (dark to light, perceptually uniform).
_SEQUENTIAL = (
    (0.000, (0, 0, 4)),
    (0.125, (28, 16, 68)),
    (0.250, (79, 18, 123)),
    (0.375, (129, 37, 129)),
    (0.500, (181, 54, 122)),
    (0.625, (229, 80, 100)),
    (0.750, (251, 135, 97)),
    (0.875, (254, 194, 135)),
    (1.000, (252, 253, 191)),
)

# Control points of a blue-white-red diverging scale (ColorBrewer RdBu, reversed).
_DIVERGING = (
    (0.00, (5, 48, 97)),
    (0.25, (67, 147, 195)),
    (0.50, (247, 247, 247)),
    (0.75, (214, 96, 77)),
    (1.00, (103, 0, 31)),
)

# Control points sampled from "viridis", used for categorical colours ordered by bits.
_VIRIDIS = (
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
)


def _lut(points: tuple[tuple[float, tuple[int, int, int]], ...], n: int = 256) -> np.ndarray:
    xs = np.array([p for p, _ in points], dtype=np.float64)
    rgb = np.array([c for _, c in points], dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    return (
        np.stack([np.interp(t, xs, rgb[:, i]) for i in range(3)], axis=-1).round().astype(np.uint8)
    )


SEQUENTIAL = _lut(_SEQUENTIAL)
DIVERGING = _lut(_DIVERGING)
VIRIDIS = _lut(_VIRIDIS)


def apply(lut: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB through ``lut``; NaN maps to the lowest colour."""
    v = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    idx = np.clip((v * (len(lut) - 1)).round(), 0, len(lut) - 1).astype(np.intp)
    return lut[idx]


def _rgb(row: np.ndarray) -> tuple[int, int, int]:
    return (int(row[0]), int(row[1]), int(row[2]))


def categorical(n: int) -> list[tuple[int, int, int]]:
    """``n`` distinct colours sampled evenly along viridis, darkest first."""
    if n <= 0:
        return []
    if n == 1:
        return [_rgb(VIRIDIS[len(VIRIDIS) // 2])]
    idx = np.linspace(0, len(VIRIDIS) - 1, n).round().astype(int)
    return [_rgb(VIRIDIS[i]) for i in idx]
