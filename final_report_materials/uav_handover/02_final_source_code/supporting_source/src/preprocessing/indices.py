"""
Vegetation indices computed from RGB.

All functions accept a float32 array shaped (..., 3) with values in [0, 1]
(R, G, B in the last axis) and return a same-shape-minus-last-axis float32 array.

Indices implemented:
    ExG   = 2G - R - B                       Excess Green
    ExGR  = ExG - (1.4 R - G)                Excess Green minus Excess Red
    CIVE  = 0.441 R - 0.811 G + 0.385 B + c  Color Index of Vegetation Extraction
    GLI   = (2G - R - B) / (2G + R + B)      Green Leaf Index
    NGRDI = (G - R) / (G + R)                Normalized Green-Red Diff Index

All are normalized to roughly [-1, 1] for stable neural network input.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _split(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split (..., 3) array into r, g, b each (...)."""
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def exg(rgb: np.ndarray) -> np.ndarray:
    """Excess Green. Range roughly [-2, 2], typically [-0.5, 0.5] for natural scenes."""
    r, g, b = _split(rgb)
    val = 2.0 * g - r - b
    # Normalize to roughly [-1, 1]: divide by 2 (theoretical max)
    return (val / 2.0).astype(np.float32)


def exgr(rgb: np.ndarray) -> np.ndarray:
    """ExG - ExR. Better at suppressing soil response than plain ExG."""
    r, g, b = _split(rgb)
    val = (2.0 * g - r - b) - (1.4 * r - g)
    return (val / 3.0).astype(np.float32)


def cive(rgb: np.ndarray) -> np.ndarray:
    """Color Index of Vegetation Extraction. Lower = more vegetation."""
    r, g, b = _split(rgb)
    # Original CIVE has an offset of 18.78745 calibrated for uint8 input;
    # for [0,1] input that offset would dwarf the signal, so we drop it and rescale.
    val = 0.441 * r - 0.811 * g + 0.385 * b
    # Invert so higher = more vegetation, and scale to [-1, 1]
    return (-val / 0.811).astype(np.float32)


def gli(rgb: np.ndarray) -> np.ndarray:
    """Green Leaf Index. Inherently in [-1, 1]."""
    r, g, b = _split(rgb)
    num = 2.0 * g - r - b
    den = 2.0 * g + r + b + _EPS
    return np.clip(num / den, -1.0, 1.0).astype(np.float32)


def ngrdi(rgb: np.ndarray) -> np.ndarray:
    """Normalized Green-Red Difference Index. Inherently in [-1, 1]."""
    r, g, _ = _split(rgb)
    num = g - r
    den = g + r + _EPS
    return np.clip(num / den, -1.0, 1.0).astype(np.float32)


# All indices in the order we stack them as input channels (4..8)
INDEX_FUNCS = (exg, exgr, cive, gli, ngrdi)
INDEX_NAMES = ("ExG", "ExGR", "CIVE", "GLI", "NGRDI")


def compute_all_indices(rgb_float01: np.ndarray) -> np.ndarray:
    """
    Compute all 5 vegetation indices and return them stacked.

    Args:
        rgb_float01: (H, W, 3) float32 with values in [0, 1]

    Returns:
        (5, H, W) float32, channels in order: ExG, ExGR, CIVE, GLI, NGRDI
    """
    if rgb_float01.dtype != np.float32:
        rgb_float01 = rgb_float01.astype(np.float32)
    if rgb_float01.max() > 1.0 + _EPS:
        raise ValueError(
            f"compute_all_indices expects values in [0,1]; got max={rgb_float01.max()}"
        )
    out = np.stack([f(rgb_float01) for f in INDEX_FUNCS], axis=0)
    return out
