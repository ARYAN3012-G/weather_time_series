"""
Auxiliary binary masks for preprocessing.

Three masks computed per image:
    valid_field_mask: True where the image is actual field data (not background/no-data)
    glare_mask:       True where specular highlights/sun glare dominate
    shadow_mask:      True where cast shadows are deep enough to suppress useful signal

All functions take a uint8 (H, W, 3) RGB array and return a (H, W) bool array.
"""
from __future__ import annotations

import cv2
import numpy as np


def _rgb_to_hsv_float(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to float HSV in [0, 1] for H, S, V."""
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] /= 179.0  # OpenCV H is [0, 179]
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0
    return hsv


def glare_mask(
    rgb_uint8: np.ndarray,
    v_thresh: float = 0.95,
    s_thresh: float = 0.10,
) -> np.ndarray:
    """
    Detect specular highlights / sun glare.

    A pixel is glare iff:
        Value (brightness) > v_thresh AND Saturation < s_thresh
    i.e., extremely bright and nearly desaturated.

    This catches sun glare on leaves (which would otherwise look like dry/drought) and
    sun glare on soil (which would otherwise look like cotton balls).
    """
    hsv = _rgb_to_hsv_float(rgb_uint8)
    return (hsv[..., 2] > v_thresh) & (hsv[..., 1] < s_thresh)


def shadow_mask(
    rgb_uint8: np.ndarray,
    v_thresh: float = 0.15,
    s_thresh: float = 0.20,
) -> np.ndarray:
    """
    Detect deep cast shadows.

    A pixel is shadow iff Value < v_thresh AND Saturation < s_thresh.
    Deep shadows under canopy can otherwise look like dark/dead vegetation.
    """
    hsv = _rgb_to_hsv_float(rgb_uint8)
    return (hsv[..., 2] < v_thresh) & (hsv[..., 1] < s_thresh)


def valid_field_mask(
    rgb_uint8: np.ndarray,
    dark_threshold: int = 10,
    saturation_threshold: int = 15,
    min_border_component_area: int = 500,
) -> np.ndarray:
    """
    Compute a binary mask: True = valid field data, False = background/no-data.

    Detects TWO types of Metashape no-data padding:
        1. Dark/black padding  — max channel <= dark_threshold (original behaviour)
        2. White/near-white    — max channel >= 248 AND saturation <= 8
           Metashape fills outside-coverage areas with white (255,255,255) by default.

    Threshold for white padding (max>=248, sat<=8) is safe:
        cotton_ball pixels: saturation ~15  → NOT masked
        bright soil:        saturation ~60  → NOT masked
        glare on leaves:    saturation ~15  → NOT masked

    Uses cv2.connectedComponentsWithStats (~50x faster than scipy.ndimage.label).
    """
    h, w = rgb_uint8.shape[:2]

    # Candidate background: dark pixels OR white/near-white padding
    max_channel = rgb_uint8.max(axis=2)
    min_channel = rgb_uint8.min(axis=2)
    saturation  = max_channel.astype(np.int16) - min_channel.astype(np.int16)

    dark_bg  = (max_channel <= dark_threshold) & (saturation <= saturation_threshold)
    white_bg = (max_channel >= 248) & (saturation <= 8)   # ← NEW: white padding
    candidate_bg = (dark_bg | white_bg).astype(np.uint8)

    if not candidate_bg.any():
        return np.ones((h, w), dtype=bool)

    # cv2 connected components (int32, SIMD-accelerated)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_bg, connectivity=8
    )
    if n_labels <= 1:
        return np.ones((h, w), dtype=bool)

    # Find labels touching any image border
    border_labels = set()
    border_labels.update(labels[0,  :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:,  0].tolist())
    border_labels.update(labels[:, -1].tolist())
    border_labels.discard(0)

    if not border_labels:
        return np.ones((h, w), dtype=bool)

    # Keep only border-touching components above minimum size
    keep = [
        lbl for lbl in border_labels
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_border_component_area
    ]
    if not keep:
        return np.ones((h, w), dtype=bool)

    # Build background mask and dilate slightly to catch soft edges
    bg = np.isin(labels, keep).astype(np.uint8)
    del labels, stats   # free immediately (2.5 GB for 25k×25k)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bg = cv2.dilate(bg, kernel, iterations=2)
    return ~bg.astype(bool)


def compute_aux_masks(
    rgb_uint8: np.ndarray,
    *,
    glare_v: float = 0.95,
    glare_s: float = 0.10,
    shadow_v: float = 0.15,
    shadow_s: float = 0.20,
    field_dark: int = 10,
    field_sat: int = 15,
    field_min_area: int = 500,
) -> dict[str, np.ndarray]:
    """One-shot computation of all auxiliary masks. Returns a dict of bool arrays."""
    return {
        "valid_field": valid_field_mask(rgb_uint8, field_dark, field_sat, field_min_area),
        "glare": glare_mask(rgb_uint8, glare_v, glare_s),
        "shadow": shadow_mask(rgb_uint8, shadow_v, shadow_s),
    }
