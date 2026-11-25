"""
Legend and overlay rendering utilities shared across pseudo-labeling outputs.

Provides:
    draw_legend_bar()     — returns a (H, W_legend, 3) legend strip that can be
                            concatenated onto the right side of any image.
    overlay_with_legend() — composite: image + colour overlay + legend strip.
    apply_palette()       — convert a class-index mask to an RGB colour mask.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.constants import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, PALETTE


# Minimum fraction of valid pixels for a class to appear in the legend
_LEGEND_MIN_FRAC: float = 0.001


def apply_palette(mask: np.ndarray) -> np.ndarray:
    """
    Convert a uint8 class-index mask to an RGB colour mask.

    Args:
        mask: (H, W) uint8 — values 0..NUM_CLASSES-1 or IGNORE_INDEX (255)

    Returns:
        (H, W, 3) uint8 RGB — ignore pixels are rendered as mid-grey (128,128,128)
    """
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(PALETTE):
        rgb[mask == cls_idx] = color
    rgb[mask == IGNORE_INDEX] = (90, 90, 90)   # dark grey = "uncertain / ignored"
    return rgb


def draw_legend_bar(
    present_classes: list[int],
    bar_width: int = 180,
    row_height: int = 28,
    font_scale: float = 0.52,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    Build a vertical legend strip showing each present class as a colour swatch + label.

    Args:
        present_classes: list of class indices to show.
        bar_width:        pixel width of the legend strip.
        row_height:       pixel height per class row.

    Returns:
        (H_legend, bar_width, 3) uint8 RGB legend image.
    """
    # Always include ignore entry at the bottom
    rows = present_classes + [-1]
    h    = row_height * len(rows) + row_height   # top padding

    canvas = np.full((h, bar_width, 3), 240, dtype=np.uint8)   # light grey bg

    # Title
    cv2.putText(canvas, 'Classes', (8, row_height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.05,
                (30, 30, 30), font_thickness + 1, cv2.LINE_AA)

    swatch_w = 26
    swatch_h = row_height - 8
    x_swatch = 8
    x_text   = swatch_w + 14
    margin   = 4

    for row_idx, cls_idx in enumerate(rows):
        y_top = row_height + row_idx * row_height

        if cls_idx == -1:
            # Ignore / uncertain entry
            color  = (90, 90, 90)
            label  = 'uncertain'
        else:
            color = tuple(int(v) for v in PALETTE[cls_idx])
            label = CLASS_NAMES[cls_idx]

        # Colour swatch (filled rectangle with thin black border)
        cv2.rectangle(canvas,
                      (x_swatch, y_top + margin),
                      (x_swatch + swatch_w, y_top + margin + swatch_h),
                      color, thickness=-1)
        cv2.rectangle(canvas,
                      (x_swatch, y_top + margin),
                      (x_swatch + swatch_w, y_top + margin + swatch_h),
                      (50, 50, 50), thickness=1)

        # Class label text
        cv2.putText(canvas, label,
                    (x_text, y_top + margin + swatch_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (30, 30, 30), font_thickness, cv2.LINE_AA)

    return canvas


def overlay_with_legend(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.50,
    min_frac: float = _LEGEND_MIN_FRAC,
    max_display_size: int = 2048,
) -> np.ndarray:
    """
    Produce a single image: RGB overlay + legend strip on the right.

    Args:
        img_rgb:           (H, W, 3) uint8 original image.
        mask:              (H, W) uint8 class-index mask (IGNORE_INDEX = uncertain).
        alpha:             Opacity of the class colour overlay.
        min_frac:          Classes covering less than this fraction are omitted from legend.
        max_display_size:  Longest side is capped at this for readable output.

    Returns:
        (H_out, W_out + legend_width, 3) uint8 composite.
    """
    H, W = img_rgb.shape[:2]

    # Determine present classes for the legend
    valid      = mask != IGNORE_INDEX
    n_valid    = max(valid.sum(), 1)
    present    = [
        cls_idx for cls_idx in range(NUM_CLASSES)
        if (mask == cls_idx).sum() / n_valid >= min_frac
    ]

    # Colour overlay
    color_mask = apply_palette(mask)
    overlay    = img_rgb.copy()
    overlay[valid] = (
        alpha * color_mask[valid].astype(np.float32) +
        (1.0 - alpha) * img_rgb[valid].astype(np.float32)
    ).astype(np.uint8)
    # Ignored pixels: tint grey slightly so the user knows they exist
    ignore     = mask == IGNORE_INDEX
    overlay[ignore] = (
        0.25 * np.array([90,90,90], dtype=np.float32) +
        0.75 * img_rgb[ignore].astype(np.float32)
    ).astype(np.uint8)

    # Legend bar — height must match the overlay
    legend  = draw_legend_bar(present, bar_width=180, row_height=max(24, H // 25))
    leg_h   = legend.shape[0]
    if leg_h < H:
        pad     = np.full((H - leg_h, legend.shape[1], 3), 240, dtype=np.uint8)
        legend  = np.vstack([legend, pad])
    else:
        legend  = legend[:H]

    composite = np.hstack([overlay, legend])

    # Resize if very large (for quick visual inspection)
    ch, cw = composite.shape[:2]
    if max(ch, cw) > max_display_size:
        scale = max_display_size / max(ch, cw)
        composite = cv2.resize(composite,
                               (int(cw * scale), int(ch * scale)),
                               interpolation=cv2.INTER_AREA)

    return composite


def add_stat_bar(
    composite: np.ndarray,
    mask: np.ndarray,
    conf: np.ndarray | None = None,
    bar_height: int = 36,
    font_scale: float = 0.48,
) -> np.ndarray:
    """
    Append a one-line statistics bar at the bottom of a composite image.
    Shows per-class percentages + ignore fraction + mean confidence.

    Args:
        composite: (H, W, 3) composite image (from overlay_with_legend).
        mask:      (H, W) original full-resolution mask (before any resize).
        conf:      (H, W) float32 confidence map, or None.
    """
    H, W = composite.shape[:2]
    bar  = np.full((bar_height, W, 3), 50, dtype=np.uint8)   # dark bg

    valid     = mask != IGNORE_INDEX
    n_pixels  = mask.size
    n_valid   = max(valid.sum(), 1)
    ignore_pc = (mask == IGNORE_INDEX).sum() / n_pixels * 100

    parts = [f'ignore={ignore_pc:.0f}%']
    for cls_idx in range(NUM_CLASSES):
        frac = (mask == cls_idx).sum() / n_valid
        if frac >= 0.01:
            name = CLASS_NAMES[cls_idx][:5]
            parts.append(f'{name}={frac:.0%}')

    if conf is not None:
        mean_conf = conf[valid].mean() if valid.any() else 0.0
        parts.append(f'mean_conf={mean_conf:.2f}')

    text = '  |  '.join(parts)
    cv2.putText(bar, text, (8, bar_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (220, 220, 220), 1, cv2.LINE_AA)

    return np.vstack([composite, bar])
