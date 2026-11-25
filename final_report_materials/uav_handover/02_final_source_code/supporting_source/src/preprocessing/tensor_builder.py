"""
Build the multi-channel input tensor used across the whole pipeline.

Three flavors:
    build_9channel  — RGB + 5 vegetation indices + HSV-V. Used by patch classifier and
                      segmentation model.
    build_12channel — adds local-std, GLCM-contrast, FFT-row-score. Used by pseudo-
                      labeling and orthomosaic inference where the bigger context lets
                      the structural cues become meaningful. (Computed on demand;
                      relatively expensive.)
    build_rgb_norm  — just the ImageNet-normalized RGB, for cases where indices are
                      computed separately.

All builders return float32 tensors shaped (C, H, W).
"""
from __future__ import annotations

import cv2
import numpy as np

from .indices import compute_all_indices

# ImageNet stats for RGB normalization
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    """uint8 (H,W,3) → ImageNet-normalized float32 (3,H,W)."""
    rgb = rgb_uint8.astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return rgb.transpose(2, 0, 1).astype(np.float32)


def _rgb_to_v_channel(rgb_uint8: np.ndarray) -> np.ndarray:
    """Extract HSV V channel only, in [-1, 1]."""
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2].astype(np.float32) / 255.0
    # Center around 0 for symmetry with the indices
    return (v * 2.0 - 1.0).astype(np.float32)


def build_9channel(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Build the 9-channel input tensor: RGB(3) + indices(5) + V(1).

    Args:
        rgb_uint8: (H, W, 3) uint8 RGB image.

    Returns:
        (9, H, W) float32 tensor.
    """
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3) RGB; got shape {rgb_uint8.shape}")
    if rgb_uint8.dtype != np.uint8:
        raise ValueError(f"Expected uint8; got {rgb_uint8.dtype}")

    rgb_norm = _normalize_rgb(rgb_uint8)            # (3, H, W)
    rgb_float01 = rgb_uint8.astype(np.float32) / 255.0
    indices = compute_all_indices(rgb_float01)      # (5, H, W)
    v = _rgb_to_v_channel(rgb_uint8)                # (H, W)

    return np.concatenate([rgb_norm, indices, v[None]], axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# Structural channels (computed only on demand)
# -----------------------------------------------------------------------------
def _local_std(gray_uint8: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Local std of luminance over a ksize×ksize window. Boll-detection proxy."""
    g = gray_uint8.astype(np.float32)
    mean = cv2.boxFilter(g, ddepth=-1, ksize=(ksize, ksize), normalize=True)
    mean_sq = cv2.boxFilter(g * g, ddepth=-1, ksize=(ksize, ksize), normalize=True)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    # Scale to roughly [-1, 1] — typical std for natural scenes is 10-40 in [0,255]
    return (std / 40.0 * 2.0 - 1.0).astype(np.float32)


def _glcm_contrast_blocked(gray_uint8: np.ndarray, block: int = 32) -> np.ndarray:
    """
    Approximate GLCM contrast computed on blocks for speed, upsampled back.
    Uses a fast per-block variance proxy (Laplacian variance), which is highly
    correlated with GLCM contrast for our purposes.
    """
    h, w = gray_uint8.shape
    lap = cv2.Laplacian(gray_uint8.astype(np.float32), ddepth=cv2.CV_32F, ksize=3)
    lap_sq = lap * lap
    # Downsample by mean-pooling
    h_blocks = h // block
    w_blocks = w // block
    if h_blocks == 0 or w_blocks == 0:
        return np.zeros_like(gray_uint8, dtype=np.float32)
    trimmed = lap_sq[: h_blocks * block, : w_blocks * block]
    pooled = trimmed.reshape(h_blocks, block, w_blocks, block).mean(axis=(1, 3))
    # Upsample to original size (nearest is fine — these are coarse cues)
    upsampled = cv2.resize(pooled, (w, h), interpolation=cv2.INTER_LINEAR)
    # Scale to [-1, 1]; typical values for natural scenes saturate around 200
    return np.clip(upsampled / 200.0 * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)


def _fft_row_periodicity_blocked(gray_uint8: np.ndarray, block: int = 256) -> np.ndarray:
    """
    Score row-crop periodicity per block using FFT.
    Returns a (H, W) map; higher = stronger periodic row structure.
    """
    h, w = gray_uint8.shape
    score = np.zeros((h, w), dtype=np.float32)
    g = gray_uint8.astype(np.float32) - gray_uint8.mean()

    for by in range(0, h - block + 1, block):
        for bx in range(0, w - block + 1, block):
            patch = g[by:by + block, bx:bx + block]
            # 2D FFT and power spectrum
            f = np.fft.fft2(patch)
            ps = np.abs(f) ** 2
            ps[0, 0] = 0  # remove DC
            # Sum of top-K bins divided by total energy → periodicity score
            total = ps.sum()
            if total <= 0:
                continue
            top_k = np.partition(ps.flatten(), -10)[-10:].sum()
            s = top_k / total
            # Typical range ~0.02-0.2; map to roughly [-1, 1]
            score[by:by + block, bx:bx + block] = np.clip(s * 10.0 - 1.0, -1.0, 1.0)

    return score


def build_12channel(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Build the 12-channel input tensor: 9-channel + local-std + GLCM-contrast + FFT-row.

    Used during pseudo-labeling and orthomosaic inference, where the larger spatial
    context makes the structural cues informative.
    """
    base = build_9channel(rgb_uint8)
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    s = _local_std(gray)
    c = _glcm_contrast_blocked(gray)
    f = _fft_row_periodicity_blocked(gray)
    structural = np.stack([s, c, f], axis=0).astype(np.float32)
    return np.concatenate([base, structural], axis=0).astype(np.float32)


CHANNEL_NAMES_9 = ("R", "G", "B", "ExG", "ExGR", "CIVE", "GLI", "NGRDI", "V")
CHANNEL_NAMES_12 = CHANNEL_NAMES_9 + ("LocalStd", "GLCM", "FFTrow")
