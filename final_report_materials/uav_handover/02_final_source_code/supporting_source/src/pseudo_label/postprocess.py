"""
Post-processing for pseudo-label probability maps.

Two stages:
    1. Dense CRF refinement — uses RGB pairwise terms to snap class boundaries
       to actual color/texture edges in the image. Corrects soft blob-like
       boundaries produced by the sliding window.

    2. Structural-cue overrides — rule-based corrections for specific known
       error modes that the classifier consistently makes. Applied ONLY to
       pseudo-labels (not at orthomosaic inference time); they clean noisy
       labels before segmentation training.

Override rules (from the plan):
    A. Pixel labeled cotton_ball BUT local_std < threshold AND V > 0.9
       → likely bright soil or glare, not a boll → set to ignore
    B. Pixel labeled drought BUT FFT row score high AND surrounded by crop
       → likely inter-row soil in a healthy field → relabel soil
    C. Pixel labeled crop BUT ExG < -0.05 AND GLI < 0 everywhere in patch
       → likely shadow on soil misclassified as crop → set to ignore
    D. Pixel in glare_mask (very bright, desaturated) → set to ignore
       (any class — glare gives unreliable signal)
    E. Pixel labeled drought BUT ExG > 0.15
       → clearly green, not drought → relabel crop or green_weed
"""
from __future__ import annotations

import numpy as np
import pydensecrf.densecrf as dcrf

from src.preprocessing.indices import exg, gli
from src.preprocessing.masks import glare_mask
from src.preprocessing.tensor_builder import (
    _local_std,
    _fft_row_periodicity_blocked,
)
from src.utils.constants import CLASS_TO_IDX, IGNORE_INDEX


# ---------------------------------------------------------------------------
# Dense CRF
# ---------------------------------------------------------------------------

def apply_crf(
    img_rgb: np.ndarray,
    prob_map: np.ndarray,
    n_classes: int,
    crf_iter: int = 5,
    gaussian_sxy: float = 3.0,
    gaussian_compat: float = 3.0,
    bilateral_sxy: float = 80.0,
    bilateral_srgb: float = 13.0,
    bilateral_compat: float = 10.0,
) -> np.ndarray:
    """
    Refine a probability map with Dense CRF.

    Args:
        img_rgb:   (H, W, 3) uint8
        prob_map:  (H, W, n_classes) float32
        n_classes: number of classes

    Returns:
        (H, W) int32 refined class map
    """
    h, w = img_rgb.shape[:2]
    # CRF expects (n_classes, H*W) C-contiguous float32
    prob_flat = prob_map.reshape(-1, n_classes).T.astype(np.float32)
    prob_flat = np.clip(prob_flat, 1e-8, 1.0)
    unary = np.ascontiguousarray(-np.log(prob_flat))

    d = dcrf.DenseCRF2D(w, h, n_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=gaussian_sxy, compat=gaussian_compat)
    d.addPairwiseBilateral(
        sxy=bilateral_sxy,
        srgb=bilateral_srgb,
        rgbim=np.ascontiguousarray(img_rgb),
        compat=bilateral_compat,
    )
    Q = np.array(d.inference(crf_iter)).reshape(n_classes, h, w)
    return Q.argmax(axis=0).astype(np.int32)


# ---------------------------------------------------------------------------
# Structural-cue overrides
# ---------------------------------------------------------------------------

def apply_structural_overrides(
    class_map: np.ndarray,
    prob_map: np.ndarray,
    img_rgb: np.ndarray,
    confidence_map: np.ndarray,
    local_std_thresh: float = 15.0,
    fft_row_thresh: float = 0.3,
    exg_drought_thresh: float = 0.15,
    exg_crop_thresh: float = -0.05,
) -> np.ndarray:
    """
    Apply rule-based corrections to the class map.
    Sets unreliable pixels to IGNORE_INDEX.

    Args:
        class_map:        (H, W) int32 argmax predictions
        prob_map:         (H, W, n_classes) float32
        img_rgb:          (H, W, 3) uint8
        confidence_map:   (H, W) float32 — max probability per pixel
        *_thresh:         tuning thresholds

    Returns:
        (H, W) int32 with some pixels set to IGNORE_INDEX (255)
    """
    import cv2
    result = class_map.copy()
    h, w   = result.shape

    rgb01  = img_rgb.astype(np.float32) / 255.0
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v_norm = hsv[:, :, 2] / 255.0

    exg_map  = exg(rgb01)   # (H, W)
    gli_map  = gli(rgb01)   # (H, W)
    lstd_map = _local_std(gray, ksize=5)
    # local_std is normalized to [-1,1] in tensor_builder; recover absolute value
    lstd_abs = (lstd_map * 0.5 + 0.5) * 40.0   # approx inversion of normalization

    idx = CLASS_TO_IDX

    # ---- Rule A: cotton_ball in flat-brightness region → ignore ----
    cotton_mask = result == idx.get('cotton_ball', -1)
    flat_bright = (lstd_abs < local_std_thresh) & (v_norm > 0.90)
    result[cotton_mask & flat_bright] = IGNORE_INDEX

    # ---- Rule B: drought surrounded by crop with row structure → soil ----
    drought_mask = result == idx.get('drought', -1)
    if drought_mask.any():
        fft_map = _fft_row_periodicity_blocked(gray, block=256)
        # fft_map is in [-1,1]; convert back: row_score = (fft+1)/2 * 0.2
        row_score = (fft_map + 1.0) / 2.0 * 0.2
        high_row  = row_score > fft_row_thresh

        # Crop neighbourhood: dilate crop mask
        crop_mask = result == idx.get('crop', -1)
        kernel    = np.ones((15, 15), np.uint8)
        import cv2 as _cv2
        crop_dilated = _cv2.dilate(crop_mask.astype(np.uint8), kernel) > 0

        relabel_drought = drought_mask & high_row & crop_dilated
        soil_idx = idx.get('soil', 1)
        result[relabel_drought] = soil_idx

    # ---- Rule C: crop in region with no greenness → ignore ----
    crop_mask = result == idx.get('crop', -1)
    not_green = (exg_map < exg_crop_thresh) & (gli_map < 0.0)
    result[crop_mask & not_green] = IGNORE_INDEX

    # ---- Rule D: glare pixels → ignore (regardless of class) ----
    glare = glare_mask(img_rgb, v_thresh=0.95, s_thresh=0.10)
    result[glare] = IGNORE_INDEX

    # ---- Rule E: drought pixel is clearly green → relabel to crop ----
    drought_mask = result == idx.get('drought', -1)  # refreshed after rule B
    clearly_green = exg_map > exg_drought_thresh
    crop_idx = idx.get('crop', 2)
    result[drought_mask & clearly_green] = crop_idx

    return result


# ---------------------------------------------------------------------------
# Combined post-processing
# ---------------------------------------------------------------------------

def postprocess_probmap(
    img_rgb: np.ndarray,
    prob_map: np.ndarray,
    n_classes: int,
    confidence_threshold: float = 0.55,
    use_crf: bool = True,
    crf_iter: int = 5,
    max_crf_pixels: int = 1024 * 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full post-processing pipeline for one image's probability map.

    Args:
        max_crf_pixels: If the image has more pixels than this, CRF is
            automatically skipped (Dense CRF is O(N log N) and becomes
            impractically slow on large orthomosaic tiles). Default 1M px
            (~1024×1024). For full UAV images (8064×4536 = 36M px) set
            use_crf=False or process tiles individually.

    Returns:
        class_map:        (H, W) uint8 — predicted class per pixel,
                          IGNORE_INDEX (255) for low-confidence / overridden pixels
        confidence_map:   (H, W) float32 — max probability per pixel
    """
    confidence_map = prob_map.max(axis=2)   # (H, W)
    h, w = img_rgb.shape[:2]

    crf_active = use_crf and (h * w <= max_crf_pixels)
    if use_crf and not crf_active:
        import warnings
        warnings.warn(
            f'CRF skipped: image {h}×{w} ({h*w/1e6:.1f}M px) exceeds '
            f'max_crf_pixels={max_crf_pixels/1e6:.1f}M. '
            'Use --no-crf flag or reduce max_crf_pixels to suppress this warning.',
            stacklevel=2,
        )

    if crf_active:
        class_map = apply_crf(img_rgb, prob_map, n_classes,
                              crf_iter=crf_iter).astype(np.int32)
    else:
        class_map = prob_map.argmax(axis=2).astype(np.int32)

    # Low-confidence pixels → ignore
    class_map[confidence_map < confidence_threshold] = IGNORE_INDEX

    # Structural overrides
    class_map = apply_structural_overrides(
        class_map, prob_map, img_rgb, confidence_map
    )

    # Final clamp: any remaining out-of-range class → ignore
    valid = (class_map >= 0) & (class_map < n_classes)
    class_map[~valid & (class_map != IGNORE_INDEX)] = IGNORE_INDEX

    return class_map.astype(np.uint8), confidence_map
