"""Preprocessing utilities for drought detection."""
from .gsd import (
    CameraIntrinsics,
    gsd_mm_per_pixel,
    pixel_area_m2,
    pixels_to_hectares,
    pixels_to_m2,
)
from .indices import (
    INDEX_FUNCS,
    INDEX_NAMES,
    compute_all_indices,
    cive,
    exg,
    exgr,
    gli,
    ngrdi,
)
from .masks import (
    compute_aux_masks,
    glare_mask,
    shadow_mask,
    valid_field_mask,
)
from .tensor_builder import (
    CHANNEL_NAMES_9,
    CHANNEL_NAMES_12,
    build_9channel,
    build_12channel,
)

__all__ = [
    # gsd
    "CameraIntrinsics",
    "gsd_mm_per_pixel",
    "pixel_area_m2",
    "pixels_to_hectares",
    "pixels_to_m2",
    # indices
    "INDEX_FUNCS",
    "INDEX_NAMES",
    "compute_all_indices",
    "cive",
    "exg",
    "exgr",
    "gli",
    "ngrdi",
    # masks
    "compute_aux_masks",
    "glare_mask",
    "shadow_mask",
    "valid_field_mask",
    # tensor_builder
    "CHANNEL_NAMES_9",
    "CHANNEL_NAMES_12",
    "build_9channel",
    "build_12channel",
]
