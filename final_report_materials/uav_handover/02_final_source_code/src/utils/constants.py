"""Shared constants for the drought detection pipeline."""
from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Classes (must match configs/config.yaml exactly)
# -----------------------------------------------------------------------------
CLASS_NAMES: tuple[str, ...] = (
    "background",   # 0 — outside valid field mask
    "soil",         # 1 — bare cultivated soil (barren merged here)
    "crop",         # 2 — healthy/maturing vegetation (maize, redgram, cotton)
    "cotton_ball",  # 3 — open cotton bolls (harvest indicator)
    "green_weed",   # 4
    "dry_weed",     # 5
    "drought",      # 6
)

NUM_CLASSES = len(CLASS_NAMES)

# index ↔ name maps
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}

# RGB palette (uint8). Matches CLASS_NAMES order.
PALETTE: np.ndarray = np.array(
    [
        [0, 0, 0],          # 0 background — black
        [139, 90, 43],      # 1 soil — brown (barren merged)
        [0, 180, 0],        # 2 crop — green
        [255, 255, 255],    # 3 cotton_ball — white
        [140, 255, 140],    # 4 green_weed — light green
        [210, 170, 90],     # 5 dry_weed — tan
        [220, 30, 30],      # 6 drought — red
    ],
    dtype=np.uint8,
)

# Pairs the model historically confuses. Used by the margin loss in the classifier.
CONFUSION_PAIRS: tuple[tuple[str, str], ...] = (
    ("drought", "dry_weed"),
    ("drought", "soil"),      # barren merged into soil
    ("drought", "crop"),
    ("soil", "cotton_ball"),
    ("crop", "green_weed"),
)

# Ignore index for pseudo-mask training
IGNORE_INDEX: int = 255

# Folder name → class name (handles patches/ folder mapping)
PATCH_FOLDER_TO_CLASS: dict[str, str] = {
    "crop": "crop",
    "cotton_ball": "cotton_ball",
    "dry_weed": "dry_weed",
    "green_weed": "green_weed",
    "drought": "drought",
    "soil": "soil",
    # barren merged into soil — move barren/ images into soil/ before training
    # 'background' is computed at inference time, never has training patches
}
