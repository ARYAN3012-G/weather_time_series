"""
Synthetic compositor for segmentation training.

Generates (image, mask) pairs by pasting labeled patches from data/patches/
onto random crops from data/unlabeled/.

Why this is critical
--------------------
The patch classifier hit a ceiling because patches lack spatial context.
The segmentation model needs to see classes ADJACENT to one another in
realistic configurations:
    - drought patch next to soil patch  → teaches the drought/soil boundary
    - drought patch next to dry_weed    → teaches drought/dry_weed
    - green_weed next to crop           → teaches weed/crop separation
    - small cotton_ball on soil         → teaches small-object detection

These adjacencies are RARE in raw pseudo-labels (the classifier already
struggles there) and HARD to find in the small clean set (only 25 tiles).
Synthetic composites provide unlimited training examples of exactly these
hard cases.

Design choices
--------------
- 60% of composites force one of the 5 confusion pairs side-by-side.
- 40% are random multi-class compositions.
- Feathered alpha edges (Gaussian sigma=5) prevent the model from learning
  to detect paste seams instead of real class boundaries.
- Patch sizes randomized within a realistic range (64-256 px on 512-px tile).
- Background tiles are randomly cropped from unlabeled/ to preserve
  domain-realistic illumination, soil colour, lens characteristics.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from src.utils.constants import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    CONFUSION_PAIRS,
    IGNORE_INDEX,
    NUM_CLASSES,
    PATCH_FOLDER_TO_CLASS,
)

# Tile size composites are generated at
DEFAULT_TILE_SIZE = 512
# Patch sizes in pixels (will be picked at random)
PATCH_SIZE_RANGE = (64, 256)
# Special small-patch range for cotton_ball (model misses small ones)
COTTON_SMALL_RANGE = (16, 80)
# How many patches per composite
N_PATCHES_RANGE = (3, 7)
# Probability of forcing a confusion-pair composition
CONFUSION_PAIR_PROB = 0.60
# Probability of forcing a small-cotton composition
SMALL_COTTON_PROB = 0.20
# Feather sigma for paste edges
FEATHER_SIGMA = 5.0


class SyntheticCompositor:
    """
    Builds (image, mask) training pairs on the fly.

    Args:
        patches_dir:   data/patches root (folders per class).
        backgrounds_dir: data/unlabeled root for background tiles.
        tile_size:     output square size.
        seed:          optional fixed seed for reproducibility.
    """

    def __init__(
        self,
        patches_dir: str | Path,
        backgrounds_dir: str | Path,
        tile_size: int = DEFAULT_TILE_SIZE,
        seed: int | None = None,
    ) -> None:
        self.patches_dir   = Path(patches_dir)
        self.backgrounds_dir = Path(backgrounds_dir)
        self.tile_size     = tile_size
        self.rng           = random.Random(seed)

        # Index all patches by class
        self._patches_by_class: dict[int, list[Path]] = {}
        for folder, cls_name in PATCH_FOLDER_TO_CLASS.items():
            cls_idx = CLASS_TO_IDX.get(cls_name)
            if cls_idx is None:
                continue
            folder_path = self.patches_dir / folder
            if not folder_path.is_dir():
                continue
            exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
            files = [p for p in folder_path.rglob('*') if p.suffix.lower() in exts]
            self._patches_by_class[cls_idx] = files

        if not self._patches_by_class:
            raise FileNotFoundError(
                f"No patches found under {patches_dir}. Check folder names."
            )

        # Index all background images
        bg_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        self._bg_files = [
            p for p in self.backgrounds_dir.rglob('*')
            if p.is_file() and p.suffix.lower() in bg_exts
        ]
        if not self._bg_files:
            raise FileNotFoundError(f"No backgrounds found under {backgrounds_dir}")

        # Confusion pairs as class indices
        self._confusion_pairs_idx = [
            (CLASS_TO_IDX[a], CLASS_TO_IDX[b])
            for a, b in CONFUSION_PAIRS
            if a in CLASS_TO_IDX and b in CLASS_TO_IDX
        ]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate one (image, mask) composite.

        Returns:
            image: (H, W, 3) uint8 RGB
            mask:  (H, W) uint8 — pixel values 0..6 (and 255 over un-pasted
                   background, treated as ignore so the model focuses on
                   the deliberate paste regions).
        """
        bg = self._load_background()
        # Mask starts as IGNORE everywhere — pasted patches will overwrite
        mask = np.full(bg.shape[:2], IGNORE_INDEX, dtype=np.uint8)

        # Decide composition type
        r = self.rng.random()
        if r < CONFUSION_PAIR_PROB:
            self._paste_confusion_pair(bg, mask)
        elif r < CONFUSION_PAIR_PROB + SMALL_COTTON_PROB:
            self._paste_small_cotton(bg, mask)
        else:
            self._paste_random_mix(bg, mask)

        return bg, mask

    # ------------------------------------------------------------------
    # Composition strategies
    # ------------------------------------------------------------------

    def _paste_confusion_pair(self, bg: np.ndarray, mask: np.ndarray) -> None:
        """Paste both classes of a confusion pair adjacent to each other."""
        cls_a, cls_b = self.rng.choice(self._confusion_pairs_idx)

        # Skip if either class has no patches
        if cls_a not in self._patches_by_class or cls_b not in self._patches_by_class:
            self._paste_random_mix(bg, mask)
            return

        # Pick a random center
        cx = self.rng.randint(self.tile_size // 4, 3 * self.tile_size // 4)
        cy = self.rng.randint(self.tile_size // 4, 3 * self.tile_size // 4)

        # Patch A on the left/top, Patch B on the right/bottom (random orientation)
        sz_a = self.rng.randint(*PATCH_SIZE_RANGE)
        sz_b = self.rng.randint(*PATCH_SIZE_RANGE)

        if self.rng.random() < 0.5:
            # Horizontal split
            x_a = max(0, cx - sz_a)
            y_a = max(0, cy - sz_a // 2)
            x_b = min(self.tile_size - sz_b, cx)
            y_b = max(0, cy - sz_b // 2)
        else:
            # Vertical split
            x_a = max(0, cx - sz_a // 2)
            y_a = max(0, cy - sz_a)
            x_b = max(0, cx - sz_b // 2)
            y_b = min(self.tile_size - sz_b, cy)

        self._paste_one(bg, mask, cls_a, sz_a, x_a, y_a)
        self._paste_one(bg, mask, cls_b, sz_b, x_b, y_b)

        # Add 1-3 more random patches to make it look like a real field
        for _ in range(self.rng.randint(1, 3)):
            cls = self.rng.choice(list(self._patches_by_class.keys()))
            sz  = self.rng.randint(*PATCH_SIZE_RANGE)
            x   = self.rng.randint(0, self.tile_size - sz)
            y   = self.rng.randint(0, self.tile_size - sz)
            self._paste_one(bg, mask, cls, sz, x, y)

    def _paste_small_cotton(self, bg: np.ndarray, mask: np.ndarray) -> None:
        """Generate composites with small cotton_ball patches on soil/crop bg."""
        cotton_idx = CLASS_TO_IDX.get('cotton_ball', 3)
        crop_idx   = CLASS_TO_IDX.get('crop', 2)
        soil_idx   = CLASS_TO_IDX.get('soil', 1)

        if cotton_idx not in self._patches_by_class:
            self._paste_random_mix(bg, mask)
            return

        # Paste a base crop or soil region
        for _ in range(self.rng.randint(1, 3)):
            cls = self.rng.choice([crop_idx, soil_idx])
            if cls not in self._patches_by_class:
                continue
            sz  = self.rng.randint(150, 300)
            x   = self.rng.randint(0, self.tile_size - sz)
            y   = self.rng.randint(0, self.tile_size - sz)
            self._paste_one(bg, mask, cls, sz, x, y)

        # Sprinkle small cotton balls
        n_cotton = self.rng.randint(3, 10)
        for _ in range(n_cotton):
            sz = self.rng.randint(*COTTON_SMALL_RANGE)
            x  = self.rng.randint(0, self.tile_size - sz)
            y  = self.rng.randint(0, self.tile_size - sz)
            self._paste_one(bg, mask, cotton_idx, sz, x, y)

    def _paste_random_mix(self, bg: np.ndarray, mask: np.ndarray) -> None:
        """Paste 3-7 random-class patches anywhere on the tile."""
        n = self.rng.randint(*N_PATCHES_RANGE)
        for _ in range(n):
            cls = self.rng.choice(list(self._patches_by_class.keys()))
            sz  = self.rng.randint(*PATCH_SIZE_RANGE)
            x   = self.rng.randint(0, self.tile_size - sz)
            y   = self.rng.randint(0, self.tile_size - sz)
            self._paste_one(bg, mask, cls, sz, x, y)

    # ------------------------------------------------------------------
    # Low-level paste
    # ------------------------------------------------------------------

    def _paste_one(
        self,
        bg: np.ndarray,
        mask: np.ndarray,
        cls_idx: int,
        size: int,
        x: int,
        y: int,
    ) -> None:
        """Paste one patch into bg at (x, y) with feathered edges + update mask."""
        files = self._patches_by_class.get(cls_idx)
        if not files:
            return
        path = self.rng.choice(files)
        patch = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if patch is None:
            return
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        # Resize to target size
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)

        # Build feathered alpha
        alpha = np.ones((size, size), dtype=np.float32)
        feather = max(3, int(size * 0.08))
        # Linear fade from border
        for i in range(feather):
            v = i / feather
            alpha[i, :]            = np.minimum(alpha[i, :], v)
            alpha[-1-i, :]         = np.minimum(alpha[-1-i, :], v)
            alpha[:, i]            = np.minimum(alpha[:, i], v)
            alpha[:, -1-i]         = np.minimum(alpha[:, -1-i], v)
        # Smooth with Gaussian
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=FEATHER_SIGMA)
        alpha = alpha.clip(0, 1)

        # Bounds check
        h, w = bg.shape[:2]
        x2 = min(x + size, w)
        y2 = min(y + size, h)
        psz_x = x2 - x
        psz_y = y2 - y
        if psz_x <= 0 or psz_y <= 0:
            return
        patch = patch[:psz_y, :psz_x]
        alpha = alpha[:psz_y, :psz_x]

        # Composite: bg' = alpha * patch + (1-alpha) * bg
        a3 = alpha[..., None]
        bg[y:y2, x:x2] = (
            a3 * patch.astype(np.float32) +
            (1 - a3) * bg[y:y2, x:x2].astype(np.float32)
        ).astype(np.uint8)

        # Update mask wherever alpha is meaningfully nonzero (avoid edge blur in mask)
        # Use a stricter threshold so mask doesn't bleed into background
        update = alpha > 0.5
        mask[y:y2, x:x2][update] = cls_idx

    def _load_background(self) -> np.ndarray:
        """Load a random TILE_SIZE crop from a random unlabeled image."""
        for _ in range(10):                          # retry on read failures
            path = self.rng.choice(self._bg_files)
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            if h < self.tile_size or w < self.tile_size:
                # Image too small — resize up
                scale = max(self.tile_size / h, self.tile_size / w) * 1.05
                img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_LINEAR)
                h, w = img.shape[:2]
            y = self.rng.randint(0, h - self.tile_size)
            x = self.rng.randint(0, w - self.tile_size)
            return img[y:y+self.tile_size, x:x+self.tile_size].copy()
        # Fallback if we couldn't read any: synthetic noise
        return np.random.randint(60, 180, (self.tile_size, self.tile_size, 3),
                                 dtype=np.uint8)
