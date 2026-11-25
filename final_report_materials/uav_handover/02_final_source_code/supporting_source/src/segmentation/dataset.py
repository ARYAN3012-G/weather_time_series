"""
Segmentation training dataset.

Combines three sources, each contributing per-batch in fixed proportions:

    CLEAN:      data/clean_masks/ (hand-corrected)
                + corresponding tiles from data/clean_images/
                Loss weight: 5.0 (high confidence — gold standard)

    PSEUDO:     runs/pseudo_labels/masks/ (noisy classifier predictions)
                + corresponding originals from data/unlabeled/
                Loss weight: 1.0 (large quantity, noisy)

    COMPOSITE:  generated on the fly by SyntheticCompositor
                Loss weight: 1.0 (unlimited, exact masks)

All sources return the same dict format:
    {
        'image':       (9, H, W) float32 tensor (RGB + 5 indices + V)
        'mask':        (H, W)    int64    class indices (IGNORE_INDEX=255 for ignored)
        'source':      str       'clean' | 'pseudo' | 'composite'
        'loss_weight': float     per-sample loss weight
        'path':        str       source path for debugging
    }
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.tensor_builder import build_9channel
from src.segmentation.compositor import SyntheticCompositor
from src.utils.constants import IGNORE_INDEX, NUM_CLASSES


# Per-source loss weights (clean is more trustworthy → higher weight)
# Higher clean weight reflects: pseudo-labels are heavily biased toward
# 'crop' (~84%) while clean masks accurately reflect the field composition
# (soil 64%, crop 5%, etc). Pseudo-labels get a low weight so they don't
# pull soil predictions toward 'crop'.
LOSS_WEIGHTS = {
    'clean':     8.0,
    'pseudo':    0.5,
    'composite': 1.0,
}


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment_image_and_mask(
    img: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    train: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Geometric and color augmentation. For training, applies random crop
    + flips + 90° rotations + HSV jitter. For val, just centre-crops.
    """
    h, w = img.shape[:2]

    if train:
        # Random crop
        if h > crop_size:
            y = random.randint(0, h - crop_size)
        else:
            y = 0
        if w > crop_size:
            x = random.randint(0, w - crop_size)
        else:
            x = 0
    else:
        # Centre crop
        y = max(0, (h - crop_size) // 2)
        x = max(0, (w - crop_size) // 2)

    # Crop (and pad with reflection if smaller than crop_size)
    img  = img[y:y+crop_size, x:x+crop_size]
    mask = mask[y:y+crop_size, x:x+crop_size]
    if img.shape[0] < crop_size or img.shape[1] < crop_size:
        pad_y = crop_size - img.shape[0]
        pad_x = crop_size - img.shape[1]
        img  = cv2.copyMakeBorder(img,  0, pad_y, 0, pad_x,
                                  cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, pad_y, 0, pad_x,
                                  cv2.BORDER_CONSTANT,
                                  value=IGNORE_INDEX)

    if train:
        # Flips
        if random.random() < 0.5:
            img  = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.5:
            img  = np.ascontiguousarray(img[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        # 90° rotations
        k = random.randint(0, 3)
        if k:
            img  = np.ascontiguousarray(np.rot90(img,  k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
        # Mild HSV jitter
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-25, 25), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-25, 25), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return img, mask


# ---------------------------------------------------------------------------
# Sub-datasets
# ---------------------------------------------------------------------------

class CleanMaskDataset(Dataset):
    """Hand-corrected (image, mask) pairs."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        crop_size: int = 512,
        train: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.crop_size  = crop_size
        self.train      = train

        # Pair files by stem
        mask_files = sorted(self.masks_dir.glob('*.png'))
        self.pairs: list[tuple[Path, Path]] = []
        for m in mask_files:
            # Image could be .jpg or .png
            for ext in ('.jpg', '.jpeg', '.png'):
                img = self.images_dir / (m.stem + ext)
                if img.exists():
                    self.pairs.append((img, m))
                    break

        if not self.pairs:
            raise FileNotFoundError(
                f'No (image, mask) pairs in {images_dir} ↔ {masks_dir}. '
                'Run scripts/prepare_clean_set.py first.'
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask    = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img, mask = _augment_image_and_mask(img, mask, self.crop_size, self.train)
        return {
            'image':       torch.from_numpy(build_9channel(img)),
            'mask':        torch.from_numpy(mask.astype(np.int64)),
            'source':      'clean',
            'loss_weight': LOSS_WEIGHTS['clean'],
            'path':        str(img_path),
        }


class PseudoLabelDataset(Dataset):
    """Pseudo-labeled (image, mask) pairs from the full unlabeled run."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        crop_size: int = 512,
        train: bool = True,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.crop_size  = crop_size
        self.train      = train

        mask_files = sorted(self.masks_dir.glob('*.png'))
        self.pairs: list[tuple[Path, Path]] = []
        for m in mask_files:
            for ext in ('.jpg', '.jpeg', '.png'):
                img = self.images_dir / (m.stem + ext)
                if img.exists():
                    self.pairs.append((img, m))
                    break

        if not self.pairs:
            raise FileNotFoundError(
                f'No (image, mask) pairs in {images_dir} ↔ {masks_dir}. '
                'Run pseudo-labeling first.'
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask    = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Random crop from this large image
        img, mask = _augment_image_and_mask(img, mask, self.crop_size, self.train)
        return {
            'image':       torch.from_numpy(build_9channel(img)),
            'mask':        torch.from_numpy(mask.astype(np.int64)),
            'source':      'pseudo',
            'loss_weight': LOSS_WEIGHTS['pseudo'],
            'path':        str(img_path),
        }


class CompositeDataset(Dataset):
    """On-the-fly synthetic composites. Returns a fixed virtual length per epoch."""

    def __init__(
        self,
        patches_dir: str | Path,
        backgrounds_dir: str | Path,
        crop_size: int = 512,
        virtual_length: int = 1000,
        train: bool = True,
        seed: int | None = None,
    ) -> None:
        self.compositor = SyntheticCompositor(
            patches_dir=patches_dir,
            backgrounds_dir=backgrounds_dir,
            tile_size=crop_size,
            seed=seed,
        )
        self.crop_size      = crop_size
        self.virtual_length = virtual_length
        self.train          = train

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, idx: int) -> dict:
        img, mask = self.compositor.sample()
        # Composites already at crop_size; still apply train-time flips/HSV
        if self.train:
            img, mask = _augment_image_and_mask(img, mask, self.crop_size, train=True)
        return {
            'image':       torch.from_numpy(build_9channel(img)),
            'mask':        torch.from_numpy(mask.astype(np.int64)),
            'source':      'composite',
            'loss_weight': LOSS_WEIGHTS['composite'],
            'path':        'synthetic',
        }


# ---------------------------------------------------------------------------
# Mixed dataset / sampler
# ---------------------------------------------------------------------------

class MixedSegmentationDataset(Dataset):
    """
    Mixes the three sources at fixed ratios per "virtual" sample index.

    With ratios (clean=4, pseudo=4, composite=8) in a batch of 16, every
    full pass through this dataset draws each source proportionally.
    The composite source effectively oversamples the clean+pseudo sources
    by repeating their indices.
    """

    def __init__(
        self,
        clean: CleanMaskDataset | None,
        pseudo: PseudoLabelDataset | None,
        composite: CompositeDataset | None,
        ratio_clean: int = 4,
        ratio_pseudo: int = 4,
        ratio_composite: int = 8,
    ) -> None:
        self.sources    : list[Dataset]       = []
        self.weights    : list[int]           = []
        if clean is not None and len(clean) > 0:
            self.sources.append(clean); self.weights.append(ratio_clean)
        if pseudo is not None and len(pseudo) > 0:
            self.sources.append(pseudo); self.weights.append(ratio_pseudo)
        if composite is not None and len(composite) > 0:
            self.sources.append(composite); self.weights.append(ratio_composite)
        assert self.sources, "MixedSegmentationDataset has no sources"

        # Compute virtual length: large enough that one epoch passes
        # through composite ~once and respects the ratios
        composite_len = (len(composite) if composite is not None else
                         max(len(s) for s in self.sources))
        total_ratio = sum(self.weights)
        self.virtual_length = (
            composite_len * total_ratio // max(ratio_composite, 1)
        )

        # Build sampling indices, refreshed at each training epoch.
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        total_ratio = sum(self.weights)
        self._table: list[tuple[int, int]] = []
        for src_idx, (src, w) in enumerate(zip(self.sources, self.weights)):
            n_appear = (self.virtual_length * w) // total_ratio
            offset = (epoch * n_appear) % len(src)
            for i in range(n_appear):
                self._table.append((src_idx, (offset + i) % len(src)))
        random.Random(epoch).shuffle(self._table)
        self.virtual_length = len(self._table)

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, idx: int) -> dict:
        src_idx, sub_idx = self._table[idx]
        return self.sources[src_idx][sub_idx]
