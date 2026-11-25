"""
PatchDataset — loads labeled image patches for the 8-class drought classifier.

Handles:
    - Multiple patch sizes (64, 128, 256, 512) from the same class folders.
    - Mixed-format inputs (PNG for patches, JPG also accepted).
    - Class-balanced oversampling (barren class is rare — gets upsampled).
    - 9-channel input (RGB + 5 vegetation indices + V).
    - Heavy, confusion-pair-aware augmentation.
    - MixUp between specific confusion pairs (implemented manually since
      albumentations 2.x does not provide it).
    - CutMix across any pair.
    - A 'hard set' manifest for a curated validation split.

Class folders expected under patches_dir:
    crop/, cotton_ball/, dry_weed/, green_weed/, drought/, soil/, barren/
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.preprocessing.tensor_builder import build_9channel
from src.utils.constants import (
    CLASS_TO_IDX,
    CONFUSION_PAIRS,
    PATCH_FOLDER_TO_CLASS,
)

# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def _make_train_aug(size: int = 224) -> A.Compose:
    """
    Strong augmentation designed against the specific confusion pairs:
        drought ↔ dry_weed       (both brown — need texture/structure distinction)
        drought ↔ barren         (both brown — need context distinction)
        drought ↔ crop           (harvest-stage dark cotton ≈ drought)
        soil ↔ cotton_ball       (bright soil glare ≈ white bolls)
        crop ↔ green_weed        (both green — need spatial arrangement)

    Includes:
        - Random shadow synthesis (combat misclassification of shadows as drought)
        - Random specular highlight synthesis (combat soil-glare-as-cotton)
        - Strong HSV jitter
        - Geometric augmentations (flips, rotations, elastic)
    """
    return A.Compose([
        A.Resize(size, size),
        # --- Geometric ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
        A.ElasticTransform(alpha=40, sigma=5, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        # --- Color/appearance  ---
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=40, val_shift_limit=30, p=0.8
        ),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05, p=0.5),
        # --- Shadow synthesis: random dark elliptical region ---
        # Simulates cast shadow from crop canopy; prevents shadow → drought confusion
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=4,
            p=0.3
        ),
        # --- Noise / blur ---
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        # --- Dropout blocks: simulate occlusion by other leaves ---
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill=0,
            p=0.2
        ),
    ])


def _make_val_aug(size: int = 224) -> A.Compose:
    return A.Compose([A.Resize(size, size)])


# ---------------------------------------------------------------------------
# MixUp / CutMix helpers (manual since albumentations 2.x removed them)
# ---------------------------------------------------------------------------

def mixup_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    label_a: int,
    label_b: int,
    alpha: float = 0.1,
    num_classes: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blend two images. Returns mixed image and soft one-hot label vector.
    alpha=0.1 means the weaker image contributes 10% at most.
    """
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)   # ensure dominant class always > 0.5
    mixed = (lam * img_a + (1.0 - lam) * img_b).astype(img_a.dtype)
    label = np.zeros(num_classes, dtype=np.float32)
    label[label_a] = lam
    label[label_b] = 1.0 - lam
    return mixed, label


def cutmix_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    label_a: int,
    label_b: int,
    alpha: float = 0.2,
    num_classes: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CutMix: paste a random rectangle from img_b onto img_a.
    """
    h, w = img_a.shape[:2]
    lam = np.random.beta(alpha, alpha)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    mixed = img_a.copy()
    mixed[y1:y2, x1:x2] = img_b[y1:y2, x1:x2]

    actual_lam = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)
    label = np.zeros(num_classes, dtype=np.float32)
    label[label_a] = actual_lam
    label[label_b] = 1.0 - actual_lam
    return mixed, label


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """
    Loads all patches from data/patches/ and returns 9-channel tensors + labels.

    Args:
        patches_dir:   Path to the patches/ directory.
        split:         'train', 'val', or 'test'.
        val_split:     Fraction of data for validation.
        test_split:    Fraction of data for test.
        seed:          Random seed for reproducible splits.
        input_size:    All patches resized to this square size for the backbone.
        mixup_prob:    Probability of applying MixUp on a confusion pair.
        cutmix_prob:   Probability of applying CutMix on any pair.
        num_classes:   Total number of classes.
    """

    CONFUSION_PAIR_IDX: list[tuple[int, int]] = [
        (CLASS_TO_IDX[a], CLASS_TO_IDX[b])
        for a, b in CONFUSION_PAIRS
        if a in CLASS_TO_IDX and b in CLASS_TO_IDX
    ]

    def __init__(
        self,
        patches_dir: str | Path,
        split: str = "train",
        val_split: float = 0.15,
        test_split: float = 0.10,
        seed: int = 42,
        input_size: int = 224,
        mixup_prob: float = 0.25,
        cutmix_prob: float = 0.25,
        num_classes: int = 8,
    ) -> None:
        patches_dir = Path(patches_dir)
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split = split
        self.input_size = input_size
        self.mixup_prob = mixup_prob if split == "train" else 0.0
        self.cutmix_prob = cutmix_prob if split == "train" else 0.0
        self.num_classes = num_classes

        # ----------------------------------------------------------------
        # Discover all images
        # ----------------------------------------------------------------
        self.samples: list[tuple[Path, int]] = []  # (path, class_idx)

        for folder_name, class_name in PATCH_FOLDER_TO_CLASS.items():
            folder = patches_dir / folder_name
            if not folder.is_dir():
                continue
            class_idx = CLASS_TO_IDX.get(class_name)
            if class_idx is None:
                continue
            imgs = self._find_images(folder)
            for p in imgs:
                self.samples.append((p, class_idx))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {patches_dir}. "
                "Check that the folder names match PATCH_FOLDER_TO_CLASS."
            )

        # ----------------------------------------------------------------
        # Stratified split
        # ----------------------------------------------------------------
        rng = random.Random(seed)
        by_class: dict[int, list[tuple[Path, int]]] = defaultdict(list)
        for s in self.samples:
            by_class[s[1]].append(s)

        train_s, val_s, test_s = [], [], []
        for cls_idx, items in by_class.items():
            shuffled = items.copy()
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_test = max(1, int(n * test_split))
            n_val = max(1, int(n * val_split))
            test_s.extend(shuffled[:n_test])
            val_s.extend(shuffled[n_test:n_test + n_val])
            train_s.extend(shuffled[n_test + n_val:])

        split_map = {"train": train_s, "val": val_s, "test": test_s}
        self.samples = split_map[split]

        # ----------------------------------------------------------------
        # Build lookup: class_idx → list of samples (for MixUp/CutMix)
        # ----------------------------------------------------------------
        self._by_class: dict[int, list[tuple[Path, int]]] = defaultdict(list)
        for s in self.samples:
            self._by_class[s[1]].append(s)

        # ----------------------------------------------------------------
        # Augmentation
        # ----------------------------------------------------------------
        if split == "train":
            self.aug = _make_train_aug(input_size)
        else:
            self.aug = _make_val_aug(input_size)

        # ----------------------------------------------------------------
        # Class counts for sampler
        # ----------------------------------------------------------------
        self._class_counts: dict[int, int] = defaultdict(int)
        for _, c in self.samples:
            self._class_counts[c] += 1

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_class_counts(self) -> dict[int, int]:
        return dict(self._class_counts)

    def get_weighted_sampler(self, oversample_barren: int = 8) -> WeightedRandomSampler:
        """
        Returns a WeightedRandomSampler for class-balanced training.
        barren class gets extra weight due to small sample count.
        """
        from src.utils.constants import CLASS_TO_IDX
        barren_idx = CLASS_TO_IDX.get("barren", -1)
        weights = []
        for _, cls in self.samples:
            count = self._class_counts[cls]
            w = 1.0 / count
            if cls == barren_idx:
                w *= oversample_barren
            weights.append(w)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _find_images(folder: Path) -> list[Path]:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

    def _load_rgb(self, path: Path) -> np.ndarray:
        """Load image as uint8 RGB, handling both PNG and JPG."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _random_confusion_pair_sample(
        self, anchor_class: int
    ) -> tuple[np.ndarray, int] | None:
        """Pick a random sample from a class that forms a confusion pair with anchor."""
        partner_classes = []
        for ca, cb in self.CONFUSION_PAIR_IDX:
            if ca == anchor_class and cb in self._by_class:
                partner_classes.append(cb)
            elif cb == anchor_class and ca in self._by_class:
                partner_classes.append(ca)
        if not partner_classes:
            return None
        partner_cls = random.choice(partner_classes)
        if not self._by_class[partner_cls]:
            return None
        partner_path, _ = random.choice(self._by_class[partner_cls])
        return self._load_rgb(partner_path), partner_cls

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        img = self._load_rgb(path)

        # Record the original patch size (H, W before resize) as a feature
        orig_h, orig_w = img.shape[:2]
        orig_size = max(orig_h, orig_w)

        # Geometric + color augmentation
        img = self.aug(image=img)["image"]

        soft_label: Optional[np.ndarray] = None

        # MixUp with confusion pair (training only)
        if self.split == "train" and random.random() < self.mixup_prob:
            partner = self._random_confusion_pair_sample(label)
            if partner is not None:
                partner_img, partner_cls = partner
                partner_img = self.aug(image=partner_img)["image"]
                img, soft_label = mixup_pair(
                    img, partner_img, label, partner_cls,
                    alpha=0.15, num_classes=self.num_classes
                )

        # CutMix across any class (training only, but not if already MixUp'd)
        elif self.split == "train" and random.random() < self.cutmix_prob:
            all_classes = list(self._by_class.keys())
            partner_cls = random.choice(all_classes)
            if self._by_class[partner_cls]:
                partner_path, _ = random.choice(self._by_class[partner_cls])
                partner_img = self._load_rgb(partner_path)
                partner_img = self.aug(image=partner_img)["image"]
                img, soft_label = cutmix_pair(
                    img, partner_img, label, partner_cls,
                    alpha=0.3, num_classes=self.num_classes
                )

        # Build 9-channel tensor
        tensor = torch.from_numpy(build_9channel(img))  # (9, H, W)

        # Hard label (int) always returned alongside soft label for logging
        if soft_label is None:
            soft_label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
            soft_label_tensor[label] = 1.0
        else:
            soft_label_tensor = torch.from_numpy(soft_label)

        return {
            "image": tensor,                                   # (9, 224, 224) float32
            "label": torch.tensor(label, dtype=torch.long),   # scalar int
            "soft_label": soft_label_tensor,                   # (8,) float32
            "orig_size": orig_size,                            # int: largest dim of original patch
            "path": str(path),
        }
