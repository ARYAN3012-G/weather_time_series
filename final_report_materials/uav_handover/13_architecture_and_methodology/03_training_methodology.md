# 03 training methodology

Use the following exact implementation as the authoritative technical reference.

## src/segmentation/train_v11.py

```python
"""
SegFormer-B2 training loop.

Usage:
    python -m src.segmentation.train --config configs/config.yaml

What it does:
    1. Builds mixed dataset (clean + pseudo + synthetic).
    2. Loads SegFormer-B2 with 9-channel input adapter.
    3. Computes per-class loss weights from training pixel counts.
    4. Trains with CE + Dice + Lovász, with per-sample loss weights.
    5. Validates each epoch on held-out clean tiles.
    6. Saves best mIoU checkpoint, writes training_report.json.
    7. Dumps val_predictions/ with PNG overlays of best-epoch predictions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.preprocessing.tensor_builder import build_9channel
from src.segmentation.dataset import (
    CleanMaskDataset,
    CompositeDataset,
    MixedSegmentationDataset,
    PseudoLabelDataset,
    _augment_image_and_mask,
)
from src.segmentation.losses import SegmentationLoss, make_class_weights
from src.segmentation.model import SegFormerB2
from src.utils.constants import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IGNORE_INDEX,
    NUM_CLASSES,
    PALETTE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Stack the dicts from PatchDataset → batched dict."""
    images       = torch.stack([b['image'] for b in batch])
    masks        = torch.stack([b['mask']  for b in batch])
    weights      = torch.tensor([b['loss_weight'] for b in batch],
                                dtype=torch.float32)
    sources      = [b['source'] for b in batch]
    return {
        'image':         images,
        'mask':          masks,
        'sample_weight': weights,
        'source':        sources,
    }


def compute_iou_per_class(
    confusion: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Per-class IoU from a (num_classes, num_classes) confusion matrix
    where confusion[true, pred] = pixel count.
    """
    iou_per = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = tp + fp + fn
        iou_per[c] = tp / denom if denom > 0 else np.nan
    miou = np.nanmean(iou_per)
    return iou_per, miou


def update_confusion(
    confusion: np.ndarray,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    """Add to confusion matrix in-place. preds/targets are torch (B, H, W)."""
    valid = (targets != IGNORE_INDEX)
    p = preds[valid].cpu().numpy().astype(np.int32)
    t = targets[valid].cpu().numpy().astype(np.int32)
    # bincount approach is fast
    idx = t * num_classes + p
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes)


def save_prediction_overlay(
    img_rgb: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: Path,
) -> None:
    """Save a 3-panel comparison: original | true mask | predicted mask."""
    H, W = img_rgb.shape[:2]

    def colorize(m):
        out = np.zeros((*m.shape, 3), dtype=np.uint8)
        for i, c in enumerate(PALETTE):
            out[m == i] = c
        out[m == IGNORE_INDEX] = (90, 90, 90)
        return out

    true_color = colorize(true_mask)
    pred_color = colorize(pred_mask)

    # Side-by-side
    composite = np.hstack([img_rgb, true_color, pred_color])

    # Label bar at top
    bar = np.full((30, composite.shape[1], 3), 230, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar, 'Original', (10, 22), font, 0.55, (20,20,20), 1, cv2.LINE_AA)
    cv2.putText(bar, 'Ground truth', (W + 10, 22), font, 0.55, (20,20,20), 1, cv2.LINE_AA)
    cv2.putText(bar, 'Prediction', (2*W + 10, 22), font, 0.55, (20,20,20), 1, cv2.LINE_AA)
    composite = np.vstack([bar, composite])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(config: dict, resume_path=None) -> None:
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU:    {torch.cuda.get_device_name(0)}')
        print(f'VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    slurm_keys = ['SLURM_JOB_ID','SLURM_JOB_NAME','SLURM_NODELIST','SLURM_CPUS_PER_TASK']
    slurm_info = {k: os.environ[k] for k in slurm_keys if k in os.environ}
    if slurm_info:
        print('SLURM:', slurm_info)

    paths   = config['paths']
    cfg_seg = config['segmentation']

    out_dir = Path(paths['runs_dir']) / 'segmentation'
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir / cfg_seg.get('run_name', 'default') / 'tb_logs')

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    crop_size = cfg_seg['input_size']

    clean_images_dir = Path(paths.get('clean_images_dir',
                                      'data/clean_images'))
    clean_masks_dir  = Path(paths.get('clean_masks_dir',
                                      'data/clean_masks'))

    # Stratified train/val split: ensure drought-rich tiles go to val
    all_clean_masks = sorted(
        m for m in clean_masks_dir.glob('*.png')
        if any(
            (clean_images_dir / (m.stem + ext)).is_file()
            for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff',
                        '.PNG', '.JPG', '.JPEG', '.TIF', '.TIFF')
        )
    )
    if len(all_clean_masks) < 10:
        print(f'[WARNING] Only {len(all_clean_masks)} clean masks found. '
              'Validation will be unreliable.')
    val_n = max(3, int(len(all_clean_masks) * 0.2))

    # Compute drought fraction per tile for stratification
    drought_fracs = {}
    for m in all_clean_masks:
        mask_arr = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        valid    = mask_arr != 255
        n_valid  = max(valid.sum(), 1)
        drought_fracs[m.stem] = float((mask_arr == 6).sum()) / n_valid

    # Sort by drought fraction descending, interleave into val/train
    sorted_stems = sorted(drought_fracs, key=drought_fracs.get, reverse=True)
    # Take every (1/val_ratio)-th tile from the top-drought end for val
    val_stems = set()
    step = max(1, len(sorted_stems) // val_n)
    for i in range(0, len(sorted_stems), step):
        if len(val_stems) >= val_n:
            break
        val_stems.add(sorted_stems[i])
    # Fallback: fill val_n if not reached
    for s in sorted_stems:
        if len(val_stems) >= val_n: break
        val_stems.add(s)

    print(f'Clean masks total: {len(all_clean_masks)}, val: {len(val_stems)}')
    print(f'Val tile drought fracs: {[round(drought_fracs[s],3) for s in val_stems]}')

    # Build sub-datasets
    # CleanMaskDataset filters by directory listing — for the train/val split
    # we create two filtered subfolder views using symlinks-or-list
    # Simpler: instantiate one ClanMaskDataset and split via Subset
    full_clean = CleanMaskDataset(
        images_dir=clean_images_dir,
        masks_dir=clean_masks_dir,
        crop_size=crop_size,
        train=True,
    )
    train_idx, val_idx = [], []
    for i, (img, mask) in enumerate(full_clean.pairs):
        if mask.stem in val_stems:
            val_idx.append(i)
        else:
            train_idx.append(i)
    print(f'  Clean train: {len(train_idx)}, val: {len(val_idx)}')

    from torch.utils.data import Subset
    clean_train = Subset(full_clean, train_idx) if train_idx else None

    # Val-mode clean (no augmentation, centre crop)
    full_clean_val = CleanMaskDataset(
        images_dir=clean_images_dir,
        masks_dir=clean_masks_dir,
        crop_size=crop_size,
        train=False,
    )
    val_subset = Subset(full_clean_val, val_idx) if val_idx else None

    pseudo_train = PseudoLabelDataset(
        images_dir=paths['unlabeled_dir'],
        masks_dir=Path(paths.get('pseudo_masks_dir', '/content/v5_data/pseudo_masks')),
        crop_size=crop_size,
        train=True,
    )

    composite_train = CompositeDataset(
        patches_dir=paths['patches_dir'],
        backgrounds_dir=paths['unlabeled_dir'],
        crop_size=crop_size,
        virtual_length=cfg_seg.get('composite_per_epoch', 1000),
        train=True,
        seed=cfg_seg.get('seed', 42),
    )

    mixed = MixedSegmentationDataset(
        clean=clean_train,
        pseudo=pseudo_train,
        composite=composite_train,
        ratio_clean    = cfg_seg.get('batch_manual',    4),
        ratio_pseudo   = cfg_seg.get('batch_pseudo',    4),
        ratio_composite= cfg_seg.get('batch_composites', 8),
    )
    print(f'  Mixed train length: {len(mixed)}')
    print(f'  Composite per epoch: {cfg_seg.get("composite_per_epoch", 1000)}')

    dl_train = DataLoader(
        mixed,
        batch_size=cfg_seg['batch_size'],
        shuffle=True,
        num_workers=cfg_seg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    dl_val = (
        DataLoader(val_subset,
                   batch_size=cfg_seg['batch_size'],
                   shuffle=False, num_workers=2,
                   pin_memory=True, collate_fn=collate_fn)
        if val_subset is not None else None
    )

    # ------------------------------------------------------------------
    # Class pixel counts (from training clean masks, for weighting)
    # ------------------------------------------------------------------
    print('\nComputing class pixel counts from clean training masks...')
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    for i in train_idx:
        _, mask_path = full_clean.pairs[i]
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        for c in range(NUM_CLASSES):
            class_counts[c] += int((m == c).sum())
    total = max(sum(class_counts.values()), 1)
    print('  Pixel distribution (training clean):')
    for c in range(NUM_CLASSES):
        pct = class_counts[c] / total * 100
        print(f'    {CLASS_NAMES[c]:14} {pct:5.2f}%')

    drought_idx    = CLASS_TO_IDX.get('drought', 6)
    cotton_idx     = CLASS_TO_IDX.get('cotton_ball', 3)

    # Drought weight override: force high weight so model must learn drought.
    drought_override = cfg_seg.get('drought_override_weight', 30.0)

    # Cotton_ball weight override: auto-computed weight (~32) was too high with
    # 151 clean tiles, causing cotton_ball false positives on bright soil pixels
    # (Balaji v4 showed 15.3% cotton vs 1.1% in v3). Lowering to 12 reduces
    # sensitivity while still giving adequate supervision for the rare class.
    cotton_override = cfg_seg.get('cotton_override_weight', None)

    overrides = {drought_idx: drought_override}
    if cotton_override is not None:
        overrides[cotton_idx] = float(cotton_override)

    class_weights = make_class_weights(
        class_counts, NUM_CLASSES,
        overrides=overrides,
    )
    print('  Computed class weights (drought + cotton overridden):')
    for ci, w in enumerate(class_weights.tolist()):
        print(f'    {CLASS_NAMES[ci]:14} {w:6.2f}')

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print('\nBuilding SegFormer-B2 ...')
    model = SegFormerB2(
        num_classes=NUM_CLASSES,
        input_size=crop_size,
        pretrained_id=cfg_seg.get('pretrained_id', 'nvidia/mit-b2'),
        pretrained_local_path=cfg_seg.get('pretrained_local_path'),
        in_channels=9,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  parameters: {n_params:.1f}M')

    # ------------------------------------------------------------------
    # Loss + optimizer
    # ------------------------------------------------------------------
    criterion = SegmentationLoss(
        class_weights=class_weights.to(device),
        w_ce=cfg_seg.get('ce_weight', 1.0),
        w_dice=cfg_seg.get('dice_weight', 2.5),
        w_lovasz=cfg_seg.get('boundary_weight', 1.0),
        ignore_index=IGNORE_INDEX,
        focal_gamma=cfg_seg.get('focal_gamma', 2.0),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_seg['lr'],
        weight_decay=cfg_seg['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_seg['epochs'], eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_miou   = 0.0
    run_name    = cfg_seg.get('run_name', 'best_v2')
    best_path   = out_dir / f'{run_name}.pt'
    history     = []
    start_epoch = 1

    # Resume from checkpoint if requested
    if resume_path is not None and Path(resume_path).exists():
        print(f'\nResuming from {resume_path} ...')
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        best_miou   = 0.0
        start_epoch = 1
        history     = []
        print('  V4 weights loaded; starting V5 from epoch 1.')
        print(f'  Resumed at epoch {start_epoch}, best mIoU so far: {best_miou:.4f}')
    elif resume_path is not None:
        print(f'[WARNING] --resume path not found: {resume_path}. Starting fresh.')

    for epoch in range(start_epoch, cfg_seg['epochs'] + 1):
        model.train()
        train_losses = {'loss': 0.0, 'focal': 0.0, 'dice': 0.0, 'lovasz': 0.0}
        n_batches = 0
        t0 = time.time()

        mixed.set_epoch(epoch)
        for batch in tqdm(dl_train, desc=f'Epoch {epoch:3d} [train]', leave=False):
            images        = batch['image'].to(device, non_blocking=True)
            masks         = batch['mask'].to(device, non_blocking=True)
            sample_weight = batch['sample_weight'].to(device)

            optimizer.zero_grad()
            logits = model(images)
            ld     = criterion(logits, masks, sample_weight)
            ld['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for k in train_losses:
                train_losses[k] += ld[k].item()
            n_batches += 1

        scheduler.step()
        for k in train_losses:
            train_losses[k] /= max(n_batches, 1)

        # ---- Validation ----
        val_miou = float('nan')
        val_iou  = np.full(NUM_CLASSES, np.nan)
        if dl_val is not None and len(dl_val) > 0:
            model.eval()
            confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            with torch.no_grad():
                for batch in tqdm(dl_val, desc=f'Epoch {epoch:3d} [val]  ', leave=False):
                    images = batch['image'].to(device, non_blocking=True)
                    masks  = batch['mask'].to(device, non_blocking=True)
                    logits = model(images)
                    preds  = logits.argmax(dim=1)
                    update_confusion(confusion, preds, masks, NUM_CLASSES)
            val_iou, val_miou = compute_iou_per_class(confusion)

        elapsed = time.time() - t0

        # ---- Log ----
        print(
            f'Epoch {epoch:3d}/{cfg_seg["epochs"]} | '
            f'train_loss={train_losses["loss"]:.4f} '
            f'(focal={train_losses["focal"]:.3f} dice={train_losses["dice"]:.3f} '
            f'lov={train_losses["lovasz"]:.3f}) | '
            f'val_mIoU={val_miou:.4f} | {elapsed:.0f}s'
        )
        if dl_val is not None:
            for c in range(NUM_CLASSES):
                print(f'    {CLASS_NAMES[c]:14} IoU={val_iou[c]:.3f}')

        # ---- TB ----
        writer.add_scalar('Loss/train', train_losses['loss'], epoch)
        writer.add_scalar('Loss/train_focal', train_losses['focal'], epoch)
        writer.add_scalar('Loss/train_dice', train_losses['dice'], epoch)
        writer.add_scalar('mIoU/val', val_miou, epoch)
        for c in range(NUM_CLASSES):
            if not np.isnan(val_iou[c]):
                writer.add_scalar(f'IoU/val_{CLASS_NAMES[c]}', val_iou[c], epoch)

        history.append({
            'epoch':      epoch,
            'train_loss': train_losses['loss'],
            'val_mIoU':   val_miou,
            'val_iou_per_class': {
                CLASS_NAMES[c]: (float(val_iou[c])
                                 if not np.isnan(val_iou[c]) else None)
                for c in range(NUM_CLASSES)
            },
        })

        # ---- Checkpoint ----
        if not np.isnan(val_miou) and val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_mIoU':         val_miou,
                'val_iou_per_class': val_iou.tolist(),
                'config':           config,
                'history':          history,
            }, best_path)
            print(f'  ✓ Saved best → {best_path.name} (val_mIoU={val_miou:.4f})')

    # ------------------------------------------------------------------
    # Final: dump val predictions from best checkpoint
    # ------------------------------------------------------------------
    print(f'\nLoading best checkpoint for final val predictions ...')
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    if dl_val is not None:
        viz_dir = out_dir / 'val_predictions_v11'
        viz_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dl_val, desc='Val viz')):
                images = batch['image'].to(device)
                masks  = batch['mask']
                logits = model(images)
                logits[:, 6] -= 1.00  # Drought calibration
                logits[:, 3] -= 0.50  # Cotton-ball calibration
                preds  = logits.argmax(dim=1).cpu().numpy()
                # Rebuild RGB from first 3 channels (denormalize)
                for i in range(images.shape[0]):
                    img_t = images[i, :3].cpu().numpy()
                    # Undo ImageNet normalization
                    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
                    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
                    img  = (img_t * std + mean).clip(0, 1) * 255
                    img  = img.transpose(1, 2, 0).astype(np.uint8)
                    out_path = viz_dir / f'val_{batch_idx:03d}_{i:02d}.jpg'
                    save_prediction_overlay(img, masks[i].numpy(), preds[i], out_path)
        print(f'Val predictions saved to {viz_dir}')

    # Final report
    report = {
        'best_val_mIoU': best_miou,
        'best_epoch':    ckpt['epoch'],
        'best_val_iou_per_class': {
            CLASS_NAMES[c]: ckpt['val_iou_per_class'][c]
            for c in range(NUM_CLASSES)
        },
        'class_pixel_counts': class_counts,
        'history':            history,
    }
    with open(out_dir / 'training_report_v11.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nDone. Best val mIoU: {best_miou:.4f}')
    print(f'Report: {out_dir / "training_report_v11.json"}')

    writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path('configs/config.yaml'))
    ap.add_argument('--resume', type=Path, default=None,
                    help='Resume from a previous checkpoint .pt file')
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, resume_path=args.resume)


if __name__ == '__main__':
    main()

```

## src/segmentation/losses.py

```python
"""
Segmentation losses — v2 (batch 4.1).

Key change from v1: replaces plain CrossEntropy with FocalLoss.
Focal loss (Lin et al. 2017) multiplies each pixel's CE by (1-p_t)^gamma
where p_t is the model's probability for the correct class.

Effect for drought:
    - Easy soil predictions (p_soil = 0.95): focal weight = (1-0.95)^2 = 0.0025
      → these gradients are suppressed 400x
    - Hard drought misclassified as soil (p_drought = 0.08): focal weight = 0.846
      → these gradients are barely reduced
    - Net ratio easy:hard = 340x larger for hard examples → forces model to
      learn drought instead of ignoring it.

Also added: explicit per-class weight overrides (drought_override_weight) so
drought doesn't rely solely on inverse-frequency weighting which gave only 2.3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import CLASS_TO_IDX, IGNORE_INDEX


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def make_class_weights(
    class_pixel_counts: dict[int, int],
    num_classes: int,
    floor: float = 0.1,
    cap: float = 50.0,
    overrides: dict[int, float] | None = None,
) -> torch.Tensor:
    """
    Inverse-frequency class weights with optional manual overrides.

    Args:
        class_pixel_counts: {class_idx: pixel_count}
        overrides:          {class_idx: weight} — manual override, bypasses
                            inverse-frequency for specified classes.
                            Example: {6: 30.0} forces drought to weight 30.
    """
    total   = sum(class_pixel_counts.values())
    weights = torch.ones(num_classes, dtype=torch.float32)
    for idx in range(num_classes):
        count = class_pixel_counts.get(idx, 0)
        if count <= 0:
            weights[idx] = 1.0
        else:
            weights[idx] = total / (num_classes * count)
    weights = weights.clamp(min=floor, max=cap)
    # Apply manual overrides AFTER clipping
    if overrides:
        for idx, w in overrides.items():
            if 0 <= idx < num_classes:
                weights[idx] = w
    return weights


# ---------------------------------------------------------------------------
# Focal Loss (replaces plain CE)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Per-pixel focal loss with class weights and ignore_index.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        alpha_t  = class_weights[class]
        p_t      = softmax probability of the correct class
        gamma    = focusing exponent (2.0 is standard)

    Args:
        class_weights: (C,) alpha values per class.
        gamma:         focusing exponent. Higher → focus more on hard examples.
        ignore_index:  pixels to skip.
        reduction:     'none' (per-pixel) for use in SegmentationLoss.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
        reduction: str = 'none',
    ) -> None:
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.reduction    = reduction
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W)
            targets: (B, H, W) int64

        Returns:
            (B, H, W) per-pixel focal loss (ignored pixels = 0)
        """
        B, C, H, W = logits.shape
        # Compute log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)          # (B, C, H, W)
        probs     = torch.exp(log_probs)                  # (B, C, H, W)

        # Mask ignored pixels — clamp target to valid range for indexing
        valid   = (targets != self.ignore_index)          # (B, H, W) bool
        t_safe  = targets.clone()
        t_safe[~valid] = 0

        # Gather log-prob and prob for the correct class
        t_safe_expand = t_safe.unsqueeze(1)                # (B, 1, H, W)
        log_pt = log_probs.gather(1, t_safe_expand).squeeze(1)   # (B, H, W)
        pt     = probs.gather(1, t_safe_expand).squeeze(1)       # (B, H, W)

        # Focal factor: (1 - p_t)^gamma
        focal_factor = (1.0 - pt).pow(self.gamma)         # (B, H, W)

        # Per-pixel CE with focal scaling
        loss = -focal_factor * log_pt                     # (B, H, W)

        # Apply class weights (alpha_t)
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            alpha_t = w[t_safe]                           # (B, H, W)
            loss    = alpha_t * loss

        # Zero out ignored pixels
        loss = loss * valid.float()

        if self.reduction == 'mean':
            return loss.sum() / valid.float().sum().clamp(min=1.0)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss    # 'none'


# ---------------------------------------------------------------------------
# Dice Loss (unchanged from v1)
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Multi-class soft Dice loss with ignore-index support."""

    def __init__(
        self,
        smooth: float = 1.0,
        per_image: bool = False,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.smooth       = smooth
        self.per_image    = per_image
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        valid          = (targets != self.ignore_index)
        t_clamped      = targets.clone()
        t_clamped[~valid] = 0
        one_hot        = F.one_hot(t_clamped, C).permute(0, 3, 1, 2).float()
        one_hot        = one_hot * valid.unsqueeze(1).float()
        probs          = probs   * valid.unsqueeze(1).float()

        if self.per_image:
            inter = (probs * one_hot).sum(dim=(2, 3))
            union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
            dice  = (2 * inter + self.smooth) / (union + self.smooth)
            return 1.0 - dice.mean()
        else:
            inter = (probs * one_hot).sum(dim=(0, 2, 3))
            union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
            dice  = (2 * inter + self.smooth) / (union + self.smooth)
            return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Lovász boundary loss (unchanged from v1)
# ---------------------------------------------------------------------------

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union        = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard      = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(
    probs_flat: torch.Tensor, labels_flat: torch.Tensor
) -> torch.Tensor:
    C      = probs_flat.shape[1]
    losses = []
    for c in range(C):
        fg = (labels_flat == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs_flat[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return probs_flat.sum() * 0.0
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs       = F.softmax(logits, dim=1)
        B, C, H, W  = probs.shape
        probs_flat  = probs.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = targets.reshape(-1)
        valid       = labels_flat != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        return lovasz_softmax_flat(probs_flat[valid], labels_flat[valid])


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """
    Combined: w_focal * FocalCE  +  w_dice * Dice  +  w_lovasz * Lovász

    Key change from v1: plain CE → FocalLoss with gamma=2.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        w_ce: float = 1.0,          # weight on focal CE
        w_dice: float = 2.5,        # higher than v1 — Dice rescues rare classes
        w_lovasz: float = 1.0,      # higher than v1 — boundary sharpening
        ignore_index: int = IGNORE_INDEX,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.w_ce      = w_ce
        self.w_dice    = w_dice
        self.w_lovasz  = w_lovasz
        self.ignore_index = ignore_index

        self.focal  = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            ignore_index=ignore_index,
            reduction='none',
        )
        self.dice   = DiceLoss(ignore_index=ignore_index)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Focal CE — per-pixel, then weight by sample importance
        focal_pixel = self.focal(logits, targets)          # (B, H, W), ignored=0
        if sample_weights is not None:
            sw          = sample_weights.view(-1, 1, 1).to(focal_pixel.device)
            focal_pixel = focal_pixel * sw
        valid   = (targets != self.ignore_index).float()
        n_valid = valid.sum().clamp(min=1.0)
        focal_loss = (focal_pixel * valid).sum() / n_valid

        dice_loss   = self.dice(logits, targets)
        lovasz_loss = self.lovasz(logits, targets)

        total = (self.w_ce * focal_loss +
                 self.w_dice * dice_loss +
                 self.w_lovasz * lovasz_loss)

        return {
            'loss':   total,
            'focal':  focal_loss.detach(),
            'dice':   dice_loss.detach(),
            'lovasz': lovasz_loss.detach(),
        }

```

## src/segmentation/dataset.py

```python
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

```

## src/segmentation/compositor.py

```python
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

```