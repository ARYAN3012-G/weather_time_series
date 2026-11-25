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
        viz_dir = out_dir / 'val_predictions'
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
    with open(out_dir / 'training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nDone. Best val mIoU: {best_miou:.4f}')
    print(f'Report: {out_dir / "training_report.json"}')

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
