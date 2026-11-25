"""
Pseudo-label the unlabeled UAV images using the trained patch classifier.

For each image in data/unlabeled/:
    1. Two-pass sliding-window inference (256×256 + 128×128).
    2. Dense CRF refinement on RGB pairwise terms.
    3. Structural-cue overrides (glare, FFT row score, ExG rules).
    4. Confidence thresholding → IGNORE_INDEX (255) for uncertain pixels.
    5. Save PNG mask (palettized) + confidence map.

Outputs (under runs/pseudo_labels/):
    masks/          one PNG per image — uint8, pixel value = class index
                    (255 = ignore)
    overlays/       semi-transparent overlay on original image (for inspection)
    confidence/     float32 confidence map saved as 8-bit PNG (0=0.0, 255=1.0)
    manifest.yaml   per-image stats: path, ignore_fraction, class_fractions

Usage:
    python -m src.pseudo_label.label --config configs/config.yaml
    python -m src.pseudo_label.label --config configs/config.yaml --limit 5  # test on 5 images
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.classifier.model import MultiscalePatchClassifier
from src.pseudo_label.postprocess import postprocess_probmap
from src.pseudo_label.sliding_window import SlidingWindowInference
from src.utils.constants import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    PALETTE,
)
from src.pseudo_label.viz import (
    overlay_with_legend,
    add_stat_bar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray | None:
    """Load a JPG/PNG as uint8 RGB. Returns None on failure."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Save a uint8 class-index mask as a palettized PNG."""
    from PIL import Image
    img = Image.fromarray(mask, mode='L')
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


def save_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    path: Path,
    conf: np.ndarray | None = None,
    alpha: float = 0.50,
) -> None:
    """Save overlay with colour legend and statistics bar."""
    composite = overlay_with_legend(img_rgb, mask, alpha=alpha)
    composite = add_stat_bar(composite, mask, conf=conf)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])


def save_confidence_png(conf: np.ndarray, path: Path) -> None:
    """Save float32 confidence map as 8-bit PNG (0→0.0, 255→1.0)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (conf * 255).clip(0, 255).astype(np.uint8))


def compute_class_fractions(mask: np.ndarray, n_classes: int) -> dict[str, float]:
    """Compute fraction of valid pixels belonging to each class."""
    valid = mask != IGNORE_INDEX
    total_valid = valid.sum()
    if total_valid == 0:
        return {CLASS_NAMES[i]: 0.0 for i in range(n_classes)}
    fractions = {}
    for i in range(n_classes):
        fractions[CLASS_NAMES[i]] = round(float((mask == i).sum() / total_valid), 4)
    return fractions


def find_unlabeled_images(unlabeled_dir: Path) -> list[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    imgs = sorted([p for p in unlabeled_dir.rglob('*')
                   if p.is_file() and p.suffix.lower() in exts])
    return imgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def label(
    config: dict,
    limit: int | None = None,
    no_crf: bool = False,
    shuffle: bool = False,
    confidence_override: float | None = None,
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    paths   = config['paths']
    cfg_pl  = config['pseudo_label']
    cfg_cls = config['classifier']

    out_dir = Path(paths['runs_dir']) / 'pseudo_labels'
    mask_dir      = out_dir / 'masks'
    overlay_dir   = out_dir / 'overlays'
    conf_dir      = out_dir / 'confidence'
    for d in [mask_dir, overlay_dir, conf_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load classifier
    # ------------------------------------------------------------------
    ckpt_path = Path(paths['runs_dir']) / 'classifier' / 'best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f'Classifier checkpoint not found: {ckpt_path}\n'
            'Run src.classifier.train first.'
        )
    print(f'Loading classifier from {ckpt_path} ...')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MultiscalePatchClassifier(
        num_classes=NUM_CLASSES,
        pretrained=False,
        img_size=cfg_cls['input_size'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  Checkpoint val macro-F1: {ckpt.get("val_macro_f1", "N/A")}')

    # ------------------------------------------------------------------
    # Build sliding-window engine
    # ------------------------------------------------------------------
    engine = SlidingWindowInference(
        model=model,
        device=device,
        batch_size=config.get('pseudo_label_batch_size', 256),
        window_size=cfg_pl.get('window_size', 512),
        stride=cfg_pl.get('stride', 256),
        input_size=cfg_cls['input_size'],
        num_classes=NUM_CLASSES,
    )

    # ------------------------------------------------------------------
    # Process images
    # ------------------------------------------------------------------
    unlabeled_dir = Path(paths['unlabeled_dir'])
    images = find_unlabeled_images(unlabeled_dir)
    if not images:
        raise FileNotFoundError(f'No images found in {unlabeled_dir}')
    if shuffle:
        import random
        random.Random(42).shuffle(images)
    if limit:
        images = images[:limit]

    print(f'\nFound {len(images)} unlabeled images. Processing...')
    conf_thresh = confidence_override if confidence_override is not None else cfg_pl['confidence_threshold']
    use_crf     = (not no_crf)
    print(f'Confidence threshold: {conf_thresh}')
    crf_iter    = cfg_pl['crf_iterations']

    manifest: list[dict] = []

    for img_path in tqdm(images, desc='Pseudo-labeling'):
        t0 = time.time()
        img_rgb = load_image(img_path)
        if img_rgb is None:
            print(f'  ⚠ Could not read {img_path.name} — skipping')
            continue

        # Two-pass sliding window
        prob_map = engine.run(img_rgb)           # (H, W, 7)

        # CRF + overrides + confidence threshold
        mask, conf = postprocess_probmap(
            img_rgb, prob_map, NUM_CLASSES,
            confidence_threshold=conf_thresh,
            use_crf=use_crf,
            crf_iter=crf_iter,
        )

        # Save outputs
        stem = img_path.stem
        save_mask_png(mask, mask_dir / f'{stem}.png')
        save_overlay(img_rgb, mask, overlay_dir / f'{stem}.jpg', conf=conf)
        save_confidence_png(conf, conf_dir / f'{stem}.png')

        # Record stats
        h, w      = mask.shape
        n_pixels  = h * w
        n_ignore  = int((mask == IGNORE_INDEX).sum())
        ignore_frac = round(n_ignore / n_pixels, 4)
        class_fracs = compute_class_fractions(mask, NUM_CLASSES)
        elapsed = round(time.time() - t0, 1)

        record = {
            'image':          img_path.name,
            'mask':           f'masks/{stem}.png',
            'overlay':        f'overlays/{stem}.jpg',
            'confidence':     f'confidence/{stem}.png',
            'height':         h,
            'width':          w,
            'ignore_fraction': ignore_frac,
            'class_fractions': class_fracs,
            'elapsed_s':      elapsed,
        }
        manifest.append(record)

        # Show all non-trivial classes
        cls_summary = ' '.join(
            f'{k[:4]}={v:.0%}'
            for k, v in class_fracs.items()
            if v >= 0.01 and k != 'background'
        )
        tqdm.write(f'  {img_path.name}: ignore={ignore_frac:.0%}  {cls_summary}  time={elapsed}s')

    # ------------------------------------------------------------------
    # Write manifest
    # ------------------------------------------------------------------
    manifest_path = out_dir / 'manifest.yaml'
    with open(manifest_path, 'w') as f:
        yaml.safe_dump({'pseudo_labels': manifest}, f,
                       default_flow_style=False, sort_keys=False)

    print(f'\nDone. Outputs: {out_dir}')
    print(f'Manifest:       {manifest_path}')
    print(f'Total images:   {len(manifest)}')

    # Summary statistics
    if manifest:
        avg_ignore   = sum(r['ignore_fraction'] for r in manifest) / len(manifest)
        avg_drought  = sum(r['class_fractions'].get('drought', 0) for r in manifest) / len(manifest)
        print(f'Avg ignore fraction:  {avg_ignore:.1%}')
        print(f'Avg drought fraction: {avg_drought:.1%}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path('configs/config.yaml'))
    ap.add_argument('--limit', type=int, default=None,
                    help='Only process first N images (for testing)')
    ap.add_argument('--no-crf', action='store_true',
                    help='Skip CRF refinement (faster, lower quality)')
    ap.add_argument('--shuffle', action='store_true',
                    help='Pick random images instead of alphabetical first N')
    ap.add_argument('--confidence', type=float, default=None,
                    help='Override confidence threshold (default 0.55 from config)')
    ap.add_argument('--batch-size', type=int, default=256,
                    help='Tiles per GPU batch (default 256)')
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config['pseudo_label_batch_size'] = args.batch_size

    label(
        config,
        limit=args.limit,
        no_crf=args.no_crf,
        shuffle=args.shuffle,
        confidence_override=args.confidence,
    )


if __name__ == '__main__':
    main()
