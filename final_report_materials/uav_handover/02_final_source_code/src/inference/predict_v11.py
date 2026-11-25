"""
Orthomosaic inference — batch 5.

Runs the trained SegFormer model over a full orthomosaic image, produces
a color-coded segmentation mask, area estimates and a drought decision.

Usage
-----
    python -m src.inference.predict \\
        --image  data/orthomosaic/field_01.jpg \\
        --model  runs/segmentation/best.pt \\
        --altitude 13.8 \\
        --out    runs/results/field_01/

Outputs in --out directory:
    mask.png        Full-resolution color-coded segmentation mask
    overlay.jpg     50% alpha overlay on original (resized to ≤8192 px wide)
    report.json     Per-class areas, drought fraction, decision
    report.pdf      One-page human-readable summary

Notes:
    - --altitude defaults to 13.8 m (from the attached Metashape report).
      Pass the actual altitude for each orthomosaic for accurate m² values.
    - --model can point to best.pt (batch4) or best_v2.pt (batch4.1).
      The pipeline is identical regardless of which model file is used.
"""
from __future__ import annotations

from pathlib import Path
import json
import yaml


import argparse
import gc
import os
import time
from pathlib import Path

# --- MUST be set before cv2 is imported anywhere in the process ---
# Removes OpenCV's hard pixel-count limit that rejects large orthomosaics.
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', str(2 ** 40))

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from src.inference.postprocess import postprocess
from src.inference.report import (
    compute_areas,
    drought_decision,
    save_json_report,
    save_mask_png,
    save_overlay_jpg,
    save_pdf_report,
)
from src.inference.tile_v11 import TileStitcher
from src.segmentation.model import SegFormerB2
from src.utils.constants import NUM_CLASSES


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_orthomosaic(path: Path) -> np.ndarray:
    """
    Load a (potentially huge) orthomosaic JPG/TIF as uint8 RGB.

    Tries three backends in order:
        1. cv2  — fastest; works after OPENCV_IO_MAX_IMAGE_PIXELS is set.
        2. PIL  — fallback; handles images cv2 still refuses.
        3. PIL with explicit LANCZOS pre-downscale if the image is so large
           that even PIL chokes on memory (>500 Mpx).

    The env var OPENCV_IO_MAX_IMAGE_PIXELS is set at module import time above,
    so cv2 should succeed for all normally-sized orthomosaics.
    """
    Image.MAX_IMAGE_PIXELS = None   # disable PIL decompression-bomb check

    # --- Backend 1: cv2 ---
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        del img
        gc.collect()
        print(f'  Loaded (cv2): {path.name}  {rgb.shape[1]}×{rgb.shape[0]} px')
        return rgb

    # --- Backend 2: PIL ---
    print(f'  cv2 failed for {path.name}, trying PIL ...')
    try:
        pil_img = Image.open(str(path)).convert('RGB')
        rgb = np.array(pil_img, dtype=np.uint8)
        del pil_img
        gc.collect()
        print(f'  Loaded (PIL): {path.name}  {rgb.shape[1]}×{rgb.shape[0]} px')
        return rgb
    except Exception as e:
        raise IOError(
            f'Could not read {path}. Tried cv2 and PIL. Last error: {e}'
        ) from e


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: Path,
    device: torch.device,
    config: dict,
) -> torch.nn.Module:
    """Load SegFormerB2 from a training checkpoint."""
    cfg_seg = config.get('segmentation', {})
    model   = SegFormerB2(
        num_classes=NUM_CLASSES,
        input_size=cfg_seg.get('input_size', 512),
        pretrained_id=None,          # no download needed — we load from checkpoint
        in_channels=9,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        val_miou = ckpt.get('val_mIoU', 'unknown')
        print(f'  Loaded checkpoint: {checkpoint_path.name}'
              f'  (val mIoU: {val_miou})')
    else:
        # Raw state dict
        model.load_state_dict(ckpt)
        print(f'  Loaded checkpoint: {checkpoint_path.name}')

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main inference function (callable from other modules / web app)
# ---------------------------------------------------------------------------



# MEMORY_SAFE_INTEGRATION_V1

import os as _memory_os

from src.inference.memory_safe_postprocess import (
    postprocess as _memory_postprocess
)

from src.inference.memory_safe_report import (
    drought_decision as _memory_drought_decision
)


def _use_memory_safe(class_map):

    return (
        class_map.size >= 100_000_000
        or _memory_os.environ.get(
            "DROUGHT_FORCE_MEMORY_SAFE", "0"
        ) == "1"
    )


def _dispatch_postprocess(
    class_map, img_rgb, *args, **kwargs
):

    if _use_memory_safe(class_map):

        print("[Memory-safe] Block-wise postprocessing")

        return _memory_postprocess(
            class_map, img_rgb,
            *args, **kwargs
        )

    return postprocess(
        class_map, img_rgb,
        *args, **kwargs
    )


def _dispatch_drought_decision(
    area_stats, harvest_mask,
    class_map, *args, **kwargs
):

    if _use_memory_safe(class_map):

        print("[Memory-safe] Disk-backed drought decision")

        return _memory_drought_decision(
            area_stats, harvest_mask,
            class_map, *args, **kwargs
        )

    return drought_decision(
        area_stats, harvest_mask,
        class_map, *args, **kwargs
    )


def predict(
    image_path: str | Path,
    model_path: str | Path,
    altitude_m: float,
    out_dir: str | Path,
    config_path: str | Path = 'configs/config.yaml',
    tile_size: int = 1024,
    tile_stride: int = 768,
    batch_size: int = 8,
    skip_pdf: bool = False,
    skip_morph: bool = False,
) -> dict:
    """
    Run full inference on one orthomosaic.

    Args:
        image_path:  Path to the orthomosaic JPG.
        model_path:  Path to best.pt or best_v2.pt.
        altitude_m:  Flight altitude in metres (used for GSD calculation).
        out_dir:     Directory where outputs are saved.
        config_path: Path to configs/config.yaml.
        tile_size:   Inference tile size (1024 recommended).
        tile_stride: Stride between tiles (768 = 256 px overlap).
        batch_size:  Tiles per GPU batch.
        skip_pdf:    If True, skip PDF generation.
        skip_morph:  If True, skip morphological cleanup (faster).

    Returns:
        dict with keys: mask_path, overlay_path, json_path, pdf_path,
                        area_stats, decision.
    """
    image_path = Path(image_path)
    model_path = Path(model_path)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Guard against null/None altitude passed from upstream
    if altitude_m is None or altitude_m <= 0:
        print(f'[WARNING] altitude_m={altitude_m!r} is invalid — '
              f'using default 13.8 m')
        altitude_m = 13.8

    t_total = time.time()

    # 1. Load image
    print('\n[1/6] Loading orthomosaic ...')
    img_rgb = load_orthomosaic(image_path)

    # 2. Load model
    print('[2/6] Loading model ...')
    model = load_model(model_path, device, config)

    # 3. Tile inference
    print(f'[3/6] Running tiled inference (tile={tile_size}, stride={tile_stride}) ...')
    t0 = time.time()
    stitcher  = TileStitcher(
        model, device,
        tile_size=tile_size,
        stride=tile_stride,
        batch_size=batch_size,
        num_classes=NUM_CLASSES,
    )
    class_map = stitcher.run(img_rgb)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    print(f'  Inference: {time.time()-t0:.1f}s')

    # 4. Post-processing
    print('[4/6] Post-processing ...')
    t0 = time.time()
    class_map, harvest_mask = _dispatch_postprocess(class_map, img_rgb,
                                          skip_morph=skip_morph)
    print(f'  Post-process: {time.time()-t0:.1f}s')

    # 5. Report
    print('[5/6] Computing areas and drought decision ...')

    registry_file = (
        Path(__file__).resolve().parents[2]
        / "configs/field_gsd_registry.json"
    )

    field_registry = json.loads(
        registry_file.read_text()
    )

    field_name = Path(image_path).name
    field_info = field_registry.get(field_name)

    if field_info is None:
        raise ValueError(
            f"No GSD registry entry for {field_name}"
        )

    expected_shape = (
        field_info["actual_height"],
        field_info["actual_width"]
    )

    if class_map.shape != expected_shape:
        raise ValueError(
            f"Unexpected mask dimensions: "
            f"{class_map.shape} vs {expected_shape}"
        )

    field_gsd = field_info["gsd_mm_per_pixel"]
    field_altitude = field_info["altitude_m"]

    print(
        f"    Reported orthomosaic GSD: "
        f"{field_gsd} mm/pixel"
    )

    if not field_info["dimensions_match"]:
        print(
            "    WARNING: GSD export dimensions "
            "require verification."
        )

    if field_info["local_coordinates"]:
        print(
            "    WARNING: Local coordinate "
            "physical scale is unverified."
        )

    area_stats = compute_areas(
        class_map,
        harvest_mask,
        field_altitude,
        gsd_override_mm=field_gsd
    )

    area_stats["gsd_source"] = field_info["source"]
    area_stats["gsd_dimensions_match"] = (
        field_info["dimensions_match"]
    )
    area_stats["physical_scale_verified"] = (
        field_info["physical_scale_verified"]
    )
    area_stats["area_is_estimate"] = True
    area_stats["area_requires_review"] = (
        not field_info["dimensions_match"]
        or field_info["local_coordinates"]
    )
    decision_cfg = yaml.safe_load(
        Path(config_path).read_text()
    )["inference"]["drought_decision"]
    decision   = _dispatch_drought_decision(
        area_stats,
        harvest_mask,
        class_map,
        tau_fraction=decision_cfg["tau1_drought_fraction"],
        tau_non_harvest=decision_cfg["tau3_nonharvest_fraction"],
        tau_cc_share=decision_cfg["tau2_largest_cc_share"]
    )

    print(f'\n  === DROUGHT DECISION: {"⚠  VISUAL DROUGHT-STRESS FLAG" if decision["drought"] else "✓  No field-level stress flag"} ===')
    print(f'  stress score:        {decision["confidence"]:.0%}')
    print(f'  drought fraction:    {decision["drought_fraction"]:.6%} of agri area')
    print(f'  non-harvest drought: {decision["non_harvest_drought_fraction"]:.6%}')
    print()
    print('  Per-class areas:')
    for cls_name, stats in area_stats['per_class'].items():
        if stats['pixels'] == 0:
            continue
        pct = stats['area_m2'] / max(area_stats['valid_field_area_m2'], 1) * 100
        print(f'    {cls_name:14} {stats["area_m2"]:>12,.1f} m²  ({pct:.1f}%)')

    # 6. Save outputs
    print('\n[6/6] Saving outputs ...')
    mask_path    = out_dir / 'mask.png'
    overlay_path = out_dir / 'overlay.jpg'
    json_path    = out_dir / 'report.json'
    pdf_path     = out_dir / 'report.pdf'

    print('  Saving mask.png ...')
    save_mask_png(class_map, mask_path)

    print('  Saving overlay.jpg ...')
    save_overlay_jpg(img_rgb, class_map, overlay_path)

    print('  Saving report.json ...')
    save_json_report(area_stats, decision, str(image_path), json_path)

    if not skip_pdf:
        print('  Saving report.pdf ...')
        save_pdf_report(img_rgb, class_map, area_stats, decision,
                        image_path.name, pdf_path)

    elapsed = time.time() - t_total
    # Release large arrays now that all outputs are saved
    del class_map, harvest_mask, img_rgb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'\nDone in {elapsed:.1f}s. Outputs in {out_dir}')

    return {
        'mask_path':    str(mask_path),
        'overlay_path': str(overlay_path),
        'json_path':    str(json_path),
        'pdf_path':     str(pdf_path) if not skip_pdf else None,
        'area_stats':   area_stats,
        'decision':     decision,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description='Orthomosaic drought inference')
    ap.add_argument('--image',    type=Path, required=True,
                    help='Path to orthomosaic JPG')
    ap.add_argument('--model',    type=Path,
                    default=Path('runs/segmentation/best.pt'),
                    help='Path to model checkpoint (default: best.pt)')
    ap.add_argument('--altitude', type=float, default=13.8,
                    help='Flight altitude in metres (default: 13.8)')
    ap.add_argument('--out',      type=Path, default=None,
                    help='Output directory (default: runs/results/<image_stem>/)')
    ap.add_argument('--config',   type=Path,
                    default=Path('configs/config.yaml'))
    ap.add_argument('--tile-size',   type=int, default=1024)
    ap.add_argument('--tile-stride', type=int, default=768)
    ap.add_argument('--batch-size',  type=int, default=8)
    ap.add_argument('--skip-pdf',  action='store_true')
    ap.add_argument('--skip-morph', action='store_true')
    args = ap.parse_args()

    if args.out is None:
        args.out = Path('runs/results') / args.image.stem

    predict(
        image_path=args.image,
        model_path=args.model,
        altitude_m=args.altitude,
        out_dir=args.out,
        config_path=args.config,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        batch_size=args.batch_size,
        skip_pdf=args.skip_pdf,
        skip_morph=args.skip_morph,
    )


if __name__ == '__main__':
    main()
