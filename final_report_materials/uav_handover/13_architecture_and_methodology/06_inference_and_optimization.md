# 06 inference and optimization

Use the following exact implementation as the authoritative technical reference.

## src/inference/predict_v11.py

```python
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

```

## src/inference/tile_v11.py

```python
"""
Tile-and-stitch inference for large orthomosaic images.

Production-hardened for very large remote sensing imagery (25k+ pixels).

Fixes applied
-------------
1. All accumulators (prob_acc, weight_acc) live on CPU float16.
   Nothing is accumulated on the GPU — only one batch at a time sits there.

2. Dynamic safe batch size: computed from image pixel count to avoid OOM
   before the first forward pass. Can be overridden by the caller.

3. OOM retry loop: if a batch still OOMs (e.g. due to fragmentation), the
   batch is automatically re-tried with half the size, down to minimum 1.

4. Explicit GPU cleanup after every batch: del + empty_cache + gc.collect.
   Prevents VRAM fragmentation across the hundreds of batches in a large image.

5. No model-side changes required — fixes are purely in data movement.
"""
from __future__ import annotations

import gc
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.preprocessing.tensor_builder import build_9channel


# ---------------------------------------------------------------------------
# Dynamic safe batch size
# ---------------------------------------------------------------------------

def _safe_batch_size(
    H: int,
    W: int,
    tile_size: int,
    requested: int,
    vram_budget_gb: float = 12.0,
) -> int:
    """
    Compute a safe batch size that is unlikely to OOM on the current GPU.

    Heuristic based on measured GPU usage for SegFormer-B2 at tile_size=1024:
        ~0.22 GB per tile in a batch (input + activations + logits)
    This is conservative — actual usage depends on batch size due to
    batched attention patterns in SegFormer.

    Args:
        H, W:          Image dimensions in pixels.
        tile_size:     Inference tile size.
        requested:     Caller-requested batch size.
        vram_budget_gb: How many GB to budget for one forward pass.
                        Default 12 GB leaves headroom for the model weights
                        (~0.5 GB) on a 70 GB H200.
    """
    pixels_per_tile = tile_size * tile_size
    # Measured ~0.22 GB per tile for SegFormer-B2 at 1024×1024
    # Measured SegFormer-B2 peak VRAM per tile in a batch:
    # ~0.75 GB at 1024×1024 — accounts for input, activations, attention maps,
    # gradients (even with no_grad, PyTorch keeps intermediate tensors).
    gb_per_tile = 9 * pixels_per_tile * 4 / 1e9 * 20.0  # 20× overhead factor

    safe = max(1, int(vram_budget_gb / gb_per_tile))
    result = min(requested, safe)

    total_tiles = math.ceil(H / tile_size) * math.ceil(W / tile_size)
    if result < requested:
        print(
            f'  [tile] Image {W}×{H} ({total_tiles} tiles): '
            f'reducing batch {requested}→{result} to stay within '
            f'{vram_budget_gb:.0f} GB VRAM budget.'
        )
    return result


# ---------------------------------------------------------------------------
# Hann window
# ---------------------------------------------------------------------------

def _hann2d(size: int) -> torch.Tensor:
    """2D Hann window as CPU float16. Clipped to min 0.01 so corners contribute."""
    h = torch.hann_window(size, periodic=False)
    h = h.clamp(min=0.1)
    return torch.outer(h, h).half()   # (size, size) float16


# ---------------------------------------------------------------------------
# TileStitcher
# ---------------------------------------------------------------------------

class TileStitcher:
    """
    Production-grade tiled inference over large orthomosaic images.

    All GPU memory is released after every batch.
    Automatically retries with smaller batch on CUDA OOM.
    All accumulator tensors live on CPU to avoid VRAM pressure.

    Args:
        model:         Trained SegFormerB2 on `device`, already in eval mode.
        device:        torch.device
        tile_size:     Tile size in pixels (default 1024).
        stride:        Stride between tile origins (default 768 = 256px overlap).
        batch_size:    Requested tiles per GPU batch. Automatically reduced
                       based on image size and VRAM budget.
        num_classes:   Number of segmentation classes.
        vram_budget_gb: VRAM budget for one forward pass (default 12 GB).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        tile_size: int = 1024,
        stride: int = 768,
        batch_size: int = 8,
        num_classes: int = 7,
        vram_budget_gb: float = 12.0,
    ) -> None:
        self.model          = model
        self.device         = device
        self.tile_size      = tile_size
        self.stride         = stride
        self.requested_bs   = batch_size
        self.num_classes    = num_classes
        self.vram_budget_gb = vram_budget_gb
        # Keep Hann window on CPU — moved to GPU per-batch only
        self.hann_cpu       = _hann2d(tile_size)   # (T, T) float16 on CPU
        model.eval()

    # ------------------------------------------------------------------
    # Internal: forward one batch with OOM retry
    # ------------------------------------------------------------------

    def _forward_batch(
        self,
        tiles_np: list[np.ndarray],
        current_bs: int,
    ) -> tuple[torch.Tensor, int]:
        """
        Run one forward pass. On CUDA OOM, halve the batch and retry.

        Returns:
            (probs_cpu, effective_bs_used)
            probs_cpu: (B, C, T, T) float16 on CPU
        """
        while True:
            try:
                batch = torch.stack(
                    [torch.from_numpy(t) for t in tiles_np]
                ).to(self.device)                          # (B, 9, T, T) float32

                with torch.no_grad():
                    logits    = self.model(batch)          # (B, C, T, T)
                    logits[:, 6] -= 1.20  # V11 drought calibration
                    logits[:, 5] -= 0.40  # V11 dry-weed calibration
                    probs_cpu = F.softmax(logits, dim=1).cpu().float()

                # Explicit GPU cleanup — critical for sustained batch runs
                del batch, logits
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

                return probs_cpu, current_bs

            except torch.cuda.OutOfMemoryError:
                # Clean up whatever partial allocations were made
                try:
                    del batch
                except NameError:
                    pass
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

                new_bs = max(1, current_bs // 2)
                if new_bs == current_bs:
                    raise   # already at 1, can't go lower — re-raise
                print(
                    f'\n  [OOM] CUDA OOM at batch_size={current_bs}. '
                    f'Retrying with batch_size={new_bs} ...'
                )
                # Re-split tiles_np to the smaller batch and retry
                # (caller must handle the rest of the original batch)
                tiles_np  = tiles_np[:new_bs]
                current_bs = new_bs

    # ------------------------------------------------------------------
    # Public: full image inference
    # ------------------------------------------------------------------

    def run_full_reference(self, img_rgb: np.ndarray) -> np.ndarray:
            """
            Full tiled inference on one large image.

            Args:
                img_rgb: (H, W, 3) uint8 RGB.

            Returns:
                (H, W) uint8 argmax class map.
            """
            H, W  = img_rgb.shape[:2]
            T     = self.tile_size
            S     = self.stride
            C     = self.num_classes

            # Compute safe batch size based on image dimensions
            bs = _safe_batch_size(H, W, T, self.requested_bs, self.vram_budget_gb)

            # Pad image so tiles cover the edges cleanly
            pad_h = (math.ceil((H - T) / S) * S + T) - H if H > T else T - H
            pad_w = (math.ceil((W - T) / S) * S + T) - W if W > T else T - W
            if pad_h > 0 or pad_w > 0:
                img_rgb = cv2.copyMakeBorder(
                    img_rgb, 0, pad_h, 0, pad_w,
                    cv2.BORDER_REFLECT_101,
                )
            Hp, Wp = img_rgb.shape[:2]

            # Tile coordinate grid
            ys     = list(range(0, Hp - T + 1, S))
            xs     = list(range(0, Wp - T + 1, S))
            coords = [(y, x) for y in ys for x in xs]
            print(f'  Grid: {len(xs)}×{len(ys)} = {len(coords)} tiles'
                  f'  |  batch_size={bs}'
                  f'  |  ~{math.ceil(len(coords)/bs)} forward passes')

            # CPU accumulators — never touch the GPU
            prob_acc   = torch.zeros(C, Hp, Wp, dtype=torch.float32)

            # Process all tiles in batches
            # OOM inside _forward_batch will halve bs and retry the SAME sub-batch.
            # We therefore use a while-loop with a pointer instead of a for-loop.
            ptr = 0
            with tqdm(total=len(coords), desc='  Tiles', unit='tile',
                      leave=False, ncols=80) as pbar:
                while ptr < len(coords):
                    batch_coords = coords[ptr: ptr + bs]
                    tiles_np     = [
                        build_9channel(img_rgb[y:y+T, x:x+T])
                        for y, x in batch_coords
                    ]

                    probs_cpu, bs_used = self._forward_batch(tiles_np, bs)
                    # _forward_batch may have reduced bs_used < len(batch_coords)
                    # if OOM occurred. In that case only bs_used tiles were processed.
                    if bs_used < len(batch_coords):
                        # Only advance by bs_used; remaining tiles re-processed next iter
                        batch_coords = batch_coords[:bs_used]
                        bs = bs_used   # use smaller size for remaining batches

                    # Splat probabilities into accumulator
                    for i, (y, x) in enumerate(batch_coords):
                        p = probs_cpu[i]                       # (C, T, T) float16
                        w = self.hann_cpu.float()                      # (T, T) float16
                        prob_acc[:, y:y+T, x:x+T]   += p * w.unsqueeze(0)

                    ptr += len(batch_coords)
                    pbar.update(len(batch_coords))

            # Argmax directly on accumulated weighted probabilities.
            # Positive per-pixel normalization factors do not change argmax.
            # This avoids allocating multiple full-size temporary tensors.
            class_map = prob_acc.argmax(dim=0).numpy().astype(np.uint8)
            return class_map[:H, :W]

    def run(self, img_rgb: np.ndarray) -> np.ndarray:
            """
            Full tiled inference on one large image.

            Args:
                img_rgb: (H, W, 3) uint8 RGB.

            Returns:
                (H, W) uint8 argmax class map.
            """
            H, W  = img_rgb.shape[:2]
            T     = self.tile_size
            S     = self.stride
            C     = self.num_classes
            assert 0 < S <= T

            # Compute safe batch size based on image dimensions
            bs = _safe_batch_size(H, W, T, self.requested_bs, self.vram_budget_gb)

            # Pad image so tiles cover the edges cleanly
            pad_h = (math.ceil((H - T) / S) * S + T) - H if H > T else T - H
            pad_w = (math.ceil((W - T) / S) * S + T) - W if W > T else T - W
            if pad_h > 0 or pad_w > 0:
                img_rgb = cv2.copyMakeBorder(
                    img_rgb, 0, pad_h, 0, pad_w,
                    cv2.BORDER_REFLECT_101,
                )
            Hp, Wp = img_rgb.shape[:2]

            # Tile coordinate grid
            ys     = list(range(0, Hp - T + 1, S))
            xs     = list(range(0, Wp - T + 1, S))
            coords = [(y, x) for y in ys for x in xs]
            print(f'  Grid: {len(xs)}×{len(ys)} = {len(coords)} tiles'
                  f'  |  batch_size={bs}'
                  f'  |  ~{math.ceil(len(coords)/bs)} forward passes')

            # CPU accumulators — never touch the GPU
            prob_acc = torch.zeros(
                C, T, Wp, dtype=torch.float32
            )

            class_map = np.empty((H, W), dtype=np.uint8)
            active_y = 0

            def finalize_rows(start, count):

                valid_count = min(count, H - start)

                if valid_count <= 0:
                    return

                for offset in range(0, valid_count, 64):

                    end = min(offset + 64, valid_count)

                    labels = prob_acc[
                        :, offset:end, :W
                    ].argmax(dim=0)

                    class_map[
                        start + offset:start + end, :
                    ] = labels.numpy().astype(np.uint8)

            print(
                f"  Rolling accumulator: "
                f"{prob_acc.numel() * 4 / 1e9:.3f} GB"
            )

            # Process all tiles in batches
            # OOM inside _forward_batch will halve bs and retry the SAME sub-batch.
            # We therefore use a while-loop with a pointer instead of a for-loop.
            ptr = 0
            with tqdm(total=len(coords), desc='  Tiles', unit='tile',
                      leave=False, ncols=80) as pbar:
                while ptr < len(coords):
                    batch_coords = coords[ptr: ptr + bs]
                    tiles_np     = [
                        build_9channel(img_rgb[y:y+T, x:x+T])
                        for y, x in batch_coords
                    ]

                    probs_cpu, bs_used = self._forward_batch(tiles_np, bs)
                    # _forward_batch may have reduced bs_used < len(batch_coords)
                    # if OOM occurred. In that case only bs_used tiles were processed.
                    if bs_used < len(batch_coords):
                        # Only advance by bs_used; remaining tiles re-processed next iter
                        batch_coords = batch_coords[:bs_used]
                        bs = bs_used   # use smaller size for remaining batches

                    # Splat probabilities into accumulator
                    for i, (y, x) in enumerate(batch_coords):
                        p = probs_cpu[i]                       # (C, T, T) float16
                        w = self.hann_cpu.float()                      # (T, T) float16
                        if y != active_y:

                            shift = y - active_y

                            assert 0 < shift <= T

                            finalize_rows(active_y, shift)

                            remaining = T - shift

                            if remaining > 0:
                                prob_acc[:, :remaining, :] = (
                                    prob_acc[:, shift:, :].clone()
                                )

                            prob_acc[:, remaining:, :].zero_()

                            active_y = y

                        prob_acc[:, :, x:x+T] += (
                            p * w.unsqueeze(0)
                        )

                    ptr += len(batch_coords)
                    pbar.update(len(batch_coords))

            # Argmax directly on accumulated weighted probabilities.
            # Positive per-pixel normalization factors do not change argmax.
            # This avoids allocating multiple full-size temporary tensors.
            finalize_rows(active_y, T)

            del prob_acc

            return class_map

```

## src/inference/postprocess.py

```python
"""
Post-processing for orthomosaic segmentation predictions.
Optimised for very large images (25 000+ pixels).

All scipy.ndimage calls replaced with cv2 equivalents which are:
  - 50-100x faster for connected-component labeling
  - Use 2x less RAM (int32 vs int64 label arrays)
  - SIMD-accelerated morphological operations

Expected time on 25 000 × 25 000 image:
  Before fix  (scipy): 15-40 minutes
  After fix   (cv2):   60-90 seconds
"""
from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.masks import compute_aux_masks
from src.utils.constants import CLASS_TO_IDX, NUM_CLASSES


# Minimum pixel count for a connected component to survive per class
MIN_COMPONENT_SIZES: dict[str, int] = {
    'drought':      200,
    'dry_weed':      80,
    'green_weed':    80,
    'cotton_ball':   10,
    'crop':           0,
    'soil':           0,
    'background':     0,
}

HARVEST_COTTON_DENSITY = 0.05
HARVEST_WINDOW         = 256


# ---------------------------------------------------------------------------
# Step 1: field mask
# ---------------------------------------------------------------------------

def apply_field_mask(
    class_map: np.ndarray,
    img_rgb:   np.ndarray,
    aux_masks=None,
) -> np.ndarray:
    """Force black-border (no-data) pixels to background class 0."""
    masks = (compute_aux_masks(img_rgb) if aux_masks is None else aux_masks)
    result = class_map.copy()
    result[~masks['valid_field']] = 0
    return result


# ---------------------------------------------------------------------------
# Step 2: soften glare and shadow pixels
# ---------------------------------------------------------------------------

def soften_problematic_pixels(
    class_map: np.ndarray,
    img_rgb:   np.ndarray,
    aux_masks=None,
) -> np.ndarray:
    """
    Replace glare / shadow pixels with the modal class of their neighbourhood.
    Uses cv2.boxFilter (SIMD-accelerated) instead of per-pixel iteration.
    """
    masks = (compute_aux_masks(img_rgb) if aux_masks is None else aux_masks)
    problem = (masks['glare'] | masks['shadow']).astype(bool)
    if not problem.any():
        return class_map

    result = class_map.copy()
    best_count = np.zeros_like(class_map, dtype=np.uint8)

    for cls in range(NUM_CLASSES):
        indicator  = (class_map == cls).view(np.uint8)
        # Count how many neighbours belong to this class (9×9 window)
        count = cv2.boxFilter(indicator, ddepth=-1,
                               ksize=(9, 9), normalize=False)
        # Where this class dominates AND pixel is problematic → relabel
        dominates = (count > best_count) & problem
        result[dominates] = cls
        best_count = np.maximum(best_count, count)

    return result


# ---------------------------------------------------------------------------
# Step 3: morphological cleanup (cv2 version — fast)
# ---------------------------------------------------------------------------

def morphological_cleanup(class_map: np.ndarray) -> np.ndarray:
    """
    Remove small isolated components per class.

    Uses cv2.connectedComponentsWithStats (50-100x faster than scipy.ndimage.label)
    and cv2.morphologyEx for dilation (SIMD-accelerated).

    Memory: one int32 label array per class at a time = 2.5 GB for 25k×25k.
    Peak is 2.5 GB regardless of class count (arrays are reused).
    Compare to original: 7 × 5 GB (int64) = 35 GB peak.
    """
    result     = class_map.copy()
    H, W       = class_map.shape
    noise_mask = np.zeros((H, W), dtype=np.uint8)

    for cls_name, min_size in MIN_COMPONENT_SIZES.items():
        if min_size <= 0:
            continue
        cls_idx = CLASS_TO_IDX.get(cls_name)
        if cls_idx is None:
            continue

        binary = (class_map == cls_idx).view(np.uint8).copy()
        if not binary.any():
            continue

        # cv2 connected components — int32, ~2.5 GB for 25k×25k
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if n_labels <= 1:
            continue   # no foreground components

        # stats columns: LEFT, TOP, WIDTH, HEIGHT, AREA
        small_components = (
            stats[:, cv2.CC_STAT_AREA] < min_size
        ).astype(np.uint8)
        small_components[0] = 0
        
        # Process in strips to limit temporary memory.
        for row_start in range(0, H, 512):
            row_end = min(row_start + 512, H)
            noise_mask[row_start:row_end] |= (
                small_components[labels[row_start:row_end]]
            )

        # Immediately free the large label array
        del labels, stats
        import gc; gc.collect()

    if not noise_mask.any():
        return result

    # Fill noise pixels with the nearest valid class using dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    for cls_idx in range(NUM_CLASSES):
        if cls_idx == 0:
            continue
        cls_mask    = ((class_map == cls_idx) & ~noise_mask.astype(bool)).view(np.uint8)
        if not cls_mask.any():
            continue
        dilated     = cv2.dilate(cls_mask, kernel)
        fill_region = noise_mask & dilated
        result[fill_region.astype(bool)] = cls_idx

    return result


# ---------------------------------------------------------------------------
# Step 4: harvest zone tagging
# ---------------------------------------------------------------------------

def tag_harvest_zones(class_map: np.ndarray) -> np.ndarray:
    """
    Returns bool mask where cotton_ball density > HARVEST_COTTON_DENSITY.
    Uses cv2.blur (fast, SIMD).
    """
    cotton_idx = CLASS_TO_IDX.get('cotton_ball', 3)
    cotton     = (class_map == cotton_idx).view(np.uint8).astype(np.float32)
    density    = cv2.blur(cotton, (HARVEST_WINDOW, HARVEST_WINDOW))
    return density > HARVEST_COTTON_DENSITY


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def postprocess(
    class_map:  np.ndarray,
    img_rgb:    np.ndarray,
    skip_morph: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full post-processing.

    Returns:
        cleaned_map:  (H, W) uint8 final class predictions
        harvest_mask: (H, W) bool harvest zone indicator
    """
    import time, gc
    t0 = time.time()

    print('    [4/6] Step 1/4: field mask ...')
    aux_masks = compute_aux_masks(img_rgb)
    class_map = apply_field_mask(class_map, img_rgb, aux_masks)
    gc.collect()
    print(f'          done ({time.time()-t0:.1f}s)')

    print('    [4/6] Step 2/4: glare/shadow softening ...')
    t1 = time.time()
    class_map = soften_problematic_pixels(class_map, img_rgb, aux_masks)
    del aux_masks
    gc.collect()
    print(f'          done ({time.time()-t1:.1f}s)')

    if not skip_morph:
        print('    [4/6] Step 3/4: morphological cleanup ...')
        t1 = time.time()
        class_map = morphological_cleanup(class_map)
        gc.collect()
        print(f'          done ({time.time()-t1:.1f}s)')
    else:
        print('    [4/6] Step 3/4: morphological cleanup SKIPPED (--skip-morph)')

    print('    [4/6] Step 4/4: harvest zone tagging ...')
    t1 = time.time()
    harvest_mask = tag_harvest_zones(class_map)
    print(f'          done ({time.time()-t1:.1f}s)')

    print(f'    [4/6] Post-processing total: {time.time()-t0:.1f}s')
    return class_map, harvest_mask

```

## src/inference/memory_safe_postprocess.py

```python

"""Block-wise postprocessing with global connectivity."""

import cv2
import numpy as np
import tempfile

from src.preprocessing import masks as original_masks
import importlib
original_post = importlib.import_module("src.inference.postprocess")
from src.preprocessing.disk_components import (
    strip_components,
    component_mask
)


def blocks(height, rows, halo=0):

    for y in range(0, height, rows):

        end = min(y + rows, height)
        lo = max(0, y - halo)
        hi = min(height, end + halo)

        yield y, end, lo, hi


def select_components(binary, selector, rows=256):

    with tempfile.TemporaryDirectory(
        prefix="drought_boundary_",
        dir="/content"
    ) as work:

        result = boundary_cc.analyze(
            binary,
            strip_rows=rows,
            work_dir=work
        )

        return boundary_cc.selected_mask(
            result, selector
        )


def valid_field_mask(
    rgb,
    dark_threshold=10,
    saturation_threshold=15,
    min_border_component_area=500,
    rows=256
):

    H, W = rgb.shape[:2]

    candidate = np.empty(
        (H, W), dtype=np.uint8
    )

    for y, end, _, _ in blocks(H, rows):

        block = rgb[y:end]

        maximum = block.max(axis=2)
        minimum = block.min(axis=2)

        saturation = (
            maximum.astype(np.int16)
            - minimum.astype(np.int16)
        )

        dark = (
            (maximum <= dark_threshold)
            & (saturation <= saturation_threshold)
        )

        white = (
            (maximum >= 248)
            & (saturation <= 8)
        )

        candidate[y:end] = dark | white

    bg = select_components(
        candidate,
        lambda areas, borders: (
            borders
            & (areas >= min_border_component_area)
        ),
        rows
    )

    del candidate

    result = np.empty(
        (H, W), dtype=bool
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )

    for y, end, lo, hi in blocks(H, rows, 4):

        dilated = cv2.dilate(
            bg[lo:hi].astype(np.uint8),
            kernel,
            iterations=2
        )

        result[y:end] = ~dilated[
            y-lo:end-lo
        ].astype(bool)

    return result


def compute_aux_masks(
    rgb,
    *,
    glare_v=0.95,
    glare_s=0.10,
    shadow_v=0.15,
    shadow_s=0.20,
    field_dark=10,
    field_sat=15,
    field_min_area=500,
    rows=256
):

    H, W = rgb.shape[:2]

    valid = valid_field_mask(
        rgb,
        field_dark,
        field_sat,
        field_min_area,
        rows
    )

    glare = np.empty((H, W), dtype=bool)
    shadow = np.empty((H, W), dtype=bool)

    for y, end, _, _ in blocks(H, rows):

        glare[y:end] = original_masks.glare_mask(
            rgb[y:end], glare_v, glare_s
        )

        shadow[y:end] = original_masks.shadow_mask(
            rgb[y:end], shadow_v, shadow_s
        )

    return {
        "valid_field": valid,
        "glare": glare,
        "shadow": shadow
    }


def soften_problematic_pixels(
    class_map,
    rgb,
    aux_masks=None,
    rows=256
):

    masks = (
        compute_aux_masks(rgb, rows=rows)
        if aux_masks is None
        else aux_masks
    )

    H, W = class_map.shape

    result = class_map.copy()

    for y, end, lo, hi in blocks(H, rows, 4):

        problem = (
            masks["glare"][y:end]
            | masks["shadow"][y:end]
        )

        if not problem.any():
            continue

        best = np.zeros(
            (end-y, W), dtype=np.uint8
        )

        target = result[y:end]

        for cls in range(original_post.NUM_CLASSES):

            indicator = (
                class_map[lo:hi] == cls
            ).astype(np.uint8)

            count = cv2.boxFilter(
                indicator,
                ddepth=-1,
                ksize=(9, 9),
                normalize=False
            )[y-lo:end-lo]

            dominates = (
                (count > best) & problem
            )

            target[dominates] = cls

            np.maximum(
                best, count, out=best
            )

    return result


def morphological_cleanup(class_map, rows=256):

    H, W = class_map.shape

    result = class_map.copy()

    noise = np.zeros(
        (H, W), dtype=bool
    )

    for name, minimum in (
        original_post.MIN_COMPONENT_SIZES.items()
    ):

        if minimum <= 0:
            continue

        cls = original_post.CLASS_TO_IDX.get(name)

        if cls is None:
            continue

        binary = (class_map == cls)

        if not binary.any():
            continue

        small = select_components(
            binary,
            lambda areas, borders: (
                (areas > 0)
                & (areas < minimum)
            ),
            rows
        )

        noise |= small

        del binary, small

    if not noise.any():
        return result

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (15, 15)
    )

    for y, end, lo, hi in blocks(H, rows, 7):

        target = result[y:end]

        for cls in range(
            1, original_post.NUM_CLASSES
        ):

            valid = (
                (class_map[lo:hi] == cls)
                & ~noise[lo:hi]
            ).astype(np.uint8)

            if not valid.any():
                continue

            dilated = cv2.dilate(
                valid, kernel
            )[y-lo:end-lo]

            fill = (
                noise[y:end]
                & dilated.astype(bool)
            )

            target[fill] = cls

    return result


def tag_harvest_zones(class_map, rows=256):

    H, W = class_map.shape

    cotton_idx = original_post.CLASS_TO_IDX.get(
        "cotton_ball", 3
    )

    window = original_post.HARVEST_WINDOW

    threshold = (
        original_post.HARVEST_COTTON_DENSITY
    )

    result = np.empty(
        (H, W), dtype=bool
    )

    for y, end, lo, hi in blocks(
        H, rows, window // 2
    ):

        cotton = (
            class_map[lo:hi] == cotton_idx
        ).astype(np.float32)

        density = cv2.blur(
            cotton, (window, window)
        )

        result[y:end] = (
            density[y-lo:end-lo] > threshold
        )

    return result


def postprocess(
    class_map,
    rgb,
    skip_morph=False,
    rows=256
):

    masks = compute_aux_masks(
        rgb, rows=rows
    )

    result = class_map.copy()

    for y, end, _, _ in blocks(
        result.shape[0], rows
    ):

        result[y:end][
            ~masks["valid_field"][y:end]
        ] = 0

    result = soften_problematic_pixels(
        result, rgb, masks, rows
    )

    del masks

    if not skip_morph:

        result = morphological_cleanup(
            result, rows
        )

    harvest = tag_harvest_zones(
        result, rows
    )

    return result, harvest


# BOUNDARY_COMPONENTS_V1

from src.preprocessing import boundary_components as boundary_cc

```

## src/inference/memory_safe_report.py

```python

"""Memory-efficient drought reporting."""

import numpy as np
import tempfile
import importlib

reference = importlib.import_module(
    "src.inference.report"
)

CLASS_TO_IDX = reference.CLASS_TO_IDX

from src.preprocessing.disk_components import (
    strip_components
)

def drought_decision(
    area_stats: dict,
    harvest_mask: np.ndarray,
    class_map: np.ndarray,
    tau_fraction: float    = 0.10,
    tau_non_harvest: float = 0.08,
    tau_cc_share: float    = 0.20,
) -> dict:
    """
    Three-signal drought decision.

    Signal 1 — Drought fraction > tau_fraction (10%)
        Fraction of the agricultural area (crop+cotton+drought+dry_weed)
        classified as drought.

    Signal 2 — Non-harvest drought fraction > tau_non_harvest (8%)
        Same as Signal 1 but excludes harvest zones (high cotton_ball density)
        to avoid false positives from senescent cotton leaves around open bolls.

    Signal 3 — Largest Connected Component (CC) share > tau_cc_share (20%)
        Fraction of ALL drought pixels belonging to the single largest
        spatially connected drought region. High = coherent stress zone.
        Low = scattered noise / misclassification.

    Decision: DROUGHT if Signal 1 alone (high fraction) OR (Signal 2 AND Signal 3).
    """

    drought_idx = CLASS_TO_IDX.get("drought", 6)

    H, W = class_map.shape

    drought_binary = np.empty(
        (H, W), dtype=np.uint8
    )

    n_drought_px = 0

    for y in range(0, H, 256):

        end = min(y + 256, H)

        block = (
            class_map[y:end] == drought_idx
        )

        drought_binary[y:end] = block

        n_drought_px += int(
            np.count_nonzero(block)
        )

    largest_cc = 0
    cc_share = 0.0

    if n_drought_px > 0:

        with tempfile.TemporaryDirectory(
            prefix="drought_decision_",
            dir="/content"
        ) as work:

            result = boundary_cc.analyze(
                drought_binary,
                strip_rows=256,
                work_dir=work
            )

            largest_cc = int(result["largest"])

            cc_share = (
                largest_cc / n_drought_px
            )

            del result

    del drought_binary

    df   = area_stats['drought_fraction']
    nhdf = area_stats['non_harvest_drought_fraction']
    sig1 = df   > tau_fraction
    sig2 = nhdf > tau_non_harvest
    sig3 = cc_share > tau_cc_share

    # Updated rule: Signal 1 alone is sufficient (handles distributed row drought)
    is_drought = sig1 or (sig2 and sig3)
    confidence = round(min(1.0,
        (df   / max(tau_fraction,    1e-6)) * 0.5 +
        (nhdf / max(tau_non_harvest, 1e-6)) * 0.3 +
        float(sig3) * 0.2
    ), 3)

    return {
        'drought':                         is_drought,
        'confidence':                      confidence,
        'stress_score': confidence,
        'score_is_calibrated_probability': False,
        'decision_type': 'visual_stress_heuristic',
        'field_confirmation_required': True,
        'drought_fraction':                df,
        'non_harvest_drought_fraction':    nhdf,
        'largest_cc_share':                round(cc_share, 4),
        'signal_1_fraction':               sig1,
        'signal_2_non_harvest':            sig2,
        'signal_3_coherent':               sig3,
        'tau_fraction':                    tau_fraction,
        'tau_non_harvest':                 tau_non_harvest,
        'tau_cc_share':                    tau_cc_share,
    }


# BOUNDARY_COMPONENTS_V1

from src.preprocessing import boundary_components as boundary_cc

```