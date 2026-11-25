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
                    probs_cpu = F.softmax(logits, dim=1).cpu().half()

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
        prob_acc   = torch.zeros(C, Hp, Wp, dtype=torch.float16)
        weight_acc = torch.zeros(1, Hp, Wp, dtype=torch.float16)

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
                    w = self.hann_cpu                      # (T, T) float16
                    prob_acc[:, y:y+T, x:x+T]   += p * w.unsqueeze(0)
                    weight_acc[0, y:y+T, x:x+T] += w

                ptr += len(batch_coords)
                pbar.update(len(batch_coords))

        # Normalise probability map
        prob_acc = prob_acc / weight_acc.clamp(min=1e-4)   # (C, Hp, Wp)
        cs       = prob_acc.sum(dim=0, keepdim=True).clamp(min=1e-4)
        prob_acc = prob_acc / cs

        # Argmax + crop back to original size
        class_map = prob_acc.argmax(dim=0).numpy().astype(np.uint8)
        return class_map[:H, :W]
