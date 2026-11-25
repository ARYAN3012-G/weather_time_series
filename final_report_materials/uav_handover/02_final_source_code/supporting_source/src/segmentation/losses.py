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
