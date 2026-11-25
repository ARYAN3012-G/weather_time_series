"""
Loss functions for the patch classifier.

Two losses combined:
    1. SoftCrossEntropyLoss — handles both hard labels and soft labels from
       MixUp/CutMix. Class-weighted to handle barren imbalance.

    2. PairwiseMarginLoss — pulls embeddings of chronic-confusion pairs apart
       in feature space. Applied on the 388-dim embedding (before the head).
       For each confusion pair (A, B) present in the batch, the loss penalizes
       any A-embedding that is closer to a B-embedding than to another A-embedding,
       and vice versa. This directly forces the model to learn discriminative
       features for the pairs it finds hardest.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import CLASS_TO_IDX, CONFUSION_PAIRS, NUM_CLASSES


# ---------------------------------------------------------------------------
# Soft cross-entropy (handles MixUp/CutMix soft labels)
# ---------------------------------------------------------------------------

class SoftCrossEntropyLoss(nn.Module):
    """
    Cross-entropy that accepts soft (one-hot or mixed) label vectors.

    For a hard one-hot label this is identical to nn.CrossEntropyLoss.
    For soft labels from MixUp/CutMix it computes the expected log-prob.

    Class weights are applied to address the barren class imbalance.
    """

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        logits: torch.Tensor,      # (B, C)
        soft_labels: torch.Tensor, # (B, C) — each row sums to 1
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            # Weight each class contribution by its class weight
            weighted_log_probs = log_probs * w.unsqueeze(0)
            loss = -(soft_labels * weighted_log_probs).sum(dim=-1).mean()
        else:
            loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        return loss


# ---------------------------------------------------------------------------
# Pairwise margin loss
# ---------------------------------------------------------------------------

class PairwiseMarginLoss(nn.Module):
    """
    For each confusion pair (class_a, class_b) present in the current batch:

        For every sample with hard label = class_a:
            Find its nearest neighbour among class_a embeddings (positive).
            Find its nearest neighbour among class_b embeddings (negative).
            Apply margin loss: max(0, d_pos - d_neg + margin)

    This is a within-batch online triplet mining restricted to the
    confusion pairs defined in CONFUSION_PAIRS.

    Args:
        margin:  Minimum distance gap we want between same-class and cross-class.
        weight:  Scalar multiplier on the total loss.
    """

    PAIR_IDX: list[tuple[int, int]] = [
        (CLASS_TO_IDX[a], CLASS_TO_IDX[b])
        for a, b in CONFUSION_PAIRS
        if a in CLASS_TO_IDX and b in CLASS_TO_IDX
    ]

    def __init__(self, margin: float = 0.3, weight: float = 0.3) -> None:
        super().__init__()
        self.margin = margin
        self.weight = weight

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D) L2-normalized
        hard_labels: torch.Tensor, # (B,) int
    ) -> torch.Tensor:
        if embeddings.shape[0] < 2:
            return embeddings.sum() * 0.0

        # L2-normalize
        emb = F.normalize(embeddings, p=2, dim=-1)
        # Pairwise L2 distances (B, B)
        dist = torch.cdist(emb, emb, p=2)

        total_loss = torch.tensor(0.0, device=embeddings.device)
        n_terms = 0

        for cls_a, cls_b in self.PAIR_IDX:
            mask_a = hard_labels == cls_a
            mask_b = hard_labels == cls_b
            if mask_a.sum() < 1 or mask_b.sum() < 1:
                continue

            # For each anchor in cls_a:
            #   positive = nearest cls_a sample (excluding self)
            #   negative = nearest cls_b sample
            a_indices = mask_a.nonzero(as_tuple=True)[0]
            for anc_idx in a_indices:
                # Positive distances (same class, excluding self)
                pos_dists = dist[anc_idx][mask_a]
                # Exclude self (distance = 0)
                pos_dists = pos_dists[pos_dists > 1e-6]
                if pos_dists.numel() == 0:
                    continue
                d_pos = pos_dists.min()

                # Negative distances (confusion partner class)
                neg_dists = dist[anc_idx][mask_b]
                d_neg = neg_dists.min()

                loss = F.relu(d_pos - d_neg + self.margin)
                total_loss = total_loss + loss
                n_terms += 1

            # Symmetric: for each anchor in cls_b
            b_indices = mask_b.nonzero(as_tuple=True)[0]
            for anc_idx in b_indices:
                pos_dists = dist[anc_idx][mask_b]
                pos_dists = pos_dists[pos_dists > 1e-6]
                if pos_dists.numel() == 0:
                    continue
                d_pos = pos_dists.min()
                neg_dists = dist[anc_idx][mask_a]
                d_neg = neg_dists.min()
                loss = F.relu(d_pos - d_neg + self.margin)
                total_loss = total_loss + loss
                n_terms += 1

        if n_terms == 0:
            return embeddings.sum() * 0.0

        return self.weight * total_loss / n_terms


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class ClassifierLoss(nn.Module):
    """
    Combined loss = SoftCE + PairwiseMargin.

    Args:
        class_counts: dict mapping class_idx → sample count (used for weighting).
        barren_oversample: extra multiplier for barren weight (rare class).
        margin_weight: weight on the pairwise margin loss.
        margin: margin value for PairwiseMarginLoss.
    """

    def __init__(
        self,
        class_counts: dict[int, int] | None = None,
        barren_oversample: int = 8,
        margin_weight: float = 0.3,
        margin: float = 0.3,
    ) -> None:
        super().__init__()

        # Build class weights (inverse frequency, normalized to mean=1)
        if class_counts is not None:
            barren_idx = CLASS_TO_IDX.get("barren", -1)
            weights = torch.ones(NUM_CLASSES)
            for idx, count in class_counts.items():
                if count > 0:
                    weights[idx] = 1.0 / count
                    if idx == barren_idx:
                        weights[idx] *= barren_oversample
            weights = weights / weights.mean()
        else:
            weights = None

        self.ce_loss = SoftCrossEntropyLoss(class_weights=weights)
        self.margin_loss = PairwiseMarginLoss(margin=margin, weight=margin_weight)

    def forward(
        self,
        logits: torch.Tensor,      # (B, C)
        embeddings: torch.Tensor,  # (B, D) — pre-head features
        soft_labels: torch.Tensor, # (B, C)
        hard_labels: torch.Tensor, # (B,)
    ) -> dict[str, torch.Tensor]:
        """Returns a dict with 'loss', 'ce', 'margin' for logging."""
        ce = self.ce_loss(logits, soft_labels)
        margin = self.margin_loss(embeddings, hard_labels)
        total = ce + margin
        return {"loss": total, "ce": ce, "margin": margin}
