"""
MultiscalePatchClassifier

Architecture:
    9-channel input (RGB + ExG + ExGR + CIVE + GLI + NGRDI + V)
        ↓
    DINOv2 ViT-S/14 backbone (ImageNet-pretrained, 9-channel via weight surgery)
        ↓
    CLS token → 384-dim embedding
        ↓
    Concatenate with 4-dim patch scale embedding (learned, not one-hot)
        ↓
    MLP head [388 → 256 → 8 classes]

Scale embedding:
    The network receives a learned embedding of the original patch size bucket
    (64 / 128 / 256 / 512 / other→256). This tells the model at what physical
    resolution it is looking, which is critical because drought vs. dry_weed
    look identical at 64px but diverge strongly at 512px.

Weight surgery for 9-channel input:
    The pretrained 3-channel patch_embed.proj conv is loaded, its weights are
    copied to the first 3 channels of a new 9-channel conv (divided by 3 to
    preserve approximate activation magnitude), and channels 3-8 are zero-init.
    The zero init is intentional: the vegetation index channels start silent and
    learn to activate only when they provide signal beyond RGB.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import NUM_CLASSES

# Backbone model name in timm
_BACKBONE_NAME = "vit_base_patch14_dinov2.lvd142m"
_EMBED_DIM = 768   # ViT-B embed dim

# Map patch sizes to bucket indices for the scale embedding
_SIZE_BUCKETS = [64, 128, 256, 512]


def _size_to_bucket(orig_size: int) -> int:
    """Map the largest patch dimension to a bucket index 0-3."""
    for i, threshold in enumerate([96, 192, 384, 999999]):
        if orig_size <= threshold:
            return i
    return 3  # 512+


class PatchScaleEmbedding(nn.Module):
    """Learned embedding for the 4 patch-size buckets."""
    def __init__(self, num_buckets: int = 4, embed_dim: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_buckets, embed_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, orig_sizes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            orig_sizes: (B,) int tensor of original patch sizes (px).
        Returns:
            (B, embed_dim) float tensor.
        """
        buckets = torch.tensor(
            [_size_to_bucket(s.item()) for s in orig_sizes],
            dtype=torch.long,
            device=self.emb.weight.device,
        )
        return self.emb(buckets)


class MultiscalePatchClassifier(nn.Module):
    """
    Args:
        num_classes:  Number of output classes (default 8).
        pretrained:   Load ImageNet weights for the backbone (default True).
        freeze_backbone_epochs: How many epochs to freeze the backbone.
                      Managed externally by the training loop; this flag is
                      read by the trainer, not used internally here.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        pretrained_weights_path: str | None = None,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        # ----------------------------------------------------------------
        # Backbone: DINOv2 ViT-B/14, loaded with 9-channel input
        # ----------------------------------------------------------------
        # Step 1: Load 3-channel pretrained weights (HuggingFace or local .pt/.pth file)
        w3 = None
        if pretrained:
            try:
                backbone_3ch = timm.create_model(
                    _BACKBONE_NAME,
                    pretrained=True,
                    num_classes=0,
                    img_size=img_size,
                    in_chans=3,
                )
                w3 = backbone_3ch.patch_embed.proj.weight.data.clone()  # (768, 3, 14, 14)
                del backbone_3ch
            except Exception:
                pass  # fall through to local weights path

        # Local weights override: pass pretrained_weights_path to __init__
        if w3 is None and pretrained_weights_path is not None:
            import os
            if os.path.isfile(pretrained_weights_path):
                sd = torch.load(pretrained_weights_path, map_location="cpu", weights_only=True)
                # Handle different save formats
                sd = sd.get("model", sd.get("state_dict", sd))
                tmp_backbone = timm.create_model(
                    _BACKBONE_NAME, pretrained=False, num_classes=0,
                    img_size=img_size, in_chans=3,
                )
                missing, unexpected = tmp_backbone.load_state_dict(sd, strict=False)
                if len(missing) < 10:  # tolerate head keys missing
                    w3 = tmp_backbone.patch_embed.proj.weight.data.clone()
                    print(f"[INFO] Loaded local DINOv2 weights from {pretrained_weights_path}")
                del tmp_backbone
            else:
                print(f"[WARNING] pretrained_weights_path not found: {pretrained_weights_path}")

        if w3 is None and pretrained:
            print("[WARNING] Could not load pretrained weights — using random init.")

        # Step 2: Create 9-channel model
        self.backbone = timm.create_model(
            _BACKBONE_NAME,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
            in_chans=9,
        )

        # Step 3: Weight surgery — copy RGB weights, zero-init index channels
        if pretrained and w3 is not None:
            with torch.no_grad():
                w9 = self.backbone.patch_embed.proj.weight.data
                # First 3 channels: copy from pretrained, scale to preserve magnitude
                # (9-ch input has 3x more channels contributing to each feature)
                w9[:, :3, :, :] = w3 / 3.0
                # Channels 3-8: zero-init (learn from scratch, don't interfere with RGB signal)
                w9[:, 3:, :, :] = 0.0

        # ----------------------------------------------------------------
        # Scale embedding
        # ----------------------------------------------------------------
        self.scale_emb = PatchScaleEmbedding(num_buckets=4, embed_dim=4)

        # ----------------------------------------------------------------
        # MLP head: backbone_dim + scale_dim → classes
        # ----------------------------------------------------------------
        head_in = _EMBED_DIM + 4   # 384 + 4 = 388
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self._init_head()

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        orig_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images:     (B, 9, H, W) float32
            orig_sizes: (B,) int64 — largest original patch dimension
        Returns:
            (B, num_classes) logits (unnormalized)
        """
        feat = self.backbone(images)           # (B, 384)
        scale = self.scale_emb(orig_sizes)     # (B, 4)
        combined = torch.cat([feat, scale], dim=1)  # (B, 388)
        return self.head(combined)             # (B, num_classes)

    def get_embeddings(
        self,
        images: torch.Tensor,
        orig_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 388-dim embedding before the head (used for feature bank)."""
        feat = self.backbone(images)
        scale = self.scale_emb(orig_sizes)
        return torch.cat([feat, scale], dim=1)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters. Head and scale_emb remain trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_optimizer_param_groups(
        self,
        lr_head: float = 3e-4,
        lr_backbone: float = 3e-5,
    ) -> list[dict]:
        """
        Returns param groups with separate LRs for head/scale_emb vs backbone.
        Called by the trainer to set up AdamW.
        """
        head_params = list(self.head.parameters()) + list(self.scale_emb.parameters())
        backbone_params = list(self.backbone.parameters())
        return [
            {"params": head_params, "lr": lr_head, "name": "head"},
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        ]
