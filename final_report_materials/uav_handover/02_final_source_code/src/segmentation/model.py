"""
SegFormer-B2 wrapper for the drought segmentation task.

Why SegFormer:
    - Hierarchical (multi-scale) features — natural fit for our spatial
      classes that span small (cotton_ball) to large (whole-field drought).
    - MiT (Mix Transformer) encoder is efficient — fits batch=16 at 512²
      on H200 comfortably.
    - Lightweight MLP decoder — fast inference at orthomosaic scale.

Key adaptations from stock SegFormer:
    - 9-channel input (RGB + 5 vegetation indices + V) instead of 3
    - 7-class output head
    - Pretrained ImageNet weights are loaded into the encoder where possible
      (first conv inflated 3→9 with weight surgery)
    - Output upsampled from H/4 × W/4 to full H × W via bilinear interpolation
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
)

from src.utils.constants import NUM_CLASSES


# SegFormer-B2 architecture specs (from the paper)
_SEGFORMER_B2_HIDDEN  = [64, 128, 320, 512]
_SEGFORMER_B2_DEPTHS  = [3, 4, 6, 3]
_SEGFORMER_B2_DEC_DIM = 768


class SegFormerB2(nn.Module):
    """
    SegFormer-B2 with 9-channel input and configurable num_classes.

    Args:
        num_classes:           Output classes (default 7).
        input_size:            Input resolution (must be divisible by 16).
        pretrained_id:         HF model ID for ImageNet pretrained encoder
                               (e.g., 'nvidia/mit-b2'). None → random init.
        pretrained_local_path: Path to a local .pt/.bin file with state_dict
                               from a previous run. Overrides pretrained_id.
        in_channels:           Input channels (default 9 for our pipeline).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_size: int = 512,
        pretrained_id: str | None = 'nvidia/mit-b2',
        pretrained_local_path: str | None = None,
        in_channels: int = 9,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size  = input_size
        self.in_channels = in_channels

        # Build config — uses 3 channels for compatibility with pretrained
        # weights, then we surgically replace the first conv with a 9-ch one.
        config_kwargs = dict(
            num_channels=3,
            num_labels=num_classes,
            hidden_sizes=_SEGFORMER_B2_HIDDEN,
            depths=_SEGFORMER_B2_DEPTHS,
            decoder_hidden_size=_SEGFORMER_B2_DEC_DIM,
            semantic_loss_ignore_index=255,
        )
        cfg = SegformerConfig(**config_kwargs)

        # Try to download pretrained encoder. Fall back to random init.
        pretrained_3ch_sd = None
        if pretrained_local_path is not None and os.path.isfile(pretrained_local_path):
            try:
                sd = torch.load(pretrained_local_path, map_location='cpu',
                                weights_only=True)
                if 'model_state_dict' in sd:
                    sd = sd['model_state_dict']
                pretrained_3ch_sd = sd
                print(f'[INFO] Loaded local SegFormer weights from {pretrained_local_path}')
            except Exception as e:
                print(f'[WARNING] Failed local load: {e}')

        if pretrained_3ch_sd is None and pretrained_id:
            try:
                # Force safetensors (avoids torch.load CVE check on older torch)
                hf_model = SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_id, num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    use_safetensors=True,
                )
                pretrained_3ch_sd = hf_model.state_dict()
                del hf_model
                print(f'[INFO] Loaded SegFormer-B2 from HuggingFace ({pretrained_id})')
            except Exception as e:
                print(f'[WARNING] HF download failed: {e}')
                # Fallback: try loading from local path if available
                if pretrained_local_path and os.path.isfile(pretrained_local_path):
                    try:
                        sd = torch.load(pretrained_local_path, map_location='cpu',
                                        weights_only=True)
                        if isinstance(sd, dict) and 'model_state_dict' in sd:
                            sd = sd['model_state_dict']
                        pretrained_3ch_sd = sd
                        print(f'[INFO] Loaded SegFormer-B2 from local {pretrained_local_path}')
                    except Exception as e2:
                        print(f'[WARNING] Local load also failed: {e2}')
                if pretrained_3ch_sd is None:
                    print('         Encoder will use random init (NOT recommended).')

        # Build the actual model with our channel count
        self.model = SegformerForSemanticSegmentation(cfg)

        # Load pretrained weights (3-channel)
        if pretrained_3ch_sd is not None:
            missing, unexpected = self.model.load_state_dict(
                pretrained_3ch_sd, strict=False
            )
            if len(missing) > 0:
                print(f'[INFO] Missing {len(missing)} keys (head re-init expected)')

        # Now inflate the first patch_embed conv from 3 → in_channels channels
        # Locate: model.segformer.encoder.patch_embeddings[0].proj
        first_conv = self.model.segformer.stages[0].patch_embeddings.proj
        old_w = first_conv.weight.data        # (out_C, 3, kH, kW)
        out_c, _, kh, kw = old_w.shape

        new_conv = nn.Conv2d(
            in_channels, out_c,
            kernel_size=(kh, kw),
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_w = new_conv.weight.data
            # Copy RGB channels (scaled to preserve magnitude)
            new_w[:, :3, :, :] = old_w / 3.0
            # Zero-init the new channels — they learn from scratch but don't
            # interfere with the pretrained RGB signal
            new_w[:, 3:, :, :] = 0.0
            new_conv.weight.data = new_w
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        self.model.segformer.stages[0].patch_embeddings.proj = new_conv

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) float32
        Returns:
            (B, num_classes, H, W) logits — upsampled to input resolution
        """
        H, W = x.shape[-2:]
        outputs = self.model(pixel_values=x)
        logits  = outputs.logits           # (B, num_classes, H/4, W/4)
        # Upsample to full input size
        logits = F.interpolate(logits, size=(H, W),
                               mode='bilinear', align_corners=False)
        return logits
