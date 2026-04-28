"""Three-stem multi-resolution segmentation network for MADOS.

Design:
    Each of the three native resolutions (10 / 20 / 60 m) has its own stem that
    projects the raw band stack to a common channel dimension. The three stem
    feature maps are bilinearly aligned to the 10 m grid and channel-concatenated
    into the fused input of a shared MiT encoder plus an all-MLP SegFormer head.

    Alongside the fused prediction, three lightweight auxiliary heads expose
    per-branch logits computed from *only* that branch's stem features. These
    per-branch logits are what the CRC consistency loss operates on.

    A pre-classifier feature tensor is also returned so that the CGD prototype
    loss can pool class-conditional prototypes from the same shared feature
    space used for the fused prediction.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder


# ---------------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------------
class _Stem(nn.Module):
    """Two-conv stem: Conv-BN-GELU x2, preserves spatial resolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# SegFormer-style all-MLP head
# ---------------------------------------------------------------------------
class _SegFormerHead(nn.Module):
    """Minimal all-MLP head from the SegFormer paper.

    Projects every stage to ``embed_dim`` via a 1x1 conv, upsamples to the
    stride-4 grid, concatenates, fuses with a 1x1 conv, and finally produces
    per-class logits. Both the fused feature tensor (pre-classifier) and the
    logits are returned.
    """

    def __init__(self, in_channels: List[int], embed_dim: int, num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, feats: List[torch.Tensor]):
        target_hw = feats[0].shape[-2:]
        projected = []
        for f, proj in zip(feats, self.projs):
            f = proj(f)
            if f.shape[-2:] != target_hw:
                f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            projected.append(f)
        x = self.fuse(torch.cat(projected, dim=1))
        emb = x
        x = self.drop(x)
        logits = self.classifier(x)
        return logits, emb


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------
class CRCCGDNet(nn.Module):
    """Three-stem MiT + SegFormer head for MADOS.

    Parameters
    ----------
    num_classes : int
        Number of output classes (15 for MADOS).
    stem_ch : int
        Output channel dimension of each resolution stem.
    backbone : str
        Encoder name recognised by ``segmentation_models_pytorch``. Default
        ``mit_b0`` is SegFormer-Tiny.
    head_dim : int
        Embedding dimension inside the SegFormer head.
    pretrained : bool
        If True, tries to load the encoder's published weights (SMP will silently
        refuse when in_channels != 3, so this is effectively from-scratch).
    """

    def __init__(
        self,
        num_classes: int = 15,
        in_ch_10: int = 4,
        in_ch_20: int = 6,
        in_ch_60: int = 1,
        stem_ch: int = 32,
        backbone: str = "mit_b0",
        head_dim: int = 256,
        pretrained: bool = False,
        use_bands: tuple = ("10", "20", "60"),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_bands = tuple(use_bands)
        assert "10" in self.use_bands, "10 m branch is mandatory (target grid)"

        self.stem10 = _Stem(in_ch_10, stem_ch)
        self.stem20 = _Stem(in_ch_20, stem_ch) if "20" in self.use_bands else None
        self.stem60 = _Stem(in_ch_60, stem_ch) if "60" in self.use_bands else None

        fused_ch = stem_ch * len(self.use_bands)
        # SMP's MiT encoder returns 6 stages: [input-echo, dummy-zero,
        # stage1, stage2, stage3, stage4]. We keep only the 4 real stages.
        self.encoder = get_encoder(
            backbone,
            in_channels=fused_ch,
            depth=5,
            weights="imagenet" if pretrained else None,
        )
        enc_ch = self.encoder.out_channels[2:]        # 4 real stages
        self.head = _SegFormerHead(enc_ch, embed_dim=head_dim, num_classes=num_classes)

        # Auxiliary per-branch heads operate on stem features only.
        self.aux10 = nn.Conv2d(stem_ch, num_classes, 1)
        self.aux20 = nn.Conv2d(stem_ch, num_classes, 1) if "20" in self.use_bands else None
        self.aux60 = nn.Conv2d(stem_ch, num_classes, 1) if "60" in self.use_bands else None

    # ------------------------------------------------------------------
    def forward(self, x10: torch.Tensor, x20: torch.Tensor,
                x60: torch.Tensor) -> Dict[str, torch.Tensor]:
        H, W = x10.shape[-2:]

        f10 = self.stem10(x10)
        f20 = self.stem20(x20) if self.stem20 is not None else None
        f60 = self.stem60(x60) if self.stem60 is not None else None

        branch_logits = {
            "10": F.interpolate(self.aux10(f10), size=(H, W), mode="bilinear", align_corners=False),
        }
        if f20 is not None:
            branch_logits["20"] = F.interpolate(self.aux20(f20), size=(H, W), mode="bilinear", align_corners=False)
        if f60 is not None:
            branch_logits["60"] = F.interpolate(self.aux60(f60), size=(H, W), mode="bilinear", align_corners=False)

        fused_list = [f10]
        if f20 is not None:
            fused_list.append(F.interpolate(f20, size=(H, W), mode="bilinear", align_corners=False))
        if f60 is not None:
            fused_list.append(F.interpolate(f60, size=(H, W), mode="bilinear", align_corners=False))
        fused_in = torch.cat(fused_list, dim=1)

        feats = self.encoder(fused_in)[2:]       # keep only the 4 real stages
        logits_lr, emb_lr = self.head(feats)
        logits = F.interpolate(logits_lr, size=(H, W), mode="bilinear", align_corners=False)

        return {
            "logits": logits,              # [B, K, H, W] — main prediction
            "emb":    emb_lr,              # [B, D, H/4, W/4] — for CGD prototypes
            "branch": branch_logits,        # subset of {"10","20","60"} → [B, K, H, W]
        }
