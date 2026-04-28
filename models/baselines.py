"""Baseline models for the paper: U-Net, DeepLabV3+, SegFormer-Tiny.

All three follow the standard MADOS preprocessing: nearest-upsample 20 m and
60 m bands to the 10 m grid, then concatenate into an 11-channel tensor. This
is exactly the "single-grid" view the paper critiques.

Each wrapper exposes the same ``(logits, emb, branch)`` dict interface as
``CRCCGDNet`` so that training and evaluation code can treat it uniformly.
The ``branch`` dict and the ``emb`` tensor are populated with the logits and a
dummy zero tensor respectively — CRC / CGD losses are disabled for baselines
via ``--no-crc --no-cgd``.
"""
from __future__ import annotations

from typing import Dict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


IN_CHANS_TOTAL = 11  # 4 (10m) + 6 (20m) + 1 (60m)


def _stack_to_10m(x10: torch.Tensor, x20: torch.Tensor, x60: torch.Tensor) -> torch.Tensor:
    H, W = x10.shape[-2:]
    x20u = F.interpolate(x20, size=(H, W), mode="nearest")
    x60u = F.interpolate(x60, size=(H, W), mode="nearest")
    # Order matches MADOS's sorted-by-wavelength layout after combining: the
    # exact ordering only matters within a baseline; we keep it consistent.
    return torch.cat([x10, x20u, x60u], dim=1)


class _Wrapper(nn.Module):
    """Wraps a plain SMP segmentation model into the (logits, emb, branch) dict API.

    SMP checks the input shape is divisible by 32. MADOS crops are 240 x 240,
    so we reflect-pad to the next multiple of 32 and crop the logits back.
    """

    def __init__(self, net: nn.Module, num_classes: int, stride: int = 32) -> None:
        super().__init__()
        self.net = net
        self.num_classes = num_classes
        self.stride = stride

    def forward(self, x10: torch.Tensor, x20: torch.Tensor, x60: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = _stack_to_10m(x10, x20, x60)
        H, W = x.shape[-2:]
        pad_h = (self.stride - H % self.stride) % self.stride
        pad_w = (self.stride - W % self.stride) % self.stride
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        logits = self.net(x)
        logits = logits[..., :H, :W]
        B, K, _, _ = logits.shape
        zeros = logits.new_zeros((B, 1, H // 4, W // 4))  # dummy emb
        return {
            "logits": logits,
            "emb":    zeros,
            "branch": {"10": logits, "20": logits, "60": logits},
        }


def build_unet(num_classes: int = 15, encoder: str = "resnet34") -> nn.Module:
    net = smp.Unet(
        encoder_name=encoder, encoder_weights=None,
        in_channels=IN_CHANS_TOTAL, classes=num_classes,
    )
    return _Wrapper(net, num_classes)


def build_deeplabv3p(num_classes: int = 15, encoder: str = "resnet34") -> nn.Module:
    net = smp.DeepLabV3Plus(
        encoder_name=encoder, encoder_weights=None,
        in_channels=IN_CHANS_TOTAL, classes=num_classes,
    )
    return _Wrapper(net, num_classes)


def build_segformer(num_classes: int = 15, encoder: str = "mit_b0") -> nn.Module:
    net = smp.Segformer(
        encoder_name=encoder, encoder_weights=None,
        in_channels=IN_CHANS_TOTAL, classes=num_classes,
    )
    return _Wrapper(net, num_classes)


BASELINES = {
    "unet":       build_unet,
    "deeplabv3p": build_deeplabv3p,
    "segformer":  build_segformer,
}
