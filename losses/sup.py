"""Supervised loss: weighted CE + soft Dice, restricted to High-confidence pixels.

Non-High-confidence pixels are pushed to IGNORE_INDEX *before* this loss runs,
so both CE and Dice see a single valid / invalid distinction and do not need
to know the confidence map.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCEDiceLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
        dice_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w = self.class_weights if self.class_weights is not None else None
        ce = F.cross_entropy(
            logits, y,
            weight=w, ignore_index=self.ignore_index, reduction="mean",
        )

        # Soft multi-class Dice over valid pixels.
        B, K, H, W = logits.shape
        valid = (y != self.ignore_index).unsqueeze(1).float()                   # [B,1,H,W]
        y_safe = torch.where(y == self.ignore_index, torch.zeros_like(y), y).clamp_(0, K - 1)
        y_oh = F.one_hot(y_safe, num_classes=K).permute(0, 3, 1, 2).float()     # [B,K,H,W]
        y_oh = y_oh * valid

        probs = F.softmax(logits, dim=1) * valid
        inter = (probs * y_oh).sum(dim=(0, 2, 3))
        denom = probs.sum(dim=(0, 2, 3)) + y_oh.sum(dim=(0, 2, 3))
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        return ce + self.dice_weight * dice.mean()


def gen_class_weights(class_distr: torch.Tensor, c: float = 1.02) -> torch.Tensor:
    """Same weighting used by MADOS: w_k = 1 / log(c + p_k)."""
    return 1.0 / torch.log(c + class_distr)
