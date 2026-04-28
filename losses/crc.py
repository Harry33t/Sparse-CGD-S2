"""Cross-Resolution Consistency (CRC) loss.

The three resolution branches each emit a full-size class posterior. On pixels
that do *not* carry a High-confidence annotation we require the three branches
to agree. Agreement is measured by the mean per-pixel KL to the branch-mean
posterior, which is symmetric in the three branches and avoids picking one of
them as the "teacher".
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRCLoss(nn.Module):
    """KL-to-mean consistency over 3 branches on non-High-confidence pixels.

    Parameters
    ----------
    conf_high : int
        Value in the confidence map that represents the High level. Everything
        else (Moderate, Low, unlabeled) is eligible for the consistency term.
    eps : float
        Numerical floor for log / softmax stability.
    """

    def __init__(self, conf_high: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.conf_high = int(conf_high)
        self.eps = eps

    def forward(self, branch_logits: Dict[str, torch.Tensor],
                conf: torch.Tensor) -> torch.Tensor:
        # Only enforce consistency across branches that are actually present.
        # Fewer than 2 branches -> consistency is vacuous.
        keys = sorted(branch_logits.keys())
        if len(keys) < 2:
            return next(iter(branch_logits.values())).new_zeros(())

        # [R, B, K, H, W] where R = number of active branches
        stacked = torch.stack([branch_logits[k] for k in keys], dim=0)
        p = F.softmax(stacked, dim=2)
        p_mean = p.mean(dim=0).clamp_min(self.eps)                      # [B, K, H, W]
        log_p = torch.log(p.clamp_min(self.eps))
        log_p_mean = torch.log(p_mean)                                  # [B, K, H, W]

        # KL(p_r || p_mean) per pixel, per branch: sum_k p_r * (log p_r - log p_mean)
        kl_per_branch = (p * (log_p - log_p_mean.unsqueeze(0))).sum(dim=2)   # [3, B, H, W]
        kl = kl_per_branch.mean(dim=0)                                    # [B, H, W]

        # Mask out High-confidence pixels (where strong supervision already applies).
        mask = (conf != self.conf_high).to(kl.dtype)                     # [B, H, W]
        denom = mask.sum().clamp_min(1.0)
        return (kl * mask).sum() / denom
