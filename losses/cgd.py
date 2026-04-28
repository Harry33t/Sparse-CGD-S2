"""Confidence-Guided Disambiguation (CGD) loss.

Maintains an EMA prototype per class in the hard subset. High-confidence pixels
of any hard class act as anchors for a triplet-margin term against the hardest
other hard-class prototype. Moderate- and Low-confidence pixels are softly
pulled toward the argmax-matching prototype, weighted by their confidence.

Both the prototypes and the per-pixel features live in the head's pre-
classifier feature space (``emb``), which is at stride 4 of the input. Labels
and the confidence map are downsampled to that stride with nearest-neighbour.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# 0-indexed MADOS ids for the hard subset (see src/data/mados_multires.py).
DEFAULT_HARD_IDS: tuple[int, ...] = (0, 5, 7, 8, 9, 11)


class CGDLoss(nn.Module):
    """Prototype-margin + soft-pull loss on the hard subset.

    Parameters
    ----------
    feat_dim : int
        Channel dimension of the embedding tensor ``emb`` emitted by the net.
    num_classes : int
        Total number of classes (used to size the prototype bank; only the
        ``hard_ids`` slots are actually updated).
    hard_ids : sequence of int
        0-indexed class ids that take part in the prototype bank.
    margin : float
        Triplet margin ``m`` in the High-confidence term.
    ema : float
        EMA coefficient for prototype updates (higher = faster adaptation).
    w_moderate, w_low : float
        Weights on the soft-pull loss for Moderate- and Low-confidence pixels.
    conf_levels : dict[str, int]
        Confidence map encoding.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hard_ids: Sequence[int] = DEFAULT_HARD_IDS,
        margin: float = 0.5,
        ema: float = 0.1,
        w_moderate: float = 0.5,
        w_low: float = 0.25,
        conf_high: int = 1,
        conf_moderate: int = 2,
        conf_low: int = 3,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hard_ids = tuple(int(x) for x in hard_ids)
        self.margin = margin
        self.ema = ema
        self.w_moderate = w_moderate
        self.w_low = w_low
        self.conf_high = conf_high
        self.conf_moderate = conf_moderate
        self.conf_low = conf_low
        self.ignore_index = ignore_index

        # Prototype bank, L2-normalized. Persistent buffers so they survive
        # checkpoint / EMA resume.
        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.register_buffer("proto_init", torch.zeros(num_classes, dtype=torch.bool))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_prototypes(self, feat_flat: torch.Tensor, y_flat: torch.Tensor,
                           conf_flat: torch.Tensor) -> None:
        hi = conf_flat == self.conf_high
        if not hi.any():
            return
        feat_hi = feat_flat[hi]
        y_hi = y_flat[hi]
        for k in self.hard_ids:
            sel = y_hi == k
            if sel.sum() == 0:
                continue
            centroid = feat_hi[sel].mean(dim=0)
            centroid = F.normalize(centroid, dim=0)
            if not bool(self.proto_init[k]):
                self.prototypes[k] = centroid
                self.proto_init[k] = True
            else:
                new = (1.0 - self.ema) * self.prototypes[k] + self.ema * centroid
                self.prototypes[k] = F.normalize(new, dim=0)

    # ------------------------------------------------------------------
    def forward(self, emb: torch.Tensor, y: torch.Tensor,
                conf: torch.Tensor) -> torch.Tensor:
        """emb: [B, D, h, w]; y, conf: [B, H, W] (any H, W divisible by stride)."""
        B, D, h, w = emb.shape

        # Downsample labels / confidence to the embedding grid.
        y_ds = F.interpolate(y.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
        conf_ds = F.interpolate(conf.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()

        feat = F.normalize(emb, dim=1).permute(0, 2, 3, 1).reshape(-1, D)   # [N, D]
        y_flat = y_ds.reshape(-1)
        conf_flat = conf_ds.reshape(-1)

        # ---- Update prototypes in training mode only -------------------
        if self.training:
            self._update_prototypes(feat.detach(), y_flat, conf_flat)

        # Need at least two initialized prototypes for a triplet.
        init_ids = [k for k in self.hard_ids if bool(self.proto_init[k])]
        if len(init_ids) < 2:
            return feat.new_zeros(())

        protos = self.prototypes[list(init_ids)]                            # [K_h, D]
        dist = torch.cdist(feat, protos)                                    # [N, K_h]

        # ---- High-confidence triplet term ------------------------------
        hi_mask = (conf_flat == self.conf_high) & torch.isin(
            y_flat, torch.tensor(init_ids, device=feat.device)
        )
        if hi_mask.any():
            d_hi = dist[hi_mask]                                            # [M, K_h]
            y_hi = y_flat[hi_mask]
            col = torch.tensor(
                [init_ids.index(int(c)) for c in y_hi.tolist()],
                device=feat.device,
                dtype=torch.long,
            )
            d_pos = d_hi.gather(1, col.unsqueeze(1)).squeeze(1)
            # Mask out the positive column when selecting the hardest negative.
            neg = d_hi.clone()
            neg.scatter_(1, col.unsqueeze(1), float("inf"))
            d_neg, _ = neg.min(dim=1)
            triplet = F.relu(d_pos - d_neg + self.margin).mean()
        else:
            triplet = feat.new_zeros(())

        # ---- Moderate / Low soft-pull term ----------------------------
        soft = feat.new_zeros(())
        for lvl, w in [(self.conf_moderate, self.w_moderate),
                       (self.conf_low,      self.w_low)]:
            m = conf_flat == lvl
            if not m.any():
                continue
            d_m = dist[m]
            # Pull each pixel toward its argmax-similarity prototype (= argmin dist).
            d_min, _ = d_m.min(dim=1)
            soft = soft + w * d_min.pow(2).mean()

        return triplet + soft
