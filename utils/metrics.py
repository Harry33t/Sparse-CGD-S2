"""Segmentation metrics for MADOS: mIoU, per-class F1, and look-alike confusion."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


# 0-indexed class ids. 1..15 in the raw file minus one.
POLLUTANT_IDS: tuple[int, ...] = (0, 5)              # Marine Debris, Oil Spill
LOOKALIKE_IDS: tuple[int, ...] = (7, 8, 9, 11)       # Sediment, Foam, Turbid, Waves & Wakes


class ConfusionMatrix:
    """Row = ground truth, column = prediction."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.K = int(num_classes)
        self.ignore = int(ignore_index)
        self.cm = np.zeros((self.K, self.K), dtype=np.int64)

    # ------------------------------------------------------------------
    def update(self, pred: torch.Tensor, y: torch.Tensor) -> None:
        mask = y != self.ignore
        p = pred[mask].detach().to("cpu", dtype=torch.int64).numpy()
        t = y[mask].detach().to("cpu", dtype=torch.int64).numpy()
        if p.size == 0:
            return
        idx = t * self.K + p
        bc = np.bincount(idx, minlength=self.K * self.K)
        self.cm += bc.reshape(self.K, self.K)

    # ------------------------------------------------------------------
    def per_class_iou(self) -> np.ndarray:
        tp = np.diag(self.cm).astype(np.float64)
        fp = self.cm.sum(axis=0) - tp
        fn = self.cm.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / np.maximum(denom, 1.0), np.nan)
        return iou

    def per_class_f1(self) -> np.ndarray:
        tp = np.diag(self.cm).astype(np.float64)
        fp = self.cm.sum(axis=0) - tp
        fn = self.cm.sum(axis=1) - tp
        prec = tp / np.maximum(tp + fp, 1.0)
        rec = tp / np.maximum(tp + fn, 1.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), np.nan)
        return f1

    def miou(self) -> float:
        iou = self.per_class_iou()
        return float(np.nanmean(iou))

    def macro_f1(self) -> float:
        f1 = self.per_class_f1()
        return float(np.nanmean(f1))

    # ------------------------------------------------------------------
    def lcr(
        self,
        pollutant_ids: Sequence[int] = POLLUTANT_IDS,
        lookalike_ids: Sequence[int] = LOOKALIKE_IDS,
    ) -> float:
        """Look-alike confusion rate: fraction of pollutant-labelled pixels
        predicted as one of the look-alike classes."""
        total = self.cm[list(pollutant_ids), :].sum()
        if total == 0:
            return float("nan")
        confused = self.cm[np.ix_(list(pollutant_ids), list(lookalike_ids))].sum()
        return float(confused / total)

    # ------------------------------------------------------------------
    def summary(self, class_names: Sequence[str] | None = None) -> dict:
        iou = self.per_class_iou()
        f1 = self.per_class_f1()
        out: dict = {
            "miou": self.miou(),
            "macro_f1": self.macro_f1(),
            "lcr": self.lcr(),
            "per_class_iou": iou.tolist(),
            "per_class_f1": f1.tolist(),
        }
        if class_names is not None:
            out["class_names"] = list(class_names)
        return out
