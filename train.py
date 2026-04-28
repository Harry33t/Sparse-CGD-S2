"""Train CRC-CGD on MADOS.

Example:
    python -m src.train \
        --data-root /root/autodl-tmp/MADOS \
        --splits    /root/autodl-tmp/MADOS/splits \
        --out-dir   /root/autodl-tmp/mados_proj/logs/run01 \
        --epochs 100 --batch-size 8

Ablation flags:
    --no-crc / --no-cgd  : disable the respective module loss
    --keep-label-ratio 0.5 / 0.25 : sparse-label regime

Auto-shutdown (AutoDL):
    pass --shutdown to ``shutdown -h +2`` after the run completes.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import MADOSMultiRes, CLASS_DISTR, IGNORE_INDEX
from .losses import CRCLoss, CGDLoss, WeightedCEDiceLoss, gen_class_weights
from .models import CRCCGDNet, BASELINES
from .utils import ConfusionMatrix


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--lambda-crc", type=float, default=0.5)
    ap.add_argument("--lambda-cgd", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.1)
    ap.add_argument("--no-crc", action="store_true")
    ap.add_argument("--no-cgd", action="store_true")
    ap.add_argument("--keep-label-ratio", type=float, default=1.0)
    ap.add_argument("--model", default="crc_cgd",
                    choices=["crc_cgd", "unet", "deeplabv3p", "segformer"],
                    help="crc_cgd = ours, others = single-grid baselines")
    ap.add_argument("--backbone", default="mit_b0")
    ap.add_argument("--stem-ch", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-bands", default="10,20,60",
                    help="comma-separated subset of {10,20,60}; '10' forces single-stem mode.")
    ap.add_argument("--supervision-mode", default="high",
                    choices=["high", "high_mod", "all"],
                    help="Which confidence tiers feed the supervised loss: "
                         "'high' (default) = H only; 'high_mod' = H+M; 'all' = H+M+L.")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--shutdown", action="store_true",
                    help="shutdown -h +2 after training (for AutoDL auto-off)")
    return ap.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace):
    train_set = MADOSMultiRes(
        root=args.data_root, splits_dir=args.splits, mode="train",
        augment=True, keep_label_ratio=args.keep_label_ratio, preload=True,
    )
    val_set = MADOSMultiRes(
        root=args.data_root, splits_dir=args.splits, mode="val",
        augment=False, keep_label_ratio=1.0, preload=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def cosine_lr(base: float, cur_iter: int, total_iters: int, warmup_iters: int) -> float:
    if cur_iter < warmup_iters:
        return base * cur_iter / max(1, warmup_iters)
    t = (cur_iter - warmup_iters) / max(1, total_iters - warmup_iters)
    return 0.5 * base * (1.0 + float(np.cos(np.pi * t)))


def apply_confidence_ignore(y: torch.Tensor, conf: torch.Tensor,
                            mode: str = "high",
                            ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Which confidence tiers contribute to the supervised loss.

    MADOS conf codes: 1 = High, 2 = Moderate, 3 = Low, others = NA/unlabeled.

    mode:
      - "high"     : H only (clean anchors, default for all prior runs).
      - "high_mod" : H + M.
      - "all"      : H + M + L (every confidence tier).
    """
    if mode == "high":
        keep = conf == 1
    elif mode == "high_mod":
        keep = (conf == 1) | (conf == 2)
    elif mode == "all":
        keep = (conf >= 1) & (conf <= 3)
    else:
        raise ValueError(f"unknown supervision mode: {mode!r}")
    return torch.where(keep, y, torch.full_like(y, ignore_index))


# ---------------------------------------------------------------------------
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             num_classes: int) -> Dict[str, float]:
    model.eval()
    cm = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for batch in loader:
            x10 = batch["x10"].to(device, non_blocking=True)
            x20 = batch["x20"].to(device, non_blocking=True)
            x60 = batch["x60"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            out = model(x10, x20, x60)
            pred = out["logits"].argmax(dim=1)
            cm.update(pred, y)
    model.train()
    return {
        "miou": cm.miou(),
        "macro_f1": cm.macro_f1(),
        "lcr": cm.lcr(),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}  cuda_name={torch.cuda.get_device_name(0) if device.type == 'cuda' else None}")

    train_loader, val_loader = build_loaders(args)
    num_classes = 15

    if args.model == "crc_cgd":
        use_bands = tuple(b.strip() for b in args.use_bands.split(",") if b.strip())
        model = CRCCGDNet(
            num_classes=num_classes,
            stem_ch=args.stem_ch,
            backbone=args.backbone,
            head_dim=args.head_dim,
            use_bands=use_bands,
        )
        # With only one active branch, CRC has no peers to compare against.
        if len(use_bands) < 2:
            args.no_crc = True
    else:
        model = BASELINES[args.model](num_classes=num_classes)
        # Baselines do not carry a meaningful emb / branch -> disable aux losses.
        args.no_crc = True
        args.no_cgd = True
    model = model.to(device)

    sup_loss = WeightedCEDiceLoss(
        class_weights=gen_class_weights(CLASS_DISTR),
        ignore_index=IGNORE_INDEX,
    ).to(device)
    crc_loss = CRCLoss().to(device)
    cgd_loss = CGDLoss(
        feat_dim=args.head_dim, num_classes=num_classes,
        margin=args.margin, ema=args.ema,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    writer = SummaryWriter(out_dir / "tb")

    total_iters = args.epochs * len(train_loader)
    warmup_iters = args.warmup_epochs * len(train_loader)
    it = 0
    best_miou = -1.0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            x10 = batch["x10"].to(device, non_blocking=True)
            x20 = batch["x20"].to(device, non_blocking=True)
            x60 = batch["x60"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            conf = batch["conf"].to(device, non_blocking=True)

            lr_now = cosine_lr(args.lr, it, total_iters, warmup_iters)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            amp_ctx = autocast(device_type="cuda", enabled=args.amp) if device.type == "cuda" else nullcontext()
            with amp_ctx:
                out = model(x10, x20, x60)
                y_hi = apply_confidence_ignore(y, conf, mode=args.supervision_mode)
                l_sup = sup_loss(out["logits"], y_hi)
                l_crc = crc_loss(out["branch"], conf) if not args.no_crc else out["logits"].new_zeros(())
                l_cgd = cgd_loss(out["emb"], y, conf) if not args.no_cgd else out["logits"].new_zeros(())
                loss = l_sup + args.lambda_crc * l_crc + args.lambda_cgd * l_cgd

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            if it % 20 == 0:
                writer.add_scalar("train/loss", loss.item(), it)
                writer.add_scalar("train/l_sup", l_sup.item(), it)
                writer.add_scalar("train/l_crc", float(l_crc), it)
                writer.add_scalar("train/l_cgd", float(l_cgd), it)
                writer.add_scalar("train/lr", lr_now, it)
            it += 1

        # ---- end of epoch: validate -----------------------------------
        metrics = validate(model, val_loader, device, num_classes)
        writer.add_scalar("val/miou", metrics["miou"], epoch)
        writer.add_scalar("val/macro_f1", metrics["macro_f1"], epoch)
        writer.add_scalar("val/lcr", metrics["lcr"], epoch)
        elapsed = time.time() - t0
        print(f"[epoch {epoch:03d}] loss={loss.item():.4f} "
              f"miou={metrics['miou']:.4f} mF1={metrics['macro_f1']:.4f} "
              f"lcr={metrics['lcr']:.4f} lr={lr_now:.2e} elapsed={elapsed/60:.1f}min")

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save(
                {"model": model.state_dict(),
                 "epoch": epoch,
                 "metrics": metrics,
                 "args": vars(args)},
                out_dir / "best.pt",
            )
        torch.save(
            {"model": model.state_dict(),
             "epoch": epoch,
             "metrics": metrics,
             "args": vars(args)},
            out_dir / "last.pt",
        )

    (out_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    writer.close()
    print(f"[done] best_val_miou={best_miou:.4f}  total={time.time()-t0:.0f}s  out={out_dir}")

    if args.shutdown:
        print("[shutdown] issuing `shutdown -h +2` ...")
        try:
            subprocess.run(["/usr/sbin/shutdown", "-h", "+2"], check=False)
        except Exception as e:
            print(f"[shutdown] failed: {e}")


if __name__ == "__main__":
    main()
