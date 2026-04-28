"""Evaluate a trained checkpoint on the MADOS test split.

Writes:
    <out-dir>/test_metrics.json   mIoU / macro-F1 / per-class F1 / LCR
    <out-dir>/confusion.npy       confusion matrix [K, K]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import MADOSMultiRes, IGNORE_INDEX
from .models import CRCCGDNet, BASELINES
from .utils import ConfusionMatrix

MD, OS = 0, 5  # pollutant class indices in MADOS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--save-probs", action="store_true",
                    help="Save flat per-pixel GT / pred / p_MD / p_OS arrays to probs.npz "
                         "(valid pixels only, for C-lite threshold/PR analysis).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = state.get("args", {})

    model_name = train_args.get("model", "crc_cgd")
    use_bands_raw = train_args.get("use_bands", "")
    if model_name == "crc_cgd" or (use_bands_raw and use_bands_raw != "10,20,60"):
        use_bands = tuple(b.strip() for b in str(use_bands_raw).split(",") if b.strip())
        model = CRCCGDNet(
            num_classes=15,
            stem_ch=train_args.get("stem_ch", 32),
            backbone=train_args.get("backbone", "mit_b0"),
            head_dim=train_args.get("head_dim", 256),
            use_bands=use_bands,
        )
    else:
        model = BASELINES[model_name](num_classes=15)
    model = model.to(device)
    model.load_state_dict(state["model"])
    model.eval()

    ds = MADOSMultiRes(args.data_root, args.splits, mode=args.split,
                       augment=False, keep_label_ratio=1.0, preload=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    cm = ConfusionMatrix(num_classes=15)
    gt_list, pred_list, pmd_list, pos_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x10 = batch["x10"].to(device)
            x20 = batch["x20"].to(device)
            x60 = batch["x60"].to(device)
            y = batch["y"].to(device)
            out = model(x10, x20, x60)
            logits = out["logits"]
            pred = logits.argmax(dim=1)
            cm.update(pred, y)

            if args.save_probs:
                prob = F.softmax(logits, dim=1)      # [B, K, H, W]
                y_flat = y.reshape(-1)
                valid = y_flat != IGNORE_INDEX
                gt_list.append(y_flat[valid].cpu().numpy().astype(np.uint8))
                pred_list.append(pred.reshape(-1)[valid].cpu().numpy().astype(np.uint8))
                pmd_list.append(prob[:, MD].reshape(-1)[valid].cpu().numpy().astype(np.float16))
                pos_list.append(prob[:, OS].reshape(-1)[valid].cpu().numpy().astype(np.float16))

    metrics = cm.summary()
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out_dir / "confusion.npy", cm.cm)
    print(json.dumps(metrics, indent=2))

    if args.save_probs:
        np.savez_compressed(
            out_dir / "probs.npz",
            gt=np.concatenate(gt_list),
            pred=np.concatenate(pred_list),
            p_md=np.concatenate(pmd_list),
            p_os=np.concatenate(pos_list),
        )
        print(f"[save-probs] wrote {out_dir / 'probs.npz'}")


if __name__ == "__main__":
    main()
