"""Generate a qualitative comparison panel for the paper.

For each of N selected test crops, produce one row:
    RGB | Ground truth | DeepLabV3+ prediction | CRC-CGD (ours) prediction

Saves `paper/figs/qualitative_<name>.png` per crop and a combined
`paper/figs/qualitative.pdf` grid.

Usage on the AutoDL server:
    python -m src.qualitative \\
        --data-root /root/autodl-tmp/MADOS \\
        --splits    /root/autodl-tmp/MADOS/splits \\
        --ckpt-base /root/autodl-tmp/mados_proj/logs/b_deeplabv3p/best.pt \\
        --ckpt-ours /root/autodl-tmp/mados_proj/logs/full/best.pt \\
        --out-dir   /root/autodl-tmp/mados_proj/qualitative \\
        --n 4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from .data import MADOSMultiRes, MEAN_10, STD_10, IGNORE_INDEX
from .models import CRCCGDNet, BASELINES


CLASS_NAMES = [
    "Marine Debris", "Dense Sargassum", "Sparse Algae", "Natural Organic",
    "Ship", "Oil Spill", "Marine Water", "Sediment", "Foam", "Turbid",
    "Shallow", "Waves", "Oil Platform", "Jellyfish", "Sea snot",
]
# 0-indexed class ids of interest
MD, OS = 0, 5

# Color palette matching the MADOS assets.py mapping (approximate RGB hex).
CLASS_COLORS = [
    "#ff0000",  # Marine Debris - red
    "#008000",  # Dense Sargassum - green
    "#32cd32",  # Sparse Algae - limegreen
    "#a52a2a",  # Natural Organic - brown
    "#ffa500",  # Ship - orange
    "#d8bfd8",  # Oil Spill - thistle
    "#000080",  # Marine Water - navy
    "#ffd700",  # Sediment - gold
    "#800080",  # Foam - purple
    "#bdb76b",  # Turbid - darkkhaki
    "#00ced1",  # Shallow - darkturquoise
    "#ffe4c4",  # Waves - bisque
    "#696969",  # Oil Platform - dimgrey
    "#ff69b4",  # Jellyfish - hotpink
    "#ffff00",  # Sea snot - yellow
]
CMAP = ListedColormap(CLASS_COLORS)
IGNORE_RGB = (0, 0, 0)   # black for ignore / unlabeled


def denorm_rgb(x10: torch.Tensor) -> np.ndarray:
    """Build an [H,W,3] 0-1 RGB from the 10 m tensor.

    MADOS 10 m band order: 492, 559, 665, 833 (B2, B3, B4, B8).
    RGB uses B4, B3, B2.
    """
    arr = x10.cpu().numpy() * STD_10 + MEAN_10          # [4,H,W]
    r = arr[2]; g = arr[1]; b = arr[0]
    rgb = np.stack([r, g, b], axis=-1)
    # per-channel 2-98 percentile stretch
    lo = np.nanpercentile(rgb, 2,  axis=(0, 1), keepdims=True)
    hi = np.nanpercentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return rgb


def colorize(y: np.ndarray, rgb_bg: np.ndarray | None = None) -> np.ndarray:
    """[H,W] int64 → [H,W,3] float in [0,1] with CLASS_COLORS.
    If ``rgb_bg`` is given (float 0-1 [H,W,3]), ignore pixels show that RGB
    darkened; otherwise ignore pixels are mid-grey."""
    H, W = y.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    if rgb_bg is not None:
        out[...] = 0.6 * rgb_bg            # darken RGB so overlay pops
    else:
        out[...] = 0.5
    for k, hexc in enumerate(CLASS_COLORS):
        m = y == k
        if not m.any():
            continue
        rgb = np.array([int(hexc[i:i+2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0
        out[m] = rgb
    return np.clip(out, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--ckpt-base", required=True,
                    help="Baseline checkpoint (e.g. b_deeplabv3p/best.pt)")
    ap.add_argument("--ckpt-ours", required=True,
                    help="Our CRC-CGD checkpoint (e.g. full/best.pt)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=4,
                    help="How many scenes to include in the figure")
    ap.add_argument("--min-pollutant", type=int, default=40,
                    help="Require at least this many High-confidence pollutant pixels")
    return ap.parse_args()


def _build_model(state, device):
    args = state.get("args", {})
    name = args.get("model", "crc_cgd")
    if name == "crc_cgd":
        use_bands_raw = args.get("use_bands", "10,20,60")
        use_bands = tuple(b.strip() for b in str(use_bands_raw).split(",") if b.strip())
        m = CRCCGDNet(
            num_classes=15,
            stem_ch=args.get("stem_ch", 32),
            backbone=args.get("backbone", "mit_b0"),
            head_dim=args.get("head_dim", 256),
            use_bands=use_bands,
        )
    else:
        m = BASELINES[name](num_classes=15)
    m.load_state_dict(state["model"])
    return m.to(device).eval()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_b = torch.load(args.ckpt_base, map_location=device, weights_only=False)
    state_o = torch.load(args.ckpt_ours, map_location=device, weights_only=False)
    model_b = _build_model(state_b, device)
    model_o = _build_model(state_o, device)

    ds = MADOSMultiRes(args.data_root, args.splits, mode="test",
                       augment=False, keep_label_ratio=1.0, preload=True)

    # Rank crops by pollutant pixel count (High confidence MD + OS), but
    # diversify by keeping at most one crop per Scene to avoid dominance.
    scores: list[tuple[int, int]] = []
    for i in range(len(ds)):
        item = ds[i]
        conf = item["conf"].numpy()
        y = item["y"].numpy()
        mask = (conf == 1) & ((y == MD) | (y == OS))
        scores.append((int(mask.sum()), i))
    scores.sort(reverse=True)

    picks: list[int] = []
    used_scenes: set[str] = set()
    for cnt, i in scores:
        if cnt < args.min_pollutant:
            break
        scene = ds._index[i]["crop_name"].rsplit("_", 1)[0]
        if scene in used_scenes:
            continue
        used_scenes.add(scene)
        picks.append(i)
        if len(picks) >= args.n:
            break
    if len(picks) < args.n:
        print(f"[warn] only {len(picks)} distinct scenes meet --min-pollutant, "
              f"padding with top-remaining.")
        for cnt, i in scores:
            if i not in picks:
                picks.append(i)
            if len(picks) >= args.n:
                break

    # Single-column vertical grid: rows = {RGB, GT, DeepLabV3+, Ours},
    # columns = scenes. This fits within one IEEE column and avoids the
    # page-wide figure* that wastes half a page below.
    N = len(picks)
    row_titles = ["RGB", "Ground truth", "DeepLabV3+", "CGD (ours)"]
    panel_in = 1.05
    fig, axes = plt.subplots(4, N, figsize=(panel_in * N + 0.3, panel_in * 4 + 0.2),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.06})
    if N == 1:
        axes = axes[:, None]

    # Row labels on the left, horizontal for readability
    for i, t in enumerate(row_titles):
        axes[i, 0].set_ylabel(t, fontsize=9, rotation=0, labelpad=32,
                              ha="right", va="center")

    rows_data = [[None] * N for _ in range(4)]
    scene_names: list[str] = []
    with torch.no_grad():
        for col, idx in enumerate(picks):
            s = ds[idx]
            scene_names.append(s["name"])
            x10 = s["x10"].unsqueeze(0).to(device)
            x20 = s["x20"].unsqueeze(0).to(device)
            x60 = s["x60"].unsqueeze(0).to(device)
            rgb = denorm_rgb(s["x10"])
            gt = s["y"].numpy()
            gt_color = colorize(np.where(gt == IGNORE_INDEX, -1, gt), rgb_bg=rgb)
            pred_b = model_b(x10, x20, x60)["logits"].argmax(1)[0].cpu().numpy()
            pred_o = model_o(x10, x20, x60)["logits"].argmax(1)[0].cpu().numpy()
            rows_data[0][col] = rgb
            rows_data[1][col] = gt_color
            rows_data[2][col] = colorize(pred_b)
            rows_data[3][col] = colorize(pred_o)

    for i in range(4):
        for j in range(N):
            ax = axes[i, j]
            ax.imshow(rows_data[i][j])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    # Scene names as column titles above the top row.
    for j, name in enumerate(scene_names):
        axes[0, j].set_title(name, fontsize=7)

    # Compact multi-column legend below.
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k])
               for k in range(15)]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=6.0, frameon=False, handlelength=1.0,
               columnspacing=0.8, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.14, right=0.995, top=0.94, bottom=0.20)
    out_pdf = out / "qualitative.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200, pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print("picks:", [ds._index[i]["crop_name"] for i in picks])


if __name__ == "__main__":
    main()
