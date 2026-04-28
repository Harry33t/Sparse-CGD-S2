"""Fig 2: Examples of MADOS scenes, labels, confidence masks.

Picks 3 representative test-split crops:
  row 1  = maximum High-confidence Oil Spill footprint
  row 2  = maximum High-confidence Marine Debris footprint
  row 3  = maximum look-alike (Foam/Turbid/Sediment/Waves) footprint

For each row, plots 4 columns:
  RGB | Ground-truth overlay | Confidence map (H/M/L) | Pollutant+look-alike only

Usage on AutoDL:
    python -m src.scene_examples \\
        --data-root /root/autodl-tmp/MADOS \\
        --splits    /root/autodl-tmp/MADOS/splits \\
        --out-dir   /root/autodl-tmp/mados_proj/scene_examples
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

from .data import MADOSMultiRes, MEAN_10, STD_10, IGNORE_INDEX


# Same class palette as qualitative.py for consistency
CLASS_NAMES = [
    "Marine Debris", "Dense Sargassum", "Sparse Algae", "Natural Organic",
    "Ship", "Oil Spill", "Marine Water", "Sediment", "Foam", "Turbid",
    "Shallow", "Waves", "Oil Platform", "Jellyfish", "Sea snot",
]
CLASS_COLORS = [
    "#ff0000", "#008000", "#32cd32", "#a52a2a", "#ffa500",
    "#d8bfd8", "#000080", "#ffd700", "#800080", "#bdb76b",
    "#00ced1", "#ffe4c4", "#696969", "#ff69b4", "#ffff00",
]
MD, OS = 0, 5
SEDIMENT, FOAM, TURBID, WAVES = 7, 8, 9, 11
LOOKALIKE = {FOAM, TURBID, SEDIMENT, WAVES}
POLLUTANT = {MD, OS}
HIGHLIGHT = sorted(POLLUTANT | LOOKALIKE)   # classes shown in column 4


def denorm_rgb(x10):
    arr = x10.cpu().numpy() * STD_10 + MEAN_10
    r, g, b = arr[2], arr[1], arr[0]
    rgb = np.stack([r, g, b], axis=-1)
    lo = np.nanpercentile(rgb, 2,  axis=(0, 1), keepdims=True)
    hi = np.nanpercentile(rgb, 98, axis=(0, 1), keepdims=True)
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def colorize(y, rgb_bg=None):
    H, W = y.shape
    out = 0.6 * rgb_bg if rgb_bg is not None else np.full((H, W, 3), 0.5, dtype=np.float32)
    out = out.astype(np.float32).copy()
    for k, hexc in enumerate(CLASS_COLORS):
        m = y == k
        if not m.any():
            continue
        rgb = np.array([int(hexc[i:i+2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0
        out[m] = rgb
    return np.clip(out, 0, 1)


def colorize_highlight(y, rgb_bg):
    """Same as colorize() but only shows pollutant and look-alike classes;
    everything else is the faded RGB background."""
    H, W = y.shape
    out = (0.5 * rgb_bg).astype(np.float32).copy()
    for k in HIGHLIGHT:
        m = y == k
        if not m.any():
            continue
        rgb = np.array([int(CLASS_COLORS[k][i:i+2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0
        out[m] = rgb
    return np.clip(out, 0, 1)


def conf_map(conf):
    """Confidence: 1=H, 2=M, 3=L. Others = unlabeled.
    Render as 3-tier red/orange/yellow over grey."""
    H, W = conf.shape
    out = np.full((H, W, 3), 0.85, dtype=np.float32)      # grey = unlabeled
    out[conf == 1] = np.array([0.85, 0.10, 0.10])          # High = red
    out[conf == 2] = np.array([0.95, 0.55, 0.10])          # Moderate = orange
    out[conf == 3] = np.array([0.95, 0.90, 0.25])          # Low = yellow
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def pick_crops(ds):
    """Return 3 distinct indices: best OS, best MD, best look-alike."""
    os_scores, md_scores, la_scores = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        y = s["y"].numpy()
        conf = s["conf"].numpy()
        hi = conf == 1
        os_ct = int((hi & (y == OS)).sum())
        md_ct = int((hi & (y == MD)).sum())
        la_ct = sum(int((y == k).sum()) for k in LOOKALIKE)
        os_scores.append((os_ct, i))
        md_scores.append((md_ct, i))
        la_scores.append((la_ct, i))
    os_scores.sort(reverse=True)
    md_scores.sort(reverse=True)
    la_scores.sort(reverse=True)

    picks = []
    used_scenes = set()

    def add(sorted_scores, label):
        for cnt, i in sorted_scores:
            scene = ds._index[i]["crop_name"].rsplit("_", 1)[0]
            if scene in used_scenes:
                continue
            picks.append((i, label, cnt))
            used_scenes.add(scene)
            return

    add(os_scores, "Oil Spill")
    add(md_scores, "Marine Debris")
    add(la_scores, "Look-alikes")
    return picks


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = MADOSMultiRes(args.data_root, args.splits, mode="test",
                       augment=False, keep_label_ratio=1.0, preload=True)

    picks = pick_crops(ds)
    print("picks:", [(ds._index[i]["crop_name"], lbl, cnt) for i, lbl, cnt in picks])

    N = len(picks)
    panel_in = 1.4
    fig, axes = plt.subplots(N, 4, figsize=(panel_in * 4 + 0.5, panel_in * N + 0.3),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    col_titles = ["RGB", "Ground truth", "Confidence map", "Pollutant + look-alikes"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=9)

    for row, (idx, lbl, cnt) in enumerate(picks):
        s = ds[idx]
        rgb  = denorm_rgb(s["x10"])
        y    = s["y"].numpy()
        conf = s["conf"].numpy()
        gt_color = colorize(np.where(y == IGNORE_INDEX, -1, y), rgb_bg=rgb)
        cm_color = conf_map(conf)
        hl_color = colorize_highlight(np.where(y == IGNORE_INDEX, -1, y), rgb_bg=rgb)

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(gt_color)
        axes[row, 2].imshow(cm_color)
        axes[row, 3].imshow(hl_color)

        scene_name = ds._index[idx]["crop_name"]
        axes[row, 0].set_ylabel(f"{lbl}\n({scene_name})", fontsize=8, rotation=0,
                                labelpad=38, ha="right", va="center")

        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

    # Confidence legend
    conf_handles = [
        mpatches.Patch(color=(0.85, 0.10, 0.10), label="High"),
        mpatches.Patch(color=(0.95, 0.55, 0.10), label="Moderate"),
        mpatches.Patch(color=(0.95, 0.90, 0.25), label="Low"),
        mpatches.Patch(color=(0.85, 0.85, 0.85), label="Unlabeled"),
    ]
    # Class legend (only the HIGHLIGHT subset + key classes)
    legend_classes = [MD, OS, FOAM, TURBID, SEDIMENT, WAVES]
    class_handles = [mpatches.Patch(color=CLASS_COLORS[k], label=CLASS_NAMES[k])
                     for k in legend_classes]

    fig.legend(handles=class_handles + conf_handles,
               loc="lower center", ncol=5, fontsize=7, frameon=False,
               handlelength=1.2, columnspacing=1.0, handletextpad=0.4,
               bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(left=0.16, right=0.995, top=0.94, bottom=0.10)
    out_pdf = out / "scene_examples.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200, pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
