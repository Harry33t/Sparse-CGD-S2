"""Quick sanity check for MADOSMultiRes without needing the Zenodo splits.

Synthesizes a tiny train_X.txt from the first few crops found under the MADOS
root and loads one batch. Shapes and dtypes are printed to stdout.

Run:
    python -m src.data._smoketest --root /path/to/MADOS
"""
from __future__ import annotations

import argparse
import os
import tempfile
from glob import glob

from torch.utils.data import DataLoader

from .mados_multires import MADOSMultiRes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=6, help="how many crops to include")
    args = ap.parse_args()

    crop_names = []
    for scene_dir in sorted(glob(os.path.join(args.root, "Scene_*"))):
        if len(crop_names) >= args.n:
            break
        base = os.path.basename(scene_dir)
        for cl_path in sorted(glob(os.path.join(scene_dir, "10", "*_cl_*.tif"))):
            if len(crop_names) >= args.n:
                break
            idx = os.path.basename(cl_path).split("_cl_")[-1].split(".tif")[0]
            crop_names.append(f"{base}_{idx}")

    print(f"Using {len(crop_names)} synthesized crops: {crop_names[:3]} ...")

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "train_X.txt"), "w") as f:
            for name in crop_names:
                f.write(name + "\n")

        ds = MADOSMultiRes(root=args.root, splits_dir=tmp, mode="train",
                           augment=True, keep_label_ratio=0.5)

        print(f"Dataset size: {len(ds)}")
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        for k, v in batch.items():
            if hasattr(v, "shape"):
                print(f"  {k:5s} shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"  {k:5s} = {v}")

        uniq = sorted(set(batch["y"].flatten().tolist()))
        print(f"Unique label values in batch: {uniq[:20]}")


if __name__ == "__main__":
    main()
