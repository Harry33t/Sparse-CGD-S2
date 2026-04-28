"""Multi-resolution MADOS dataloader.

Unlike the reference ``mados-master/utils/dataset.py`` which nearest-upsamples
every band to 10 m and stacks them, this loader keeps the three native
resolutions in separate tensors so the model can process them as a 10/20/60 m
stack, which is the setup our paper needs.

Returned sample (dict):
    x10  : FloatTensor [4, H,   W  ]  10 m bands (492, 559, 665, 833 nm)
    x20  : FloatTensor [6, H/2, W/2]  20 m bands (704, 739, 780, 864, 1610, 2186 nm)
    x60  : FloatTensor [1, H/6, W/6]  60 m band  (442 nm)
    y    : LongTensor  [H, W]         class label in [0, K-1]; 255 = ignore
    conf : ByteTensor  [H, W]         0=NA, 1=High, 2=Moderate, 3=Low
    name : str                        crop identifier (for logging / eval)
"""
from __future__ import annotations

import os
from glob import glob
from typing import Dict, List, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Band grouping (fixed by MADOS file layout under <Scene>/{10,20,60}/)
# ---------------------------------------------------------------------------
BANDS_10 = [492, 559, 665, 833]               # B2 B3 B4 B8
BANDS_20 = [704, 739, 780, 864, 1610, 2186]   # B5 B6 B7 B8A B11 B12
BANDS_60 = [442]                              # B1

# Per-resolution mean / std, derived from the original MADOS stats which were
# computed post-upsampling; the same physical reflectance so the per-band
# numbers carry over. Original ordering was sorted by wavelength number:
#   [442, 492, 559, 665, 704, 739, 780, 833, 864, 1610, 2186]
_BANDS_MEAN = np.array([0.0582676,  0.05223386, 0.04381474, 0.0357083,  0.03412902,
                        0.03680401, 0.03999107, 0.03566642, 0.03965081, 0.0267993,
                        0.01978944], dtype=np.float32)
_BANDS_STD  = np.array([0.03240627, 0.03432253, 0.0354812,  0.0375769,  0.03785412,
                        0.04992323, 0.05884482, 0.05545856, 0.06423746, 0.04211187,
                        0.03019115], dtype=np.float32)
_WL_ORDER = [442, 492, 559, 665, 704, 739, 780, 833, 864, 1610, 2186]


def _stats_for(bands: Sequence[int]):
    idx = [_WL_ORDER.index(b) for b in bands]
    return _BANDS_MEAN[idx].reshape(-1, 1, 1), _BANDS_STD[idx].reshape(-1, 1, 1)


MEAN_10, STD_10 = _stats_for(BANDS_10)
MEAN_20, STD_20 = _stats_for(BANDS_20)
MEAN_60, STD_60 = _stats_for(BANDS_60)

# Pixel-level class distribution from the MADOS reference loader.
# Useful for weighted CE (applied on High-confidence pixels only in our setup).
CLASS_DISTR = torch.tensor([
    0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452,
    0.34775, 0.20638, 0.00062, 0.1169,  0.09188, 0.01309,
    0.00917, 0.00176, 0.00963,
])

# 0-indexed class ids for the hard subset used by CGD.
HARD_CLASS_IDS = {
    "marine_debris": 0,        # "Marine Debris" = 1 in MADOS -> 0 after y-1
    "oil_spill": 5,            # "Oil Spill" = 6
    "sediment": 7,             # "Sediment-Laden Water" = 8
    "foam": 8,                 # "Foam" = 9
    "turbid": 9,               # "Turbid Water" = 10
    "waves_wakes": 11,         # "Waves & Wakes" = 12
}

IGNORE_INDEX = 255


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_band(path: str) -> np.ndarray:
    """Read a single-band GeoTIFF at its native resolution."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _read_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int64)


def _impute_nan(arr: np.ndarray, mean_per_band: np.ndarray) -> np.ndarray:
    """Fill NaNs band-wise with the provided mean vector."""
    if not np.isnan(arr).any():
        return arr
    out = arr.copy()
    c = out.shape[0]
    for k in range(c):
        m = np.isnan(out[k])
        if m.any():
            out[k][m] = float(mean_per_band[k])
    return out


def _stack_bands(scene_dir: str, res_dir: str, crop_idx: str,
                 expected_n: int) -> np.ndarray:
    """Build a [C, h, w] array from whatever rhorc bands the scene has at this
    resolution. Wavelength numbers differ slightly across scenes (e.g. 559 vs
    560 nm, 442 vs 443 nm) because ACOLITE reports effective band centres, but
    the *positional* order after sorting by wavelength is stable (B2 B3 B4 B8
    at 10 m, B5 B6 B7 B8A B11 B12 at 20 m, B1 at 60 m)."""
    base = os.path.basename(scene_dir)
    paths = glob(os.path.join(scene_dir, res_dir,
                              f"{base}_L2R_rhorc_*_{crop_idx}.tif"))
    paths.sort(key=lambda p: int(os.path.basename(p).split("_")[-2]))
    if len(paths) != expected_n:
        raise RuntimeError(
            f"expected {expected_n} rhorc bands in {scene_dir}/{res_dir} "
            f"for crop {crop_idx}, got {len(paths)}"
        )
    return np.stack([_read_band(p) for p in paths], axis=0)


def _apply_flip_rot(img: np.ndarray, flip_code: int, rot_k: int) -> np.ndarray:
    """img: [C, H, W]. Apply the same flip/rotation to every channel."""
    if flip_code == 0:
        img = np.flip(img, axis=1)
    elif flip_code == 1:
        img = np.flip(img, axis=2)
    elif flip_code == -1:
        img = np.flip(np.flip(img, axis=1), axis=2)
    if rot_k:
        img = np.rot90(img, rot_k, axes=(1, 2))
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MADOSMultiRes(Dataset):
    """Three-branch MADOS dataset.

    Parameters
    ----------
    root : str
        Path to the MADOS root (the directory containing ``Scene_*``).
    splits_dir : str
        Directory containing ``train_X.txt`` / ``val_X.txt`` / ``test_X.txt``
        from the official Zenodo release.
    mode : {"train", "val", "test"}
    augment : bool
        Random flips + 90 deg rotations. Default True in train, False otherwise.
    keep_label_ratio : float
        Sparse-label ablation. Randomly drops this fraction of **High-confidence**
        pixels to ``IGNORE_INDEX`` while leaving the confidence map intact so
        that CRC/CGD can still see them. 1.0 = keep all.
    """

    def __init__(
        self,
        root: str,
        splits_dir: str,
        mode: str = "train",
        augment: bool | None = None,
        keep_label_ratio: float = 1.0,
        preload: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("train", "val", "test"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.root = root
        self.augment = (mode == "train") if augment is None else augment
        self.keep_label_ratio = float(keep_label_ratio)
        self.preload = preload

        split_file = os.path.join(splits_dir, f"{mode}_X.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                "Download the MADOS splits from Zenodo and pass splits_dir."
            )
        self.crop_names: List[str] = np.genfromtxt(split_file, dtype="str").tolist()

        # Enumerate scenes under root and map crop_name -> (scene_dir, crop_idx).
        self._index: List[Dict] = []
        scene_dirs = sorted(glob(os.path.join(root, "*")))
        for scene_dir in scene_dirs:
            if not os.path.isdir(scene_dir):
                continue
            base = os.path.basename(scene_dir)
            for cl_path in glob(os.path.join(scene_dir, "10", "*_cl_*.tif")):
                crop_idx = os.path.basename(cl_path).split("_cl_")[-1].split(".tif")[0]
                crop_name = f"{base}_{crop_idx}"
                if crop_name in self.crop_names:
                    self._index.append({
                        "scene_dir": scene_dir,
                        "crop_idx": crop_idx,
                        "crop_name": crop_name,
                    })

        if not self._index:
            raise RuntimeError(
                f"No crops matched from split {split_file} under {root}."
            )

        self._cache: Dict[str, Dict] = {}
        if self.preload:
            for entry in tqdm(self._index, desc=f"Preload MADOS {mode}"):
                self._cache[entry["crop_name"]] = self._load_raw(entry)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def _load_raw(self, entry: Dict) -> Dict:
        scene_dir = entry["scene_dir"]
        crop_idx = entry["crop_idx"]
        base = os.path.basename(scene_dir)

        x10 = _stack_bands(scene_dir, "10", crop_idx, expected_n=len(BANDS_10))
        x20 = _stack_bands(scene_dir, "20", crop_idx, expected_n=len(BANDS_20))
        x60 = _stack_bands(scene_dir, "60", crop_idx, expected_n=len(BANDS_60))

        cl_path   = os.path.join(scene_dir, "10", f"{base}_L2R_cl_{crop_idx}.tif")
        conf_path = os.path.join(scene_dir, "10", f"{base}_L2R_conf_{crop_idx}.tif")
        y = _read_mask(cl_path)          # values in {0, 1..15}
        conf = _read_mask(conf_path)     # values in {0, 1(H), 2(M), 3(L)}

        return {
            "x10": x10, "x20": x20, "x60": x60,
            "y": y, "conf": conf,
        }

    # ------------------------------------------------------------------
    def __getitem__(self, i: int) -> Dict:
        entry = self._index[i]
        raw = self._cache.get(entry["crop_name"]) or self._load_raw(entry)

        x10 = _impute_nan(raw["x10"], _BANDS_MEAN[[_WL_ORDER.index(b) for b in BANDS_10]])
        x20 = _impute_nan(raw["x20"], _BANDS_MEAN[[_WL_ORDER.index(b) for b in BANDS_20]])
        x60 = _impute_nan(raw["x60"], _BANDS_MEAN[[_WL_ORDER.index(b) for b in BANDS_60]])
        y_raw = raw["y"].copy()
        conf = raw["conf"].astype(np.uint8).copy()

        # MADOS stores labels as 1..15 with 0 for unlabeled; shift to 0..14 and
        # map 0 -> IGNORE_INDEX so CE can safely skip it.
        y = np.where(y_raw > 0, y_raw - 1, IGNORE_INDEX).astype(np.int64)

        # Sparse-label ablation: randomly drop a fraction of High-confidence
        # labels *only* so that CRC/CGD can still use the confidence map.
        if self.mode == "train" and self.keep_label_ratio < 1.0:
            hi_mask = (conf == 1) & (y != IGNORE_INDEX)
            if hi_mask.any():
                idx = np.flatnonzero(hi_mask.ravel())
                drop_n = int(len(idx) * (1.0 - self.keep_label_ratio))
                if drop_n > 0:
                    drop = np.random.choice(idx, size=drop_n, replace=False)
                    flat_y = y.ravel()
                    flat_y[drop] = IGNORE_INDEX
                    y = flat_y.reshape(y.shape)

        # Normalize each branch.
        x10 = (x10 - MEAN_10) / STD_10
        x20 = (x20 - MEAN_20) / STD_20
        x60 = (x60 - MEAN_60) / STD_60

        # Data augmentation: same flip / rotation across all three branches.
        if self.augment:
            flip_choice = np.random.randint(0, 5)          # 0,1,-1 or 2,2 (no-op)
            flip_code = [1, 0, -1, 2, 2][flip_choice]
            rot_k = np.random.randint(0, 4) if np.random.rand() < 0.8 else 0

            if flip_code != 2:
                x10 = _apply_flip_rot(x10, flip_code, 0)
                x20 = _apply_flip_rot(x20, flip_code, 0)
                x60 = _apply_flip_rot(x60, flip_code, 0)
                # Treat y / conf as single-channel tensors for the helper.
                y    = _apply_flip_rot(y[None],    flip_code, 0)[0]
                conf = _apply_flip_rot(conf[None], flip_code, 0)[0]
            if rot_k:
                x10 = _apply_flip_rot(x10, 2, rot_k)
                x20 = _apply_flip_rot(x20, 2, rot_k)
                x60 = _apply_flip_rot(x60, 2, rot_k)
                y    = _apply_flip_rot(y[None],    2, rot_k)[0]
                conf = _apply_flip_rot(conf[None], 2, rot_k)[0]

        return {
            "x10":  torch.from_numpy(x10.astype(np.float32)),
            "x20":  torch.from_numpy(x20.astype(np.float32)),
            "x60":  torch.from_numpy(x60.astype(np.float32)),
            "y":    torch.from_numpy(y.astype(np.int64)),
            "conf": torch.from_numpy(conf.astype(np.uint8)),
            "name": entry["crop_name"],
        }
