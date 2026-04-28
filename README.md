# Sparse-CGD-S2

Code for the paper:

> **Toward Operational Marine Pollution Monitoring from Sentinel-2 Under Sparse Annotations:
> Confidence-Guided Disambiguation and Empirical Assessment of Spectral-Resolution Utility.**

This repository contains the training, evaluation, and figure-generation code used to
produce the experiments reported on the [MADOS](https://marine-pollution.github.io/MADOS)
benchmark, including the proposed **Confidence-Guided Disambiguation (CGD)** module.

## Repository layout

```
.
|-- data/                # Multi-resolution Sentinel-2 dataloader (MADOS)
|-- losses/              # CGD prototype-margin loss, CRC, supervised CE+Dice
|-- models/              # CGD network, baselines (U-Net, DeepLabV3+, SegFormer-Tiny)
|-- utils/               # Metrics and small helpers
|-- train.py             # Training entry point
|-- eval.py              # Evaluation on the test split
|-- qualitative.py       # Generates qualitative comparison figures
`-- scene_examples.py    # Generates representative MADOS scene examples
```

## Installation

Tested with Python 3.10, PyTorch 2.x, CUDA 12.x.

```bash
git clone https://github.com/Harry33t/Sparse-CGD-S2.git
cd Sparse-CGD-S2
pip install torch torchvision  # match your CUDA
pip install numpy einops timm transformers tensorboard tqdm rasterio scikit-learn
```

## Data

Download MADOS from the official project page
(<https://marine-pollution.github.io/MADOS>) and place it as

```
/path/to/MADOS/
    Scene_xxx_x/        # individual MADOS scenes
    splits/
        train_X.txt
        val_X.txt
        test_X.txt
```

A quick sanity check is provided:

```bash
python -m data._smoketest --root /path/to/MADOS
```

## Training

The matched 10 m-only configuration with CGD on the High-only supervision regime
(this is the main configuration of the paper):

```bash
python train.py \
    --data-root /path/to/MADOS \
    --splits   /path/to/MADOS/splits \
    --use-bands 10 \
    --supervision-mode high \
    --seed 42 \
    --out-dir runs/cgd_band10_s42
```

Common flags:

* `--use-bands {10, 10,20, 10,20,60}` selects the spectral configuration.
* `--supervision-mode {high, high_mod, all}` selects the label-quality regime.
  Combine with `--label-fraction 0.5` or `0.25` to subsample the high-confidence labels.
* `--seed {42, 123, 456}` for the three-seed reproduction used in the paper.

## Evaluation

```bash
python eval.py \
    --data-root /path/to/MADOS \
    --splits   /path/to/MADOS/splits \
    --ckpt     runs/cgd_band10_s42/best.pt \
    --save-probs        # writes test/probs.npz for threshold sweeps
```

`eval.py` writes:

* `test/test_metrics.json` -- per-class F1, mIoU, LCR
* `test/confusion.npy`     -- 15x15 confusion matrix
* `test/probs.npz`         -- per-pixel ground truth, prediction, p_MD, p_OS

## Figures

```bash
# Fig 2: representative MADOS scenes
python scene_examples.py \
    --data-root /path/to/MADOS \
    --splits   /path/to/MADOS/splits \
    --out-dir  figures/

# Fig 12: qualitative RGB / GT / DLV3+ / CGD comparison
python qualitative.py \
    --data-root /path/to/MADOS \
    --splits   /path/to/MADOS/splits \
    --ckpt-base runs/deeplabv3p/best.pt \
    --ckpt-ours runs/cgd_band10_s42/best.pt \
    --out-dir   figures/ \
    --n 4
```

## Reproducing the paper tables

* **Table 2 (main pollutant-centric performance)** -- run training for each of
  U-Net, DeepLabV3+, SegFormer-Tiny, and CGD with seeds 42 / 123 / 456.
* **Table 3 (error decomposition)** -- aggregate per-class confusion-pair rates
  from `test/confusion.npy` for DeepLabV3+ vs CGD across the three seeds.
* **Table 4 (alert reliability)** -- run `eval.py --save-probs` to produce
  `test/probs.npz`, then sweep thresholds tau in {0.5, 0.7, 0.9}.
* **Table 5 (supervision regimes / spectral configs)** -- vary
  `--supervision-mode`, `--label-fraction`, and `--use-bands` on the CGD model.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{li2026cgd,
  author  = {Li, Fangfang and Huang, Guanxiong and Yang, Lei},
  title   = {Toward Operational Marine Pollution Monitoring from {S}entinel-2
             Under Sparse Annotations: Confidence-Guided Disambiguation and
             Empirical Assessment of Spectral-Resolution Utility},
  journal = {Environmental Monitoring and Assessment},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
