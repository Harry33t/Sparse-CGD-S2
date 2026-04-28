"""CPU forward-shape check for CRCCGDNet.

Run:
    python -m src.models._smoketest
"""
from __future__ import annotations

import torch

from .crc_cgd_net import CRCCGDNet


def main() -> None:
    model = CRCCGDNet(num_classes=15)
    model.eval()
    B, H, W = 2, 240, 240
    x10 = torch.randn(B, 4, H,      W     )
    x20 = torch.randn(B, 6, H // 2, W // 2)
    x60 = torch.randn(B, 1, H // 6, W // 6)
    with torch.no_grad():
        out = model(x10, x20, x60)
    print("logits:", out["logits"].shape)
    print("emb:   ", out["emb"].shape)
    for k, v in out["branch"].items():
        print(f"branch {k}: {v.shape}")
    n = sum(p.numel() for p in model.parameters())
    print(f"#params: {n/1e6:.2f} M")


if __name__ == "__main__":
    main()
