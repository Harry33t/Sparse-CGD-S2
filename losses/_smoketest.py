"""Sanity-check CRC + CGD + sup losses with random inputs."""
from __future__ import annotations

import torch

from ..models import CRCCGDNet
from .crc import CRCLoss
from .cgd import CGDLoss
from .sup import WeightedCEDiceLoss, gen_class_weights


def main() -> None:
    torch.manual_seed(0)
    B, H, W, K = 2, 240, 240, 15
    x10 = torch.randn(B, 4, H, W)
    x20 = torch.randn(B, 6, H // 2, W // 2)
    x60 = torch.randn(B, 1, H // 6, W // 6)
    # Label map with a mix of classes and an ignore region.
    y = torch.randint(low=0, high=K, size=(B, H, W))
    y[:, :30, :] = 255  # ignore
    conf = torch.randint(low=0, high=4, size=(B, H, W))  # 0..3

    model = CRCCGDNet(num_classes=K).train()
    out = model(x10, x20, x60)

    # Class-dist placeholder
    cd = torch.ones(K) / K
    sup = WeightedCEDiceLoss(class_weights=gen_class_weights(cd), ignore_index=255)
    crc = CRCLoss()
    cgd = CGDLoss(feat_dim=256, num_classes=K)

    y_hi = torch.where(conf == 1, y, torch.full_like(y, 255))
    l_sup = sup(out["logits"], y_hi)
    l_crc = crc(out["branch"], conf)
    l_cgd = cgd(out["emb"], y, conf)

    total = l_sup + 0.5 * l_crc + 0.5 * l_cgd
    total.backward()
    print(f"l_sup={l_sup.item():.4f}  l_crc={l_crc.item():.4f}  l_cgd={l_cgd.item():.4f}")
    print(f"total={total.item():.4f}")
    # Check a param got gradient.
    g = next(p for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"grad on one param: shape={tuple(g.shape)}  sum_abs={g.grad.abs().sum().item():.4f}")
    print("OK")


if __name__ == "__main__":
    main()
