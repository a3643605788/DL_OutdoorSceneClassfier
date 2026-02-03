from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from pathlib import Path


# ------------------------------------------------------------
# 你需要依你專案實際位置調整這兩個 import
# 例如：
# from src.models.cnn_baseline import CNNBaseline
# from src.data.dataloaders import build_dataloaders
# ------------------------------------------------------------
from src.model import CNNBaseline
from src.data_preprocessing import build_dataloaders


@dataclass
class Cfg:
    data_dir: str = "data/processed/splits"  # 依你的 processed 輸出調整
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 0  # Windows 常用 0 先穩定
    lr: float = 1e-3
    steps_overfit: int = 100
    seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


def main() -> None:
    cfg = Cfg()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # 1) DataLoader：只需要 train loader 就夠做 smoke test
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        split_root=Path("data/processed/splits"),
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    num_classes = len(class_names)
    print("num_classes =", num_classes, "class_names =", class_names)

    # 2) Model / Loss / Optimizer
    model = CNNBaseline(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ============================================================
    # 產出 1：1 batch forward + backward + step
    # 1 batch：就是 DataLoader「吐出一次」的那一包資料。
    # ============================================================
    model.train()

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    print("images.shape =", tuple(images.shape))
    print("labels.shape =", tuple(labels.shape), "labels.dtype =", labels.dtype)

    # CrossEntropyLoss 需要 labels 是 Long 且為 class index
    if labels.dtype != torch.long:
        labels = labels.long()

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    print("logits.shape =", tuple(logits.shape))

    # shape guard
    assert logits.ndim == 2, f"logits should be 2D [B,C], got {logits.ndim}D"
    assert logits.shape[0] == labels.shape[0], "batch size mismatch"
    assert logits.shape[1] == num_classes, f"class mismatch: logits C={logits.shape[1]} vs {num_classes}"

    loss = criterion(logits, labels)
    print("loss =", float(loss.detach().cpu()))

    assert math.isfinite(float(loss.detach().cpu())), "loss is NaN/Inf"

    loss.backward()

    # grad sanity check（抽幾個參數看）
    grad_stats = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                grad_stats.append((name, "None"))
            else:
                grad_ok = torch.isfinite(p.grad).all().item()
                grad_stats.append((name, f"finite={grad_ok}, mean={p.grad.abs().mean().item():.3e}"))
    print("\n[grad check] sample:")
    for row in grad_stats[:10]:
        print(" -", row[0], ":", row[1])

    # step 前後參數是否有變（抽一個參數張量）
    with torch.no_grad():
        p0 = None
        for p in model.parameters():
            if p.requires_grad:
                p0 = p.detach().clone()
                break

    optimizer.step()

    with torch.no_grad():
        p1 = None
        for p in model.parameters():
            if p.requires_grad:
                p1 = p.detach().clone()
                break
        delta = (p1 - p0).abs().mean().item()
    print("\nparam delta mean abs =", delta)
    assert delta > 0, "parameters did not change after optimizer.step()"

    print("\n✅ 產出1完成：1 batch forward/backward/step OK")

    # ============================================================
    # 產出 2：Overfit single batch，確認可訓練
    # ============================================================
    print(f"\n[Overfit single batch] steps={cfg.steps_overfit}")
    images_fixed, labels_fixed = images, labels  # 固定同一個 batch
    for step in range(1, cfg.steps_overfit + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images_fixed)
        loss = criterion(logits, labels_fixed)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == labels_fixed).float().mean().item()
            print(f"step={step:03d} loss={loss.item():.4f} acc={acc*100:.1f}%")

            # 避免 silent 爆炸
            assert is_finite_tensor(loss.detach()), "loss became NaN/Inf during overfit"
            assert is_finite_tensor(logits.detach()), "logits became NaN/Inf during overfit"

    print("\n✅ 產出2完成：Overfit single batch 跑完（loss 應明顯下降、acc 應上升）")


if __name__ == "__main__":
    main()
