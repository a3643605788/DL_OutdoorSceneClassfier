from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 這個程式主要處理的內容
# 1.讀取原始資料（找出每類有哪些圖片）
# 2.做切分（決定哪些圖屬於 train/val/test）
# 3.建立前處理規則（Resize/Normalize）
# 4.建立 DataLoader，實際抽一個 batch 驗證
# data_preprocessing.py 做的是把原始影像資料集（data/raw/...）整理成一份訓練可直接使用的資料版本（data/processed/...）

# DataLoader 是 train_loader / val_loader / test_loader 這三個變數的型別
# DataLoader會從Dataset取出多筆資料，組合成一個batch，然後在訓練迴圈中一個batch一個batch地餵給模型。


# ----------------------------
# 設定
# ----------------------------
@dataclass
class SplitConfig:
    raw_dir: Path
    out_dir: Path
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    copy_mode: str = "copy"  # "copy" | "hardlink" | "symlink"


# ----------------------------
# 掃描資料夾，把每個類別的圖片列出來
# 到seg_train看有幾個子資料夾(buildings/forest...)
# 找出每個子資料夾的所有圖片
# 回傳一個dict
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def list_images_by_class(raw_dir: Path) -> Dict[str, List[Path]]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir.resolve()}")

    classes = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not classes:
        raise ValueError(f"No class folders under: {raw_dir.resolve()}")

    out: Dict[str, List[Path]] = {}
    for cls_dir in sorted(classes, key=lambda p: p.name):
        imgs = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not imgs:
            raise ValueError(f"No images found in class folder: {cls_dir.resolve()}")
        out[cls_dir.name] = imgs
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# 真的把檔案變成 train/val/test 的資料夾結構
# ----------------------------
def transfer_file(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return

    if mode == "copy":              #真正複製一份檔案（耗時間、占空間）
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":          #不複製內容，只建立一個指向同一份檔案的連結（快、省空間，但要同磁碟）
        os.link(src, dst)
        return

    if mode == "symlink":           #捷徑式連結（Windows 可能需要權限）
        os.symlink(src, dst)
        return

    raise ValueError(f"Unknown copy_mode: {mode}")


# ----------------------------
# 決定每一類要怎麼切 train/val/test，只處理索引不處理圖片內容
# 例如某類有1000張圖
# 先把0~999順序打亂
# 算出 test:150張 val:150張 train:700張
# ----------------------------
def split_indices(n: int, val_ratio: float, test_ratio: float, rng: random.Random) -> Tuple[List[int], List[int], List[int]]:
    if not (0 < val_ratio < 1) or not (0 < test_ratio < 1) or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio and test_ratio must be in (0,1) and val_ratio+test_ratio < 1")

    indices = list(range(n))
    rng.shuffle(indices)

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ----------------------------
# --把原始資料切成訓練用結構--
# 建立輸出資料夾(data/processed/splits/train, data/processed/splits/val, data/processed/splits/test)
# 對每個類別(buildings/forest/...)算出哪些圖片分到 train/val/test，把對應的圖片搬（copy/link）到各類別(buildings、forest、glacier、mountain、sea、street)
# ----------------------------
def create_splits(cfg: SplitConfig) -> Dict[str, int]:
    rng = random.Random(cfg.seed)
    images_by_class = list_images_by_class(cfg.raw_dir)

    split_root = cfg.out_dir / "splits"
    for split in ["train", "val", "test"]:
        ensure_dir(split_root / split)

    counts = {"train": 0, "val": 0, "test": 0}

    for cls, imgs in images_by_class.items():
        n = len(imgs)
        train_idx, val_idx, test_idx = split_indices(n, cfg.val_ratio, cfg.test_ratio, rng)

        def put(split_name: str, idxs: List[int]) -> None:
            nonlocal counts
            for i in idxs:
                src = imgs[i]
                # 保留副檔名，檔名加上原資料夾名避免重名
                dst = split_root / split_name / cls / f"{src.stem}{src.suffix.lower()}"
                transfer_file(src, dst, cfg.copy_mode)
                counts[split_name] += 1

        put("train", train_idx)
        put("val", val_idx)
        put("test", test_idx)

    return counts


# ----------------------------
# --定義送進模型前的影像處理--
# 調整每張圖片的格式，讓模型能夠更好吸收
# ----------------------------
def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    回傳 (train_transform, eval_transform)
    Normalize 使用 ImageNet mean/std，適合之後接 pretrained backbone（ResNet/EfficientNet 等）
    """
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_tfm, eval_tfm


# ----------------------------
# --把資料變成訓練會用的 batch--
# 用 ImageFolder + DataLoader 做可訓練的資料管線
# 建立並回傳多個 DataLoader
# ----------------------------
def build_dataloaders(split_root: Path, img_size: int, batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_tfm, eval_tfm = get_transforms(img_size)

    train_ds = datasets.ImageFolder(split_root / "train", transform=train_tfm)
    val_ds = datasets.ImageFolder(split_root / "val", transform=eval_tfm)
    test_ds = datasets.ImageFolder(split_root / "test", transform=eval_tfm)

    # ImageFolder 會自動用資料夾名稱排序當作 class list
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names


# ----------------------------
# 抽一個 batch 看形狀對不對
# ----------------------------
def sanity_check_loader(loader: DataLoader, class_names: List[str]) -> None:
    x, y = next(iter(loader))
    print(f"[Sanity] batch x shape: {tuple(x.shape)}")   # (B, C, H, W)
    print(f"[Sanity] batch y shape: {tuple(y.shape)}")
    print(f"[Sanity] y sample: {y[:10].tolist()}")
    print(f"[Sanity] label->class sample: {[class_names[i] for i in y[:10].tolist()]}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, default="data/raw/seg_train")
    p.add_argument("--out_dir", type=str, default="data/processed")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy_mode", type=str, choices=["copy", "hardlink", "symlink"], default="copy")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


# 1.讀參數
# 2.印出設定（你看到的 raw_dir/out_dir/ratio）
# 3.create_splits() 產生資料夾
# 4.build_dataloaders() 建 loader
# 5.sanity_check_loader() 驗收
def main() -> None:
    args = parse_args()
    cfg = SplitConfig(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_mode=args.copy_mode,
    )

    print(f"raw_dir = {cfg.raw_dir.resolve()}")
    print(f"out_dir = {cfg.out_dir.resolve()}")
    print(f"split ratios: train={1-cfg.val_ratio-cfg.test_ratio:.2f}, val={cfg.val_ratio:.2f}, test={cfg.test_ratio:.2f}")
    print(f"copy_mode = {cfg.copy_mode}, seed = {cfg.seed}")

    counts = create_splits(cfg)
    print(f"[Done] Split counts: {counts}")

    split_root = cfg.out_dir / "splits"
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        split_root=split_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"[Data] classes ({len(class_names)}): {class_names}")
    print(f"[Data] train batches: {len(train_loader)}, val batches: {len(val_loader)}, test batches: {len(test_loader)}")

    sanity_check_loader(train_loader, class_names)
    print("[OK] DataLoader callable and batch shape verified.")


if __name__ == "__main__":
    main()
