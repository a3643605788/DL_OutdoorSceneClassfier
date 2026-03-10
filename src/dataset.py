# src/dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_transforms(img_size: int, is_train: bool = False):
    """
    獲取影像轉換組合
    is_train: True 使用針對性數據增強；False 僅進行標準化縮放
    """
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    if is_train:
        # Day 13 針對性數據增強
        return transforms.Compose([
            # 1. 解決「只看局部」：隨機裁剪 50%~100% 的區域並縮放，強迫模型學習不同尺度的特徵
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)), 
            # 2. 增加左右對稱性
            transforms.RandomHorizontalFlip(),
            # 3. 解決「光影誤判」：隨機調整亮度、對比、飽和度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        # 評估與測試用的轉換 (保持影像原始比例與色調)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

def get_train_val_loaders(processed_root='data/processed/splits', img_size=224, batch_size=32):
    """新增：獲取訓練與驗證集的 Loader"""
    root = Path(processed_root)
    
    # 訓練集使用增強 (is_train=True)
    train_tfm = get_transforms(img_size, is_train=True)
    train_ds = datasets.ImageFolder(root / "train", transform=train_tfm)
    
    # 驗證集不使用增強 (is_train=False)
    val_tfm = get_transforms(img_size, is_train=False)
    val_ds = datasets.ImageFolder(root / "val", transform=val_tfm)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, train_ds.classes

def get_test_loader(processed_root='data/processed/splits', img_size=224, batch_size=32):
    """專門給評估用的 function"""
    eval_tfm = get_transforms(img_size, is_train=False)
    test_path = Path(processed_root) / "test"
    
    test_ds = datasets.ImageFolder(test_path, transform=eval_tfm)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return test_loader, test_ds.classes