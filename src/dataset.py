# src/dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_transforms(img_size: int):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    
    # 評估與測試用的轉換 (不包含隨機翻轉)
    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return eval_tfm

def get_test_loader(processed_root='data/processed/splits', img_size=224, batch_size=32):
    """專門給評估用的 function"""
    eval_tfm = get_transforms(img_size)
    test_path = Path(processed_root) / "test"
    
    test_ds = datasets.ImageFolder(test_path, transform=eval_tfm)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return test_loader, test_ds.classes