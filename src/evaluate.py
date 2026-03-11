import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# 1. 引用模型與資料零件
from model import CNNBaseline, ResNetTransfer
from data_preprocessing import build_dataloaders

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def run_model_evaluation():
    # --- 1. 設定區域 ---
    model_type = "resnet"  # 改成 "resnet" 評估新模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224
    split_root = Path("data/processed/splits")
    output_dir = Path("outputs")
    
    # 2. 載入資料 (直接用 build_dataloaders 獲取 test_loader)
    _, _, test_loader, classes = build_dataloaders(
        split_root=split_root, 
        img_size=img_size, 
        batch_size=32, 
        num_workers=0
    )
    
    # 3. 初始化對應模型並載入權重
    if model_type == "resnet":
        model = ResNetTransfer(num_classes=len(classes)).to(device)
        weight_path = output_dir / "best_resnet.pth"
    else:
        model = CNNBaseline(num_classes=len(classes)).to(device)
        weight_path = output_dir / "best_baseline.pth"
        
    print(f"\n[INFO] 正在載入模型權重: {weight_path.name}")
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    all_preds, all_labels, all_images = [], [], []

    print(f"正在評估 {len(test_loader.dataset)} 張測試影像...")
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.append(images)

    all_images = torch.cat(all_images)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. 繪製混淆矩陣 (解決標籤破圖問題)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(11, 9)) # 稍微加大畫布
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12})
    
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('Actual Label', fontsize=12, labelpad=10)
    plt.title(f'Confusion Matrix: {model_type.upper()}', fontsize=15, pad=20)
    
    # 【關鍵】自動調整邊距，防止標籤被切掉
    plt.tight_layout() 
    plt.savefig(output_dir / f"confusion_matrix_{model_type}.png")
    plt.show()

    # 5. 輸出 Classification Report
    print(f"\n[{model_type.upper()} 評估報告]")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 6. 自動偵測誤判重災區：Street vs Building
    street_idx = classes.index('street')
    building_idx = classes.index('buildings')
    
    mis_indices = np.where((all_labels == street_idx) & (all_preds == building_idx))[0]
    
    if len(mis_indices) > 0:
        num_show = min(5, len(mis_indices))
        print(f"\n[分析] 發現 {len(mis_indices)} 張街道圖被誤判為建築物 (展示前 {num_show} 張)。")
        plt.figure(figsize=(15, 4))
        for i, idx in enumerate(mis_indices[:num_show]):
            plt.subplot(1, num_show, i+1)
            img = denormalize(all_images[idx]).permute(1, 2, 0).numpy()
            plt.imshow(np.clip(img, 0, 1))
            plt.title(f"Street -> Building\nIdx: {idx}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("\n[太棒了] ResNet18 在測試集中完全解決了 Street 誤判為 Building 的問題！")

if __name__ == "__main__":
    run_model_evaluation()