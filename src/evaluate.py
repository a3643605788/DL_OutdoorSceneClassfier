import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from model import CNNBaseline  # 確保你之前定義的 model 在這
from dataset import get_test_loader

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def run_day12_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 載入資料 (對應你 data_preprocessing.py 產出的路徑)
    test_loader, classes = get_test_loader(processed_root='data/processed/splits', img_size=224)
    
    # 2. 初始化模型並載入權重
    model = CNNBaseline(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load('outputs/best_baseline.pth'))
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

    # 3. 繪製混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted (Model Evaluate)')
    plt.ylabel('Actual (Real Label)', rotation=0, labelpad=40, va='center')
    plt.yticks(rotation=0)
    plt.title('Day 12: Confusion Matrix')
    plt.show()

    # 4. 深度分析：Street (index 5) vs Building (index 0)
    # 根據 Intel Dataset 預設排序，通常 buildings 是 0, street 是 5
    street_idx = classes.index('street')
    building_idx = classes.index('buildings')
    
    # 找出模型「太過自信」但卻判斷錯誤的案例 (Street 誤判成 Building)
    mis_indices = np.where((all_labels == street_idx) & (all_preds == building_idx))[0]
    
    if len(mis_indices) > 0:
        print(f"\n[分析結果] 發現 {len(mis_indices)} 張街道圖被誤判為建築物。")
        plt.figure(figsize=(15, 4))
        for i, idx in enumerate(mis_indices[:5]):
            plt.subplot(1, 5, i+1)
            img = denormalize(all_images[idx]).permute(1, 2, 0).numpy()
            plt.imshow(np.clip(img, 0, 1))
            plt.title(f"Street -> Building\n(Index: {idx})")
            plt.axis('off')
        plt.show()

if __name__ == "__main__":
    run_day12_evaluation()