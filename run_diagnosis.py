import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 引用你的專案零件
from src.model import ResNetTransfer
from src.data_preprocessing import build_dataloaders
from src.gradcam import GradCAM, overlay_heatmap

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs")
    
    # 1. 載入資料與類別
    _, _, test_loader, classes = build_dataloaders(
        split_root=Path("data/processed/splits"), img_size=224, batch_size=32, num_workers=0
    )
    
    # 2. 載入模型
    model = ResNetTransfer(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(output_dir / "best_resnet.pth"))
    model.eval()

    # 3. 初始化 Grad-CAM (針對 ResNet18 的最後一個卷積層 layer4)
    target_layer = model.model.layer4[-1] 
    cam_tool = GradCAM(model, target_layer)

    # 4. 找出 Street -> Building 的誤判案例
    street_idx = classes.index('street')
    building_idx = classes.index('buildings')
    
    misclassified_samples = []
    print("正在搜尋誤判案例...")
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        
        # 找出符合「真實是街道，預測是建築」的索引
        mask = (labels == street_idx) & (preds == building_idx)
        if mask.any():
            indices = torch.where(mask)[0]
            for idx in indices:
                misclassified_samples.append((images[idx].unsqueeze(0), labels[idx].item(), preds[idx].item()))
        
        if len(misclassified_samples) >= 5: break # 只抓前 5 個做展示

    # 5. 繪圖
    print(f"開始產出 Grad-CAM 熱力圖 (共 {len(misclassified_samples)} 張)...")
    plt.figure(figsize=(18, 8))
    
    for i, (img_tensor, true_label, pred_label) in enumerate(misclassified_samples[:5]):
        # 生成熱力圖
        heatmap = cam_tool.generate_heatmap(img_tensor, pred_label) # 觀察模型為何判斷為 Building
        
        # 疊加圖層
        original_img = denormalize(img_tensor.squeeze())
        combined_img = overlay_heatmap(original_img, heatmap)
        
        # 繪圖
        plt.subplot(2, 5, i+1)
        plt.imshow(original_img)
        plt.title(f"Original (True: Street)")
        plt.axis('off')
        
        plt.subplot(2, 5, i+6)
        plt.imshow(combined_img)
        plt.title(f"Grad-CAM (Pred: Building)")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "resnet_diagnosis_gradcam.png")
    print(f"診斷結果已儲存至: {output_dir / 'resnet_diagnosis_gradcam.png'}")
    plt.show()

if __name__ == "__main__":
    main()