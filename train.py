import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt  # 新增：用於繪圖
import json                     # 新增：用於儲存數據紀錄
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 新增：匯入 Scheduler

# 1. 引用你寫好的零件
from src.model import CNNBaseline
from src.model import ResNetTransfer
from src.data_preprocessing import build_dataloaders

# 模型的「學習」階段 (學習特徵、更新權重)
# 執行一個完整的「訓練輪次」(Epoch)，即讓模型把所有的訓練資料看過一遍並修正錯誤。
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()                                       # 告訴模型現在進入「訓練模式」

    # 這三個變數:建立累加器，用來計算這一輪結束後的平均損失和準確率。
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")            # 建立進度條，讓你可以在終端機即時看到訓練進度
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)       # 將資料搬移到 GPU（如果有）或 CPU，確保計算在同一個硬體設備上進行
        
        # Batch 迴圈（核心五步驟）
        optimizer.zero_grad()                           # 第一步：清空梯度。每次計算新誤差前，必須把上一次留下的舊紀錄（導數）歸零，否則會累加導致錯誤。
        outputs = model(images)                         # 第二步：前向傳播。將圖片輸入模型，得到預測結果
        loss = criterion(outputs, labels)               # 第三步：計算損失。比對預測結果與正確標籤（labels），算出誤差有多大。
        loss.backward()                                 # 第四步：反向傳播。PyTorch 會自動計算誤差對模型中每個參數（權重）的「貢獻度」（即梯度）。
        optimizer.step()                                # 第五步：更新權重。根據剛剛算出的梯度，動手修改模型參數，讓下次預測更準確。
        
        running_loss += loss.item()                     # 累加當前 Batch 的損失

        # max(1) 會回傳兩個值：最大值的「數值」和最大值的「索引」。我們只需要索引（類別編號），所以用底線忽略不需要的第一個數值。
        _, predicted = outputs.max(1)                   # 找出模型預測機率最高的類別索引
        
        total += labels.size(0)                         # 累加目前處理過的圖片總數
        correct += predicted.eq(labels).sum().item()    # 統計模型猜對了幾個標籤
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})   # 一個Betch的表現

    # 計算這一整輪的平均數值並回傳
    avg_loss = running_loss / len(dataloader)
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc


# 模型的「期中考試」 (評估泛化能力、檢查是否過擬合)
# 模型學習完一輪後，使用它沒看過的驗證集資料來評估其實力
def validate(model, dataloader, criterion, device):
    model.eval()                                        # 告訴模型進入「評估模式」
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():                               # 最重要的優化。告訴 PyTorch 接下來不需要記錄梯度。這能大幅節省記憶體，並讓運算變快。
        # for迴圈與訓練的for迴圈相似，但完全沒有 optimizer.zero_grad(), backward(), 或 step()。
        # 模型只是純粹地輸入資料並輸出預測，不會進行自我修正。
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return val_loss / len(dataloader), 100. * correct / total


# 新增：繪圖與存檔函式
def save_learning_curves(history, output_dir, model_type):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Loss 曲線
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], '-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], '-o', label='Val Loss')
    plt.title(f'{model_type} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy 曲線
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], '-o', label='Train Acc')
    plt.plot(epochs, history['val_acc'], '-o', label='Val Acc')
    plt.title(f'{model_type} - Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"learning_curve_{model_type}.png")
    print(f"\n[INFO] Learning curve saved to outputs/learning_curve_{model_type}.png")


def main():
    # 【在這裡切換實驗】
    model_type = "resnet"  # 可選 "baseline" 或 "resnet"

    # --- 參數設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    batch_size = 32
    img_size = 224
    
    # 【自動切換 LR】Transfer Learning 建議用較小的 0.0001
    lr = 0.0001 if model_type == "resnet" else 0.001
    
    split_root = Path("data/processed/splits")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 2. 建立 Loaders
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        split_root=split_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=0 
    )
    
    # 3. 根據選擇初始化模型
    if model_type == "resnet":
        model = ResNetTransfer(num_classes=len(class_names)).to(device)
        print(f"\n>>> use Pre-trained ResNet18 (LR={lr})")
    else:
        model = CNNBaseline(num_classes=len(class_names)).to(device)
        print(f"\n>>> mode：use CNN Baseline (LR={lr})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 5 # 稍微拉長耐心，讓 Scheduler 有發揮空間
    trigger_times = 0

    # 4. 訓練迴圈
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 這裡會根據 model_type 自動命名權重檔
        model_filename = f"best_{model_type}.pth"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / model_filename)
            print(f">>> found better model！saved to {model_filename}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f">>> Val loss didn't improve ({trigger_times}/{patience})")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Summary - LR: {current_lr:.6f} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if trigger_times >= patience:
            print("Early stopping!")
            break

    # 5. 存檔與繪圖
    save_learning_curves(history, output_dir, model_type)
    with open(output_dir / f"metrics_{model_type}.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()


# ---表格---
# 特性	    train_one_epoch()	       validate()
# ==============================================================
# 主要目的	學習特徵、更新權重           評估泛化能力、檢查是否過擬合
# 資料來源	訓練集 (Train Set)	        驗證集 (Val Set)
# 梯度更新	會 更新權重 (backward())	不會 更新權重 (no_grad())
# 執行頻率	每個 Epoch 跑一次	        每個 Epoch 結束後跑一次