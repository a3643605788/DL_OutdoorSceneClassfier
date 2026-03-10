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
def save_learning_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))

    # Loss 曲線
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], '-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], '-o', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy 曲線
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], '-o', label='Train Acc')
    plt.plot(epochs, history['val_acc'], '-o', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png")
    print(f"\n[INFO] Learning curve saved to {output_dir / 'learning_curve.png'}")


def main():
    # --- 參數設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20
    batch_size = 32
    img_size = 224
    lr = 0.001
    split_root = Path("data/processed/splits") # 依照 Day 4 的輸出路徑
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True) # 自動建立 outputs 資料夾

    # --- 2. 呼叫 Day 4 的函式建立 Loaders ---
    # 這行解決了你提到的 train_loader 和 val_loader 未定義問題
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        split_root=split_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=0 # Windows 下建議先設為 0 以避免錯誤
    )
    
    # --- 3. 初始化模型、損失函數與優化器 ---
    model = CNNBaseline(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 【新增：初始化 Scheduler】
    # mode='min': 監控指標（val_loss）不再下降時觸發
    # factor=0.1: 學習率降低為原來的 1/10
    # patience=2: 如果連續 2 輪指標沒改善，就調降 LR
    # verbose=True: 降速時會在終端機印出通知
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # 新增：建立歷史紀錄字典
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    
    best_val_loss = float('inf')
    patience = 3  # 如果連續 3 輪 Val Loss 沒下降就停止
    trigger_times = 0


    # --- 4. 執行 Epoch 迴圈 ---
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 執行訓練
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 執行驗證
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        

        # 【核心修改：更新 Scheduler】
        # Scheduler 必須放在 validate 之後，因為它需要根據當前的 val_loss 來判斷是否降速
        scheduler.step(val_loss)

        # 獲取當前 Learning Rate (用於視覺化觀察)
        current_lr = optimizer.param_groups[0]['lr']

        # Model Checkpoint: 永遠儲存最強的版本
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_baseline.pth")
            print(f">>> Found better model! Saved to best_baseline.pth")
            trigger_times = 0 # 重置計數器
        else:
            trigger_times += 1
            print(f">>> Val loss did not improve. (Count: {trigger_times}/{patience})")

        # 新增：將數值存入歷史紀錄
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early Stopping: 防止像 Day 10 那樣最後一輪崩壞
        if trigger_times >= patience:
            print("Early stopping! Stopping training to prevent overfitting.")
            break

        # 單一 Epoch 的平均
        # loss:預測值與真實標籤之間差距的指標，值越小差距越小  Acc:猜對的機率有多高
        print(f"Summary - LR: {current_lr:.6f} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "                   # 訓練集資料的表現 
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")                                 # 驗證集資料的表現

    # --- 5. 訓練結束：繪圖並存檔 ---
    save_learning_curves(history, output_dir)

    # (進階建議) 將原始數據存成 JSON，方便以後分析
    with open(output_dir / "metrics.json", "w") as f:
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