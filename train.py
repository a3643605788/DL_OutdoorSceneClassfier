import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

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

def main():
    # --- 參數設定 (解決 epochs 未定義問題) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 32
    img_size = 224
    lr = 0.001
    split_root = Path("data/processed/splits") # 依照 Day 4 的輸出路徑

    # --- 2. 呼叫 Day 4 的函式建立 Loaders ---
    # 這行解決了你提到的 train_loader 和 val_loader 未定義問題
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        split_root=split_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=0 # Windows 下建議先設為 0 以避免錯誤
    )
    
    # --- 3. 初始化模型 ---
    model = CNNBaseline(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # --- 4. 執行 Epoch 迴圈 ---
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 執行訓練
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 執行驗證
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 單一 Epoch 的平均
        # loss:預測值與真實標籤之間差距的指標，值越小差距越小  Acc:猜對的機率有多高
        print(f"Summary - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "                   # 訓練集資料的表現 
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")                                 # 驗證集資料的表現

if __name__ == "__main__":
    main()


# ---表格---
# 特性	    train_one_epoch()	       validate()
# ==============================================================
# 主要目的	學習特徵、更新權重           評估泛化能力、檢查是否過擬合
# 資料來源	訓練集 (Train Set)	        驗證集 (Val Set)
# 梯度更新	會 更新權重 (backward())	不會 更新權重 (no_grad())
# 執行頻率	每個 Epoch 跑一次	        每個 Epoch 結束後跑一次