import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
from pydantic import BaseModel
from src.model import ResNetTransfer  # 使用你在第二階段完成的 ResNet18

# 自動生成 Swagger 文件(/docs)
app = FastAPI(title="Outdoor Scene Classifier API")

# 定義標籤映射 (根據 Intel Scene Dataset 標準順序)
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# 影像預處理 (必須與訓練時使用的 Transform 完全一致)
# 將圖片檔案轉化為數學張量
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 硬體檢測與模型載入
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 自動偵測主機是否有顯卡，有的話優先使用 cuda
model = ResNetTransfer(num_classes=6, pretrained=False)

try:
    # 修正：map_location='cpu' 確保在無 GPU 環境也能載入
    # map_location=device:解決「模型在 GPU 訓練，但 API 在 CPU 執行」時會報錯的問題
    state_dict = torch.load("outputs/best_resnet.pth", map_location=device)   # 讀取 .pth 檔
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()    # 關閉只在訓練時需要的隨機機制，確保每次推論同張圖片的結果都相同
    print(f"Model loaded successfully on {device}")
except FileNotFoundError:
    print("Warning: outputs/best_resnet.pth not found. Please ensure the model file exists.")

# 定義 API 回傳的格式
class PredictionResponse(BaseModel):
    class_id: int
    label: str
    confidence: float

# --- 3. API 接口 ---
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # 檢查檔案格式
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")

    try:
        # 讀取並轉換影像
        image_bytes = await file.read()     # 從上傳的請求中提取原始二進位
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # 將二進位轉換成 PIL 格式，並強制轉為 RGB 3 通道（防止用戶上傳帶透明層的 RGBA 圖片）
        
        # 預處理 (增加 Batch 維度)
        # 圖片轉換後維度是(3, 224, 224)，模型預期的是 Batch維度是(1, 3, 224, 224)，這裡會改成Batch的維度
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # 停用梯度追蹤增加運算速度
        with torch.no_grad():
            output = model(input_tensor)
            # 使用 Softmax 取得機率
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(probabilities, 0)
            
        return PredictionResponse(
            class_id=int(pred),
            label=CLASS_NAMES[int(pred)],
            confidence=round(float(conf), 4)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device)}