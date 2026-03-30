import io
import os
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
from pydantic import BaseModel
from src.model import ResNetTransfer

# 初始化 FastAPI
app = FastAPI(title="Outdoor Scene Classifier API")

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# 預處理流程
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 硬體檢測與模型載入
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetTransfer(num_classes=6, pretrained=False)

# 修正路徑：確保能讀取到容器內的 outputs 資料夾
MODEL_PATH = "outputs/best_resnet.pth"

try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
except FileNotFoundError:
    print(f"❌ Warning: {MODEL_PATH} not found. Inference will fail.")

class PredictionResponse(BaseModel):
    class_id: int
    label: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG supported.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
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

# 讓程式具備直接啟動的能力 (Cloud Run 友善)
if __name__ == "__main__":
    # 關鍵：Cloud Run 會透過環境變數傳入 PORT
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)