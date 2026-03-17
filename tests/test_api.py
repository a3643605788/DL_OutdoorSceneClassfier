import pytest
from fastapi.testclient import TestClient
from api.main import app  # 引用你的 API 實例
import io
from PIL import Image

client = TestClient(app)

# 測試 1: 檢查健康檢查接口
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# 測試 2: 檢查正常的圖片推論
def test_predict_success():
    # 創建一張全黑的 224x224 虛擬圖片作為測試輸入
    file_name = "test_img.jpg"
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/predict",
        files={"file": (file_name, img_byte_arr, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "class_id" in data
    assert "label" in data
    assert 0 <= data["confidence"] <= 1.0

# 測試 3: 檢查錯誤的檔案格式 (防呆測試)
def test_predict_invalid_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    # 我們在 api/main.py 裡寫了檢查，所以應該回傳 400
    assert response.status_code == 400
    assert "detail" in response.json()

# 目前自動化測試涵蓋的內容
# 1.系統存活測試:呼叫 GET /health
# 2.核心業務邏輯測試:模擬一個使用者上傳一張「合格的圖片」（224x224 的紅色 JPEG）
# 3.強健性與安全性測試:模擬一個調皮的使用者上傳一個「文字檔 (.txt)」而非圖片