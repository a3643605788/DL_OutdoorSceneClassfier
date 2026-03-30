# --- 第一階段：Builder ---
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

# 安裝 CPU 版 Torch 節省空間
RUN pip install --no-cache-dir --user \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# --- 第二階段：Final ---
FROM python:3.12-slim

WORKDIR /app

# 複製編譯好的套件
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 複製程式碼與模型
COPY api/ ./api/
COPY src/ ./src/
COPY outputs/best_resnet.pth ./outputs/best_resnet.pth

# Cloud Run 預設監聽 8080，但這行只是聲明
EXPOSE 8080

# 關鍵：使用 sh -c 來讀取環境變數 $PORT，不要寫死 8000！
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]