# --- 第一階段：Builder (安裝依賴) ---
FROM python:3.12-slim AS builder

WORKDIR /app

# 僅複製 requirements 以利用 Docker 快取機制
COPY requirements.txt .

# 安裝套件到特定的資料夾
RUN pip install --no-cache-dir --user -r requirements.txt

# --- 第二階段：Final (運行服務) ---
FROM python:3.12-slim

WORKDIR /app

# 從第一階段只把安裝好的套件搬過來
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 複製必要的原始碼與模型權重 (排除測試與筆記本)
COPY src/ ./src/
COPY api/ ./api/
COPY outputs/best_resnet.pth ./outputs/best_resnet.pth

# 宣告對外埠號
EXPOSE 8000

# 啟動命令
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]