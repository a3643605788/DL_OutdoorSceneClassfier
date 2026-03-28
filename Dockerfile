# --- 第一階段：Builder ---
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

# 強制使用 CPU 版本的 torch 指令，並在安裝後清理快取
RUN pip install --no-cache-dir --user \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# --- 第二階段：Final ---
FROM python:3.12-slim

WORKDIR /app

# 只複製必要的套件
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 確保 .dockerignore 有生效，只複製需要的檔案
COPY api/ ./api/
COPY src/ ./src/
COPY outputs/best_resnet.pth ./outputs/best_resnet.pth

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]