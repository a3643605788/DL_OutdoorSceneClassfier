# tools/print_model_summary.py
from __future__ import annotations

import torch
from torchinfo import summary

from src.model import CNNBaseline

def main():
    model = CNNBaseline(num_classes=6)
    # 等同於輸入一張 224x224 RGB 圖
    summary(model, input_size=(1, 3, 224, 224), col_names=("input_size", "output_size", "num_params", "kernel_size"))

if __name__ == "__main__":
    main()