# src/model.py
from __future__ import annotations

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    A simple CNN baseline for image classification.
    Input:  (N, 3, 224, 224)
    Output: (N, num_classes)
    """

    def __init__(self, num_classes: int = 6, dropout: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # /2

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # /2

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # /2

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # /2
        )

        # 不依賴固定 feature map 大小，避免你之後改 img_size 會壞
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int = 6) -> nn.Module:
    return CNNBaseline(num_classes=num_classes)
