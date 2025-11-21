"""
Simple CNN architecture for Model Version 1 (User 1)
Save this file as: models/model_v1.py
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A lightweight CNN suitable for small-to-medium image classification tasks.
    Architecture:
    - Conv(3→32) + BN + ReLU + MaxPool
    - Conv(32→64) + BN + ReLU + MaxPool
    - Conv(64→128) + BN + ReLU + AdaptiveAvgPool
    - FC: 128 → 64 → num_classes
    """

    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # downsample ×2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # downsample ×2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # adaptive pooling → output shape (128,1,1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(num_classes=2, device="cpu"):
    """
    Helper function used by training notebooks and train.py
    """
    model = SimpleCNN(num_classes=num_classes)
    return model.to(device)
