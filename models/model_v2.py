import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """
    Simple custom CNN for Cat vs Dog classification.

    Features:
    - 3 convolutional blocks with BatchNorm + ReLU + MaxPool
    - Dropout in the classifier to improve generalization
    - AdaptiveAvgPool2d to work with different input sizes (e.g., 224x224)
    """

    def __init__(self, num_classes: int = 2):
        super(CustomCNN, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global pooling to make it resolution-agnostic
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CustomCNN(num_classes=2)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)
