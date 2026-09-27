import torch
import torch.nn as nn


class YOLOv10Backbone(nn.Module):
    """Lightweight CNN backbone for monocular 3D detection."""

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if x is None:
            x = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.net(x)
