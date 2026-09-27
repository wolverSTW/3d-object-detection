import torch
import torch.nn as nn


class YOLOv10Neck(nn.Module):
    """Simple feature aggregation neck used before the detection head."""

    def __init__(self, in_channels=128, out_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        if isinstance(features, dict):
            feat = features.get('feature_map', features.get('feat'))
            if feat is None:
                return features
            return {'feature_map': self.conv(feat), 'feat': self.conv(feat)}
        if isinstance(features, torch.Tensor):
            return {'feature_map': self.conv(features), 'feat': self.conv(features)}
        return {'feature_map': features, 'feat': features}
