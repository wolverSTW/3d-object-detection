import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """Multi-task detection head for class, box, dimension, depth, location, and orientation."""

    def __init__(self, in_channels=256, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )
        self.box_head = nn.Linear(in_channels, 4)
        self.dim_head = nn.Linear(in_channels, 3)
        self.depth_head = nn.Linear(in_channels, 1)
        self.location_head = nn.Linear(in_channels, 3)
        self.orientation_head = nn.Linear(in_channels, 1)
        self.uncertainty_head = nn.Linear(in_channels, 1)

    def forward(self, features):
        feat = features.get('feat') if isinstance(features, dict) else features
        if feat is None:
            feat = features.get('feature_map') if isinstance(features, dict) else features
        if feat is None:
            feat = torch.zeros(1, 256, 1, 1, device=next(self.parameters()).device)
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        if feat.dim() == 4:
            pooled = nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        else:
            pooled = feat.flatten(1)

        return {
            'class_logits': self.classifier(feat).squeeze(-1).squeeze(-1) if feat.dim() == 4 else self.classifier(feat),
            'box_pred': self.box_head(pooled),
            'dim_pred': self.dim_head(pooled),
            'depth_pred': self.depth_head(pooled),
            'location_pred': self.location_head(pooled),
            'orientation_pred': self.orientation_head(pooled),
            'uncertainty_pred': self.uncertainty_head(pooled),
        }
