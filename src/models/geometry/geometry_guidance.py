import torch
import torch.nn as nn


class GeometryGuidance(nn.Module):
    """Geometry-aware feature refinement block for a monocular 3D model."""

    def __init__(self, enabled=True, guidance_type='camera_geometry', depth_channels=128):
        super().__init__()
        self.enabled = enabled
        self.guidance_type = guidance_type
        self.depth_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(depth_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    @staticmethod
    def _estimate_depth(calibration=None):
        if isinstance(calibration, dict):
            for key in ('P2', 'P3', 'P0'):
                if key in calibration and isinstance(calibration[key], (list, tuple)) and len(calibration[key]) >= 2:
                    return float(calibration[key][0][0]) / 100.0
        return 8.0

    def forward(self, features, calibration=None):
        if not self.enabled:
            return features

        feat = features.get('feature_map', features.get('feat')) if isinstance(features, dict) else features
        if feat is None:
            return features

        depth = self.depth_head(feat).squeeze(-1).squeeze(-1)
        depth = torch.clamp(depth, min=1.0, max=100.0)

        if isinstance(features, dict):
            features = dict(features)
            features['geometry_guidance'] = {
                'depth': depth.detach().mean().item() if depth.numel() > 0 else self._estimate_depth(calibration),
                'projected_2d': (0.0, 0.0),
                'type': self.guidance_type,
            }
            features['depth_prior'] = depth
            return features

        return {'feature_map': feat, 'feat': feat, 'depth_prior': depth}
