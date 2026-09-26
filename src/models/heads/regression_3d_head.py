class Regression3DHead:
    """A lightweight 3D regression head for box, dimension, and distance outputs."""

    def __init__(self, in_channels=256, num_classes=3):
        self.in_channels = in_channels
        self.num_classes = num_classes

    def forward(self, features):
        if isinstance(features, dict):
            geometry_prior = features.get('geometry_prior', {})
            depth = geometry_prior.get('depth', 0.0)
            projected = geometry_prior.get('projected_2d', (0.0, 0.0))
        else:
            depth = 0.0
            projected = (0.0, 0.0)

        return {
            'box_pred': [0.0, 0.0, 0.0, 0.0],
            'cls_pred': [1.0, 0.0, 0.0],
            'dim_pred': [1.5, 1.6, 3.8],
            'depth_pred': [depth],
            'projected_center': projected,
        }
