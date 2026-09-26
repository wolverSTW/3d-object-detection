class DetectionHead:
    """Placeholder head for 2D/3D detection output generation."""

    def __init__(self, in_channels=256, num_classes=3, reg_dims=7):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_dims = reg_dims

    def forward(self, features):
        return {
            'class_logits': None,
            'box_predictions': None,
            'dim_predictions': None,
            'depth_predictions': None,
        }
