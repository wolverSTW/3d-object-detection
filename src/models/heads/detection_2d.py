class Detection2DHead:
    """Outputs a 2D detection prediction block for multi-task training."""

    def __init__(self, in_channels=256, num_classes=3):
        self.in_channels = in_channels
        self.num_classes = num_classes

    def forward(self, features):
        return {
            'class_logits': [1.0, 0.0, 0.0],
            'bbox_2d': [0.0, 0.0, 100.0, 200.0],
            'confidence': [0.8],
        }
