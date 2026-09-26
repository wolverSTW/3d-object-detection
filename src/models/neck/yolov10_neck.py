class YOLOv10Neck:
    """Placeholder neck for feature aggregation before the 3D detection head."""

    def __init__(self, in_channels=256, out_channels=(256, 512, 512)):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, features):
        return features
