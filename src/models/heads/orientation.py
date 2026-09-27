class OrientationHead:
    """Predicts orientation angle for the 3D box."""

    def __init__(self, in_channels=256):
        self.in_channels = in_channels

    def forward(self, features):
        return {
            'orientation_pred': [0.0],
        }
