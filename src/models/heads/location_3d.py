class Location3DHead:
    """Predicts 3D center coordinates in camera space."""

    def __init__(self, in_channels=256):
        self.in_channels = in_channels

    def forward(self, features):
        return {
            'location_pred': [0.0, 0.0, 8.0],
        }
