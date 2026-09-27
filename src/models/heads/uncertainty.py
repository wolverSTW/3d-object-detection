class UncertaintyHead:
    """Predicts uncertainty for depth and classification outputs."""

    def __init__(self, in_channels=256):
        self.in_channels = in_channels

    def forward(self, features):
        return {
            'uncertainty_pred': [0.1],
        }
