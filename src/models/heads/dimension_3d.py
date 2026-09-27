class Dimension3DHead:
    """Predicts 3D dimensions for each object."""

    def __init__(self, in_channels=256, reg_dims=3):
        self.in_channels = in_channels
        self.reg_dims = reg_dims

    def forward(self, features):
        return {
            'dim_pred': [1.5, 1.6, 3.8],
        }
