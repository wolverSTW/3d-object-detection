class GeometryGuidance:
    """Geometry-aware feature refinement block for a monocular 3D model."""

    def __init__(self, enabled=True, guidance_type='camera_geometry'):
        self.enabled = enabled
        self.guidance_type = guidance_type

    def forward(self, features, calibration=None):
        if not self.enabled:
            return features

        if isinstance(features, dict):
            features = dict(features)
            features['geometry_guidance'] = {
                'type': self.guidance_type,
                'camera_calibration_used': calibration is not None,
                'enabled': self.enabled,
            }
        return features
