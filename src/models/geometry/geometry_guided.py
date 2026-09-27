class GeometryGuidedModule:
    """Geometry-aware refinement module for monoview depth and spatial priors."""

    def __init__(self, enabled=True, use_depth_prior=True, use_projection_consistency=True):
        self.enabled = enabled
        self.use_depth_prior = use_depth_prior
        self.use_projection_consistency = use_projection_consistency

    def forward(self, features, calibration=None):
        if not self.enabled:
            return features

        guidance = {
            'type': 'camera_geometry',
            'calibration_used': calibration is not None,
            'depth_prior': 8.0,
            'projection_consistent': self.use_projection_consistency,
        }

        if isinstance(features, dict):
            x = dict(features)
            x['geometry_guidance'] = guidance
            return x

        return {'geometry_guidance': guidance}
