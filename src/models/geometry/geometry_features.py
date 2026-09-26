import math


class GeometryFeatures:
    """Constructs camera-geometry-aware features from 3D box and calibration priors."""

    def __init__(self, fx=707.0912, fy=707.0912, cx=601.8873, cy=183.1104):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def project_3d_to_2d(self, x, y, z):
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return u, v

    def compute_depth_prior(self, bbox_2d, dimensions_3d, location_3d):
        if not location_3d or len(location_3d) < 3:
            return 0.0
        x, y, z = location_3d
        depth = max(float(z), 1e-6)
        width = float(dimensions_3d[0]) if len(dimensions_3d) > 0 else 1.0
        height = float(dimensions_3d[1]) if len(dimensions_3d) > 1 else 1.0
        projected = self.project_3d_to_2d(x, y, z)
        return {
            'depth': depth,
            'projected_2d': projected,
            'scale_factor': depth / max(width * height, 1e-6),
        }

    def forward(self, features, calibration=None, object_info=None):
        if object_info is None:
            return features

        if isinstance(object_info, dict):
            location = object_info.get('location_3d', [0.0, 0.0, 0.0])
            dims = object_info.get('dimensions_3d', [1.0, 1.0, 1.0])
            bbox = object_info.get('bbox_2d', [0.0, 0.0, 1.0, 1.0])
            prior = self.compute_depth_prior(bbox, dims, location)
            if isinstance(features, dict):
                features = dict(features)
                features['geometry_prior'] = prior
                return features

        return features
