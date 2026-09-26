import unittest

from src.models.geometry.geometry_features import GeometryFeatures
from src.models.geometry.csamm import CSAMM
from src.models.heads.regression_3d_head import Regression3DHead


class TestGeometryModules(unittest.TestCase):
    def test_geometry_features_compute_depth_prior(self):
        geometry = GeometryFeatures()
        prior = geometry.compute_depth_prior([10, 20, 100, 200], [1.5, 1.6, 3.8], [0.0, 0.0, 10.0])
        self.assertIn('depth', prior)
        self.assertIn('projected_2d', prior)
        self.assertGreater(prior['depth'], 0.0)

    def test_csamm_runs(self):
        module = CSAMM(channels=256)
        output = module.forward({'x': [1, 2, 3]})
        self.assertIsNotNone(output)

    def test_regression_3d_head_runs(self):
        head = Regression3DHead(in_channels=256, num_classes=3)
        outputs = head.forward({'geometry_prior': {'depth': 10.0, 'projected_2d': (100.0, 200.0)}})
        self.assertIn('box_pred', outputs)
        self.assertIn('dim_pred', outputs)
        self.assertIn('depth_pred', outputs)


if __name__ == '__main__':
    unittest.main()
