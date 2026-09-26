import unittest

from src.models.baseline import YOLO3DBaseline
from src.models.proposed import YOLO3DGeometryGuided


class TestModelSkeleton(unittest.TestCase):
    def test_baseline_model_runs(self):
        model = YOLO3DBaseline(in_channels=3, num_classes=3, reg_dims=7)
        outputs = model.forward(None)
        self.assertIsNotNone(outputs)

    def test_geometry_model_runs(self):
        model = YOLO3DGeometryGuided(in_channels=3, num_classes=3, reg_dims=7, enabled=True)
        outputs = model.forward(None)
        self.assertIsNotNone(outputs)


if __name__ == '__main__':
    unittest.main()
