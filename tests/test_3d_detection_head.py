import unittest

from src.models.heads.three_d_detection_head import ThreeDDetectionHead


class TestThreeDDetectionHead(unittest.TestCase):
    def test_3d_detection_head_runs(self):
        head = ThreeDDetectionHead(in_channels=256, num_classes=3)
        output = head.forward({'geometry_guidance': {'camera_calibration_used': True}})
        self.assertIn('box_pred', output)
        self.assertIn('dim_pred', output)
        self.assertIn('depth_pred', output)


if __name__ == '__main__':
    unittest.main()
