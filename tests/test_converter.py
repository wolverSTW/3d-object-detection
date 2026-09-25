import unittest
import torch
import numpy as np
from src.data.kitti_parser import KITTILabel
from src.data.converter import TargetConverter

class TestTargetConverter(unittest.TestCase):

    def test_convert_labels_to_target(self):
        dummy_label_line = "Car 0.00 0 -1.57 100.0 200.0 300.0 400.0 1.5 1.6 3.5 1.0 2.0 5.0 -1.57"
        label = KITTILabel(dummy_label_line)
        
        converter = TargetConverter()
        img_shape = (800, 1000) # (H, W)
        
        target_tensor = converter.convert_labels_to_target([label], img_shape)
        
        self.assertEqual(target_tensor.shape, (1, 13))
        # Class id check for Car -> 0
        self.assertEqual(target_tensor[0, 0].item(), 0)
        # Normalized 2D BBox check: x1 = 100/1000 = 0.1, y1 = 200/800 = 0.25
        self.assertAlmostEqual(target_tensor[0, 1].item(), 0.1)
        self.assertAlmostEqual(target_tensor[0, 2].item(), 0.25)
        # Check metric distance Euclidean (sqrt(1^2 + 2^2 + 5^2) = sqrt(30) ~ 5.4772)
        expected_dist = np.sqrt(1**2 + 2**2 + 5**2)
        self.assertAlmostEqual(target_tensor[0, 12].item(), expected_dist, places=3)

if __name__ == '__main__':
    unittest.main()
