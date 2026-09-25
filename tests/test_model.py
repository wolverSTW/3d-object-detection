import unittest
import torch
from src.models.baseline import YOLO3DBaseline

class TestYOLO3DBaseline(unittest.TestCase):

    def test_baseline_forward_pass(self):
        # Dummy batch of 2 images with KITTI resolution (375, 1242)
        dummy_input = torch.randn(2, 3, 375, 1242)
        model = YOLO3DBaseline(num_classes=3)
        
        output = model(dummy_input)
        
        self.assertIn("cls_logits", output)
        self.assertIn("bbox2d", output)
        self.assertIn("dimensions", output)
        self.assertIn("location", output)
        self.assertIn("rotation_y", output)
        
        # Verify shape after 16x stride backbone reduction
        expected_h = 375 // 16 # 23
        expected_w = 1242 // 16 # 77
        
        self.assertEqual(output["cls_logits"].shape, (2, 3, expected_h, expected_w))
        self.assertEqual(output["bbox2d"].shape, (2, 4, expected_h, expected_w))
        self.assertEqual(output["dimensions"].shape, (2, 3, expected_h, expected_w))
        self.assertEqual(output["location"].shape, (2, 3, expected_h, expected_w))
        self.assertEqual(output["rotation_y"].shape, (2, 1, expected_h, expected_w))

if __name__ == '__main__':
    unittest.main()
