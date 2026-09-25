import unittest
import torch
from src.losses.total_loss import YOLO3DTotalLoss

class TestYOLO3DLoss(unittest.TestCase):

    def test_loss_computation_and_backward(self):
        loss_fn = YOLO3DTotalLoss()
        
        # Dummy prediction tensors
        preds = {
            "cls_logits": torch.randn(2, 3, 24, 78, requires_grad=True),
            "bbox2d": torch.randn(2, 4, 24, 78, requires_grad=True),
            "dimensions": torch.randn(2, 3, 24, 78, requires_grad=True),
            "location": torch.randn(2, 3, 24, 78, requires_grad=True),
            "rotation_y": torch.randn(2, 1, 24, 78, requires_grad=True)
        }
        
        # Dummy target tensors
        targets = {
            "cls": torch.zeros(2, 3, 24, 78),
            "bbox2d": torch.randn(2, 4, 24, 78),
            "dim3d": torch.randn(2, 3, 24, 78),
            "loc3d": torch.randn(2, 3, 24, 78),
            "rot": torch.randn(2, 1, 24, 78)
        }
        
        total_loss, loss_dict = loss_fn(preds, targets)
        
        self.assertIn("total_loss", loss_dict)
        self.assertGreater(total_loss.item(), 0)
        
        # Test backward pass capability
        total_loss.backward()
        self.assertIsNotNone(preds["cls_logits"].grad)

if __name__ == '__main__':
    unittest.main()
