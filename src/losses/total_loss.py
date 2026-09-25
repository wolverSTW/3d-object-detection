import torch
import torch.nn as nn
from src.losses.detection_loss import FocalLoss, SmoothL1Loss3D, OrientationLoss

class YOLO3DTotalLoss(nn.Module):
    """
    Aggregates Focal Loss, 2D BBox Loss, 3D Dimension Loss, 3D Location Loss, and Orientation Loss.
    """
    def __init__(self, weights=None):
        super(YOLO3DTotalLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.l1_loss = SmoothL1Loss3D()
        self.rot_loss = OrientationLoss()
        
        # Default loss weights
        if weights is None:
            self.weights = {
                "cls": 1.0,
                "bbox2d": 1.0,
                "dim3d": 2.0,
                "loc3d": 2.0,
                "rot": 1.0
            }
        else:
            self.weights = weights

    def forward(self, predictions, targets_dict):
        loss_cls = self.focal_loss(predictions["cls_logits"], targets_dict["cls"])
        loss_bbox2d = self.l1_loss(predictions["bbox2d"], targets_dict["bbox2d"])
        loss_dim3d = self.l1_loss(predictions["dimensions"], targets_dict["dim3d"])
        loss_loc3d = self.l1_loss(predictions["location"], targets_dict["loc3d"])
        loss_rot = self.rot_loss(predictions["rotation_y"], targets_dict["rot"])

        total_loss = (
            self.weights["cls"] * loss_cls +
            self.weights["bbox2d"] * loss_bbox2d +
            self.weights["dim3d"] * loss_dim3d +
            self.weights["loc3d"] * loss_loc3d +
            self.weights["rot"] * loss_rot
        )

        return total_loss, {
            "loss_cls": loss_cls.item(),
            "loss_bbox2d": loss_bbox2d.item(),
            "loss_dim3d": loss_dim3d.item(),
            "loss_loc3d": loss_loc3d.item(),
            "loss_rot": loss_rot.item(),
            "total_loss": total_loss.item()
        }
