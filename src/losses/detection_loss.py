import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in object detection.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()

class SmoothL1Loss3D(nn.Module):
    """
    Smooth L1 Loss for 3D Bounding Box attributes (Dimensions, Location, Distance).
    """
    def __init__(self, beta=1.0):
        super(SmoothL1Loss3D, self).__init__()
        self.beta = beta

    def forward(self, preds, targets):
        return F.smooth_l1_loss(preds, targets, beta=self.beta)

class OrientationLoss(nn.Module):
    """
    Cosine-based Orientation Loss for rotation angle (rotation_y).
    """
    def __init__(self):
        super(OrientationLoss, self).__init__()

    def forward(self, pred_rot, target_rot):
        # Loss = 1 - cos(pred - target)
        loss = 1.0 - torch.cos(pred_rot - target_rot)
        return loss.mean()
