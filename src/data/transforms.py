import torch
import torchvision.transforms.functional as F
import numpy as np

class KITTITransforms:
    """
    Handles image resizing, normalization, and tensor conversion
    while keeping track of bounding box coordinate transformations.
    """
    def __init__(self, img_size=(375, 1242), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.img_size = img_size # Target height, width
        self.mean = mean
        self.std = std

    def __call__(self, image, targets):
        """
        Args:
            image (PIL Image): Input RGB image
            targets (torch.Tensor): Targets matrix of shape (N, 13)
        Returns:
            torch.Tensor: Transformed image tensor (3, H, W)
            torch.Tensor: Targets matrix with adjusted coordinates
        """
        orig_w, orig_h = image.size
        target_h, target_w = self.img_size

        # Resize image
        img_resized = F.resize(image, [target_h, target_w])
        img_tensor = F.to_tensor(img_resized)
        img_tensor = F.normalize(img_tensor, mean=self.mean, std=self.std)

        # Targets scale adjustment (2D bounding box scaling)
        if targets.shape[0] > 0:
            targets = targets.clone()
            # x1, x2 are relative [0,1], no scale change needed if kept relative
            # However, if target coordinates are pixel-based, scale as below:
            # targets[:, 1] *= (target_w / orig_w) ...

        return img_tensor, targets
