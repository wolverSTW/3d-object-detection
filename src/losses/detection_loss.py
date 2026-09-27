import math

import torch


class DetectionLoss:
    """Simple regression-style loss for box, class, and 3D dimension terms."""

    def __init__(self, lambda_box=1.0, lambda_cls=0.5, lambda_3d=1.5):
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_3d = lambda_3d

    @staticmethod
    def _to_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.float()
        if isinstance(value, (list, tuple)):
            return torch.tensor(value, dtype=torch.float32)
        return torch.tensor(float(value), dtype=torch.float32)

    def _safe_list(self, value):
        if isinstance(value, (list, tuple)):
            return value
        if isinstance(value, dict):
            return list(value.values())
        return [value]

    def _l2_error(self, pred, target):
        pred_t = self._to_tensor(pred)
        target_t = self._to_tensor(target)
        if pred_t.dim() == 0:
            pred_t = pred_t.unsqueeze(0)
        if target_t.dim() == 0:
            target_t = target_t.unsqueeze(0)
        size = max(pred_t.numel(), target_t.numel())
        if pred_t.numel() < size:
            pred_t = pred_t.repeat(size // pred_t.numel())
        if target_t.numel() < size:
            target_t = target_t.repeat(size // target_t.numel())
        return torch.mean((pred_t - target_t) ** 2)

    def __call__(self, outputs, targets):
        box_pred = outputs.get('box_pred', [0.0])
        box_target = targets.get('box_target', [0.0])
        cls_pred = outputs.get('cls_pred', [0.0])
        cls_target = targets.get('cls_target', [0.0])
        dim_pred = outputs.get('dim_pred', [0.0])
        dim_target = targets.get('dim_target', [0.0])

        box_loss = self._l2_error(box_pred, box_target)
        cls_pred_t = self._to_tensor(self._safe_list(cls_pred))
        cls_target_t = self._to_tensor(self._safe_list(cls_target))
        class_loss = torch.mean((cls_pred_t - cls_target_t) ** 2)
        dim_loss = self._l2_error(dim_pred, dim_target)

        total_loss = self.lambda_box * box_loss + self.lambda_cls * class_loss + self.lambda_3d * dim_loss
        return total_loss
