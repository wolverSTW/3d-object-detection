import math

import torch
import torch.nn.functional as F


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

    @staticmethod
    def _as_float_tensor(value):
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        return tensor.float()

    @staticmethod
    def _as_long_tensor(value):
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return tensor.long()

    def _l2_error(self, pred, target):
        pred_t = self._as_float_tensor(pred)
        target_t = self._as_float_tensor(target)
        if pred_t.dim() == 0:
            pred_t = pred_t.unsqueeze(0)
        if target_t.dim() == 0:
            target_t = target_t.unsqueeze(0)
        if pred_t.shape != target_t.shape:
            try:
                target_t = target_t.expand_as(pred_t)
            except RuntimeError:
                pass
        return torch.mean((pred_t - target_t) ** 2)

    def __call__(self, outputs, targets):
        box_pred = outputs.get('box_pred', [0.0])
        box_target = targets.get('box_target', [0.0])
        cls_pred = outputs.get('class_logits', outputs.get('cls_pred', [0.0]))
        cls_target = targets.get('cls_target', [0.0])
        dim_pred = outputs.get('dim_pred', [0.0])
        dim_target = targets.get('dim_target', [0.0])

        box_loss = self._l2_error(box_pred, box_target)
        cls_pred_t = self._as_float_tensor(cls_pred)
        cls_target_t = self._as_long_tensor(cls_target)
        if cls_pred_t.dim() == 1:
            cls_pred_t = cls_pred_t.unsqueeze(0)
        if cls_target_t.dim() == 0:
            cls_target_t = cls_target_t.unsqueeze(0)
        if cls_pred_t.shape[0] != cls_target_t.shape[0]:
            cls_pred_t = cls_pred_t[: cls_target_t.shape[0]]
        class_loss = F.cross_entropy(cls_pred_t, cls_target_t)
        dim_loss = self._l2_error(dim_pred, dim_target)

        total_loss = self.lambda_box * box_loss + self.lambda_cls * class_loss + self.lambda_3d * dim_loss
        return total_loss
