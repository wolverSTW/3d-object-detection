import math


class DetectionLoss:
    """Simple regression-style loss for box, class, and 3D dimension terms."""

    def __init__(self, lambda_box=1.0, lambda_cls=0.5, lambda_3d=1.5):
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_3d = lambda_3d

    def _safe_list(self, value):
        if isinstance(value, (list, tuple)):
            return value
        if isinstance(value, dict):
            return list(value.values())
        return [value]

    def _l2_error(self, pred, target):
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        if not isinstance(target, (list, tuple)):
            target = [target]
        pred = [float(x) for x in pred]
        target = [float(x) for x in target]
        return sum((p - t) ** 2 for p, t in zip(pred, target)) / max(len(pred), 1)

    def __call__(self, outputs, targets):
        box_pred = outputs.get('box_pred', [0.0])
        box_target = targets.get('box_target', [0.0])
        cls_pred = outputs.get('cls_pred', [0.0])
        cls_target = targets.get('cls_target', [0.0])
        dim_pred = outputs.get('dim_pred', [0.0])
        dim_target = targets.get('dim_target', [0.0])

        box_loss = self._l2_error(box_pred, box_target)
        class_loss = sum((float(p) - float(t)) ** 2 for p, t in zip(self._safe_list(cls_pred), self._safe_list(cls_target))) / max(len(self._safe_list(cls_pred)), 1)
        dim_loss = self._l2_error(dim_pred, dim_target)

        total_loss = self.lambda_box * box_loss + self.lambda_cls * class_loss + self.lambda_3d * dim_loss
        return float(total_loss)
