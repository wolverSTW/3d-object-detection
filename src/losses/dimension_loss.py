class DimensionLoss:
    """Loss for 3D box dimension prediction."""

    def __call__(self, pred, target):
        pred = pred if isinstance(pred, (list, tuple)) else [pred]
        target = target if isinstance(target, (list, tuple)) else [target]
        if len(pred) < len(target):
            pred = pred + [0.0] * (len(target) - len(pred))
        elif len(target) < len(pred):
            target = target + [0.0] * (len(pred) - len(target))
        return sum((p - t) ** 2 for p, t in zip(pred, target)) / max(len(pred), 1)
