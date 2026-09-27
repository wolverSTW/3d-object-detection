class TotalLoss:
    """Composite loss for multi-task 3D object detection training."""

    def __init__(self):
        self.components = {
            'cls': 0.5,
            'box': 1.0,
            'dim': 1.5,
            'orientation': 0.5,
            'location': 1.0,
            'depth': 2.0,
            'uncertainty': 0.2,
        }

    def __call__(self, outputs, targets):
        total = 0.0
        for name, weight in self.components.items():
            key = {
                'cls': 'class_logits',
                'box': 'box_pred',
                'dim': 'dim_pred',
                'orientation': 'orientation_pred',
                'location': 'location_pred',
                'depth': 'depth_pred',
                'uncertainty': 'uncertainty_pred',
            }[name]
            if key in outputs:
                pred = outputs[key]
                target_key = {
                    'cls': 'cls_target',
                    'box': 'box_target',
                    'dim': 'dim_target',
                    'orientation': 'orientation_target',
                    'location': 'location_target',
                    'depth': 'depth_target',
                    'uncertainty': 'uncertainty_target',
                }[name]
                target = targets.get(target_key, pred)
                if isinstance(pred, (list, tuple)) and isinstance(target, (list, tuple)):
                    total += weight * sum((p - t) ** 2 for p, t in zip(pred, target)) / max(len(pred), 1)
                elif pred is not None and target is not None:
                    total += weight * (float(pred) - float(target)) ** 2
        return total
