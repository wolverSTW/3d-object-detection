class GeometryLoss:
    """Geometry-aware regression loss using depth and projection consistency terms."""

    def __init__(self, lambda_geo=0.8):
        self.lambda_geo = lambda_geo

    def _l2_error(self, pred, target):
        pred_list = pred if isinstance(pred, (list, tuple)) else [pred]
        target_list = target if isinstance(target, (list, tuple)) else [target]
        pred_list = [float(x) for x in pred_list]
        target_list = [float(x) for x in target_list]
        return sum((p - t) ** 2 for p, t in zip(pred_list, target_list)) / max(len(pred_list), 1)

    def __call__(self, outputs, targets):
        depth_pred = outputs.get('depth_pred', [0.0])
        depth_target = targets.get('depth_target', [0.0])
        proj_pred = outputs.get('proj_pred', [0.0])
        proj_target = targets.get('proj_target', [0.0])

        depth_error = self._l2_error(depth_pred, depth_target)
        proj_error = self._l2_error(proj_pred, proj_target)
        total_loss = self.lambda_geo * (depth_error + proj_error)
        return float(total_loss)
