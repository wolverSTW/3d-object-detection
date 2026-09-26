import math


class DistanceMetrics:
    """Distance estimation metrics: MAE and RMSE for monocular depth/distance prediction."""

    def __init__(self):
        pass

    @staticmethod
    def mae(predictions, targets):
        if not predictions or not targets:
            return 0.0
        n = min(len(predictions), len(targets))
        errors = [abs(float(predictions[i]) - float(targets[i])) for i in range(n)]
        return sum(errors) / n

    @staticmethod
    def rmse(predictions, targets):
        if not predictions or not targets:
            return 0.0
        n = min(len(predictions), len(targets))
        errs = [(float(predictions[i]) - float(targets[i])) ** 2 for i in range(n)]
        return math.sqrt(sum(errs) / n)
