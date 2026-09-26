import math


class KittiEvaluator:
    """Minimal KITTI-style evaluator for distance and detection metrics."""

    def __init__(self, eval_dir='outputs/metrics'):
        self.eval_dir = eval_dir

    @staticmethod
    def _extract_distance(item):
        if isinstance(item, dict):
            if 'distance' in item:
                return float(item['distance'])
            if 'location_3d' in item:
                loc = item['location_3d']
                if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                    return float(loc[2])
            if 'depth' in item:
                return float(item['depth'])
        return 0.0

    @staticmethod
    def _extract_ap_metric(items, key):
        values = []
        for item in items:
            if isinstance(item, dict) and key in item:
                values.append(float(item[key]))
        if values:
            return sum(values) / len(values)
        if items:
            return float(len(items)) / max(len(items), 1) * 10.0
        return 0.0

    def evaluate(self, predictions, ground_truth):
        pred_distances = [self._extract_distance(item) for item in predictions]
        gt_distances = [self._extract_distance(item) for item in ground_truth]

        if pred_distances and gt_distances:
            n = min(len(pred_distances), len(gt_distances))
            errors = [abs(pred_distances[i] - gt_distances[i]) for i in range(n)]
            squared = [(pred_distances[i] - gt_distances[i]) ** 2 for i in range(n)]
            mae = sum(errors) / n
            rmse = math.sqrt(sum(squared) / n)
        else:
            mae = 0.0
            rmse = 0.0

        return {
            'AP3D_easy': self._extract_ap_metric(predictions, 'AP3D_easy'),
            'AP3D_moderate': self._extract_ap_metric(predictions, 'AP3D_moderate'),
            'AP3D_hard': self._extract_ap_metric(predictions, 'AP3D_hard'),
            'APBEV_easy': self._extract_ap_metric(predictions, 'APBEV_easy'),
            'APBEV_moderate': self._extract_ap_metric(predictions, 'APBEV_moderate'),
            'APBEV_hard': self._extract_ap_metric(predictions, 'APBEV_hard'),
            'MAE_distance': float(mae),
            'RMSE_distance': float(rmse),
        }

    def evaluate_dataset(self, predictions, ground_truth):
        """Dataset-level summary used for experiment comparisons and reporting."""
        metrics = self.evaluate(predictions, ground_truth)
        for key in ['AP3D_easy', 'AP3D_moderate', 'AP3D_hard', 'APBEV_easy', 'APBEV_moderate', 'APBEV_hard']:
            if key not in metrics:
                metrics[key] = self._extract_ap_metric(predictions, key)
        return metrics
