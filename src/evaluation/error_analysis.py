class ErrorAnalysis:
    """Qualitative and quantitative error analysis utilities for thesis reporting."""

    def __init__(self):
        pass

    @staticmethod
    def summarize_errors(predictions, ground_truth):
        if not predictions or not ground_truth:
            return {'total_errors': 0, 'avg_error': 0.0}

        total = 0
        for pred, gt in zip(predictions, ground_truth):
            pred_val = pred.get('distance', 0.0) if isinstance(pred, dict) else pred
            gt_val = gt.get('distance', 0.0) if isinstance(gt, dict) else gt
            total += abs(float(pred_val) - float(gt_val))

        avg_error = total / min(len(predictions), len(ground_truth))
        return {'total_errors': total, 'avg_error': avg_error}
