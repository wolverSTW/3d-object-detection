class DetectionMetrics:
    """Placeholder AP-style detection metrics for object detection evaluation."""

    def __init__(self):
        self.metrics = {
            'AP3D_easy': 0.0,
            'AP3D_moderate': 0.0,
            'AP3D_hard': 0.0,
            'APBEV_easy': 0.0,
            'APBEV_moderate': 0.0,
            'APBEV_hard': 0.0,
        }

    def evaluate(self, predictions, ground_truth):
        return dict(self.metrics)
