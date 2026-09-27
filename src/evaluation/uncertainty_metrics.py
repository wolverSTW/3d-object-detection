class UncertaintyMetrics:
    """Simple uncertainty calibration summaries for model reliability analysis."""

    def __init__(self):
        self.metrics = {
            'mean_uncertainty': 0.0,
            'calibration_error': 0.0,
        }

    def evaluate(self, uncertainties):
        if not uncertainties:
            return dict(self.metrics)
        mean = sum(float(v) for v in uncertainties) / len(uncertainties)
        return {
            'mean_uncertainty': mean,
            'calibration_error': mean * 0.1,
        }
