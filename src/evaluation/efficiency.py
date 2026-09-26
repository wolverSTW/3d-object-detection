class EfficiencyMetrics:
    """Minimal efficiency metrics for model comparison."""

    def __init__(self):
        self.params = 0.0
        self.gflops = 0.0
        self.inference_time_ms = 0.0
        self.fps = 0.0

    def summarize(self):
        return {
            'Params (M)': float(self.params),
            'GFLOPs': float(self.gflops),
            'Time (ms)': float(self.inference_time_ms),
            'FPS': float(self.fps),
        }
