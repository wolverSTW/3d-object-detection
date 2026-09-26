import json
from pathlib import Path


class ExperimentReporter:
    """Simple report generator for empirical experiment summaries."""

    def __init__(self, output_dir='outputs/metrics'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_summary(self, metrics, filename='experiment_summary.json'):
        path = self.output_dir / filename
        with path.open('w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        return path

    def summarize(self, model_name, metrics, efficiency=None, status='completed'):
        summary = {
            'model': model_name,
            'metrics': metrics,
            'status': status,
        }
        if efficiency is not None:
            summary['efficiency'] = efficiency
        return summary
