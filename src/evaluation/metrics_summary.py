from pathlib import Path


class MetricsSummary:
    """Utility for formatting scalar metrics for tables and reports."""

    def __init__(self, metrics=None):
        self.metrics = metrics or {}

    def as_table_row(self, model_name):
        row = {'model': model_name}
        row.update(self.metrics)
        return row

    def save_csv(self, path, model_name):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        row = self.as_table_row(model_name)
        lines = [','.join(row.keys()), ','.join(str(v) for v in row.values())]
        output_path.write_text('\n'.join(lines), encoding='utf-8')
        return output_path
