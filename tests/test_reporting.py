import unittest

from src.visualization.reporting import ExperimentReporter


class TestExperimentReporter(unittest.TestCase):
    def test_reporter_summarizes_metrics(self):
        reporter = ExperimentReporter(output_dir='outputs/test_report')
        metrics = {'MAE_distance': 1.2, 'RMSE_distance': 2.0}
        summary = reporter.summarize('baseline', metrics)
        self.assertEqual(summary['model'], 'baseline')
        self.assertIn('metrics', summary)

    def test_reporter_saves_summary(self):
        reporter = ExperimentReporter(output_dir='outputs/test_report')
        metrics = {'MAE_distance': 1.2, 'RMSE_distance': 2.0}
        path = reporter.save_summary(metrics, filename='summary.json')
        self.assertTrue(path.exists())

    def test_reporter_includes_efficiency_summary(self):
        reporter = ExperimentReporter(output_dir='outputs/test_report')
        metrics = {'MAE_distance': 1.2, 'RMSE_distance': 2.0}
        efficiency = {'Params (M)': 12.5, 'GFLOPs': 2.1, 'FPS': 30.0}
        summary = reporter.summarize('baseline', metrics, efficiency=efficiency)
        self.assertIn('efficiency', summary)
        self.assertEqual(summary['efficiency']['FPS'], 30.0)


if __name__ == '__main__':
    unittest.main()
