import json
import unittest
from pathlib import Path

from src.evaluation.efficiency import EfficiencyMetrics
from src.evaluation.kitti_eval import KittiEvaluator
from src.visualization.reporting import ExperimentReporter


class TestExperimentRun(unittest.TestCase):
    def test_experiment_run_creates_summary_artifacts(self):
        evaluator = KittiEvaluator()
        metrics = evaluator.evaluate(
            [{'distance': 10.0, 'AP3D_easy': 0.70}, {'distance': 12.0, 'AP3D_easy': 0.68}],
            [{'distance': 9.0}, {'distance': 13.0}],
        )
        efficiency = EfficiencyMetrics()
        efficiency.params = 8.2
        efficiency.gflops = 3.4
        efficiency.inference_time_ms = 11.5
        efficiency.fps = 87.0

        reporter = ExperimentReporter(output_dir='outputs/test_experiment_run')
        summary = reporter.summarize('baseline', metrics, efficiency=efficiency.summarize())
        path = reporter.save_summary(summary, filename='experiment_summary.json')

        self.assertIn('efficiency', summary)
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(data['model'], 'baseline')

    def test_experiment_runner_comparison_summary(self):
        from src.training.experiment_runner import ExperimentRunner

        runner = ExperimentRunner(output_dir='outputs/test_experiment_compare')
        results = runner.compare_runs(
            [
                {'model_name': 'baseline', 'predictions': [{'distance': 10.0}, {'distance': 12.0}], 'ground_truth': [{'distance': 9.0}, {'distance': 13.0}]},
                {'model_name': 'geometry', 'predictions': [{'distance': 9.5}, {'distance': 11.5}], 'ground_truth': [{'distance': 9.0}, {'distance': 13.0}]},
            ]
        )

        self.assertIn('comparison', results)
        self.assertIn('baseline', results['comparison'])
        self.assertIn('geometry', results['comparison'])
        self.assertTrue(Path(results['comparison_path']).exists())


if __name__ == '__main__':
    unittest.main()
