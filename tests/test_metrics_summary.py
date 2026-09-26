import unittest

from src.evaluation.metrics_summary import MetricsSummary


class TestMetricsSummary(unittest.TestCase):
    def test_as_table_row(self):
        summary = MetricsSummary({'MAE_distance': 1.2, 'RMSE_distance': 2.0})
        row = summary.as_table_row('baseline')
        self.assertEqual(row['model'], 'baseline')
        self.assertEqual(row['MAE_distance'], 1.2)

    def test_save_csv(self):
        summary = MetricsSummary({'MAE_distance': 1.2, 'RMSE_distance': 2.0})
        path = summary.save_csv('outputs/test_metrics_summary.csv', 'baseline')
        self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
