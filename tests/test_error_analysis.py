import unittest

from src.evaluation.error_analysis import ErrorAnalysis


class TestErrorAnalysis(unittest.TestCase):
    def test_summarize_errors(self):
        predictions = [{'distance': 10.0}, {'distance': 12.0}]
        ground_truth = [{'distance': 9.0}, {'distance': 15.0}]
        result = ErrorAnalysis.summarize_errors(predictions, ground_truth)
        self.assertIn('total_errors', result)
        self.assertIn('avg_error', result)
        self.assertGreaterEqual(result['avg_error'], 0.0)


if __name__ == '__main__':
    unittest.main()
