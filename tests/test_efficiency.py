import unittest

from src.evaluation.efficiency import EfficiencyMetrics


class TestEfficiencyMetrics(unittest.TestCase):
    def test_efficiency_summary(self):
        metrics = EfficiencyMetrics()
        metrics.params = 6.5
        metrics.gflops = 10.2
        metrics.inference_time_ms = 12.3
        metrics.fps = 81.0
        summary = metrics.summarize()
        self.assertIn('Params (M)', summary)
        self.assertIn('GFLOPs', summary)
        self.assertIn('Time (ms)', summary)
        self.assertIn('FPS', summary)


if __name__ == '__main__':
    unittest.main()
