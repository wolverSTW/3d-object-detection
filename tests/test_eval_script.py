import subprocess
import sys
import unittest


class TestEvaluationScript(unittest.TestCase):
    def test_evaluation_script_runs_and_reports_metrics(self):
        result = subprocess.run(
            [sys.executable, 'scripts/evaluate.py', '--config', 'configs/experiments/baseline.yaml'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('MAE_distance', result.stdout)


if __name__ == '__main__':
    unittest.main()
