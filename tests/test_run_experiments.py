import subprocess
import sys
import unittest


class TestRunExperimentsScript(unittest.TestCase):
    def test_run_experiments_script_runs(self):
        result = subprocess.run(
            [sys.executable, 'scripts/run_experiments.py', '--output-dir', 'experiments/test_orchestration'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('baseline', result.stdout)
        self.assertIn('geometry', result.stdout)


if __name__ == '__main__':
    unittest.main()
