import subprocess
import sys
import unittest


class TestCompareExperimentsScript(unittest.TestCase):
    def test_compare_experiments_script_runs(self):
        result = subprocess.run(
            [sys.executable, 'scripts/compare_experiments.py', '--output-dir', 'experiments/test_compare'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('Comparison saved to:', result.stdout)


if __name__ == '__main__':
    unittest.main()
