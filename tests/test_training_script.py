import subprocess
import sys
import unittest


class TestTrainingScript(unittest.TestCase):
    def test_training_script_runs(self):
        result = subprocess.run(
            [sys.executable, 'scripts/train.py', '--epochs', '1', '--model', 'baseline'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('Running baseline training', result.stdout)


if __name__ == '__main__':
    unittest.main()
