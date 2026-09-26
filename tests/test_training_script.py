import os
import subprocess
import sys
import unittest

from tools.download.create_mock_kitti import create_mock_kitti


class TestTrainingScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        create_mock_kitti()

    def test_training_script_runs(self):
        result = subprocess.run(
            [sys.executable, 'scripts/train.py', '--epochs', '1', '--model', 'baseline'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('Running baseline training', result.stdout)

    def test_build_dataloader_uses_kitti_split(self):
        from scripts.train import build_dataloader

        dataloader = build_dataloader('data/KITTI/raw', 'train.txt', limit=2)
        self.assertEqual(len(dataloader), 2)
        self.assertIn('image_path', dataloader[0])
        self.assertTrue(os.path.exists(dataloader[0]['image_path']))


if __name__ == '__main__':
    unittest.main()
