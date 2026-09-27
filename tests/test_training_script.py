import os
import subprocess
import sys
import unittest

import torch

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

    def test_training_loss_changes_with_image_content(self):
        from PIL import Image

        from src.models.baseline import YOLO3DBaseline
        from src.training.trainer import Trainer

        model = YOLO3DBaseline(in_channels=3, num_classes=3, reg_dims=7)
        trainer = Trainer(model=model, optimizer=None, criterion=None, save_dir='experiments/test_loss_variation')

        dark = Image.new('RGB', (32, 32), color=(10, 20, 30))
        bright = Image.new('RGB', (64, 64), color=(200, 220, 240))

        loss_dark = trainer._compute_batch_loss({'image': dark, 'annotations': [], 'calib': {}})
        loss_bright = trainer._compute_batch_loss({'image': bright, 'annotations': [], 'calib': {}})

        self.assertNotAlmostEqual(loss_dark, loss_bright, places=6)

    def test_build_dataloader_uses_batch_and_worker_settings(self):
        from scripts.train import build_dataloader

        dataloader = build_dataloader('data/KITTI/raw', 'train.txt', limit=4, batch_size=2, num_workers=0)

        self.assertTrue(hasattr(dataloader, '__iter__'))
        self.assertEqual(dataloader.batch_size, 2)
        self.assertEqual(dataloader.num_workers, 0)

        batch = next(iter(dataloader))
        self.assertIn('image', batch)
        self.assertTrue(torch.is_tensor(batch['image']))
        self.assertEqual(batch['image'].shape[0], 2)


if __name__ == '__main__':
    unittest.main()
