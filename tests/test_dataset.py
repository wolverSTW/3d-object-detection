import os
import unittest

from src.data.dataset import KittiDataset


class TestKittiDataset(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join('data', 'KITTI', 'raw')
        self.dataset = KittiDataset(self.data_dir)

    def test_dataset_loads_samples(self):
        samples = self.dataset.load_split('train.txt')
        self.assertGreater(len(samples), 0)
        self.assertIn('image_path', samples[0])
        self.assertIn('label_path', samples[0])
        self.assertIn('calib_path', samples[0])

    def test_dataset_returns_annotations(self):
        sample = self.dataset[0]
        self.assertIn('annotations', sample)
        self.assertIsInstance(sample['annotations'], list)


if __name__ == '__main__':
    unittest.main()
