import unittest
import torch
import os
import tempfile
import shutil
from PIL import Image
from src.data.dataset import KITTIDataset, collate_fn

class TestKITTIDataset(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory structure for testing
        self.test_dir = tempfile.mkdtemp()
        self.split_dir = os.path.join(self.test_dir, "training")
        os.makedirs(os.path.join(self.split_dir, "image_2"))
        os.makedirs(os.path.join(self.split_dir, "label_2"))
        os.makedirs(os.path.join(self.split_dir, "calib"))

        # Create dummy image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(os.path.join(self.split_dir, "image_2", "000000.png"))

        # Create dummy label
        dummy_label = "Car 0.00 0 -1.57 10.0 10.0 50.0 50.0 1.5 1.6 3.5 1.0 2.0 5.0 -1.57\n"
        with open(os.path.join(self.split_dir, "label_2", "000000.txt"), "w") as f:
            f.write(dummy_label)

        # Create dummy calib
        dummy_calib = "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        with open(os.path.join(self.split_dir, "calib", "000000.txt"), "w") as f:
            f.write(dummy_calib)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_dataset_loading(self):
        dataset = KITTIDataset(root_dir=self.test_dir, split="training")
        self.assertEqual(len(dataset), 1)

        sample = dataset[0]
        self.assertIn("image", sample)
        self.assertIn("targets", sample)
        self.assertEqual(sample["targets"].shape[1], 13)

    def test_collate_fn(self):
        dataset = KITTIDataset(root_dir=self.test_dir, split="training")
        batch = [dataset[0], dataset[0]] # Batch of 2
        
        collated = collate_fn(batch)
        self.assertEqual(collated["images"].shape[0], 2)
        # Verify batch index added to targets (14 columns now: [batch_idx, cls_id, ...])
        self.assertEqual(collated["targets"].shape[1], 14)

if __name__ == '__main__':
    unittest.main()
