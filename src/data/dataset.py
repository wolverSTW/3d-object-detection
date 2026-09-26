import os
from typing import Dict, List, Optional, Tuple

from PIL import Image

from src.data.kitti_parser import KittiParser


class KittiDataset:
    """Minimal KITTI dataset wrapper used for local experiments and validation."""

    def __init__(self, data_dir: str, split_file: str = 'train.txt', transform=None):
        self.data_dir = data_dir
        self.parser = KittiParser(data_dir)
        self.split_file = split_file
        self.transform = transform
        self.samples = self.load_split(split_file)

    def load_split(self, split_file: str) -> List[Dict[str, str]]:
        split_path = os.path.join(os.path.dirname(self.data_dir), 'splits', split_file)
        if not os.path.exists(split_path):
            split_path = os.path.join(self.data_dir, 'splits', split_file)

        samples: List[Dict[str, str]] = []
        if not os.path.exists(split_path):
            return samples

        with open(split_path, 'r', encoding='utf-8') as f:
            for line in f:
                idx = line.strip()
                if not idx:
                    continue
                image_path = os.path.join(self.data_dir, 'image_2', f'{idx}.png')
                label_path = os.path.join(self.data_dir, 'label_2', f'{idx}.txt')
                calib_path = os.path.join(self.data_dir, 'calib', f'{idx}.txt')
                samples.append({
                    'id': idx,
                    'image_path': image_path,
                    'label_path': label_path,
                    'calib_path': calib_path,
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        annotations = self.parser.parse_label(sample['id'])
        calib = self.parser.parse_calib(sample['id'])

        item = {
            'image': image,
            'annotations': annotations,
            'calib': calib,
            'sample_id': sample['id'],
            'image_path': sample['image_path'],
            'label_path': sample['label_path'],
        }

        if self.transform is not None:
            item = self.transform(item)
        return item
