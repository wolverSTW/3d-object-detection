#!/usr/bin/env python3
"""Training script for the thesis project.

This is the actual project training entry point for the KITTI monocular 3D
object detection workflow.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import KittiDataset
from src.losses.detection_loss import DetectionLoss
from src.models.baseline import YOLO3DBaseline
from src.models.proposed import YOLO3DGeometryGuided
from src.training.trainer import Trainer
from src.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train the 3D detection model.')
    parser.add_argument('--config', type=str, default='configs/experiments/baseline.yaml')
    parser.add_argument('--model', type=str, default='proposed', choices=['baseline', 'geometry', 'proposed'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data-dir', type=str, default='data/KITTI/raw', help='KITTI raw dataset directory.')
    parser.add_argument('--split', type=str, default='train.txt', help='Training split file name.')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit for the number of training items.')
    return parser.parse_args()


def _kitti_collate(batch):
    images = []
    annotations = []
    calib = []
    for item in batch:
        image = item['image']
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        if hasattr(image, 'getdata'):
            image = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.size[1], image.size[0], 3).permute(2, 0, 1) / 255.0
        elif isinstance(image, torch.Tensor):
            image = image.float()
        else:
            image = torch.zeros(3, 32, 32, dtype=torch.float32)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        images.append(image)
        annotations.append(item.get('annotations', []))
        calib.append(item.get('calib', {}))
    return {'image': torch.cat(images, dim=0), 'annotations': annotations, 'calib': calib}


def build_dataloader(data_dir: str, split_file: str = 'train.txt', limit: int = None, batch_size: int | None = None, num_workers: int = 0, pin_memory: bool = False):
    dataset = KittiDataset(data_dir=data_dir, split_file=split_file)
    if limit is not None:
        dataset.samples = dataset.samples[:limit]

    items = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if 'annotations' not in sample or not sample['annotations']:
            sample['annotations'] = [{
                'location_3d': [0.0, 0.0, 8.0],
                'bbox_2d': [0.0, 0.0, 64.0, 64.0],
                'dimensions_3d': [1.5, 1.6, 3.8],
                'rotation_y': 0.0,
            }]
        items.append(sample)

    if batch_size is None:
        return items

    return DataLoader(items, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=_kitti_collate)


def build_model(model_name: str):
    if model_name in ('geometry', 'proposed'):
        return YOLO3DGeometryGuided(in_channels=3, num_classes=3, reg_dims=7, enabled=True)
    return YOLO3DBaseline(in_channels=3, num_classes=3, reg_dims=7)


def main():
    args = parse_args()
    torch.manual_seed(42)
    config = load_yaml_config(args.config)

    training_cfg = config.get('training', {})
    batch_size = args.batch_size if args.batch_size is not None else training_cfg.get('batch_size', 8)
    num_workers = args.num_workers if args.num_workers is not None else training_cfg.get('num_workers', 2 if torch.cuda.is_available() else 0)
    pin_memory = training_cfg.get('pin_memory', torch.cuda.is_available())

    model = build_model(args.model)
    criterion = DetectionLoss(lambda_box=1.0, lambda_cls=0.5, lambda_3d=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    save_dir = config.get('output', {}).get('save_dir', 'experiments/default')
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, save_dir=save_dir, device='cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = build_dataloader(args.data_dir, args.split, args.limit, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if len(dataloader.dataset) == 0:
        raise FileNotFoundError(f'No training samples found in {args.data_dir} with split {args.split}.')

    print(f'Running {args.model} training for {args.epochs} epoch(s)')
    print(f'Using KITTI dataset from {args.data_dir} with {len(dataloader.dataset)} samples, batch_size={batch_size}, num_workers={num_workers}')
    print('Training status: starting...')

    for epoch in range(args.epochs):
        epoch_loss = trainer.train_epoch(dataloader, epoch=epoch + 1, total_epochs=args.epochs)
        checkpoint_path = trainer.save_checkpoint(epoch + 1, model_name=args.model)
        print(f'checkpoint={checkpoint_path}')

    print('\nModel training completed.')
    print(f'Results saved in: {save_dir}')


if __name__ == '__main__':
    main()
