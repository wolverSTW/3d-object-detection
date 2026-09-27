#!/usr/bin/env python3
"""Project training entry point.

This file serves as the main entry point for the monocular 3D object detection
research workflow with a real trainable baseline and geometry-guided variant.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from src.data.dataset import KittiDataset
from src.losses.detection_loss import DetectionLoss
from src.models.baseline import YOLO3DBaseline
from src.models.proposed import YOLO3DGeometryGuided
from src.training.trainer import Trainer
from src.utils.config import load_yaml_config

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D object detection model.')
    parser.add_argument('--config', type=str, default='configs/experiments/baseline.yaml',
                        help='Path to YAML configuration file.')
    parser.add_argument('--model', type=str, default='proposed', choices=['baseline', 'geometry', 'proposed'],
                        help='Model variant to run.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=8, help='Mini-batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--data-dir', type=str, default='data/KITTI/raw', help='KITTI raw data directory.')
    parser.add_argument('--split', type=str, default='train.txt', help='KITTI split file.')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit for training items.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


def build_dataloader(data_dir: str, split_file: str = 'train.txt', limit: int = None):
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
    return items


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    config = load_yaml_config(args.config)

    if args.model == 'geometry':
        model = YOLO3DGeometryGuided(in_channels=3, num_classes=3, reg_dims=7, enabled=True)
    else:
        model = YOLO3DBaseline(in_channels=3, num_classes=3, reg_dims=7)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = DetectionLoss(lambda_box=1.0, lambda_cls=0.5, lambda_3d=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs('experiments', exist_ok=True)
    output_dir = config.get('output', {}).get('save_dir', 'experiments/default')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, save_dir=output_dir, device=device)
    dataloader = build_dataloader(args.data_dir, args.split, args.limit)

    print('=== 3D Object Detection Thesis Project ===')
    print(f'Loaded configuration: {args.config}')
    print(f'Device: {device}')
    print(f'Output directory: {output_dir}')
    print(f'Running {args.model} training with seed={args.seed}')
    print(f'Using {len(dataloader)} samples, batch_size={args.batch_size}, lr={args.lr}')

    for epoch in range(1, args.epochs + 1):
        epoch_loss = trainer.train_epoch(dataloader, epoch=epoch, total_epochs=args.epochs)
        trainer.save_checkpoint(epoch, model_name=args.model)

    print('Project pipeline executed successfully.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
