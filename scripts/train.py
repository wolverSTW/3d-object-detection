#!/usr/bin/env python3
"""Training script for the thesis project.

This script loads a YAML config, instantiates the model, and runs a minimal
training loop for local validation and experimentation.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import KittiDataset
from src.models.baseline import YOLO3DBaseline
from src.models.proposed import YOLO3DGeometryGuided
from src.training.trainer import Trainer
from src.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train the 3D detection model.')
    parser.add_argument('--config', type=str, default='configs/experiments/baseline.yaml')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'geometry'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--data-dir', type=str, default='data/KITTI/raw', help='KITTI raw dataset directory.')
    parser.add_argument('--split', type=str, default='train.txt', help='Training split file name.')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit for the number of training items.')
    return parser.parse_args()


def build_dataloader(data_dir: str, split_file: str = 'train.txt', limit: int = None):
    dataset = KittiDataset(data_dir=data_dir, split_file=split_file)
    if limit is not None:
        dataset.samples = dataset.samples[:limit]
    return dataset.samples


def build_model(model_name: str):
    if model_name == 'geometry':
        return YOLO3DGeometryGuided(in_channels=3, num_classes=3, reg_dims=7, enabled=True)
    return YOLO3DBaseline(in_channels=3, num_classes=3, reg_dims=7)


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    model = build_model(args.model)
    save_dir = config.get('output', {}).get('save_dir', 'experiments/default')
    trainer = Trainer(model=model, optimizer=None, criterion=None, save_dir=save_dir)

    dataloader = build_dataloader(args.data_dir, args.split, args.limit)
    if not dataloader:
        raise FileNotFoundError(f'No training samples found in {args.data_dir} with split {args.split}.')

    print(f'Running {args.model} training for {args.epochs} epoch(s)')
    print(f'Using KITTI dataset from {args.data_dir} with {len(dataloader)} samples')
    for epoch in range(args.epochs):
        epoch_loss = trainer.train_epoch(dataloader)
        print(f'Epoch {epoch + 1}: loss={epoch_loss:.4f}')
        trainer.save_checkpoint(epoch + 1)

    print(f'Model training completed. Results saved in: {save_dir}')


if __name__ == '__main__':
    main()
