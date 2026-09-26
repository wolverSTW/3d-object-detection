#!/usr/bin/env python3
"""Training script for the thesis project.

This script loads a YAML config, instantiates the model, and runs a minimal
training loop for local validation and experimentation.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.baseline import YOLO3DBaseline
from src.models.proposed import YOLO3DGeometryGuided
from src.training.trainer import Trainer
from src.utils.config import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train the 3D detection model.')
    parser.add_argument('--config', type=str, default='configs/experiments/baseline.yaml')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'geometry'])
    parser.add_argument('--epochs', type=int, default=1)
    return parser.parse_args()


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

    sample_loader = [{
        'image': None,
        'annotations': [{
            'location_3d': [10.0, 0.0, 5.0],
            'bbox_2d': [10.0, 20.0, 100.0, 200.0],
            'dimensions_3d': [1.5, 1.6, 3.8],
        }],
        'calib': {}
    } for _ in range(2)]

    print(f'Running {args.model} training for {args.epochs} epoch(s)')
    for epoch in range(args.epochs):
        epoch_loss = trainer.train_epoch(sample_loader)
        print(f'Epoch {epoch + 1}: loss={epoch_loss:.4f}')
        trainer.save_checkpoint(epoch + 1)

    print(f'Model training completed. Results saved in: {save_dir}')


if __name__ == '__main__':
    main()
