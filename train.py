#!/usr/bin/env python3
"""Project training entry point.

This file serves as the main entry point for the monocular 3D object detection
research workflow. It loads a YAML configuration and runs a minimal training pass.
"""

import argparse
import os
import sys
from pathlib import Path

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
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'geometry'],
                        help='Model variant to run.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    if args.model == 'geometry':
        model = YOLO3DGeometryGuided(in_channels=3, num_classes=3, reg_dims=7, enabled=True)
    else:
        model = YOLO3DBaseline(in_channels=3, num_classes=3, reg_dims=7)

    os.makedirs('experiments', exist_ok=True)
    output_dir = config.get('output', {}).get('save_dir', 'experiments/default')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model=model, optimizer=None, criterion=None, save_dir=output_dir)
    sample_loader = [{
        'image': None,
        'annotations': [{
            'location_3d': [10.0, 0.0, 5.0],
            'bbox_2d': [10.0, 20.0, 100.0, 200.0],
            'dimensions_3d': [1.5, 1.6, 3.8],
        }],
        'calib': {}
    } for _ in range(2)]

    print('=== 3D Object Detection Thesis Project ===')
    print(f'Loaded configuration: {args.config}')
    print(f'Output directory: {output_dir}')
    print(f'Running {args.model} training with seed={args.seed}')
    epoch_loss = trainer.train_epoch(sample_loader)
    checkpoint = trainer.save_checkpoint(1)
    print(f'Epoch 1 loss={epoch_loss:.4f}')
    print(f'Checkpoint: {checkpoint}')
    print('Project pipeline executed successfully.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
