#!/usr/bin/env python3
"""Prepare or validate the KITTI directory structure for experiments."""

from pathlib import Path


def ensure_kitti_layout(data_root: str = 'data/KITTI') -> None:
    root = Path(data_root)
    required = [
        root / 'raw' / 'image_2',
        root / 'raw' / 'label_2',
        root / 'raw' / 'calib',
        root / 'splits',
    ]
    for path in required:
        path.mkdir(parents=True, exist_ok=True)
    print(f'KITTI structure ensured at: {root}')


if __name__ == '__main__':
    ensure_kitti_layout()
