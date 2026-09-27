#!/usr/bin/env python3
"""Validate the KITTI dataset layout and create a split file if possible."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data' / 'KITTI'
RAW_ROOT = DATA_ROOT / 'raw'
SPLIT_DIR = DATA_ROOT / 'splits'

REQUIRED_DIRS = [
    'image_2',
    'label_2',
    'calib',
]


def resolve_directories(raw_root: Path) -> dict:
    resolved = {}
    for name in REQUIRED_DIRS:
        direct = raw_root / name
        training = raw_root / 'training' / name
        if direct.exists():
            resolved[name] = direct
        elif training.exists():
            resolved[name] = training
        else:
            resolved[name] = direct
    return resolved


def ensure_split(image_dir: Path, split_path: Path) -> None:
    if not image_dir.exists():
        return

    images = sorted(image_dir.glob('*.png'))
    if not images:
        print(f'No PNG files found in {image_dir}.')
        return

    ids = [p.stem for p in images]
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text('\n'.join(ids) + '\n', encoding='utf-8')
    print(f'Wrote {len(ids)} sample IDs to {split_path}')


def validate_kitti_dataset(data_root: Path = DATA_ROOT) -> dict:
    raw_root = data_root / 'raw'
    split_path = data_root / 'splits' / 'train.txt'
    resolved = resolve_directories(raw_root)

    statuses = {}
    for name in REQUIRED_DIRS:
        path = resolved[name]
        statuses[name] = path.exists() and any(path.iterdir())
        state = 'OK' if statuses[name] else 'MISSING'
        print(f'{name}: {state}')

    if all(statuses.values()):
        image_dir = resolved['image_2']
        ensure_split(image_dir, split_path)
        print('Dataset layout looks valid for training.')
    else:
        print('\nKITTI is not fully present yet. Upload or unzip the official dataset into data/KITTI/raw/.')
        print('The official zip normally creates data/KITTI/raw/training/{image_2,label_2,calib} and data/KITTI/raw/testing/{image_2}.')
        print('Then rerun this script to create the split file.')

    return statuses


def main() -> int:
    print(f'Checking dataset at: {DATA_ROOT}')
    validate_kitti_dataset(DATA_ROOT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
