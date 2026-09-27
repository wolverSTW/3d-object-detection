#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-${1:-baseline}}"
EPOCHS="${EPOCHS:-${2:-100}}"
DATA_DIR="${DATA_DIR:-data/KITTI/raw}"
SPLIT="${SPLIT:-train.txt}"
LIMIT="${LIMIT:-}"

if [[ "$MODEL" == "geometry" ]]; then
  CONFIG="configs/experiments/geometry.yaml"
else
  CONFIG="configs/experiments/baseline.yaml"
fi

python scripts/kitti_pipeline.py >/dev/null

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected. Running full training for model=$MODEL with epochs=$EPOCHS"
  if [[ -n "$LIMIT" ]]; then
    exec python scripts/train.py \
      --config "$CONFIG" \
      --model "$MODEL" \
      --epochs "$EPOCHS" \
      --data-dir "$DATA_DIR" \
      --split "$SPLIT" \
      --limit "$LIMIT"
  fi

  exec python scripts/train.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --data-dir "$DATA_DIR" \
    --split "$SPLIT"
fi

echo "No GPU detected. Running CPU validation smoke test for model=$MODEL"

if [[ ! -f data/KITTI/splits/train_cpu.txt ]]; then
  python - <<'PY'
from pathlib import Path
raw = Path('data/KITTI/raw')
image_dir = raw / 'image_2'
if not image_dir.exists():
    raise FileNotFoundError('image_2 not found in data/KITTI/raw')
ids = sorted(p.stem for p in image_dir.glob('*.png'))[:5]
if not ids:
    raise FileNotFoundError('No PNG files found in data/KITTI/raw/image_2')
Path('data/KITTI/splits').mkdir(exist_ok=True, parents=True)
Path('data/KITTI/splits/train_cpu.txt').write_text('\n'.join(ids) + '\n', encoding='utf-8')
print('Created CPU validation split with', len(ids), 'samples')
PY
fi

if [[ -n "$LIMIT" ]]; then
  exec python scripts/train.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --epochs 1 \
    --data-dir "$DATA_DIR" \
    --split train_cpu.txt \
    --limit "$LIMIT"
fi

exec python scripts/train.py \
  --config "$CONFIG" \
  --model "$MODEL" \
  --epochs 1 \
  --data-dir "$DATA_DIR" \
  --split train_cpu.txt \
  --limit 5
