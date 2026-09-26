#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-baseline}"
EPOCHS="${EPOCHS:-50}"
CONFIG="${CONFIG:-configs/experiments/baseline.yaml}"
DATA_DIR="${DATA_DIR:-data/KITTI/raw}"
SPLIT="${SPLIT:-train.txt}"
LIMIT="${LIMIT:-}"
LOG_DIR="${LOG_DIR:-experiments/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ ! -d "$DATA_DIR/image_2" ] || [ ! -d "$DATA_DIR/label_2" ] || [ ! -d "$DATA_DIR/calib" ]; then
  echo "KITTI dataset not found at $DATA_DIR"
  echo "Expected directories: $DATA_DIR/image_2, $DATA_DIR/label_2, and $DATA_DIR/calib"
  echo "You can generate mock data locally with: python tools/download/create_mock_kitti.py"
  exit 1
fi

TRAIN_CMD=(python scripts/train.py --config "$CONFIG" --model "$MODEL" --data-dir "$DATA_DIR" --split "$SPLIT" --epochs "$EPOCHS")
if [ -n "$LIMIT" ]; then
  TRAIN_CMD+=(--limit "$LIMIT")
fi

mkdir -p "$OUTPUT_DIR"

# Save training outputs into a stable experiment folder for later inspection.
EXPERIMENT_NAME="${MODEL}_kitti_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR/$EXPERIMENT_NAME"

echo "Starting training run for model=$MODEL on dataset=$DATA_DIR"
echo "Command: ${TRAIN_CMD[*]}"

"${TRAIN_CMD[@]}" 2>&1 | tee "$LOG_DIR/${MODEL}_kitti_train.log"

cp -r "$OUTPUT_DIR" "$OUTPUT_DIR/$EXPERIMENT_NAME" 2>/dev/null || true

echo "Training finished. Logs saved to $LOG_DIR/${MODEL}_kitti_train.log"
echo "Experiment artifacts are under $OUTPUT_DIR/$EXPERIMENT_NAME"
