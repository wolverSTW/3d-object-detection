# Lightweight Geometry-Guided Monocular 3D Detection

Quick start:

```bash
cd /workspaces/3d-object-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_training.py
```

This project is a beginner-friendly training and evaluation pipeline for monocular 3D object detection on the KITTI dataset. It includes a baseline model, a geometry-guided variant, dataset validation, training, evaluation, and experiment comparison.

## What this project does

- Prepares the official KITTI dataset for training
- Validates that the expected KITTI folders are present
- Builds a training split from the image files
- Trains a baseline model and a geometry-guided model
- Evaluates detection performance and distance metrics
- Compares model results for experiments

## 1. Install the project

From the repo root:

```bash
cd /workspaces/3d-object-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. GPU Sever Setup

Use this flow when setting up the project on a GPU server or a remote Linux machine.

### Clone or update the repository

If the project is not already available:

```bash
git clone https://github.com/wolverSTW/3d-object-detection.git
cd 3d-object-detection
```

If the project is already cloned on the server:

```bash
cd /workspaces/3d-object-detection
git pull origin main
```

### Create the virtual environment and install dependencies

```bash
cd /workspaces/3d-object-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Check GPU availability

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

If you see `CUDA available: True`, the project is ready to train on the GPU.

### Prepare KITTI dataset

```bash
python scripts/kitti_pipeline.py
```

This downloads or validates the KITTI archive, extracts the files, and creates the required dataset split.

### Start training on the GPU server

Recommended command for the thesis model:

```bash
python scripts/train.py --model proposed --epochs 100 --data-dir data/KITTI/raw --split train.txt
```

Baseline model:

```bash
python scripts/train.py --model baseline --epochs 100 --data-dir data/KITTI/raw --split train.txt
```

To run in the background and keep the output in a log file:

```bash
nohup python scripts/train.py --model proposed --epochs 100 --data-dir data/KITTI/raw --split train.txt > outputs/training.log 2>&1 &
```

Monitor training:

```bash
tail -f outputs/training.log
```

## 3. Run the project

Use the project launcher:

```bash
python run_training.py
```

This performs the full KITTI preparation and starts the correct training path for the current environment.

## 4. Prepare the KITTI dataset

Run the dataset workflow:

```bash
python scripts/kitti_pipeline.py
```

What this does:

- downloads the official KITTI archive files if they are missing
- extracts them once
- makes sure the expected dataset layout is ready
- creates the split file for training
- runs a short dataset summary

This is the safe setup flow for real KITTI data. It skips steps that are already complete.

## 5. Dataset layout

The official KITTI download usually creates a structure like this:

```text
data/KITTI/raw/
├── training/
│   ├── image_2/
│   ├── label_2/
│   └── calib/
├── testing/
│   └── image_2/
├── data_object_image_2.zip
├── data_object_label_2.zip
└── data_object_calib.zip
```

The project resolves both the official extracted structure and the flattened raw layout automatically.

## 6. Train the model

Use the launcher for the main training flow:

```bash
python run_training.py
```

For direct training commands, use:

```bash
python scripts/train.py \
  --config configs/experiments/baseline.yaml \
  --model baseline \
  --epochs 100 \
  --data-dir data/KITTI/raw \
  --split train.txt
```

For the geometry model:

```bash
python scripts/train.py \
  --config configs/experiments/geometry.yaml \
  --model geometry \
  --epochs 100 \
  --data-dir data/KITTI/raw \
  --split train.txt
```

## 7. Evaluate the models

```bash
python scripts/evaluate.py --config configs/experiments/baseline.yaml
python scripts/evaluate.py --config configs/experiments/geometry.yaml
```

## 8. Compare experiment results

```bash
python scripts/run_experiments.py --output-dir experiments/final_compare
python scripts/compare_experiments.py --output-dir experiments/final_compare
```

## 9. Optional: run tests

```bash
python -m unittest discover -s tests -v
```

## Common issue to avoid

If the dataset is missing, you will see errors like:

- `No training samples found ...`
- `image_2: MISSING`
- `label_2: MISSING`
- `calib: MISSING`

This usually means the official KITTI ZIP has not been downloaded or extracted correctly. In that case, run:

```bash
python scripts/kitti_pipeline.py
```

and then rerun training.

## Project structure

```text
3d-object-detection/
├── configs/
├── data/
├── docs/
├── experiments/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── tools/
├── README.md
├── requirements.txt
├── train.py
└── .gitignore
```

## Summary

The normal workflow is:

1. install dependencies
2. run `python run_training.py`
3. prepare the KITTI dataset if needed
4. train with the project launcher or direct training command
5. evaluate and compare results

This keeps the project simple and avoids duplicate setup instructions.

## License

This project is intended for academic and research use.
