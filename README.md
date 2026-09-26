# Lightweight YOLOv10-Based Geometry-Guided Framework for Monocular 3D Object Detection and Distance Estimation

This repository provides a thesis-ready research project for monocular 3D object detection and distance estimation on the KITTI dataset. It follows a practical workflow from dataset audit and preprocessing to baseline training, geometry-guided model experiments, evaluation, and efficiency analysis.

## Project Snapshot

- Research focus: monocular 3D object detection and depth/distance estimation
- Dataset: KITTI benchmark data with lightweight mock-data support for local validation
- Method: YOLOv10-inspired baseline with geometry-guided refinement modules
- Outputs: training pipeline, evaluation metrics, ablation comparison reports, and experiment summaries

## Project Goals

- Reproduce a lightweight YOLOv10-based baseline for monocular 3D object detection.
- Extend the baseline with geometry-guided feature refinement.
- Evaluate 3D detection metrics (AP3D / APBEV) and distance estimation metrics.
- Compare baseline and proposed models under ablation settings.
- Provide a clean, reproducible research project structure for the thesis workflow.

## Repository Structure

```text
3d-object-detection/
├── configs/
│   ├── dataset/
│   ├── models/
│   └── experiments/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── training/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
├── scripts/
├── tools/
├── tests/
├── docs/
├── experiments/
├── outputs/
├── data/
├── README.md
├── requirements.txt
├── train.py
├── .env.example
└── .gitignore
```

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate mock KITTI data for local validation:

```bash
python tools/download/create_mock_kitti.py
```

3. Run the project entry point:

```bash
python train.py --config configs/experiments/baseline.yaml
```

4. Run the training script directly:

```bash
python scripts/train.py --config configs/experiments/baseline.yaml --model baseline --epochs 1
```

5. Evaluate results:

```bash
python scripts/evaluate.py --config configs/experiments/baseline.yaml
```

6. Run the test suite:

```bash
python -m unittest discover -s tests -v
```

## Data Pipeline

The project is designed around the KITTI dataset workflow:

- dataset acquisition and integrity validation
- advanced exploratory data analysis
- preprocessing and target generation
- baseline YOLOv10-style 3D detection
- geometry-guided model extension
- training, validation, and benchmarking

## Thesis Experiment Themes

### Baseline Model
- YOLOv10 backbone/necks
- 3D detection head
- standard monocular 3D outputs

### Proposed Geometry-Guided Model
- YOLOv10 features enhanced with geometry priors and guidance modules
- camera geometry-aware representation learning
- improved 3D localization and distance prediction

### Ablation Study
- baseline only
- geometry guidance enabled
- CSAMM / geometry feature module enabled
- full model with all geometry components

### Evaluation Dimensions
- AP3D Easy / Moderate / Hard
- APBEV Easy / Moderate / Hard
- distance MAE / RMSE
- model efficiency metrics (params, GFLOPs, inference time, FPS)

## Running on a vast.ai GPU Server

These are the steps typically used to run this project on a rented GPU instance from vast.ai.

1. Launch a GPU instance on vast.ai.
   - Choose a Linux image with Python and CUDA support.
   - Recommended: Ubuntu 22.04 or 24.04 with a CUDA-enabled GPU.
   - Save the provided SSH command and your instance IP.

2. Connect to the server:

```bash
ssh root@<INSTANCE_IP>
```

3. Install the required system packages:

```bash
apt update && apt install -y git python3 python3-pip python3-venv tmux
```

4. Clone the repository and enter it:

```bash
git clone <your-repo-url> /workspace/3d-object-detection
cd /workspace/3d-object-detection
```

5. Create a virtual environment and install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

6. Create or sync the dataset:

```bash
python tools/download/create_mock_kitti.py
```

If you are using the full KITTI dataset, place it under `data/KITTI/raw/` and keep the expected folder structure from the project.

7. Start a long-running training session in the background with `tmux`:

```bash
tmux new -s thesis-train
source .venv/bin/activate
cd /workspace/3d-object-detection
python scripts/train.py --config configs/experiments/baseline.yaml --model baseline --epochs 1
```

To detach from the session:

```bash
tmux detach
```

To reconnect later:

```bash
tmux attach -t thesis-train
```

8. Run the comparison workflow after training:

```bash
python scripts/run_experiments.py --output-dir experiments/final_compare
```

9. Check the generated reports and metrics:

```bash
ls -R experiments
```

10. Copy results back to your local machine if needed:

```bash
scp -r root@<INSTANCE_IP>:/workspace/3d-object-detection/experiments ./
```

## Notes

This scaffold is intentionally organized to support the full thesis pipeline while remaining lightweight and easy to extend. The implementation can be upgraded from these templates into a full training pipeline as the research progresses.

## License

This project is intended for academic and research use.
