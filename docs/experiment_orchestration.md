# Experiment Orchestration

This project includes a lightweight orchestration flow for comparing the baseline and geometry-guided variants.

## Commands

```bash
python scripts/run_experiments.py --output-dir experiments/comparison
python scripts/compare_experiments.py --output-dir experiments/comparison
python train.py --config configs/experiments/baseline.yaml --model baseline
python train.py --config configs/experiments/geometry.yaml --model geometry
```

## Output

Each run saves a JSON summary under the configured output directory, including:

- model name
- distance metrics
- AP3D / APBEV metrics
- efficiency summary
- comparison artifact for baseline vs geometry
