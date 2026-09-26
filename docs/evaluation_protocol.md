# Evaluation Protocol

## Detection Metrics

- AP3D (Easy, Moderate, Hard)
- APBEV (Easy, Moderate, Hard)
- Standard KITTI evaluation protocol on validation/test split

## Distance Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Optional relative error metrics for analysis

## Efficiency Metrics

- Number of parameters
- GFLOPs
- Inference time
- Frames per second (FPS)

## Ablation Strategy

- baseline model
- geometry-guided variant
- geometry + CSAMM variant
- complete pipeline

Use this protocol to ensure consistency across all experiments and thesis comparisons.
