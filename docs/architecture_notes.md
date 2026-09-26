# Architecture Notes

## Baseline Path

- Input: RGB image
- Backbone: YOLOv10-inspired encoder
- Neck: feature aggregation
- Head: 3D detection regression head
- Output: 3D box parameters, depth/distance estimates, class predictions

## Geometry-Guided Path

- Input features are enriched with camera geometry priors.
- Projection consistency and triangulation-inspired reasoning guide the model.
- Additional geometry features support 3D localization and distance estimation.

## Evaluation Focus

- AP3D and APBEV at Easy, Moderate, Hard difficulty levels
- distance MAE and RMSE
- model size, FLOPs, latency, and FPS
- qualitative analysis and failure case review
