# Thesis Methodology Summary

## 1. Research Objective

This thesis investigates lightweight monocular 3D object detection and distance estimation for autonomous driving scenarios, using the KITTI dataset as the primary benchmark. The project aims to develop a compact geometry-aware detection framework that improves 3D localization and distance estimation without introducing excessive computational cost.

## 2. Overall Methodology

The proposed workflow follows a standard research pipeline:

1. Dataset acquisition and validation
2. Data audit and exploratory analysis
3. Preprocessing and target generation
4. Baseline model development
5. Geometry-guided model extension
6. Training and validation
7. Evaluation and ablation analysis
8. Performance reporting and discussion

This structure is specifically designed to compare a lightweight YOLOv10-inspired baseline against a geometry-aware variant under consistent experimental conditions.

## 3. Dataset and Data Audit

The KITTI dataset provides synchronized RGB images, calibration files, and 3D annotations. The methodology begins with a comprehensive audit of:

- object class distribution
- 2D bounding boxes and overlap patterns
- 3D dimensions and cuboid statistics
- object depth and distance distributions
- calibration consistency
- camera geometry priors

This stage is important because geometry-aware 3D detection depends heavily on accurate camera parameters and object scale priors.

## 4. Baseline Model

The baseline model follows a YOLOv10-style backbone and detection pipeline:

- RGB image input
- feature extraction via a lightweight backbone
- multi-scale feature fusion through a neck
- 3D detection head for box and dimension regression
- class predictions and distance-aware outputs

The baseline establishes a reference model for comparing the effects of geometry-aware feature enhancement.

## 5. Geometry-Guided Extension

The proposed architecture extends the baseline by introducing geometry-aware feature guidance. The design integrates camera geometry priors into the representation learning process through:

- depth-aware feature modulation
- projection-based priors
- 3D-to-2D geometric consistency
- auxiliary geometry feature extraction
- attention-driven refinement using a lightweight CSAMM-style module

The key idea is that the detector should not only learn appearance-based cues but also use geometry information derived from camera calibration and object position in 3D space. This helps improve localization and distance estimation, especially for partially occluded or depth-sensitive objects.

## 6. Loss Function Design

A weighted multi-component loss is used to supervise the model:

- box regression loss for localization
- classification loss for object category prediction
- dimension regression loss for 3D cuboid dimensions
- geometry consistency loss for depth and projection clues

The combined objective balances appearance-based detection performance with geometric supervision. This is particularly important for monocular 3D tasks where depth is ambiguous without structural priors.

## 7. Evaluation Protocol

The detection performance is evaluated using the standard KITTI protocol across difficulty levels:

- AP3D Easy / Moderate / Hard
- APBEV Easy / Moderate / Hard

For distance estimation, the methodology measures:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)

Efficiency is also reported to ensure that gains in accuracy are not achieved at unreasonable computational cost:

- parameters
- GFLOPs
- inference time
- FPS

## 8. Ablation Study

To isolate the role of geometry-aware components, the method compares several experimental settings:

- baseline model only
- geometry-guided model without full refinement
- model with geometry-aware feature module enabled
- full proposed architecture

These ablations help determine which components contribute most to 3D detection and distance accuracy.

## 9. Expected Contribution

The thesis is expected to show that lightweight geometry-guided cues can significantly improve monocular 3D detection and distance estimation while preserving a compact architecture suitable for efficient deployment. The contribution is not only predictive accuracy, but also interpretability through geometry-informed feature design and clear ablation-driven evidence.

## 10. Research Significance

This project sits at the intersection of computer vision, autonomous perception, and geometry-aware deep learning. By combining a compact YOLO-inspired architecture with camera geometry guidance, the work aims to advance practical monocular 3D perception while maintaining reproducibility and computational efficiency.
