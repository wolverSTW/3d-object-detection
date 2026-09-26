"""Loss functions for 3D detection and distance estimation."""

from .detection_loss import DetectionLoss
from .geometry_loss import GeometryLoss

__all__ = ["DetectionLoss", "GeometryLoss"]
