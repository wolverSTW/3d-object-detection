import torch

from src.models.backbone.yolov10_backbone import YOLOv10Backbone
from src.models.neck.yolov10_neck import YOLOv10Neck
from src.models.heads.detection_head import DetectionHead


class YOLO3DBaseline(torch.nn.Module):
    """Baseline monocular 3D detection model for thesis experiments."""

    def __init__(self, in_channels=3, num_classes=3, reg_dims=7):
        super().__init__()
        self.backbone = YOLOv10Backbone(in_channels=in_channels, base_channels=32)
        self.neck = YOLOv10Neck(in_channels=128, out_channels=256)
        self.head = DetectionHead(in_channels=256, num_classes=num_classes)

    def forward(self, x, calibration=None):
        feats = self.backbone(x)
        neck_out = self.neck(feats)
        return self.head(neck_out)
