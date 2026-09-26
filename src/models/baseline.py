from src.models.backbone.yolov10_backbone import YOLOv10Backbone
from src.models.neck.yolov10_neck import YOLOv10Neck
from src.models.heads.detection_head import DetectionHead


class YOLO3DBaseline:
    """Minimal baseline model skeleton consistent with the thesis proposal."""

    def __init__(self, in_channels=3, num_classes=3, reg_dims=7):
        self.backbone = YOLOv10Backbone(in_channels=in_channels)
        self.neck = YOLOv10Neck()
        self.head = DetectionHead(in_channels=256, num_classes=num_classes, reg_dims=reg_dims)

    def forward(self, x, calibration=None):
        features = self.backbone.forward(x)
        features = self.neck.forward(features)
        outputs = self.head.forward(features)
        return outputs
