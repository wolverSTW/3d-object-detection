import torch
import torch.nn as nn

class LightweightBackbone(nn.Module):
    """
    Lightweight Feature Extraction Backbone based on Efficient Conv Stacks.
    Generates multi-scale visual features.
    """
    def __init__(self, in_channels=3):
        super(LightweightBackbone, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

    def forward(self, x):
        f1 = self.stage1(x) # [B, 64, H/4, W/4]
        f2 = self.stage2(f1) # [B, 128, H/8, W/8]
        f3 = self.stage3(f2) # [B, 256, H/16, W/16]
        return f3

class YOLO3DPredictionHead(nn.Module):
    """
    Decoupled Prediction Head for 2D/3D attributes & Metric Distance.
    """
    def __init__(self, in_channels=256, num_classes=3):
        super(YOLO3DPredictionHead, self).__init__()
        
        # Classification Head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        
        # 2D Bounding Box Head
        self.bbox2d_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )
        
        # 3D Dimensions Head (Height, Width, Length)
        self.dim3d_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, 3, kernel_size=1)
        )
        
        # 3D Location Head (x, y, z)
        self.loc3d_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, 3, kernel_size=1)
        )
        
        # Orientation Head (rotation_y)
        self.rot_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        cls_logits = self.cls_head(x)
        bbox2d_preds = self.bbox2d_head(x)
        dim3d_preds = self.dim3d_head(x)
        loc3d_preds = self.loc3d_head(x)
        rot_preds = self.rot_head(x)

        return {
            "cls_logits": cls_logits,
            "bbox2d": bbox2d_preds,
            "dimensions": dim3d_preds,
            "location": loc3d_preds,
            "rotation_y": rot_preds
        }

class YOLO3DBaseline(nn.Module):
    """
    Complete Monocular 3D Detection Baseline Network.
    """
    def __init__(self, num_classes=3):
        super(YOLO3DBaseline, self).__init__()
        self.backbone = LightweightBackbone()
        self.head = YOLO3DPredictionHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions
