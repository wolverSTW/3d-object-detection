import numpy as np
import torch

class TargetConverter:
    """
    Converts KITTI raw labels into tensor targets for 3D object detection head.
    Classes mapping:
        Car: 0, Pedestrian: 1, Cyclist: 2
    """
    CLASS_MAP = {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2
    }

    def __init__(self, class_map=None):
        if class_map is not None:
            self.class_map = class_map
        else:
            self.class_map = self.CLASS_MAP

    def convert_labels_to_target(self, labels, img_shape):
        """
        Args:
            labels (list): List of KITTILabel objects
            img_shape (tuple): (height, width) of the image
        Returns:
            torch.Tensor: Target matrix of shape (N, 13)
            Columns: [class_id, x1, y1, x2, y2, h, w, l, x, y, z, rot_y, distance]
        """
        targets = []
        img_h, img_w = img_shape[:2]

        for label in labels:
            if label.type not in self.class_map:
                continue

            cls_id = self.class_map[label.type]
            
            # Normalized 2D bounding box coords [0, 1]
            x1 = label.bbox2d[0] / img_w
            y1 = label.bbox2d[1] / img_h
            x2 = label.bbox2d[2] / img_w
            y2 = label.bbox2d[3] / img_h

            # 3D bounding box attributes
            h, w, l = label.dimensions
            x, y, z = label.location
            rot_y = label.rotation_y
            dist = label.distance

            target_row = [cls_id, x1, y1, x2, y2, h, w, l, x, y, z, rot_y, dist]
            targets.append(target_row)

        if len(targets) == 0:
            return torch.zeros((0, 13), dtype=torch.float32)

        return torch.tensor(targets, dtype=torch.float32)
