import os
import numpy as np

class KittiParser:
    """Parser for KITTI 3D Object Detection dataset labels and calibration files."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "image_2")
        self.label_dir = os.path.join(data_dir, "label_2")
        self.calib_dir = os.path.join(data_dir, "calib")

    def parse_label(self, idx_str: str):
        """Parses KITTI ground truth label txt file."""
        label_path = os.path.join(self.label_dir, f"{idx_str}.txt")
        objects = []
        if not os.path.exists(label_path):
            return objects

        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                obj = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox_2d': np.array([float(x) for x in parts[4:8]]),  # [xmin, ymin, xmax, ymax]
                    'dimensions_3d': np.array([float(x) for x in parts[8:11]]), # [h, w, l]
                    'location_3d': np.array([float(x) for x in parts[11:14]]),  # [x, y, z]
                    'rotation_y': float(parts[14])
                }
                objects.append(obj)
        return objects

    def parse_calib(self, idx_str: str):
        """Parses KITTI calibration txt file."""
        calib_path = os.path.join(self.calib_dir, f"{idx_str}.txt")
        calib_data = {}
        if not os.path.exists(calib_path):
            return calib_data

        with open(calib_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, value = line.split(':', 1)
                calib_data[key] = np.array([float(x) for x in value.strip().split(' ')]).reshape(3, 4) if 'P' in key else np.array([float(x) for x in value.strip().split(' ')])
        return calib_data