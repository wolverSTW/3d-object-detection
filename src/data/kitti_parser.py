import os
import numpy as np
from PIL import Image

class KITTICalibration:
    """
    Parses KITTI calibration matrices from calibration text files.
    Focuses on P2 (rectified camera matrix for image_2).
    """
    def __init__(self, calib_filepath):
        self.calib_filepath = calib_filepath
        self.P2 = self._parse_calib_file()

    def _parse_calib_file(self):
        calib_data = {}
        with open(self.calib_filepath, 'r') as f:
            for line in f.readlines():
                if ':' in line:
                    key, val = line.split(':', 1)
                    calib_data[key.strip()] = np.array([float(x) for x in val.strip().split()])

        # P2 matrix shape is (3, 4)
        P2 = calib_data['P2'].reshape(3, 4)
        return P2

class KITTILabel:
    """
    Parses single object annotation in KITTI label format.
    Format: type, truncated, occluded, alpha, bbox2d, dimensions (h,w,l), location (x,y,z), rotation_y
    """
    def __init__(self, line):
        parts = line.strip().split(' ')
        self.type = parts[0]
        self.truncated = float(parts[1])
        self.occluded = int(parts[2])
        self.alpha = float(parts[3])
        
        # 2D Bounding Box [xmin, ymin, xmax, ymax]
        self.bbox2d = np.array([float(x) for x in parts[4:8]])
        
        # 3D Object Dimensions [height, width, length] in meters
        self.dimensions = np.array([float(x) for x in parts[8:11]])
        
        # 3D Object Location [x, y, z] in camera coordinates (meters)
        self.location = np.array([float(x) for x in parts[11:14]])
        
        # Rotation around Y-axis [-pi, pi]
        self.rotation_y = float(parts[15]) if len(parts) > 14 else float(parts[14])
        
        # Calculate metric Euclidean distance from camera origin (0,0,0) to object center (x,y,z)
        self.distance = np.sqrt(np.sum(self.location ** 2))

class KITTIParser:
    """
    Handles file path resolution and data loading for a single sample frame.
    """
    def __init__(self, root_dir, split="training"):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, split, "image_2")
        self.label_dir = os.path.join(root_dir, split, "label_2")
        self.calib_dir = os.path.join(root_dir, split, "calib")

    def get_image(self, idx):
        img_path = os.path.join(self.image_dir, f"{idx:06d}.png")
        return Image.open(img_path).convert("RGB")

    def get_calibration(self, idx):
        calib_path = os.path.join(self.calib_dir, f"{idx:06d}.txt")
        return KITTICalibration(calib_path)

    def get_labels(self, idx):
        label_path = os.path.join(self.label_dir, f"{idx:06d}.txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        labels.append(KITTILabel(line))
        return labels
