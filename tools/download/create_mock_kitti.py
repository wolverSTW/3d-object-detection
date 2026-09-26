import os
import numpy as np
from PIL import Image

TARGET_DIR = "data/KITTI/raw"

def create_mock_kitti():
    """Generates synthetic KITTI-like mock data for local testing."""
    img_dir = os.path.join(TARGET_DIR, "image_2")
    label_dir = os.path.join(TARGET_DIR, "label_2")
    calib_dir = os.path.join(TARGET_DIR, "calib")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)

    print("Creating synthetic mock data for local testing...")

    # Generate 5 mock samples
    for i in range(5):
        sample_id = f"{i:06d}"

        # 1. Dummy Image (RGB, 375x1242 - Standard KITTI resolution)
        img_path = os.path.join(img_dir, f"{sample_id}.png")
        if not os.path.exists(img_path):
            img_array = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
            Image.fromarray(img_array).save(img_path)

        # 2. Dummy Annotation Label (KITTI format)
        # Format: type truncated occluded alpha bbox_2d(4) dimensions_3d(3) location_3d(3) rotation_y
        label_path = os.path.join(label_dir, f"{sample_id}.txt")
        if not os.path.exists(label_path):
            mock_label = (
                "Car 0.00 0 -1.57 50.0 100.0 200.0 300.0 1.5 1.6 3.8 0.5 1.5 10.0 0.0\n"
                "Pedestrian 0.00 0 -1.00 150.0 120.0 220.0 280.0 1.7 0.6 0.8 -1.2 1.2 8.5 0.2\n"
            )
            with open(label_path, "w") as f:
                f.write(mock_label)

        # 3. Dummy Calibration Matrix (P2 Camera Matrix standard)
        calib_path = os.path.join(calib_dir, f"{sample_id}.txt")
        if not os.path.exists(calib_path):
            mock_calib = (
                "P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
                "P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
                "P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163785e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03\n"
                "P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.729909e-03\n"
                "R0_rect: 9.999239e-01 9.837763e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278426e-03 7.402527e-03 4.351614e-03 9.999631e-01\n"
                "Tr_velo_to_cam: 6.927708e-03 -9.999722e-01 -2.757823e-03 -2.457729e-02 -1.162982e-03 2.749836e-03 -9.999955e-01 -6.127237e-02 9.999969e-01 6.930030e-03 -1.143899e-03 -3.328607e-01\n"
                "Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-02 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482451e-02 9.998881e-01 -7.997231e-01\n"
            )
            with open(calib_path, "w") as f:
                f.write(mock_calib)

    # 4. Generate splits/train.txt
    split_dir = "data/KITTI/splits"
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "train.txt"), "w") as f:
        for i in range(5):
            f.write(f"{i:06d}\n")

    print("[SUCCESS] Mock KITTI data created successfully in data/KITTI/raw/")

if __name__ == "__main__":
    create_mock_kitti()