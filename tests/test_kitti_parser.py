import unittest
import numpy as np
import tempfile
import os
from src.data.kitti_parser import KITTICalibration, KITTILabel

class TestKITTIParser(unittest.TestCase):

    def test_calibration_parsing(self):
        # Dummy calibration contents
        dummy_calib_text = "P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163761e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03\n"
        
        with tempfile.NamedTemporaryFile('w', delete=False) as f:
            f.write(dummy_calib_text)
            temp_path = f.name

        try:
            calib = KITTICalibration(temp_path)
            self.assertEqual(calib.P2.shape, (3, 4))
            self.assertAlmostEqual(calib.P2[0, 0], 721.5377)
        finally:
            os.remove(temp_path)

    def test_label_parsing(self):
        # Dummy label line
        dummy_label_line = "Car 0.00 0 -1.57 589.28 177.10 622.25 189.23 1.48 1.60 3.69 1.84 1.47 8.41 0.01"
        label = KITTILabel(dummy_label_line)

        self.assertEqual(label.type, "Car")
        self.assertEqual(label.occluded, 0)
        self.assertEqual(len(label.bbox2d), 4)
        self.assertEqual(len(label.dimensions), 3)
        self.assertEqual(len(label.location), 3)
        
        # Verify distance calculation: sqrt(1.84^2 + 1.47^2 + 8.41^2) approx 8.733
        expected_distance = np.sqrt(1.84**2 + 1.47**2 + 8.41**2)
        self.assertAlmostEqual(label.distance, expected_distance, places=3)

if __name__ == '__main__':
    unittest.main()
