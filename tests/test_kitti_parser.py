import os
import unittest
from src.data.kitti_parser import KittiParser
from tools.download.create_mock_kitti import create_mock_kitti

class TestKittiParser(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create mock data first
        create_mock_kitti()
        cls.data_dir = "data/KITTI/raw"
        cls.parser = KittiParser(cls.data_dir)

    def test_parse_label(self):
        objects = self.parser.parse_label("000000")
        self.assertGreater(len(objects), 0, "No objects parsed from mock label!")
        self.assertIn('type', objects[0])
        self.assertIn('location_3d', objects[0])
        self.assertEqual(len(objects[0]['bbox_2d']), 4)

    def test_parse_calib(self):
        calib = self.parser.parse_calib("000000")
        self.assertIn('P2', calib)
        self.assertEqual(calib['P2'].shape, (3, 4))

if __name__ == '__main__':
    unittest.main()