import os
import torch
from torch.utils.data import Dataset
from src.data.kitti_parser import KITTIParser
from src.data.converter import TargetConverter
from src.data.transforms import KITTITransforms

class KITTIDataset(Dataset):
    """
    PyTorch Dataset implementation for KITTI 3D Object Detection.
    """
    def __init__(self, root_dir, split="training", transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.parser = KITTIParser(root_dir, split=split)
        self.converter = TargetConverter()
        self.transforms = transforms if transforms is not None else KITTITransforms()
        
        # Load available sample IDs
        self.image_dir = os.path.join(root_dir, split, "image_2")
        if os.path.exists(self.image_dir):
            self.sample_ids = [int(f.split('.')[0]) for f in os.listdir(self.image_dir) if f.endswith('.png')]
            self.sample_ids.sort()
        else:
            self.sample_ids = []

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        image = self.parser.get_image(sample_id)
        labels = self.parser.get_labels(sample_id)
        calib = self.parser.get_calibration(sample_id)
        
        orig_shape = (image.height, image.width)
        targets = self.converter.convert_labels_to_target(labels, orig_shape)
        
        image_tensor, targets = self.transforms(image, targets)
        
        return {
            "image": image_tensor,
            "targets": targets,
            "calib_p2": torch.tensor(calib.P2, dtype=torch.float32),
            "sample_id": sample_id
        }

def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes per frame.
    Adds a batch index column to targets for proper batching.
    """
    images = []
    targets_list = []
    calib_list = []
    sample_ids = []

    for i, item in enumerate(batch):
        images.append(item["image"])
        calib_list.append(item["calib_p2"])
        sample_ids.append(item["sample_id"])
        
        target = item["targets"]
        if target.shape[0] > 0:
            # Prepend batch index (i) to each target row
            batch_idx = torch.full((target.shape[0], 1), i, dtype=torch.float32)
            target = torch.cat([batch_idx, target], dim=1)
            targets_list.append(target)

    images = torch.stack(images, dim=0)
    calibs = torch.stack(calib_list, dim=0)
    
    if len(targets_list) > 0:
        targets = torch.cat(targets_list, dim=0)
    else:
        targets = torch.zeros((0, 14), dtype=torch.float32)

    return {
        "images": images,
        "targets": targets,
        "calibs": calibs,
        "sample_ids": sample_ids
    }
