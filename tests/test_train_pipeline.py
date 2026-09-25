import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.baseline import YOLO3DBaseline
from src.losses.total_loss import YOLO3DTotalLoss
from src.engine.trainer import Trainer

class TestTrainingPipeline(unittest.TestCase):

    def test_full_training_loop_mock(self):
        device = torch.device("cpu")
        
        # Mock dataset with 4 dummy images
        dummy_images = torch.randn(4, 3, 375, 1242)
        dummy_targets = torch.zeros(4, 13)
        
        dataset = TensorDataset(dummy_images, dummy_targets)
        
        def dummy_collate_fn(batch):
            images = torch.stack([b[0] for b in batch], dim=0)
            targets = torch.stack([b[1] for b in batch], dim=0)
            return {"images": images, "targets": targets}

        train_loader = DataLoader(dataset, batch_size=2, collate_fn=dummy_collate_fn)

        model = YOLO3DBaseline(num_classes=3).to(device)
        criterion = YOLO3DTotalLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            epochs=1
        )

        # Run 1 epoch training
        avg_loss = trainer.train_epoch(1)
        self.assertGreater(avg_loss, 0.0)

if __name__ == '__main__':
    unittest.main()
