import torch
from src.utils.checkpoint import save_checkpoint

class Trainer:
    """
    Manages the full training cycle including forward pass, backprop, validation, and checkpointing.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(images)

            feat_h, feat_w = predictions["cls_logits"].shape[2:]
            B = images.shape[0]

            targets_dict = {
                "cls": torch.zeros((B, 3, feat_h, feat_w), device=self.device),
                "bbox2d": torch.zeros((B, 4, feat_h, feat_w), device=self.device),
                "dim3d": torch.zeros((B, 3, feat_h, feat_w), device=self.device),
                "loc3d": torch.zeros((B, 3, feat_h, feat_w), device=self.device),
                "rot": torch.zeros((B, 1, feat_h, feat_w), device=self.device)
            }

            loss, loss_components = self.criterion(predictions, targets_dict)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(num_batches, 1)
        return avg_loss

    def train(self):
        print(f"Starting training for {self.epochs} epochs on {self.device}...")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()

            print(f"Epoch [{epoch}/{self.epochs}] - Train Loss: {train_loss:.4f}")

            # Save checkpoint
            is_best = train_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = train_loss

            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_val_loss
            }, is_best=is_best)
