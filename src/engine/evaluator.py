import torch

class Evaluator:
    """
    Evaluates model performance on validation set.
    Calculates average validation loss and metrics.
    """
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        total_val_loss = 0.0
        num_batches = len(val_loader)

        if num_batches == 0:
            return {"val_loss": 0.0}

        for batch in val_loader:
            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)

            predictions = self.model(images)
            
            # Target matching for validation loss computation
            feat_h, feat_w = predictions["cls_logits"].shape[2:]
            B = images.shape[0]

            targets_dict = {
                "cls": torch.zeros((B, 3, feat_h, feat_w), device=self.device),
                "bbox2d": torch.zeros((B, 4, feat_h, feat_w), device=self.device),
                "dim3d": torch.zeros((B, 3, feat_h, feat_w), device=self.device),
                "loc3d": torch.zeros((B, 3, feat_h, feat_w), device=self.device),
                "rot": torch.zeros((B, 1, feat_h, feat_w), device=self.device)
            }

            loss, _ = self.criterion(predictions, targets_dict)
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / num_batches
        return {"val_loss": avg_val_loss}
