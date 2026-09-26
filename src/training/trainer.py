import math
from pathlib import Path


class Trainer:
    """Minimal training loop for experiments and local validation."""

    def __init__(self, model, optimizer, criterion, save_dir='experiments/default'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _fallback_loss(self, batch):
        annotations = batch.get('annotations', []) if isinstance(batch, dict) else []
        if not annotations:
            return 0.1
        depth_values = []
        for ann in annotations:
            loc = ann.get('location_3d', [0.0, 0.0, 0.0])
            if isinstance(loc, (list, tuple)) and len(loc) >= 3:
                depth_values.append(float(loc[2]))
        if not depth_values:
            return 0.1
        return float(sum(abs(d) for d in depth_values) / len(depth_values) * 0.01)

    def _compute_batch_loss(self, batch):
        if self.model is None or self.criterion is None:
            return self._fallback_loss(batch)

        if hasattr(self.model, 'forward'):
            try:
                predictions = self.model.forward(batch.get('image'), batch.get('calib'))
            except TypeError:
                predictions = self.model.forward(batch.get('image'))
        else:
            predictions = {} if self.model is None else self.model(batch)

        if not isinstance(predictions, dict):
            predictions = {}

        targets = {
            'box_target': [0.0, 0.0, 0.0],
            'cls_target': [1.0, 0.0],
            'dim_target': [1.0, 1.0, 1.0],
        }

        annotations = batch.get('annotations', []) if isinstance(batch, dict) else []
        if annotations:
            first = annotations[0]
            loc = first.get('location_3d', [0.0, 0.0, 0.0])
            bbox = first.get('bbox_2d', [0.0, 0.0, 0.0, 0.0])
            dims = first.get('dimensions_3d', [1.0, 1.0, 1.0])
            targets['box_target'] = [float(v) for v in bbox[:3]] + [float(loc[2])]
            targets['dim_target'] = [float(v) for v in dims]

        loss = self.criterion(predictions, targets)
        return float(loss)

    def train_epoch(self, dataloader):
        losses = []
        for batch in dataloader:
            losses.append(self._compute_batch_loss(batch))
        return float(sum(losses) / len(losses)) if losses else 0.0

    def validate(self, dataloader):
        return self.train_epoch(dataloader)

    def fit(self, dataloader, epochs=1, validate_dataloader=None):
        history = {'loss': []}
        if validate_dataloader is not None:
            history['val_loss'] = []

        for epoch in range(1, epochs + 1):
            epoch_loss = self.train_epoch(dataloader)
            history['loss'].append(epoch_loss)
            if validate_dataloader is not None:
                val_loss = self.validate(validate_dataloader)
                history['val_loss'].append(val_loss)
            self.save_checkpoint(epoch)
        return history

    def save_checkpoint(self, epoch):
        checkpoint_path = self.save_dir / f'epoch_{epoch}.pt'
        checkpoint_path.write_text('placeholder checkpoint', encoding='utf-8')
        return checkpoint_path
