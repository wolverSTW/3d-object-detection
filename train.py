import argparse
import torch
from torch.utils.data import DataLoader
from src.data.dataset import KITTIDataset, collate_fn
from src.models.baseline import YOLO3DBaseline
from src.losses.total_loss import YOLO3DTotalLoss
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.utils.seed import set_seed
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Train YOLO3D Monocular Detection Model")
    parser.add_argument("--config", type=str, default="configs/dataset/kitti.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    root_dir = config['dataset']['root_dir']

    # Dataset & DataLoader
    train_dataset = KITTIDataset(root_dir=root_dir, split="training")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Model, Loss & Optimizer
    model = YOLO3DBaseline(num_classes=3).to(device)
    criterion = YOLO3DTotalLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Engine
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs
    )

    trainer.train()

if __name__ == "__main__":
    main()
