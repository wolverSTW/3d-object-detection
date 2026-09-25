import os
import torch

def save_checkpoint(state, is_best, checkpoint_dir="outputs/checkpoints", filename="latest.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_filepath)

def load_checkpoint(filepath, model, optimizer=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint
