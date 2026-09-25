import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.dataset import KITTIDataset

def load_config(config_path="configs/dataset/kitti.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_eda():
    config = load_config()
    root_dir = config['dataset']['root_dir']
    
    print("--- Starting KITTI Data Audit & EDA ---")
    if not os.path.exists(os.path.join(root_dir, "training", "image_2")):
        print(f"Dataset path {root_dir} not found or incomplete. Run this script after downloading dataset on GPU server.")
        return

    dataset = KITTIDataset(root_dir=root_dir, split="training")
    print(f"Total Training Samples: {len(dataset)}")

    class_counts = {}
    distances = []
    bbox_heights = []
    bbox_widths = []

    for i in range(len(dataset)):
        sample = dataset[i]
        targets = sample["targets"]
        
        for target in targets:
            cls_id = int(target[0].item())
            dist = target[12].item()
            h = target[5].item()
            w = target[6].item()

            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            distances.append(dist)
            bbox_heights.append(h)
            bbox_widths.append(w)

    print("\n--- Object Class Distribution ---")
    cls_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    for cls_id, count in class_counts.items():
        print(f"{cls_names.get(cls_id, 'Unknown')}: {count}")

    # Generate EDA Visual Plots
    os.makedirs("outputs/figures", exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    sns.histplot(distances, bins=50, kde=True, color='blue')
    plt.title("Object Metric Distance Distribution (Meters)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Count")
    plt.savefig("outputs/figures/eda_distance_distribution.png")
    plt.close()

    print("\nEDA plots successfully saved to 'outputs/figures/eda_distance_distribution.png'")

if __name__ == "__main__":
    run_eda()
