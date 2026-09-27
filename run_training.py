#!/usr/bin/env python3
"""Device-aware training launcher for the KITTI pipeline.

Behavior:
- GPU available: run the full KITTI setup and train baseline for 100 epochs
- CPU only: run the setup and launch a short validation pass on a small subset
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    print(f"Project root: {ROOT}")
    device = "cuda" if has_cuda() else "cpu"
    print(f"Detected device: {device}")

    setup_result = run([sys.executable, "scripts/kitti_pipeline.py"])
    if setup_result != 0:
        print("KITTI setup failed. Fix the dataset before training.")
        return setup_result

    if device == "cuda":
        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config",
            "configs/experiments/baseline.yaml",
            "--model",
            "baseline",
            "--epochs",
            "100",
            "--data-dir",
            "data/KITTI/raw",
            "--split",
            "train.txt",
        ]
        print("Running full GPU training pipeline with 100 epochs.")
        return run(cmd)

    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        "configs/experiments/baseline.yaml",
        "--model",
        "baseline",
        "--epochs",
        "1",
        "--data-dir",
        "data/KITTI/raw",
        "--split",
        "train_cpu.txt",
        "--limit",
        "5",
    ]
    print("CUDA not detected. Running CPU validation smoke test.")
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
