#!/usr/bin/env python3
"""One-time KITTI setup pipeline.

The workflow is intentionally idempotent:
1. Download the archive files once if raw data is missing.
2. Extract the archives once if they have not been extracted.
3. Allocate the dataset into the expected raw layout.
4. Run a lightweight EDA summary once.
5. Generate the training split once.

This script skips any completed stage automatically on later runs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List

import requests

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

KITTI_URLS = {
    "images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
}
REQUIRED_DIRS = ["image_2", "label_2", "calib"]


def resolve_kitti_dirs(raw_root: Path) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for name in REQUIRED_DIRS:
        direct = raw_root / name
        training = raw_root / "training" / name
        if direct.exists():
            resolved[name] = direct
        elif training.exists():
            resolved[name] = training
        else:
            resolved[name] = direct
    return resolved


def _load_state(state_path: Path) -> Dict[str, Dict[str, bool]]:
    if not state_path.exists():
        return {"completed_steps": {}}
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {"completed_steps": {}}

    if not isinstance(state, dict):
        return {"completed_steps": {}}
    steps = state.get("completed_steps", {})
    if not isinstance(steps, dict):
        return {"completed_steps": {}}
    return {"completed_steps": steps}


def _save_state(state_path: Path, state: Dict[str, Dict[str, bool]]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")


def ensure_kitti_layout(data_root: Path) -> None:
    (data_root / "raw").mkdir(parents=True, exist_ok=True)
    (data_root / "splits").mkdir(parents=True, exist_ok=True)
    for folder in REQUIRED_DIRS:
        (data_root / "raw" / folder).mkdir(parents=True, exist_ok=True)


def dataset_ready(raw_root: Path) -> bool:
    if not raw_root.exists():
        return False
    resolved = resolve_kitti_dirs(raw_root)
    return all(path.exists() and any(path.iterdir()) for path in resolved.values())


def split_ready(data_root: Path) -> bool:
    split_file = data_root / "splits" / "train.txt"
    return split_file.exists() and split_file.stat().st_size > 0


def archive_is_valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def download_kitti_dataset(data_root: Path) -> Dict[str, object]:
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    if dataset_ready(raw_root):
        return {"download_required": False, "downloaded": [], "message": "KITTI dataset already present."}

    downloaded: List[str] = []
    for name, url in KITTI_URLS.items():
        archive_name = url.rsplit("/", 1)[-1]
        archive_path = raw_root / archive_name

        if archive_is_valid(archive_path):
            downloaded.append(str(archive_path))
            continue

        if archive_path.exists():
            archive_path.unlink()

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with archive_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

        if not archive_is_valid(archive_path):
            raise ValueError(f"Downloaded archive is invalid: {archive_path}")

        downloaded.append(str(archive_path))

    return {"download_required": True, "downloaded": downloaded, "message": "KITTI archives downloaded."}


def extract_kitti_archives(data_root: Path) -> Dict[str, object]:
    raw_root = data_root / "raw"
    archives = sorted(raw_root.glob("*.zip"))
    if not archives:
        return {"extracted": 0, "message": "No KITTI archives found to extract."}

    extracted = 0
    for archive in archives:
        if not archive_is_valid(archive):
            raise ValueError(f"Skipping invalid KITTI archive: {archive}")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(raw_root)
        extracted += 1

    return {"extracted": extracted, "message": f"Extracted {extracted} archive(s)."}


def allocate_kitti_dataset(data_root: Path) -> Dict[str, object]:
    raw_root = data_root / "raw"
    training_root = raw_root / "training"

    for folder in REQUIRED_DIRS:
        source = training_root / folder
        destination = raw_root / folder
        destination.mkdir(parents=True, exist_ok=True)

        if source.exists():
            for item in source.iterdir():
                target = destination / item.name
                if item.is_dir():
                    if not target.exists():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    if not target.exists():
                        shutil.copy2(item, target)

    resolved = resolve_kitti_dirs(raw_root)
    for folder, path in resolved.items():
        if not path.exists() or not any(path.iterdir()):
            raise FileNotFoundError(f"KITTI allocation incomplete for {path}")

    return {"allocated": REQUIRED_DIRS, "message": "KITTI raw layout allocated."}


def run_eda_summary(data_root: Path) -> Dict[str, object]:
    raw_root = data_root / "raw"
    summary = {}

    for folder in REQUIRED_DIRS:
        folder_path = raw_root / folder
        files = sorted(p.name for p in folder_path.glob("*") if p.exists()) if folder_path.exists() else []
        summary[folder] = {
            "file_count": len(files),
            "sample": files[0] if files else None,
        }

    summary_path = data_root / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary_path": str(summary_path), "summary": summary}


def create_training_split(data_root: Path) -> Dict[str, object]:
    raw_root = data_root / "raw"
    image_dir = resolve_kitti_dirs(raw_root)["image_2"]
    split_dir = data_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"image_2 not found in {raw_root}")

    ids = sorted(p.stem for p in image_dir.glob("*.png"))
    if not ids:
        raise FileNotFoundError(f"No PNG files found in {image_dir}")

    split_file = split_dir / "train.txt"
    split_file.write_text("\n".join(ids) + "\n", encoding="utf-8")

    return {"created": len(ids), "split_file": str(split_file), "message": f"Wrote {len(ids)} training IDs."}


def generate_cpu_validation_split(data_root: Path, max_items: int = 5) -> Dict[str, object]:
    split_dir = data_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    image_dir = resolve_kitti_dirs(data_root / "raw")["image_2"]
    if not image_dir.exists():
        raise FileNotFoundError(f"image_2 is missing from {data_root}")

    ids = sorted(p.stem for p in image_dir.glob("*.png"))
    if not ids:
        raise FileNotFoundError(f"No PNG files found in {image_dir}")

    subset = ids[:max_items]
    split_path = split_dir / "train_cpu.txt"
    split_path.write_text("\n".join(subset) + "\n", encoding="utf-8")

    return {
        "split_file": str(split_path),
        "sample_count": len(subset),
        "samples": subset,
        "message": f"Created CPU validation split with {len(subset)} sample(s).",
    }


def run_device_aware_pipeline(data_root: str | Path = "data/KITTI") -> Dict[str, object]:
    data_root = Path(data_root)
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    pipeline_result = run_pipeline(data_root=data_root)
    if device == "cpu":
        cpu_split = generate_cpu_validation_split(data_root)
        pipeline_result["device"] = "cpu"
        pipeline_result["cpu_validation"] = cpu_split
        pipeline_result["mode"] = "cpu_smoke_test"
        pipeline_result["training_command"] = [
            "python",
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
            str(min(5, cpu_split["sample_count"])),
        ]
        return pipeline_result

    pipeline_result["device"] = "cuda"
    pipeline_result["mode"] = "gpu_full_training"
    pipeline_result["training_command"] = [
        "python",
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
    return pipeline_result


def run_pipeline(
    data_root: str | Path = "data/KITTI",
    *,
    skip_download: bool = False,
    skip_extract: bool = False,
    skip_allocate: bool = False,
    skip_eda: bool = False,
    skip_split: bool = False,
) -> Dict[str, object]:
    data_root = Path(data_root)
    raw_root = data_root / "raw"
    state_path = data_root / ".pipeline_state.json"
    state = _load_state(state_path)
    completed = state.setdefault("completed_steps", {})

    ensure_kitti_layout(data_root)

    status: Dict[str, object] = {"download_required": False, "status": {}}

    if not skip_download:
        if not dataset_ready(raw_root):
            download_result = download_kitti_dataset(data_root)
            status["download_required"] = bool(download_result.get("downloaded"))
            status["status"]["download"] = download_result
            completed["download"] = True
        else:
            completed.setdefault("download", True)
            status["status"]["download"] = {"download_required": False, "message": "Dataset already present."}
    else:
        completed.setdefault("download", True)

    if not skip_extract:
        if completed.get("extract") is True:
            status["status"]["extract"] = {"message": "Skipped: already extracted."}
        else:
            extract_result = extract_kitti_archives(data_root)
            status["status"]["extract"] = extract_result
            completed["extract"] = True
    else:
        completed.setdefault("extract", True)

    if not skip_allocate:
        if completed.get("allocation") is True:
            status["status"]["allocation"] = {"message": "Skipped: already allocated."}
        else:
            allocate_result = allocate_kitti_dataset(data_root)
            status["status"]["allocation"] = allocate_result
            completed["allocation"] = True
    else:
        completed.setdefault("allocation", True)

    if not skip_eda:
        if completed.get("eda") is True:
            status["status"]["eda"] = {"message": "Skipped: EDA already completed."}
        else:
            eda_result = run_eda_summary(data_root)
            status["status"]["eda"] = eda_result
            completed["eda"] = True
    else:
        completed.setdefault("eda", True)

    if not skip_split:
        if completed.get("split") is True:
            status["status"]["split"] = {"message": "Skipped: split already created."}
        else:
            split_result = create_training_split(data_root)
            status["status"]["split"] = split_result
            completed["split"] = True
    else:
        completed.setdefault("split", True)

    _save_state(state_path, state)
    dataset_ready_flag = dataset_ready(raw_root) and split_ready(data_root)
    status["dataset_ready"] = dataset_ready_flag
    status["completed_steps"] = completed
    status["state_path"] = str(state_path)
    return status


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the one-time KITTI dataset setup workflow.")
    parser.add_argument("--data-root", default="data/KITTI", help="Path to the KITTI dataset directory.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the archive download step.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip the ZIP extraction step.")
    parser.add_argument("--skip-allocate", action="store_true", help="Skip the raw-layout allocation step.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip the EDA summary step.")
    parser.add_argument("--skip-split", action="store_true", help="Skip the train split creation step.")
    args = parser.parse_args()

    result = run_device_aware_pipeline(data_root=args.data_root)

    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("device") == "cpu":
        print("CPU detected: running a small real-data validation pass instead of the 100-epoch GPU training pipeline.")
    else:
        print("GPU detected: running the full 100-epoch training pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
