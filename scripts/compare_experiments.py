#!/usr/bin/env python3
"""Run a simple side-by-side comparison between baseline and geometry runs."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.experiment_runner import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(description='Compare baseline and geometry experiment summaries.')
    parser.add_argument('--output-dir', type=str, default='experiments/comparison')
    return parser.parse_args()


def build_run_specs():
    ground_truth = [{'distance': 9.0}, {'distance': 13.0}]
    return [
        {
            'model_name': 'baseline',
            'predictions': [{'distance': 10.0, 'AP3D_easy': 0.70}, {'distance': 12.0, 'AP3D_easy': 0.68}],
            'ground_truth': ground_truth,
        },
        {
            'model_name': 'geometry',
            'predictions': [{'distance': 9.5, 'AP3D_easy': 0.75}, {'distance': 11.5, 'AP3D_easy': 0.72}],
            'ground_truth': ground_truth,
        },
    ]


def main():
    args = parse_args()
    runner = ExperimentRunner(output_dir=args.output_dir)
    result = runner.compare_runs(build_run_specs())

    print(f'Comparison saved to: {result["comparison_path"]}')
    for model_name, summary in result['comparison'].items():
        metrics = summary['metrics']
        print(f'{model_name}: MAE_distance={metrics.get("MAE_distance", 0.0):.4f}, RMSE_distance={metrics.get("RMSE_distance", 0.0):.4f}')


if __name__ == '__main__':
    main()
