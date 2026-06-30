"""
Batch runner for the revised rank-collapse experiment.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run_experiment(dataset: str, data_root: str, device: str, epochs: int, output_dir: str, split: str, representation: str) -> bool:
    print(f"\n{'=' * 80}\nRunning rank-collapse experiment on: {dataset}\n{'=' * 80}\n")
    cmd = [
        sys.executable,
        "rank_collapse_experiment.py",
        "--dataset",
        dataset,
        "--data_root",
        data_root,
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--split",
        split,
        "--representation",
        representation,
        "--output_dir",
        output_dir,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Failed on {dataset}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch rank-collapse experiments")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="rank_collapse_results")
    parser.add_argument("--split", type=str, default="public", choices=["public", "random", "ogb"])
    parser.add_argument("--representation", type=str, default="v_clipped", choices=["h", "z", "v", "v_clipped", "x_manifold", "logits"])
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for dataset in args.datasets:
        results[dataset] = run_experiment(
            dataset=dataset,
            data_root=args.data_root,
            device=args.device,
            epochs=args.epochs,
            output_dir=args.output_dir,
            split=args.split,
            representation=args.representation,
        )

    successful = [d for d, ok in results.items() if ok]
    failed = [d for d, ok in results.items() if not ok]
    print(f"\nSuccessful: {len(successful)}/{len(args.datasets)}")
    if successful:
        print("  " + ", ".join(successful))
    if failed:
        print(f"Failed: {len(failed)}/{len(args.datasets)}")
        print("  " + ", ".join(failed))
    print(f"\nBatch rank-collapse outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
