"""
Batch sensitivity analysis for the revised GRAPPLE methodology.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_sensitivity_on_dataset(dataset: str, data_root: str, device: str, epochs: int, output_dir: str, param: str, split: str) -> bool:
    print(f"\n{'=' * 80}\nRunning sensitivity analysis on: {dataset} | param={param}\n{'=' * 80}\n")
    cmd = [
        sys.executable,
        "sensitivity_analysis.py",
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
        "--param",
        param,
        "--output_dir",
        output_dir,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Failed on {dataset} for {param}: {exc}")
        return False


def plot_cross_dataset_sensitivity(datasets: list[str], param_name: str, output_dir: str) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        csv_file = os.path.join(output_dir, f"sensitivity_{param_name}_{dataset}.csv")
        if not os.path.exists(csv_file):
            continue
        df = pd.read_csv(csv_file)
        ax = axes[idx]
        x_values = df["param_value"].values
        test_acc = df["test_acc"].values
        ax.plot(x_values, test_acc, "o-", linewidth=2.5, markersize=8, color="#2E86AB")
        best_idx = int(test_acc.argmax())
        ax.scatter([x_values[best_idx]], [test_acc[best_idx]], s=180, marker="*", color="#F18F01", zorder=5)
        ax.set_title(dataset, fontsize=13, fontweight="bold")
        ax.set_xlabel(param_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Test Accuracy", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        if param_name in {"lambda_clu", "lambda_etf", "lambda_bal", "lambda_cap", "kappa_max"}:
            ax.set_xscale("log")

    plt.suptitle(f"Sensitivity Analysis: {param_name}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cross_dataset_sensitivity_{param_name}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_cross_dataset_summary(datasets: list[str], params: list[str], output_dir: str) -> pd.DataFrame | None:
    rows = []
    for param in params:
        for dataset in datasets:
            csv_file = os.path.join(output_dir, f"sensitivity_{param}_{dataset}.csv")
            if not os.path.exists(csv_file):
                continue
            df = pd.read_csv(csv_file)
            test_acc = df["test_acc"].values
            values = df["param_value"].values
            best_idx = int(test_acc.argmax())
            rows.append(
                {
                    "Dataset": dataset,
                    "Parameter": param,
                    "Best Value": values[best_idx],
                    "Best Acc": test_acc[best_idx],
                    "Sensitivity (%)": (test_acc.max() - test_acc.min()) / max(test_acc.max(), 1e-12) * 100.0,
                    "Std Dev": test_acc.std(),
                }
            )
    if not rows:
        return None
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(output_dir, "cross_dataset_sensitivity_summary.csv"), index=False)
    return summary_df


def plot_sensitivity_heatmap(datasets: list[str], params: list[str], output_dir: str) -> None:
    matrix = []
    for param in params:
        row = []
        for dataset in datasets:
            csv_file = os.path.join(output_dir, f"sensitivity_{param}_{dataset}.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                test_acc = df["test_acc"].values
                row.append((test_acc.max() - test_acc.min()) / max(test_acc.max(), 1e-12) * 100.0)
            else:
                row.append(np.nan)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        np.array(matrix),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=datasets,
        yticklabels=params,
        cbar_kws={"label": "Sensitivity (%)"},
        linewidths=1,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Parameter Sensitivity Across Datasets", fontsize=15, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Parameter", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensitivity_heatmap.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Batch sensitivity analysis")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output_dir", type=str, default="sensitivity_results")
    parser.add_argument("--params", nargs="+", default=["tau", "lambda_clu", "lambda_etf", "lambda_bal", "lambda_cap", "kappa_max"])
    parser.add_argument("--split", type=str, default="public", choices=["public", "random", "ogb"])
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    if not args.skip_training:
        for dataset in args.datasets:
            for param in args.params:
                results[(dataset, param)] = run_sensitivity_on_dataset(
                    dataset=dataset,
                    data_root=args.data_root,
                    device=args.device,
                    epochs=args.epochs,
                    output_dir=args.output_dir,
                    param=param,
                    split=args.split,
                )

    for param in args.params:
        plot_cross_dataset_sensitivity(args.datasets, param, args.output_dir)
    summary_df = create_cross_dataset_summary(args.datasets, args.params, args.output_dir)
    plot_sensitivity_heatmap(args.datasets, args.params, args.output_dir)

    if results:
        successful = [k for k, ok in results.items() if ok]
        failed = [k for k, ok in results.items() if not ok]
        print(f"\nSuccessful runs: {len(successful)}/{len(results)}")
        if failed:
            print(f"Failed runs: {len(failed)}")
            for dataset, param in failed:
                print(f"  {dataset} | {param}")

    if summary_df is not None:
        print("\nCross-dataset sensitivity summary:")
        print(summary_df.to_string(index=False))
    print(f"\nBatch sensitivity outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
