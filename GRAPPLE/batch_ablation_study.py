"""
Batch ablation runner for the revised GRAPPLE methodology.
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


VARIANT_ORDER = ["full", "no_clu", "no_etf", "no_bal", "no_cap", "supervised_only"]
VARIANT_LABELS = {
    "full": "Full",
    "no_clu": "w/o Clu",
    "no_etf": "w/o ETF",
    "no_bal": "w/o Bal",
    "no_cap": "w/o Cap",
    "supervised_only": "Sup Only",
}
VARIANT_COLORS = {
    "full": "#2E86AB",
    "no_clu": "#A23B72",
    "no_etf": "#F18F01",
    "no_bal": "#6A4C93",
    "no_cap": "#C73E1D",
    "supervised_only": "#90A959",
}


def run_ablation_on_dataset(dataset: str, data_root: str, device: str, epochs: int, output_dir: str, split: str) -> bool:
    print(f"\n{'=' * 80}\nRunning ablation study on: {dataset}\n{'=' * 80}\n")
    cmd = [
        sys.executable,
        "ablation_study.py",
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
        "--output_dir",
        output_dir,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Failed on {dataset}: {exc}")
        return False


def aggregate_results(datasets: list[str], output_dir: str) -> pd.DataFrame | None:
    frames = []
    for dataset in datasets:
        csv_path = os.path.join(output_dir, f"ablation_results_{dataset}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["dataset"] = dataset
            frames.append(df)
            print(f"Loaded {csv_path}")
        else:
            print(f"Missing results for {dataset}: {csv_path}")
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "ablation_results_all_datasets.csv"), index=False)
    return combined


def plot_cross_dataset_comparison(combined_df: pd.DataFrame, output_dir: str) -> None:
    datasets = list(combined_df["dataset"].unique())

    pivot = combined_df.pivot_table(values="test_acc", index="variant", columns="dataset", aggfunc="mean")
    pivot = pivot.reindex([v for v in VARIANT_ORDER if v in pivot.index])
    pivot.index = [VARIANT_LABELS.get(v, v) for v in pivot.index]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        cbar_kws={"label": "Test Accuracy"},
        linewidths=1,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Ablation Study Across Datasets", fontsize=15, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("Variant", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_dataset_heatmap.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.12
    active_variants = [v for v in VARIANT_ORDER if v in combined_df["variant"].unique()]
    for idx, variant in enumerate(active_variants):
        values = []
        for dataset in datasets:
            hit = combined_df[(combined_df["dataset"] == dataset) & (combined_df["variant"] == variant)]["test_acc"].values
            values.append(hit[0] if len(hit) else np.nan)
        offset = (idx - len(active_variants) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, "#888888"),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Test Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_title("Ablation Performance Comparison", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_dataset_barplot.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    full_scores = combined_df[combined_df["variant"] == "full"][["dataset", "test_acc"]].rename(columns={"test_acc": "full_acc"})
    merged = combined_df.merge(full_scores, on="dataset", how="left")
    merged = merged[merged["variant"] != "full"].copy()
    merged["drop_pct"] = (merged["full_acc"] - merged["test_acc"]) / merged["full_acc"].clip(lower=1e-12) * 100.0
    summary = (
        merged.groupby("variant", as_index=False)["drop_pct"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "avg_drop", "std": "std_drop"})
        .sort_values("avg_drop", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [VARIANT_LABELS.get(v, v) for v in summary["variant"]]
    colors = [VARIANT_COLORS.get(v, "#888888") for v in summary["variant"]]
    bars = ax.barh(labels, summary["avg_drop"], xerr=summary["std_drop"], color=colors, alpha=0.85, edgecolor="black")
    for bar, avg_drop, std_drop in zip(bars, summary["avg_drop"], summary["std_drop"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2.0, f" {avg_drop:.2f}% (+/- {std_drop:.2f})", va="center")
    ax.set_xlabel("Average Performance Drop (%)", fontsize=12, fontweight="bold")
    ax.set_title("Average Drop vs Full Model", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_performance_drop.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_summary_table(combined_df: pd.DataFrame, output_dir: str) -> None:
    summary = (
        combined_df.groupby("variant", as_index=False)
        .agg(
            avg_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            avg_val_acc=("val_acc", "mean"),
            avg_nmi=("nmi_val", "mean"),
            avg_kappa=("kappa", "mean"),
            avg_rho=("rho", "mean"),
        )
    )
    summary["variant_label"] = summary["variant"].map(VARIANT_LABELS)
    summary = summary[["variant", "variant_label", "avg_test_acc", "std_test_acc", "avg_val_acc", "avg_nmi", "avg_kappa", "avg_rho"]]
    summary.to_csv(os.path.join(output_dir, "ablation_summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Batch ablation study")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="ablation_results")
    parser.add_argument("--split", type=str, default="public", choices=["public", "random", "ogb"])
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    if not args.skip_training:
        for dataset in args.datasets:
            results[dataset] = run_ablation_on_dataset(dataset, args.data_root, args.device, args.epochs, args.output_dir, args.split)

    combined_df = aggregate_results(args.datasets, args.output_dir)
    if combined_df is None:
        print("No results found to aggregate.")
        return

    plot_cross_dataset_comparison(combined_df, args.output_dir)
    create_summary_table(combined_df, args.output_dir)

    if results:
        successful = [d for d, ok in results.items() if ok]
        failed = [d for d, ok in results.items() if not ok]
        print(f"\nSuccessful: {len(successful)}/{len(args.datasets)}")
        if successful:
            print("  " + ", ".join(successful))
        if failed:
            print(f"Failed: {len(failed)}/{len(args.datasets)}")
            print("  " + ", ".join(failed))

    print(f"\nBatch ablation outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
