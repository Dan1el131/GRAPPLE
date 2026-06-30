"""
Ablation study for the revised prototype-curvature GRAPPLE methodology.

The old prompting / dual-space / scattering ablations are no longer valid after
the core model rewrite. This script now ablates the actual losses used by the
current paper-aligned implementation:

1. `L_clu`: node-to-prototype clustering
2. `L_etf`: tangent-space simplex / ETF regularization
3. `L_bal`: balanced prototype usage
4. `L_cap`: capacity-matching curvature adaptation
5. `supervised_only`: prototype classifier without geometry losses
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "grapple-mpl"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from grapple.eval import nmi_score
from grapple.experiment_utils import get_representation, train_and_evaluate
from grapple.utils.seed import set_seed


ABLATION_VARIANTS: dict[str, dict] = {
    "full": {
        "description": "Full revised model",
        "train_overrides": {},
    },
    "no_clu": {
        "description": "Without clustering loss",
        "train_overrides": {"lambda_clu": 0.0},
    },
    "no_etf": {
        "description": "Without ETF simplex regularization",
        "train_overrides": {"lambda_etf": 0.0},
    },
    "no_bal": {
        "description": "Without balanced prototype usage",
        "train_overrides": {"lambda_bal": 0.0},
    },
    "no_cap": {
        "description": "Without capacity-matching curvature loss",
        "train_overrides": {"lambda_cap": 0.0},
    },
    "supervised_only": {
        "description": "Prototype classifier only",
        "train_overrides": {
            "lambda_clu": 0.0,
            "lambda_etf": 0.0,
            "lambda_bal": 0.0,
            "lambda_cap": 0.0,
            "lambda_reg": 0.0,
            "lambda_kappa": 0.0,
        },
    },
}


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


def run_variant(
    variant_name: str,
    dataset_name: str,
    data_root: str,
    device: torch.device,
    epochs: int,
    seed: int,
    split: str,
):
    spec = ABLATION_VARIANTS[variant_name]
    model, data, _, metrics = train_and_evaluate(
        dataset_name=dataset_name,
        data_root=data_root,
        device=device,
        seed=seed,
        split=split,
        checkpoint_path=f"checkpoint_{variant_name}_{dataset_name}.pt",
        train_overrides={"epochs": epochs, **spec["train_overrides"]},
    )

    out = metrics["out"]
    rep = get_representation(out, "z")
    nmi_val = nmi_score(rep, data.y.to(device), mask=data.val_mask.to(device))

    return {
        "variant": variant_name,
        "description": spec["description"],
        "train_acc": metrics["train_acc"],
        "val_acc": metrics["val_acc"],
        "test_acc": metrics["test_acc"],
        "best_val_during_train": metrics["best_val_during_train"],
        "kappa": metrics["kappa"],
        "rho": metrics["rho"],
        "nmi_val": nmi_val,
    }


def plot_ablation_results(results_df: pd.DataFrame, dataset_name: str, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ordered = results_df.copy()
    ordered["label"] = ordered["variant"].map(VARIANT_LABELS)
    ordered["color"] = ordered["variant"].map(VARIANT_COLORS)

    ax1 = axes[0]
    bars = ax1.bar(
        ordered["label"],
        ordered["test_acc"],
        color=ordered["color"],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )
    for bar, acc in zip(bars, ordered["test_acc"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_ylabel("Test Accuracy", fontsize=12, fontweight="bold")
    ax1.set_title(f"(a) Variant Accuracy - {dataset_name}", fontsize=13, fontweight="bold", loc="left")
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    ax2 = axes[1]
    full_acc = float(ordered.loc[ordered["variant"] == "full", "test_acc"].iloc[0])
    drop = (full_acc - ordered["test_acc"]) / max(full_acc, 1e-12) * 100.0
    bars = ax2.bar(
        ordered["label"],
        drop,
        color=ordered["color"],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )
    for bar, val in zip(bars, drop):
        if val > 0.05:
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax2.set_ylabel("Drop vs Full (%)", fontsize=12, fontweight="bold")
    ax2.set_title(f"(b) Relative Performance Drop - {dataset_name}", fontsize=13, fontweight="bold", loc="left")
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="public", choices=["public", "random", "ogb"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ABLATION_VARIANTS.keys()),
        choices=list(ABLATION_VARIANTS.keys()),
    )
    parser.add_argument("--output_dir", type=str, default="ablation_results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for variant_name in args.variants:
        print(f"\n{'=' * 70}\nRunning ablation variant: {variant_name}\n{'=' * 70}")
        results.append(
            run_variant(
                variant_name=variant_name,
                dataset_name=args.dataset,
                data_root=args.data_root,
                device=device,
                epochs=args.epochs,
                seed=args.seed,
                split=args.split,
            )
        )

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"ablation_results_{args.dataset}.csv")
    json_path = os.path.join(args.output_dir, f"ablation_results_{args.dataset}.json")
    fig_path = os.path.join(args.output_dir, f"ablation_comparison_{args.dataset}.pdf")
    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    plot_ablation_results(results_df, args.dataset, fig_path)

    display_df = results_df[["variant", "val_acc", "test_acc", "nmi_val", "kappa", "rho"]].copy()
    print("\nAblation summary:")
    print(display_df.to_string(index=False))
    print(f"\nSaved:\n- {csv_path}\n- {json_path}\n- {fig_path}")


if __name__ == "__main__":
    main()
