"""
Sensitivity analysis for the revised prototype-curvature GRAPPLE model.

The old script varied contrastive / prompting / hyperbolic-branch parameters
from the previous codebase. This version only varies parameters that still
exist in the migrated methodology.
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
import torch

from grapple.experiment_utils import DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG, train_and_evaluate
from grapple.utils.seed import set_seed


PARAM_SPACE: dict[str, list[float]] = {
    "tau": [0.05, 0.1, 0.2, 0.5, 1.0],
    "lambda_clu": [0.1, 0.5, 1.0, 2.0, 5.0],
    "lambda_etf": [0.1, 0.5, 1.0, 2.0, 5.0],
    "lambda_bal": [0.01, 0.1, 0.5, 1.0, 2.0],
    "lambda_cap": [0.1, 0.5, 1.0, 2.0, 5.0],
    "cap_margin": [0.0, 0.05, 0.1, 0.2, 0.4],
    "kappa_max": [0.25, 0.5, 1.0, 2.0, 4.0],
    "radius_init": [0.2, 0.35, 0.5, 0.65, 0.8],
}


def run_single_experiment(
    param_name: str,
    param_value: float,
    dataset_name: str,
    data_root: str,
    device: torch.device,
    epochs: int,
    seed: int,
    split: str,
):
    model_overrides = {}
    train_overrides = {"epochs": epochs}

    if param_name in DEFAULT_MODEL_CONFIG:
        model_overrides[param_name] = param_value
    elif param_name in DEFAULT_TRAIN_CONFIG:
        train_overrides[param_name] = param_value
    else:
        raise ValueError(f"Unknown sensitivity parameter: {param_name}")

    _, _, _, metrics = train_and_evaluate(
        dataset_name=dataset_name,
        data_root=data_root,
        device=device,
        seed=seed,
        split=split,
        model_overrides=model_overrides,
        train_overrides=train_overrides,
        checkpoint_path=f"sensitivity_{param_name}_{param_value}_{dataset_name}.pt",
    )

    return {
        "param_name": param_name,
        "param_value": param_value,
        "train_acc": metrics["train_acc"],
        "val_acc": metrics["val_acc"],
        "test_acc": metrics["test_acc"],
        "best_val_during_train": metrics["best_val_during_train"],
        "kappa": metrics["kappa"],
        "rho": metrics["rho"],
    }


def plot_sensitivity_curve(results_df: pd.DataFrame, param_name: str, dataset_name: str, output_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_values = results_df["param_value"].values
    test_acc = results_df["test_acc"].values
    val_acc = results_df["val_acc"].values

    ax1 = axes[0]
    ax1.plot(x_values, test_acc, "o-", linewidth=2.5, markersize=8, color="#2E86AB", label="Test Acc")
    ax1.plot(x_values, val_acc, "s--", linewidth=2.0, markersize=7, color="#A23B72", label="Val Acc")
    best_idx = int(test_acc.argmax())
    ax1.scatter([x_values[best_idx]], [test_acc[best_idx]], s=180, marker="*", color="#F18F01", zorder=5, label="Best")
    ax1.set_xlabel(param_name, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax1.set_title(f"(a) Accuracy vs {param_name} - {dataset_name}", fontsize=13, fontweight="bold", loc="left")
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle="--")
    if param_name in {"lambda_clu", "lambda_etf", "lambda_bal", "lambda_cap", "kappa_max"}:
        ax1.set_xscale("log")

    ax2 = axes[1]
    best_test = float(test_acc.max())
    drop = (best_test - test_acc) / max(best_test, 1e-12) * 100.0
    ax2.bar(
        [str(v) for v in x_values],
        drop,
        color=["#F18F01" if i == best_idx else "#C73E1D" for i in range(len(x_values))],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )
    ax2.set_xlabel(param_name, fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drop from Best (%)", fontsize=12, fontweight="bold")
    ax2.set_title(f"(b) Relative Drop - {dataset_name}", fontsize=13, fontweight="bold", loc="left")
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"sensitivity_{param_name}_{dataset_name}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="public", choices=["public", "random", "ogb"])
    parser.add_argument("--param", type=str, required=True, choices=sorted(PARAM_SPACE.keys()))
    parser.add_argument("--output_dir", type=str, default="sensitivity_results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    values = PARAM_SPACE[args.param]
    results = []
    for value in values:
        print(f"\nTesting {args.param} = {value}")
        results.append(
            run_single_experiment(
                param_name=args.param,
                param_value=value,
                dataset_name=args.dataset,
                data_root=args.data_root,
                device=device,
                epochs=args.epochs,
                seed=args.seed,
                split=args.split,
            )
        )

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"sensitivity_{args.param}_{args.dataset}.csv")
    json_path = os.path.join(args.output_dir, f"sensitivity_{args.param}_{args.dataset}.json")
    fig_path = plot_sensitivity_curve(results_df, args.param, args.dataset, args.output_dir)

    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("\nSensitivity summary:")
    print(results_df[["param_value", "val_acc", "test_acc", "kappa", "rho"]].to_string(index=False))
    print(f"\nSaved:\n- {csv_path}\n- {json_path}\n- {fig_path}")


if __name__ == "__main__":
    main()
