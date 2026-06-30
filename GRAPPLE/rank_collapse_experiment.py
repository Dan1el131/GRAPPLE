"""
Rank-collapse analysis for the revised GRAPPLE methodology.

The old experiment compared legacy scattering variants. After the model rewrite,
the most relevant comparison is whether the new geometric regularizers preserve
representation rank relative to weaker baselines.
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
import seaborn as sns
import torch

from grapple.experiment_utils import get_representation, train_and_evaluate
from grapple.utils.seed import set_seed


RANK_VARIANTS: dict[str, dict] = {
    "full": {
        "label": "Full",
        "train_overrides": {},
    },
    "no_etf": {
        "label": "w/o ETF",
        "train_overrides": {"lambda_etf": 0.0},
    },
    "supervised_only": {
        "label": "Sup Only",
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


def extract_embeddings(model, data, device: torch.device, representation: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        rep = get_representation(out, representation)
        mask = data.val_mask.to(device)
        if mask.dim() > 1:
            mask = mask[:, 0]
        return rep[mask].detach().cpu().numpy()


def compute_singular_values(embeddings: np.ndarray) -> np.ndarray:
    _, singular_values, _ = np.linalg.svd(embeddings, full_matrices=False)
    return singular_values


def compute_rank_metrics(singular_values: np.ndarray, threshold: float = 0.01) -> dict[str, float]:
    normalized = singular_values / (np.sum(singular_values) + 1e-12)
    entropy = -np.sum(normalized * np.log(normalized + 1e-12))
    effective_rank = float(np.exp(entropy))
    stable_rank = float(np.sum(singular_values ** 2) / (singular_values[0] ** 2 + 1e-12))
    numerical_rank = int(np.sum(singular_values >= threshold * singular_values[0]))
    spectral_gap = float(singular_values[0] / (singular_values[1] + 1e-12)) if len(singular_values) > 1 else float("inf")
    return {
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "numerical_rank": numerical_rank,
        "spectral_gap": spectral_gap,
    }


def plot_singular_value_spectrum(results: dict[str, np.ndarray], dataset_name: str, save_path: str) -> None:
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    colors = {
        "Full": "#2E86AB",
        "w/o ETF": "#A23B72",
        "Sup Only": "#F18F01",
    }
    styles = {
        "Full": "-",
        "w/o ETF": "--",
        "Sup Only": ":",
    }

    for label, singular_vals in results.items():
        values = np.log10(singular_vals + 1e-10)
        count = min(50, len(values))
        x = np.arange(1, count + 1)
        plt.plot(x, values[:count], label=label, color=colors[label], linestyle=styles[label], linewidth=2.5)

    plt.xlabel("Singular Value Index", fontsize=13, fontweight="bold")
    plt.ylabel("log10 Singular Value", fontsize=13, fontweight="bold")
    plt.title(f"Singular Value Spectrum - {dataset_name}", fontsize=15, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="public", choices=["public", "random", "ogb"])
    parser.add_argument("--representation", type=str, default="v_clipped", choices=["h", "z", "v", "v_clipped", "x_manifold", "logits"])
    parser.add_argument("--output_dir", type=str, default="rank_collapse_results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    spectra: dict[str, np.ndarray] = {}
    metrics_dict: dict[str, dict[str, float]] = {}

    for variant_name, spec in RANK_VARIANTS.items():
        print(f"\n{'=' * 70}\nRank experiment variant: {variant_name}\n{'=' * 70}")
        model, data, _, metrics = train_and_evaluate(
            dataset_name=args.dataset,
            data_root=args.data_root,
            device=device,
            seed=args.seed,
            split=args.split,
            checkpoint_path=f"checkpoint_{variant_name}_{args.dataset}.pt",
            train_overrides={"epochs": args.epochs, "eval_interval": max(10, min(args.epochs, 25)), **spec["train_overrides"]},
        )
        embeddings = extract_embeddings(model, data, device, args.representation)
        singular_values = compute_singular_values(embeddings)
        label = spec["label"]
        spectra[label] = singular_values
        metrics_dict[label] = compute_rank_metrics(singular_values)
        metrics_dict[label]["test_acc"] = metrics["test_acc"]
        metrics_dict[label]["kappa"] = metrics["kappa"]

    spectrum_path = os.path.join(args.output_dir, f"singular_spectrum_{args.dataset}.pdf")
    metrics_path = os.path.join(args.output_dir, f"rank_metrics_{args.dataset}.json")
    npz_path = os.path.join(args.output_dir, f"singular_values_{args.dataset}.npz")

    plot_singular_value_spectrum(spectra, args.dataset, spectrum_path)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_dict, fh, indent=2)
    np.savez(npz_path, **spectra)

    print("\nRank metrics:")
    for label, metrics in metrics_dict.items():
        print(
            f"{label:>10} | "
            f"eff_rank={metrics['effective_rank']:.2f} | "
            f"stable_rank={metrics['stable_rank']:.2f} | "
            f"num_rank={metrics['numerical_rank']} | "
            f"gap={metrics['spectral_gap']:.2f} | "
            f"test_acc={metrics['test_acc']:.4f} | "
            f"kappa={metrics['kappa']:.4f}"
        )

    print(f"\nSaved:\n- {spectrum_path}\n- {metrics_path}\n- {npz_path}")


if __name__ == "__main__":
    main()
