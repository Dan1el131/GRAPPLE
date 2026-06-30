from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "runs" / "aaai_analysis_ready"
OUT_DIR = ROOT / "runs" / "aaai_figures" / "analysis_plots"


def parse_mean_std(value: str) -> tuple[float, float]:
    match = re.match(r"\s*([-+]?\d+(?:\.\d+)?)\s*\+/-\s*([-+]?\d+(?:\.\d+)?)", str(value))
    if not match:
        raise ValueError(f"Cannot parse mean/std from {value!r}")
    return float(match.group(1)), float(match.group(2))


def nice_dataset(name: str) -> str:
    return {
        "amazon-computers": "Amazon\nComputers",
        "amazon-photo": "Amazon\nPhoto",
        "coauthor-cs": "Coauthor\nCS",
        "coauthor-physics": "Coauthor\nPhysics",
        "ogbn-products": "ogbn-\nproducts",
        "actor": "Actor",
        "citeseer": "Citeseer",
    }.get(name, name)


def make_curvature_plot() -> None:
    df = pd.read_csv(DATA_DIR / "curvature_analysis_aggregate.csv")
    keep = ["amazon-computers", "amazon-photo", "coauthor-cs", "coauthor-physics", "ogbn-products"]
    order = ["learned", "fixed_zero", "fixed_positive", "fixed_negative"]
    labels = {
        "learned": "Learned",
        "fixed_zero": "Fixed zero",
        "fixed_positive": "Fixed positive",
        "fixed_negative": "Fixed negative",
    }
    colors = {
        "learned": "#1f77b4",
        "fixed_zero": "#4c4c4c",
        "fixed_positive": "#2ca02c",
        "fixed_negative": "#d62728",
    }
    markers = {
        "learned": "o",
        "fixed_zero": "s",
        "fixed_positive": "^",
        "fixed_negative": "x",
    }
    rows = []
    for _, row in df[df["dataset"].isin(keep)].iterrows():
        mean, std = parse_mean_std(row["test_acc_pct"])
        rows.append({**row.to_dict(), "mean": mean, "std": std})
    plot_df = pd.DataFrame(rows)

    y_base = {dataset: idx for idx, dataset in enumerate(keep)}
    offsets = {"learned": -0.24, "fixed_zero": -0.08, "fixed_positive": 0.08, "fixed_negative": 0.24}

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    for setting in order:
        sub = plot_df[plot_df["setting"] == setting]
        ys = [y_base[d] + offsets[setting] for d in sub["dataset"]]
        ax.errorbar(
            sub["mean"],
            ys,
            xerr=sub["std"],
            fmt=markers[setting],
            ms=4.0,
            lw=0.9,
            capsize=2,
            color=colors[setting],
            label=labels[setting],
        )
    ax.set_yticks([y_base[d] for d in keep])
    ax.set_yticklabels([nice_dataset(d) for d in keep])
    ax.invert_yaxis()
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(pad=0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "curvature_robustness.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "curvature_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_sensitivity_plot() -> None:
    df = pd.read_csv(DATA_DIR / "sensitivity_aggregate.csv")
    rows = []
    for _, row in df.iterrows():
        mean, std = parse_mean_std(row["test_acc_pct"])
        setting = str(row["setting"])
        tau_match = re.search(r"tau=([^_]+)", setting)
        clu_match = re.search(r"lambda_clu=([^_]+)", setting)
        rows.append(
            {
                "dataset": row["dataset"],
                "setting": setting,
                "mean": mean,
                "std": std,
                "tau": float(tau_match.group(1)) if tau_match else None,
                "lambda_clu": float(clu_match.group(1)) if clu_match else None,
            }
        )
    plot_df = pd.DataFrame(rows)
    plot_df["drop"] = plot_df.groupby("dataset")["mean"].transform("max") - plot_df["mean"]

    datasets = ["actor", "amazon-computers", "citeseer", "coauthor-cs"]
    colors = {
        "actor": "#9467bd",
        "amazon-computers": "#1f77b4",
        "citeseer": "#ff7f0e",
        "coauthor-cs": "#2ca02c",
    }
    labels = {
        "actor": "Actor",
        "amazon-computers": "Amazon-Computers",
        "citeseer": "Citeseer",
        "coauthor-cs": "Coauthor-CS",
    }

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.05), sharey=True)

    tau_df = plot_df[plot_df["lambda_clu"].eq(0.01)].copy()
    for dataset in datasets:
        sub = tau_df[tau_df["dataset"].eq(dataset)].sort_values("tau")
        axes[0].errorbar(
            sub["tau"],
            sub["drop"],
            yerr=sub["std"],
            marker="o",
            lw=1.1,
            ms=3.6,
            capsize=2,
            color=colors[dataset],
            label=labels[dataset],
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([0.5, 1, 2])
    axes[0].set_xticklabels(["0.5", "1", "2"])
    axes[0].set_xlabel(r"Temperature $\tau$")
    axes[0].set_ylabel("Accuracy drop from dataset best (%)")
    axes[0].set_title(r"$\lambda_{\mathrm{clu}}=0.01$")

    clu_df = plot_df[plot_df["tau"].eq(2.0)].copy()
    for dataset in datasets:
        sub = clu_df[clu_df["dataset"].eq(dataset)].sort_values("lambda_clu")
        axes[1].errorbar(
            sub["lambda_clu"],
            sub["drop"],
            yerr=sub["std"],
            marker="o",
            lw=1.1,
            ms=3.6,
            capsize=2,
            color=colors[dataset],
            label=labels[dataset],
        )
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[1].set_xticks([0, 1e-4, 1e-3, 1e-2])
    axes[1].set_xticklabels(["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
    axes[1].set_xlabel(r"$\lambda_{\mathrm{clu}}$")
    axes[1].set_title(r"$\tau=2$")

    for ax in axes:
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=-0.15)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.03), frameon=False)
    fig.tight_layout(pad=0.35)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "sensitivity_drop.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "sensitivity_drop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_sensitivity_compact_plot() -> None:
    df = pd.read_csv(DATA_DIR / "sensitivity_aggregate.csv")
    rows = []
    for _, row in df.iterrows():
        mean, std = parse_mean_std(row["test_acc_pct"])
        setting = str(row["setting"])
        tau_match = re.search(r"tau=([^_]+)", setting)
        clu_match = re.search(r"lambda_clu=([^_]+)", setting)
        rows.append(
            {
                "dataset": row["dataset"],
                "mean": mean,
                "std": std,
                "tau": float(tau_match.group(1)) if tau_match else None,
                "lambda_clu": float(clu_match.group(1)) if clu_match else None,
            }
        )
    plot_df = pd.DataFrame(rows)

    # Two representative datasets keep the figure readable in a paper column:
    # one product graph and one heterophilic graph.
    datasets = [
        ("amazon-computers", "Amazon-Computers", "#2866a6"),
        ("actor", "Actor", "#8a5fbf"),
    ]

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.05))
    for row_idx, (dataset, title, color) in enumerate(datasets):
        tau_df = plot_df[
            plot_df["dataset"].eq(dataset) & plot_df["lambda_clu"].eq(0.01)
        ].sort_values("tau")
        clu_df = plot_df[
            plot_df["dataset"].eq(dataset) & plot_df["tau"].eq(2.0)
        ].sort_values("lambda_clu")

        ax_tau = axes[row_idx, 0]
        ax_clu = axes[row_idx, 1]

        ax_tau.errorbar(
            tau_df["tau"],
            tau_df["mean"],
            yerr=tau_df["std"],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.25,
            capsize=2,
        )
        ax_tau.set_xscale("log", base=2)
        ax_tau.set_xticks([0.5, 1, 2])
        ax_tau.set_xticklabels(["0.5", "1", "2"])
        ax_tau.set_xlabel(r"$\tau$")

        ax_clu.errorbar(
            clu_df["lambda_clu"],
            clu_df["mean"],
            yerr=clu_df["std"],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.25,
            capsize=2,
        )
        ax_clu.set_xscale("symlog", linthresh=1e-4)
        ax_clu.set_xticks([0, 1e-4, 1e-3, 1e-2])
        ax_clu.set_xticklabels(["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
        ax_clu.set_xlabel(r"$\lambda_{\mathrm{clu}}$")

        for ax, sub in [(ax_tau, tau_df), (ax_clu, clu_df)]:
            ymin = float((sub["mean"] - sub["std"]).min())
            ymax = float((sub["mean"] + sub["std"]).max())
            pad = max((ymax - ymin) * 0.28, 0.25)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=2.5, width=0.7)

        ax_tau.set_ylabel(f"{title}\nTest acc. (%)")

    axes[0, 0].set_title(r"$\lambda_{\mathrm{clu}}=0.01$")
    axes[0, 1].set_title(r"$\tau=2$")
    fig.tight_layout(pad=0.45, h_pad=0.6, w_pad=0.45)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "sensitivity_compact.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "sensitivity_compact.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_sensitivity_wide_plot() -> None:
    df = pd.read_csv(DATA_DIR / "sensitivity_aggregate.csv")
    rows = []
    for _, row in df.iterrows():
        mean, std = parse_mean_std(row["test_acc_pct"])
        setting = str(row["setting"])
        tau_match = re.search(r"tau=([^_]+)", setting)
        clu_match = re.search(r"lambda_clu=([^_]+)", setting)
        rows.append(
            {
                "dataset": row["dataset"],
                "mean": mean,
                "std": std,
                "tau": float(tau_match.group(1)) if tau_match else None,
                "lambda_clu": float(clu_match.group(1)) if clu_match else None,
            }
        )
    plot_df = pd.DataFrame(rows)

    datasets = [
        ("amazon-computers", "Amazon-Computers", "#2866a6"),
        ("actor", "Actor", "#8a5fbf"),
    ]
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(6.85, 1.85))
    for idx, (dataset, title, color) in enumerate(datasets):
        tau_df = plot_df[
            plot_df["dataset"].eq(dataset) & plot_df["lambda_clu"].eq(0.01)
        ].sort_values("tau")
        clu_df = plot_df[
            plot_df["dataset"].eq(dataset) & plot_df["tau"].eq(2.0)
        ].sort_values("lambda_clu")
        for ax, sub, xlabel in [
            (axes[idx * 2], tau_df, r"$\tau$"),
            (axes[idx * 2 + 1], clu_df, r"$\lambda_{\mathrm{clu}}$"),
        ]:
            x = sub["tau"] if xlabel == r"$\tau$" else sub["lambda_clu"]
            ax.errorbar(
                x,
                sub["mean"],
                yerr=sub["std"],
                color=color,
                marker="o",
                markersize=3.5,
                linewidth=1.25,
                capsize=2,
            )
            ymin = float((sub["mean"] - sub["std"]).min())
            ymax = float((sub["mean"] + sub["std"]).max())
            pad = max((ymax - ymin) * 0.28, 0.25)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=2.5, width=0.7)
            ax.set_xlabel(xlabel)
        axes[idx * 2].set_xscale("log", base=2)
        axes[idx * 2].set_xticks([0.5, 1, 2])
        axes[idx * 2].set_xticklabels(["0.5", "1", "2"])
        axes[idx * 2 + 1].set_xscale("symlog", linthresh=1e-4)
        axes[idx * 2 + 1].set_xticks([0, 1e-4, 1e-3, 1e-2])
        axes[idx * 2 + 1].set_xticklabels(["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
        axes[idx * 2].set_ylabel("Test acc. (%)")
        axes[idx * 2].set_title(title + "\n" + r"vary $\tau$")
        axes[idx * 2 + 1].set_title(title + "\n" + r"vary $\lambda_{\mathrm{clu}}$")
    fig.tight_layout(pad=0.45, w_pad=0.55)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "sensitivity_wide.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "sensitivity_wide.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_curvature_plot()
    make_sensitivity_plot()
    make_sensitivity_compact_plot()
    make_sensitivity_wide_plot()
