from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from grapple.data import load_dataset
from grapple.models.grapple import GrappleModel, ModelConfig
from grapple.trainer import masked_accuracy
from grapple.utils.seed import set_seed


DATASET_LABELS = {
    "amazon-computers": "Amazon Computers",
    "amazon-photo": "Amazon Photo",
    "coauthor-cs": "Coauthor CS",
    "coauthor-physics": "Coauthor Physics",
}

USAGE_GRID_CHECKPOINTS = {
    "amazon-computers": Path(
        "server_results/see_22141_20260608/overnight_results/results/prototype_usage/"
        "prototype_usage_amazon-computers_seed0_K2C_tau2_clu0.01_best.pt"
    ),
    "amazon-photo": Path(
        "runs/grapple_paper_figures/prototype_usage_exports/results/prototype_usage/"
        "prototype_usage_amazon-photo_seed0_K2C_tau2_clu0.01_best.pt"
    ),
    "coauthor-cs": Path(
        "server_results/see_22141_20260608/overnight_results/results/prototype_usage/"
        "prototype_usage_coauthor-cs_seed0_K2C_tau2_clu0.01_best.pt"
    ),
    "coauthor-physics": Path(
        "runs/grapple_paper_figures/prototype_usage_exports/results/prototype_usage/"
        "prototype_usage_coauthor-physics_seed0_K2C_tau2_clu0.01_best.pt"
    ),
}

PALETTE = {
    "amazon-computers": "#2866A6",
    "amazon-photo": "#D55E00",
    "coauthor-cs": "#2A9D55",
    "coauthor-physics": "#7B4AB8",
}


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def save_figure(fig: Any, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")


def load_amazon_checkpoint(path: Path, device: torch.device) -> tuple[GrappleModel, dict[str, torch.Tensor]]:
    return load_usage_checkpoint("amazon-computers", path, device)


def normalize_for(dataset: str) -> bool:
    return not dataset.startswith("amazon-")


def load_usage_checkpoint(dataset: str, path: Path, device: torch.device) -> tuple[GrappleModel, dict[str, Any]]:
    data, _, meta = load_dataset(
        dataset,
        root="data",
        normalize_features=normalize_for(dataset),
        split="random",
        train_ratio=0.1,
        val_ratio=0.1,
        test_ratio=0.8,
        seed=0,
    )
    model = GrappleModel(
        ModelConfig(
            in_dim=int(meta["num_features"]),
            num_classes=int(meta["num_classes"]),
            encoder_type="sage",
            gcn_hidden=256,
            gcn_out=128,
            gcn_layers=3,
            proj_dim=128,
            dropout=0.2,
            num_prototypes=int(meta["num_classes"]) * 2,
            tau=2.0,
            prototype_init="simplex",
        )
    )
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device).eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return model, {"data": data, "out": out, "meta": meta, "ckpt": ckpt}


def make_label_rate_curve(fewlabel_csv: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(fewlabel_csv)
    keep = ["amazon-computers", "amazon-photo", "coauthor-cs", "coauthor-physics"]
    rows: list[dict[str, float | str]] = []
    for _, row in df[df["dataset"].isin(keep)].iterrows():
        label_rate = float(str(row["setting"]).split("=")[-1]) * 100.0
        rows.append(
            {
                "dataset": str(row["dataset"]),
                "label_rate": label_rate,
                "mean": float(row["test_acc_mean"]) * 100.0,
                "std": float(row["test_acc_std"]) * 100.0,
            }
        )
    plot_df = pd.DataFrame(rows)
    plot_df.to_csv(out_dir / "label_rate_curve_data.csv", index=False)

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    markers = ["o", "s", "^", "D"]
    for marker, dataset in zip(markers, keep):
        sub = plot_df[plot_df["dataset"].eq(dataset)].sort_values("label_rate")
        ax.errorbar(
            sub["label_rate"],
            sub["mean"],
            yerr=sub["std"],
            color=PALETTE[dataset],
            marker=marker,
            markersize=3.9,
            linewidth=1.35,
            elinewidth=0.75,
            capsize=2.0,
            capthick=0.75,
            label=DATASET_LABELS[dataset],
        )
    ax.set_xlabel("Label rate (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xlim(0.5, 20.8)
    ax.set_ylim(79.0, 97.0)
    ax.set_yticks([80, 84, 88, 92, 96])
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=1.5, handletextpad=0.45)
    fig.tight_layout(pad=0.35)
    save_figure(fig, out_dir, "fig1_label_rate_curve")
    plt.close(fig)


def class_conditioned_usage(alpha: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, num_classes: int, k: int) -> np.ndarray:
    usage = torch.zeros(num_classes, k, dtype=torch.float64)
    for cls in range(num_classes):
        cls_mask = mask & y.eq(cls)
        proto_slice = slice(cls * k, (cls + 1) * k)
        if int(cls_mask.sum().item()) == 0:
            usage[cls] = torch.nan
            continue
        row = alpha[cls_mask, proto_slice].double().mean(dim=0)
        denom = row.sum().clamp_min(1e-12)
        usage[cls] = row / denom
    return usage.cpu().numpy()


def class_to_all_prototype_usage(alpha: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, num_classes: int) -> np.ndarray:
    usage = torch.zeros(num_classes, alpha.size(1), dtype=torch.float64)
    for cls in range(num_classes):
        cls_mask = mask & y.eq(cls)
        if int(cls_mask.sum().item()) == 0:
            usage[cls] = torch.nan
            continue
        row = alpha[cls_mask].double().mean(dim=0)
        usage[cls] = row / row.sum().clamp_min(1e-12)
    return usage.cpu().numpy()


def make_usage_heatmap(payload: dict[str, Any], checkpoint: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.gridspec import GridSpec

    data = payload["data"]
    out = payload["out"]
    meta = payload["meta"]
    num_classes = int(meta["num_classes"])
    k = 2
    within_usage = class_conditioned_usage(
        out["alpha"].detach().cpu(),
        data.y.detach().cpu().view(-1),
        data.test_mask.detach().cpu().view(-1).bool(),
        num_classes,
        k,
    )
    all_usage = class_to_all_prototype_usage(
        out["alpha"].detach().cpu(),
        data.y.detach().cpu().view(-1),
        data.test_mask.detach().cpu().view(-1).bool(),
        num_classes,
    )
    global_usage = out["alpha"].detach().cpu()[data.test_mask.detach().cpu().view(-1).bool()].double().mean(dim=0).numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "prototype_usage_heatmap_test_nodes.csv", within_usage, delimiter=",", fmt="%.8f")
    np.savetxt(out_dir / "prototype_usage_all_prototypes_test_nodes.csv", all_usage, delimiter=",", fmt="%.8f")
    np.savetxt(out_dir / "prototype_usage_global_test_nodes.csv", global_usage[None, :], delimiter=",", fmt="%.8f")
    metadata = {
        "dataset": "amazon-computers",
        "checkpoint": str(checkpoint),
        "seed": 0,
        "split": "random 10/10/80",
        "nodes_used": "test nodes",
        "usage_definition": "Main heatmap: for each true class, average soft assignment over all prototypes on test nodes, then row-normalize. Columns are grouped by prototype owner class. Inset data: class-internal usage over each class's own prototypes.",
        "prototypes_per_class": k,
        "test_accuracy": float(
            masked_accuracy(
                out["logits"].detach().cpu(),
                data.y.detach().cpu(),
                data.test_mask.detach().cpu(),
            )
        ),
        "kappa": float(out["kappa"].detach().cpu().item()),
        "rho": float(out["rho"].detach().cpu().item()),
    }
    (out_dir / "prototype_usage_heatmap_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    configure_matplotlib()
    fig = plt.figure(figsize=(6.85, 3.15))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.42, 3.0],
        width_ratios=[4.8, 1.35],
        hspace=0.08,
        wspace=0.18,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_in = fig.add_subplot(gs[1, 1])

    im = ax.imshow(all_usage, cmap="mako_r" if "mako_r" in plt.colormaps() else "YlGnBu", vmin=0.0, vmax=max(0.12, float(np.nanpercentile(all_usage, 99))), aspect="auto")
    ax.set_xlabel("Prototype group")
    ax.set_ylabel("True class")
    ax.set_yticks(np.arange(num_classes))
    ax.set_yticklabels([str(i) for i in range(num_classes)])
    ax.set_xticks([cls * k + 0.5 for cls in range(num_classes)])
    ax.set_xticklabels([str(cls) for cls in range(num_classes)])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0)

    for cls in range(num_classes + 1):
        x = cls * k - 0.5
        ax.axvline(x, color="white", linewidth=0.65, alpha=0.9)
        if cls < num_classes:
            rect = patches.Rectangle(
                (cls * k - 0.5, cls - 0.5),
                k,
                1,
                fill=False,
                edgecolor="#111111",
                linewidth=0.9,
            )
            ax.add_patch(rect)
    for row in range(num_classes + 1):
        ax.axhline(row - 0.5, color="white", linewidth=0.35, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax_top.bar(np.arange(num_classes * k), global_usage, width=0.82, color="#6B7280", edgecolor="none")
    ax_top.set_xlim(-0.5, num_classes * k - 0.5)
    ax_top.set_ylim(0.0, max(global_usage.max() * 1.25, 0.02))
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_ylabel("Global", rotation=0, labelpad=19, va="center", fontsize=7)
    for cls in range(num_classes + 1):
        ax_top.axvline(cls * k - 0.5, color="white", linewidth=0.65)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    im_in = ax_in.imshow(within_usage, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax_in.set_xlabel("Within class")
    ax_in.set_xticks(np.arange(k))
    ax_in.set_xticklabels([f"P{i + 1}" for i in range(k)])
    ax_in.set_yticks(np.arange(num_classes))
    ax_in.set_yticklabels([])
    ax_in.tick_params(length=0)
    ax.set_yticks(np.arange(num_classes))
    for i in range(num_classes):
        for j in range(k):
            val = within_usage[i, j]
            ax_in.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.6, color="#111111" if val < 0.62 else "white")
    for spine in ax_in.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=[ax_top, ax], fraction=0.025, pad=0.012)
    cbar.set_label("Mean assignment")
    cbar.outline.set_linewidth(0.6)
    save_figure(fig, out_dir, "fig2_prototype_usage_heatmap")
    plt.close(fig)


def make_usage_grid(out_dir: Path, device: torch.device) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    rows: list[dict[str, Any]] = []
    for dataset, checkpoint in USAGE_GRID_CHECKPOINTS.items():
        if not checkpoint.exists():
            continue
        _, payload = load_usage_checkpoint(dataset, checkpoint, device)
        data = payload["data"]
        out = payload["out"]
        meta = payload["meta"]
        num_classes = int(meta["num_classes"])
        usage = class_conditioned_usage(
            out["alpha"].detach().cpu(),
            data.y.detach().cpu().view(-1),
            data.test_mask.detach().cpu().view(-1).bool(),
            num_classes,
            2,
        )
        acc = float(masked_accuracy(out["logits"].detach().cpu(), data.y.detach().cpu(), data.test_mask.detach().cpu()))
        rows.append({"dataset": dataset, "checkpoint": checkpoint, "usage": usage, "acc": acc, "classes": num_classes})
        np.savetxt(out_dir / f"prototype_usage_within_class_{dataset}_test_nodes.csv", usage, delimiter=",", fmt="%.8f")

    if not rows:
        return

    max_classes = max(int(row["classes"]) for row in rows)
    ncols = len(rows)
    fig_width = max(2.0 * ncols + 0.35, 6.2)
    fig_height = max(0.26 * max_classes + 1.15, 2.55)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes[0]
    im = None
    for ax, row in zip(axes_flat, rows):
        usage = row["usage"]
        im = ax.imshow(usage, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"{DATASET_LABELS[row['dataset']]}\n{100.0 * row['acc']:.1f}%", pad=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["P1", "P2"])
        ax.set_yticks(np.arange(int(row["classes"])))
        ax.set_yticklabels([str(i) for i in range(int(row["classes"]))])
        ax.tick_params(length=0, pad=1.5)
        ax.set_xlabel("Prototype")
        if ax is axes_flat[0]:
            ax.set_ylabel("Class")
        else:
            ax.set_ylabel("")
        for i in range(usage.shape[0] + 1):
            ax.axhline(i - 0.5, color="white", linewidth=0.45, alpha=0.85)
        ax.axvline(0.5, color="white", linewidth=0.7, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    assert im is not None
    cbar = fig.colorbar(im, ax=list(axes_flat), fraction=0.025, pad=0.012)
    cbar.set_label("Class-normalized usage")
    cbar.outline.set_linewidth(0.6)
    meta = {
        "datasets": [
            {
                "dataset": row["dataset"],
                "checkpoint": str(row["checkpoint"]),
                "classes": int(row["classes"]),
                "test_accuracy": row["acc"],
            }
            for row in rows
        ],
        "nodes_used": "test nodes",
        "usage_definition": "For each true class, average soft assignment over that class's own two prototypes, then row-normalize.",
        "missing": [dataset for dataset, checkpoint in USAGE_GRID_CHECKPOINTS.items() if not checkpoint.exists()],
    }
    (out_dir / "prototype_usage_grid_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_figure(fig, out_dir, "fig2b_prototype_usage_grid")
    plt.close(fig)


def stratified_test_sample(y: np.ndarray, test_mask: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for cls in np.unique(y):
        idx = np.flatnonzero((y == cls) & test_mask)
        if len(idx) == 0:
            continue
        size = min(max_per_class, len(idx))
        picked.extend(rng.choice(idx, size=size, replace=False).tolist())
    return np.array(sorted(picked), dtype=int)


def clean_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        x = np.clip(x, lo, hi)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    return np.nan_to_num(x / np.maximum(scale, 1e-6), nan=0.0, posinf=0.0, neginf=0.0)


def make_tsne(payload: dict[str, Any], checkpoint: Path, out_dir: Path, seed: int) -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    data = payload["data"]
    out = payload["out"]
    meta = payload["meta"]
    y = data.y.detach().cpu().view(-1).numpy()
    test_mask = data.test_mask.detach().cpu().view(-1).bool().numpy()
    node_idx = stratified_test_sample(y, test_mask, max_per_class=160, seed=seed)

    node_emb = out["v_clipped"].detach().cpu().numpy()
    proto_emb = (out["rho"] * out["prototype_directions"]).detach().cpu().numpy()
    proto_class = out["prototype_to_class"].detach().cpu().numpy()
    combined = np.vstack([node_emb[node_idx], proto_emb])
    xy = TSNE(
        n_components=2,
        random_state=seed,
        init="random",
        learning_rate="auto",
        perplexity=30,
    ).fit_transform(clean_features(combined))
    node_xy = xy[: len(node_idx)]
    proto_xy = xy[len(node_idx) :]

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tsne_nodes_and_prototypes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "index", "x", "y", "class"])
        for local_i, global_i in enumerate(node_idx):
            writer.writerow(["node", int(global_i), float(node_xy[local_i, 0]), float(node_xy[local_i, 1]), int(y[global_i])])
        for proto_i in range(len(proto_xy)):
            writer.writerow(["prototype", proto_i, float(proto_xy[proto_i, 0]), float(proto_xy[proto_i, 1]), int(proto_class[proto_i])])
    metadata = {
        "dataset": "amazon-computers",
        "checkpoint": str(checkpoint),
        "seed": seed,
        "nodes_used": "class-balanced test-node sample",
        "max_test_nodes_per_class": 160,
        "num_sampled_nodes": int(len(node_idx)),
        "num_prototypes": int(len(proto_xy)),
        "representation": "Joint t-SNE on v_clipped node tangent vectors and rho * prototype_directions prototype tangent vectors.",
        "test_accuracy": float(masked_accuracy(out["logits"].detach().cpu(), data.y.detach().cpu(), data.test_mask.detach().cpu())),
        "kappa": float(out["kappa"].detach().cpu().item()),
        "rho": float(out["rho"].detach().cpu().item()),
    }
    (out_dir / "tsne_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    configure_matplotlib()
    num_classes = int(meta["num_classes"])
    cmap = plt.get_cmap("tab10", num_classes)
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ax.scatter(
        node_xy[:, 0],
        node_xy[:, 1],
        c=y[node_idx],
        cmap=cmap,
        s=6.5,
        alpha=0.58,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        proto_xy[:, 0],
        proto_xy[:, 1],
        c=proto_class,
        cmap=cmap,
        s=135,
        marker="*",
        edgecolors="black",
        linewidths=0.85,
        zorder=5,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#222222")
    node_handle = mlines.Line2D([], [], color="#7A7A7A", marker="o", linestyle="None", markersize=4, label="Node")
    proto_handle = mlines.Line2D(
        [],
        [],
        color="black",
        marker="*",
        linestyle="None",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        label="Prototype",
    )
    ax.legend(handles=[node_handle, proto_handle], loc="upper right", frameon=True, framealpha=0.95, borderpad=0.35)
    fig.tight_layout(pad=0.2)
    save_figure(fig, out_dir, "fig3_tsne_with_learned_prototypes")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready GRAPPLE figures.")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/grapple_paper_figures"))
    parser.add_argument(
        "--fewlabel_csv",
        type=Path,
        default=Path("infosci_4datasets_20260628/fewlabel_aggregate.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "server_results/see_22141_20260608/overnight_results/results/prototype_usage/"
            "prototype_usage_amazon-computers_seed0_K2C_tau2_clu0.01_best.pt"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_label_rate_curve(args.fewlabel_csv, args.out_dir)
    _, payload = load_amazon_checkpoint(args.checkpoint, torch.device(args.device))
    make_usage_heatmap(payload, args.checkpoint, args.out_dir)
    make_usage_grid(args.out_dir, torch.device(args.device))
    make_tsne(payload, args.checkpoint, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
