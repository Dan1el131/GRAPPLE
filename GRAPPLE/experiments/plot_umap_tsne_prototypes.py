#!/usr/bin/env python3
"""Visualize GRAPPLE node embeddings together with learned prototypes.

Example:
  python3 experiments/plot_umap_tsne_prototypes.py \
    --dataset coauthor-cs \
    --checkpoint server_results/see_22141_20260608/overnight_results/results/prototype_usage/prototype_usage_coauthor-cs_seed0_K2C_tau2_clu0.01_best.pt \
    --output_dir runs/aaai_figures/embedding_coauthor_cs \
    --method tsne
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grapple.data import load_dataset
from grapple.models.grapple import GrappleModel, ModelConfig
from grapple.trainer import masked_accuracy
from grapple.utils.seed import set_seed


def normalize_for(dataset: str) -> bool:
    return not dataset.startswith("amazon-") and dataset != "ogbn-products"


def encoder_for(dataset: str) -> str:
    if dataset.startswith(("amazon-", "coauthor-")) or dataset == "ogbn-products":
        return "sage"
    return "gcn"


def split_kwargs(dataset: str) -> dict[str, object]:
    if dataset == "ogbn-products":
        return {"split": "ogb"}
    kwargs: dict[str, object] = {"split": "random", "train_ratio": 0.1, "val_ratio": 0.1, "test_ratio": 0.8}
    if dataset in {"actor", "chameleon", "squirrel"}:
        kwargs["to_undirected"] = True
    return kwargs


def stratified_sample(y: np.ndarray, max_nodes: int, seed: int) -> np.ndarray:
    if max_nodes <= 0 or len(y) <= max_nodes:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_nodes // max(1, len(classes)))
    picked: list[int] = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0:
            continue
        size = min(len(idx), per_class)
        picked.extend(rng.choice(idx, size=size, replace=False).tolist())
    if len(picked) < max_nodes:
        remaining = np.setdiff1d(np.arange(len(y)), np.array(picked, dtype=int), assume_unique=False)
        size = min(len(remaining), max_nodes - len(picked))
        if size > 0:
            picked.extend(rng.choice(remaining, size=size, replace=False).tolist())
    return np.array(sorted(set(picked)), dtype=int)


def reduce_2d(x: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise SystemExit("UMAP is not installed. Use --method tsne or install umap-learn.") from exc
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=20, min_dist=0.15).fit_transform(x)
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(x)
    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, (len(x) - 1) // 3))
    return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw t-SNE/UMAP visualization with GRAPPLE prototypes.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/aaai_figures/embedding_with_prototypes"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--representation", choices=["v_clipped", "x_manifold", "z_unit"], default="v_clipped")
    parser.add_argument("--max_nodes", type=int, default=2500)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--prototypes_per_class", type=int, default=2)
    parser.add_argument("--tau", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(args.seed)
    data, _, meta = load_dataset(
        args.dataset,
        root=args.data_root,
        normalize_features=normalize_for(args.dataset),
        seed=args.seed,
        **split_kwargs(args.dataset),
    )
    num_classes = int(meta["num_classes"])
    model = GrappleModel(
        ModelConfig(
            in_dim=int(meta["num_features"]),
            num_classes=num_classes,
            encoder_type=encoder_for(args.dataset),
            gcn_hidden=args.hidden,
            gcn_out=args.out_dim,
            gcn_layers=args.layers,
            proj_dim=args.proj_dim,
            dropout=args.dropout,
            num_prototypes=num_classes * int(args.prototypes_per_class),
            tau=args.tau,
            prototype_init="simplex",
        )
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    device = torch.device(args.device)
    model.to(device).eval()

    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        y = data.y.detach().cpu().view(-1).numpy()
        pred = out["logits"].argmax(dim=-1).detach().cpu().numpy()
        test_acc = masked_accuracy(out["logits"].detach().cpu(), data.y.detach().cpu(), data.test_mask.detach().cpu())

    if args.representation == "v_clipped":
        node_emb = out["v_clipped"].detach().cpu().numpy()
        proto_emb = (out["rho"] * out["prototype_directions"]).detach().cpu().numpy()
        axis_note = "tangent-space embeddings"
    elif args.representation == "x_manifold":
        node_emb = out["x_manifold"].detach().cpu().numpy()
        proto_emb = out["prototypes"].detach().cpu().numpy()
        axis_note = "manifold coordinate embeddings"
    else:
        node_emb = out["z_unit"].detach().cpu().numpy()
        proto_emb = out["prototype_directions"].detach().cpu().numpy()
        axis_note = "normalized projected embeddings"

    node_idx = stratified_sample(y, args.max_nodes, args.seed)
    combined = np.vstack([node_emb[node_idx], proto_emb])
    xy = reduce_2d(combined, args.method, args.seed)
    node_xy = xy[: len(node_idx)]
    proto_xy = xy[len(node_idx) :]
    proto_class = out["prototype_to_class"].detach().cpu().numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.method}_{args.representation}_seed{args.seed}"

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    cmap = plt.get_cmap("tab20", max(num_classes, 1))
    ax.scatter(
        node_xy[:, 0],
        node_xy[:, 1],
        c=y[node_idx],
        cmap=cmap,
        s=8,
        alpha=0.56,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        proto_xy[:, 0],
        proto_xy[:, 1],
        c=proto_class,
        cmap=cmap,
        s=260,
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        label="Class prototypes",
        zorder=5,
    )
    for i, (x0, y0) in enumerate(proto_xy):
        ax.text(x0, y0, str(int(proto_class[i])), ha="center", va="center", fontsize=7, color="white", zorder=6)
    ax.set_title(f"GRAPPLE {args.dataset}: {args.method.upper()} with learned prototypes")
    ax.set_xlabel(f"{args.method.upper()}-1 ({axis_note})")
    ax.set_ylabel(f"{args.method.upper()}-2")
    ax.text(
        0.01,
        0.01,
        f"test acc={100 * test_acc:.2f}%, kappa={float(out['kappa'].item()):.4f}, rho={float(out['rho'].item()):.4f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.legend(loc="upper right", frameon=True)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / f"{stem}_points.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "index", "x", "y", "class", "pred"])
        for local_i, global_i in enumerate(node_idx):
            writer.writerow(["node", int(global_i), node_xy[local_i, 0], node_xy[local_i, 1], int(y[global_i]), int(pred[global_i])])
        for i in range(len(proto_xy)):
            writer.writerow(["prototype", i, proto_xy[i, 0], proto_xy[i, 1], int(proto_class[i]), ""])

    print(f"Saved {args.output_dir / f'{stem}.pdf'}")
    print(f"Saved {args.output_dir / f'{stem}.png'}")
    print(f"Saved {args.output_dir / f'{stem}_points.csv'}")


if __name__ == "__main__":
    main()
