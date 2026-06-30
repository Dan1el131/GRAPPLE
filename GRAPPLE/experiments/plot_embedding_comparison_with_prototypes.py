#!/usr/bin/env python3
"""Draw baseline-vs-GRAPPLE embedding visualization with GRAPPLE prototypes."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_supervised_baseline import FullGraphClassifier
from grapple.data import load_dataset
from grapple.models.grapple import GrappleModel, ModelConfig
from grapple.trainer import canonical_mask, masked_accuracy
from grapple.utils.seed import set_seed


def normalize_for(dataset: str) -> bool:
    return not dataset.startswith("amazon-") and dataset != "ogbn-products"


def encoder_for_grapple(dataset: str) -> str:
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
        if len(idx):
            picked.extend(rng.choice(idx, size=min(per_class, len(idx)), replace=False).tolist())
    if len(picked) < max_nodes:
        remaining = np.setdiff1d(np.arange(len(y)), np.array(picked, dtype=int), assume_unique=False)
        if len(remaining):
            picked.extend(rng.choice(remaining, size=min(len(remaining), max_nodes - len(picked)), replace=False).tolist())
    return np.array(sorted(set(picked)), dtype=int)


def clean_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        x = np.clip(x, lo, hi)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    x = x / np.maximum(scale, 1e-6)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def reduce_2d(x: np.ndarray, method: str, seed: int) -> np.ndarray:
    x = clean_features(x)
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(x)
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise SystemExit("UMAP is not installed. Use --method tsne or install umap-learn.") from exc
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=20, min_dist=0.15).fit_transform(x)
    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, (len(x) - 1) // 3))
    return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(x)


def train_baseline(data, meta, args, device: torch.device) -> tuple[FullGraphClassifier, float]:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = canonical_mask(data.train_mask).to(device)
    val_mask = canonical_mask(data.val_mask).to(device)
    test_mask = canonical_mask(data.test_mask).to(device)
    model = FullGraphClassifier(
        in_dim=int(meta["num_features"]),
        hidden_dim=args.hidden_dim,
        num_classes=int(meta["num_classes"]),
        num_layers=args.layers,
        dropout=args.dropout,
        encoder_type=args.baseline,
        gat_heads=4,
    ).to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = -1.0
    best_test = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                eval_logits = model(x, edge_index)
                val = masked_accuracy(eval_logits, y, val_mask)
                test = masked_accuracy(eval_logits, y, test_mask)
            if val > best_val:
                best_val = val
                best_test = test
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"{args.baseline.upper()} epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val={val:.4f} | test={test:.4f} | best_test={best_test:.4f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_test


@torch.no_grad()
def baseline_embedding(model: FullGraphClassifier, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    h = x
    for i, layer in enumerate(model.layers):
        h = layer(h) if model.encoder_type == "mlp" else layer(h, edge_index)
        if i < len(model.layers) - 1:
            h = F.relu(h)
            if i == len(model.layers) - 2:
                emb = h.detach().cpu().numpy()
            h = F.dropout(h, p=model.dropout, training=False)
    logits = h.detach().cpu().numpy()
    if "emb" not in locals():
        emb = logits
    return emb, logits


def load_grapple(data, meta, args, device: torch.device):
    model = GrappleModel(
        ModelConfig(
            in_dim=int(meta["num_features"]),
            num_classes=int(meta["num_classes"]),
            encoder_type=encoder_for_grapple(args.dataset),
            gcn_hidden=args.hidden_dim,
            gcn_out=args.out_dim,
            gcn_layers=args.layers,
            proj_dim=args.proj_dim,
            dropout=args.dropout,
            num_prototypes=int(meta["num_classes"]) * int(args.prototypes_per_class),
            tau=args.tau,
            prototype_init="simplex",
        )
    )
    ckpt = torch.load(args.grapple_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device).eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        test_acc = masked_accuracy(out["logits"].detach().cpu(), data.y.detach().cpu(), data.test_mask.detach().cpu())
    node_emb = out["v_clipped"].detach().cpu().numpy()
    proto_emb = (out["rho"] * out["prototype_directions"]).detach().cpu().numpy()
    proto_class = out["prototype_to_class"].detach().cpu().numpy()
    return node_emb, proto_emb, proto_class, test_acc, float(out["kappa"].item()), float(out["rho"].item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline vs GRAPPLE t-SNE/PCA with prototypes.")
    parser.add_argument("--dataset", default="amazon-computers")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--grapple_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs/aaai_figures/embedding_comparison"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--baseline", choices=["mlp", "gcn", "sage", "gat"], default="sage")
    parser.add_argument("--max_nodes", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--prototypes_per_class", type=int, default=2)
    parser.add_argument("--tau", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    data, _, meta = load_dataset(
        args.dataset,
        root=args.data_root,
        normalize_features=normalize_for(args.dataset),
        seed=args.seed,
        **split_kwargs(args.dataset),
    )
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.detach().cpu().view(-1).numpy()
    node_idx = stratified_sample(y, args.max_nodes, args.seed)

    baseline_model, baseline_test = train_baseline(data, meta, args, device)
    baseline_emb, baseline_logits = baseline_embedding(baseline_model, x, edge_index)
    baseline_pred = baseline_logits.argmax(axis=-1)

    gr_emb, proto_emb, proto_class, gr_test, kappa, rho = load_grapple(data, meta, args, device)

    baseline_xy = reduce_2d(baseline_emb[node_idx], args.method, args.seed)
    gr_combined = np.vstack([gr_emb[node_idx], proto_emb])
    gr_xy_all = reduce_2d(gr_combined, args.method, args.seed)
    gr_xy = gr_xy_all[: len(node_idx)]
    proto_xy = gr_xy_all[len(node_idx) :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_label = {"mlp": "MLP", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}[args.baseline]
    stem = f"{args.dataset}_{args.baseline}_vs_grapple_{args.method}_seed{args.seed}"

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    cmap = plt.get_cmap("tab20", max(int(meta["num_classes"]), 1))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)

    axes[0].scatter(baseline_xy[:, 0], baseline_xy[:, 1], c=y[node_idx], cmap=cmap, s=8, alpha=0.58, linewidths=0, rasterized=True)
    axes[0].set_title(f"{baseline_label} embedding\nTest acc={100 * baseline_test:.2f}%")
    axes[0].set_xlabel(f"{args.method.upper()}-1")
    axes[0].set_ylabel(f"{args.method.upper()}-2")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(gr_xy[:, 0], gr_xy[:, 1], c=y[node_idx], cmap=cmap, s=8, alpha=0.58, linewidths=0, rasterized=True)
    axes[1].scatter(
        proto_xy[:, 0],
        proto_xy[:, 1],
        c=proto_class,
        cmap=cmap,
        s=260,
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        label="GRAPPLE prototypes",
        zorder=5,
    )
    for i, (x0, y0) in enumerate(proto_xy):
        axes[1].text(x0, y0, str(int(proto_class[i])), ha="center", va="center", fontsize=7, color="white", zorder=6)
    axes[1].set_title(f"GRAPPLE embedding + prototypes\nTest acc={100 * gr_test:.2f}%, kappa={kappa:.4f}, rho={rho:.3f}")
    axes[1].set_xlabel(f"{args.method.upper()}-1")
    axes[1].set_ylabel(f"{args.method.upper()}-2")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend(loc="upper right", frameon=True)

    fig.suptitle(f"Prototype-centered geometry on {args.dataset}", fontsize=16)
    fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (args.output_dir / f"{stem}_points.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["panel", "kind", "index", "x", "y", "class", "pred"])
        for local_i, global_i in enumerate(node_idx):
            writer.writerow([args.baseline, "node", int(global_i), baseline_xy[local_i, 0], baseline_xy[local_i, 1], int(y[global_i]), int(baseline_pred[global_i])])
            writer.writerow(["grapple", "node", int(global_i), gr_xy[local_i, 0], gr_xy[local_i, 1], int(y[global_i]), ""])
        for i, (x0, y0) in enumerate(proto_xy):
            writer.writerow(["grapple", "prototype", i, x0, y0, int(proto_class[i]), ""])

    print(f"Saved {args.output_dir / f'{stem}.pdf'}")
    print(f"Saved {args.output_dir / f'{stem}.png'}")
    print(f"Saved {args.output_dir / f'{stem}_points.csv'}")


if __name__ == "__main__":
    main()
