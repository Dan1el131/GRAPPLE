from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grapple.data import load_dataset
from grapple.models.grapple import GrappleModel, ModelConfig
from grapple.models.stereographic import geodesic_distance
from grapple.trainer import TrainConfig, canonical_mask, masked_accuracy, train
from grapple.utils.seed import set_seed


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def split_args(dataset: str) -> dict[str, Any]:
    if dataset == "ogbn-arxiv":
        return {"split": "ogb", "to_undirected": True}
    return {"split": "random", "train_ratio": 0.1, "val_ratio": 0.1, "test_ratio": 0.8}


def encoder_for(dataset: str) -> str:
    if dataset.startswith(("amazon-", "coauthor-")) or dataset in {"citeseer", "actor", "wikics", "wiki-cs"}:
        return "sage"
    return "gcn"


def normalize_for(dataset: str) -> bool:
    return not dataset.startswith("amazon-")


def compute_usage(out: dict[str, torch.Tensor], y: torch.Tensor, num_classes: int) -> dict[str, Any]:
    alpha = out["alpha"].detach().cpu()
    dist = out["dist"].detach().cpu()
    prototypes = out["prototypes"].detach().cpu()
    kappa = out["kappa"].detach().cpu()
    assign = alpha.argmax(dim=-1)
    num_nodes = int(assign.numel())
    num_prototypes = int(alpha.size(1))
    counts = torch.bincount(assign, minlength=num_prototypes)
    entropy = (-(alpha * (alpha + 1e-12).log()).sum(dim=-1) / torch.log(torch.tensor(float(num_prototypes)))).mean()
    dead_ratio = float((counts == 0).float().mean().item())

    purity_weighted = 0.0
    purity_by_proto: list[float] = []
    class_proto_counts = torch.zeros(num_classes, num_prototypes, dtype=torch.long)
    y_cpu = y.detach().cpu().view(-1)
    for proto_id in range(num_prototypes):
        mask = assign == proto_id
        if int(mask.sum().item()) == 0:
            purity_by_proto.append(0.0)
            continue
        labels = y_cpu[mask]
        label_counts = torch.bincount(labels, minlength=num_classes)
        majority = int(label_counts.max().item())
        total = int(mask.sum().item())
        purity = majority / max(total, 1)
        purity_by_proto.append(float(purity))
        purity_weighted += purity * total / max(num_nodes, 1)
        class_proto_counts[:, proto_id] = label_counts

    proto_dist = geodesic_distance(prototypes, prototypes, kappa=kappa, eps=1e-8)
    return {
        "assignment_entropy": float(entropy.item()),
        "dead_prototype_ratio": dead_ratio,
        "weighted_purity": float(purity_weighted),
        "mean_min_distance": float(dist.min(dim=1).values.mean().item()),
        "prototype_counts": counts.tolist(),
        "purity_by_prototype": purity_by_proto,
        "class_prototype_usage": class_proto_counts.tolist(),
        "prototype_distance_matrix": proto_dist.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRAPPLE and record prototype usage metrics.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_root", type=Path, default=Path("overnight_results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--prototypes_per_class", type=int, default=2)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--lambda_clu", type=float, default=0.01)
    parser.add_argument("--lambda_etf", type=float, default=0.1)
    parser.add_argument("--lambda_bal", type=float, default=0.1)
    parser.add_argument("--lambda_cap", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    ds_kwargs = split_args(args.dataset)
    data, _, meta = load_dataset(
        args.dataset,
        root=args.data_root,
        normalize_features=normalize_for(args.dataset),
        seed=args.seed,
        **ds_kwargs,
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
            dropout=0.2,
            num_prototypes=num_classes * int(args.prototypes_per_class),
            tau=args.tau,
            prototype_init="simplex",
        )
    )
    exp_id = (
        f"prototype_usage_{args.dataset}_seed{args.seed}_"
        f"K{args.prototypes_per_class}C_tau{args.tau:g}_clu{args.lambda_clu:g}"
    )
    result_dir = args.output_root / "results" / "prototype_usage"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = result_dir / f"{exp_id}_best.pt"
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=1e-3,
        weight_decay=5e-4,
        lambda_sup=1.0,
        lambda_clu=args.lambda_clu,
        lambda_etf=args.lambda_etf,
        lambda_bal=args.lambda_bal,
        lambda_cap=args.lambda_cap,
        lambda_reg=1e-4,
        lambda_kappa=1e-4,
        eval_interval=args.eval_interval,
        checkpoint_path=str(checkpoint),
    )
    device = torch.device(args.device)
    best_val = train(model, data, device=device, cfg=cfg)
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        y = data.y.to(device)
        train_acc = masked_accuracy(out["logits"], y, data.train_mask.to(device))
        val_acc = masked_accuracy(out["logits"], y, data.val_mask.to(device))
        test_acc = masked_accuracy(out["logits"], y, data.test_mask.to(device))

    usage = compute_usage(out, data.y, num_classes)
    detail_path = result_dir / f"{exp_id}.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    detail_path.write_text(
        json.dumps({"exp_id": exp_id, "config": serializable_config, "usage": usage}, indent=2),
        encoding="utf-8",
    )

    distance_path = result_dir / f"{exp_id}_prototype_distance_matrix.csv"
    with distance_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(usage["prototype_distance_matrix"])

    row = {
        "exp_id": exp_id,
        "dataset": args.dataset,
        "seed": args.seed,
        "prototypes_per_class": args.prototypes_per_class,
        "tau": args.tau,
        "lambda_clu": args.lambda_clu,
        "best_val_acc": best_val,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "kappa": float(out["kappa"].item()),
        "rho": float(out["rho"].item()),
        "assignment_entropy": usage["assignment_entropy"],
        "dead_prototype_ratio": usage["dead_prototype_ratio"],
        "weighted_purity": usage["weighted_purity"],
        "mean_min_distance": usage["mean_min_distance"],
        "detail_path": str(detail_path),
        "distance_matrix_path": str(distance_path),
    }
    summary_path = args.output_root / "results" / "prototype_usage_summary.csv"
    append_csv(summary_path, row, list(row.keys()))
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
