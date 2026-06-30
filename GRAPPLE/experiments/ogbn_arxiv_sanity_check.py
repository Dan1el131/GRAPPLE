from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grapple.data import load_dataset
from grapple.models.grapple import GrappleModel, ModelConfig
from grapple.trainer import TrainConfig, canonical_mask, masked_accuracy, train
from grapple.utils.seed import set_seed


class SanityFullGraphClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        dropout: float,
        encoder_type: str,
        gat_heads: int,
    ):
        super().__init__()
        self.dropout = dropout
        self.encoder_type = encoder_type.strip().lower()
        dims = [in_dim] + ([hidden_dim] * max(num_layers - 1, 0)) + [num_classes]
        if self.encoder_type == "mlp":
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        elif self.encoder_type == "sage":
            self.layers = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        elif self.encoder_type == "gcn":
            self.layers = nn.ModuleList(
                [GCNConv(dims[i], dims[i + 1], cached=False, normalize=True) for i in range(len(dims) - 1)]
            )
        elif self.encoder_type == "gat":
            self.layers = nn.ModuleList()
            if len(dims) == 2:
                self.layers.append(GATConv(dims[0], dims[1], heads=1, concat=False, dropout=dropout))
            else:
                self.layers.append(GATConv(dims[0], hidden_dim, heads=gat_heads, concat=True, dropout=dropout))
                for _ in range(max(num_layers - 2, 0)):
                    self.layers.append(GATConv(hidden_dim * gat_heads, hidden_dim, heads=gat_heads, concat=True, dropout=dropout))
                self.layers.append(GATConv(hidden_dim * gat_heads, num_classes, heads=1, concat=False, dropout=dropout))
        else:
            raise ValueError("encoder_type must be one of: mlp, sage, gcn, gat.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h) if self.encoder_type == "mlp" else layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def class_hist(tensor: torch.Tensor, num_classes: int) -> list[int]:
    return torch.bincount(tensor.detach().cpu().view(-1), minlength=num_classes).tolist()


def split_stats(data, num_classes: int) -> dict[str, Any]:
    y = data.y.detach().cpu().view(-1)
    train = canonical_mask(data.train_mask).cpu()
    val = canonical_mask(data.val_mask).cpu()
    test = canonical_mask(data.test_mask).cpu()
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_classes": int(num_classes),
        "x_shape": list(data.x.shape),
        "y_shape": list(data.y.shape),
        "y_min": int(y.min().item()),
        "y_max": int(y.max().item()),
        "train_size": int(train.sum().item()),
        "val_size": int(val.sum().item()),
        "test_size": int(test.sum().item()),
        "train_val_overlap": int((train & val).sum().item()),
        "train_test_overlap": int((train & test).sum().item()),
        "val_test_overlap": int((val & test).sum().item()),
        "all_label_hist": class_hist(y, num_classes),
        "train_label_hist": class_hist(y[train], num_classes),
        "val_label_hist": class_hist(y[val], num_classes),
        "test_label_hist": class_hist(y[test], num_classes),
    }


@torch.no_grad()
def evaluate_logits(logits, y, train_mask, val_mask, test_mask, num_classes: int) -> dict[str, Any]:
    pred = logits.argmax(dim=-1)
    return {
        "train_acc": masked_accuracy(logits, y, train_mask),
        "val_acc": masked_accuracy(logits, y, val_mask),
        "test_acc": masked_accuracy(logits, y, test_mask),
        "unique_pred_classes": int(pred.unique().numel()),
        "prediction_hist": class_hist(pred, num_classes),
        "train_prediction_hist": class_hist(pred[canonical_mask(train_mask)], num_classes),
        "val_prediction_hist": class_hist(pred[canonical_mask(val_mask)], num_classes),
        "test_prediction_hist": class_hist(pred[canonical_mask(test_mask)], num_classes),
    }


def train_baseline(args, data, meta, encoder_type: str) -> dict[str, Any]:
    device = torch.device(args.device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = canonical_mask(data.train_mask).to(device)
    val_mask = canonical_mask(data.val_mask).to(device)
    test_mask = canonical_mask(data.test_mask).to(device)
    gat_heads = args.gat_heads if encoder_type == "gat" else 1
    hidden_dim = args.gat_hidden_dim if encoder_type == "gat" else args.hidden_dim
    model = SanityFullGraphClassifier(
        in_dim=int(meta["num_features"]),
        hidden_dim=int(hidden_dim),
        num_classes=int(meta["num_classes"]),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        encoder_type=encoder_type,
        gat_heads=int(gat_heads),
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = -1.0
    best_test = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                eval_logits = model(x, edge_index)
                val_acc = masked_accuracy(eval_logits, y, val_mask)
                test_acc = masked_accuracy(eval_logits, y, test_mask)
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"{encoder_type} epoch={epoch} loss={loss.item():.4f} "
                f"val={val_acc:.4f} test={test_acc:.4f} best_epoch={best_epoch}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(x, edge_index)
    metrics = evaluate_logits(final_logits, y, train_mask, val_mask, test_mask, int(meta["num_classes"]))
    metrics.update(
        {
            "method": encoder_type,
            "best_epoch": best_epoch,
            "best_val_acc": best_val,
            "best_test_acc": best_test,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
            "kappa": "",
            "rho": "",
        }
    )
    return metrics


def train_grapple(args, data, meta) -> dict[str, Any]:
    device = torch.device(args.device)
    num_classes = int(meta["num_classes"])
    model = GrappleModel(
        ModelConfig(
            in_dim=int(meta["num_features"]),
            num_classes=num_classes,
            encoder_type=args.grapple_encoder,
            gcn_hidden=args.grapple_hidden_dim,
            gcn_out=args.grapple_out_dim,
            gcn_layers=args.grapple_layers,
            proj_dim=args.grapple_proj_dim,
            dropout=args.dropout,
            num_prototypes=num_classes * args.prototypes_per_class,
            tau=args.tau,
            prototype_init="simplex",
            geometry_logit_weight=1.0,
            euclidean_head_weight=0.0,
        )
    )
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_sup=1.0,
        lambda_clu=args.lambda_clu,
        lambda_etf=args.lambda_etf,
        lambda_bal=args.lambda_bal,
        lambda_cap=args.lambda_cap,
        lambda_reg=1e-4,
        lambda_kappa=1e-4,
        geometry_warmup_epochs=args.geometry_warmup_epochs,
        geometry_warmup_start=0.0,
        eval_interval=args.eval_interval,
        checkpoint_path=str(args.output_root / "results" / "ogbn_arxiv_sanity_grapple_best.pt"),
    )
    best_val = train(model, data, device=device, cfg=cfg)
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
    metrics = evaluate_logits(
        out["logits"],
        data.y.to(device),
        data.train_mask.to(device),
        data.val_mask.to(device),
        data.test_mask.to(device),
        num_classes,
    )
    metrics.update(
        {
            "method": "grapple",
            "best_epoch": "",
            "best_val_acc": best_val,
            "best_test_acc": metrics["test_acc"],
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
            "kappa": float(out["kappa"].item()),
            "rho": float(out["rho"].item()),
        }
    )
    return metrics


def write_report(path: Path, stats: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suspicious = []
    random_acc = 1.0 / max(int(stats["num_classes"]), 1)
    for row in rows:
        if row.get("status") != "ok":
            suspicious.append(f"{row.get('method')} failed: {row.get('error', '')[:180]}")
            continue
        if int(row.get("unique_pred_classes", 0)) <= 1:
            suspicious.append(f"{row.get('method')} predicts only one class")
        if row.get("method") in {"gcn", "sage", "gat"} and float(row.get("best_val_acc", 0.0)) < max(0.15, 3.0 * random_acc):
            suspicious.append(f"{row.get('method')} validation accuracy is unusually low")
    judgement = "suspicious" if suspicious else "usable"
    lines = [
        "# ogbn-arxiv sanity check",
        "",
        f"- num_nodes: {stats['num_nodes']}",
        f"- num_edges: {stats['num_edges']}",
        f"- num_classes: {stats['num_classes']}",
        f"- x_shape: {stats['x_shape']}",
        f"- y_shape: {stats['y_shape']} min={stats['y_min']} max={stats['y_max']}",
        f"- split sizes: train={stats['train_size']} val={stats['val_size']} test={stats['test_size']}",
        (
            "- split overlap: "
            f"train_val={stats['train_val_overlap']} train_test={stats['train_test_overlap']} "
            f"val_test={stats['val_test_overlap']}"
        ),
        f"- reliability judgement: {judgement}",
        "",
        "## Label Distributions",
        "",
        "```json",
        json.dumps(
            {
                "all": stats["all_label_hist"],
                "train": stats["train_label_hist"],
                "val": stats["val_label_hist"],
                "test": stats["test_label_hist"],
            },
            indent=2,
        ),
        "```",
        "",
        "## Model Results",
        "",
        "| method | status | train | val | test | best_val | unique_pred_classes | params | kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {train:.4f} | {val:.4f} | {test:.4f} | {best:.4f} | {uniq} | {params} | {kappa} |".format(
                method=row.get("method", ""),
                status=row.get("status", ""),
                train=float(row.get("train_acc") or 0.0),
                val=float(row.get("val_acc") or 0.0),
                test=float(row.get("test_acc") or 0.0),
                best=float(row.get("best_val_acc") or 0.0),
                uniq=row.get("unique_pred_classes", ""),
                params=row.get("trainable_params", ""),
                kappa=row.get("kappa", ""),
            )
        )
    lines.extend(["", "## Prediction Histograms", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('method')}",
                "",
                "```json",
                json.dumps(
                    {
                        "all": json.loads(row.get("prediction_hist_json", "[]")),
                        "train": json.loads(row.get("train_prediction_hist_json", "[]")),
                        "val": json.loads(row.get("val_prediction_hist_json", "[]")),
                        "test": json.loads(row.get("test_prediction_hist_json", "[]")),
                    },
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    if suspicious:
        lines.extend(["## Suspicious Items", ""])
        lines.extend([f"- {item}" for item in suspicious])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def row_for_csv(metrics: dict[str, Any], status: str = "ok", error: str = "") -> dict[str, Any]:
    row = dict(metrics)
    for key in [
        "prediction_hist",
        "train_prediction_hist",
        "val_prediction_hist",
        "test_prediction_hist",
    ]:
        row[f"{key}_json"] = json.dumps(row.pop(key, []))
    row["status"] = status
    row["error"] = error
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check ogbn-arxiv splits, labels, baselines, and GRAPPLE.")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_root", type=Path, default=Path("overnight_results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gat_hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grapple_encoder", default="gcn", choices=["gcn", "sage", "gcnii", "appnp"])
    parser.add_argument("--grapple_hidden_dim", type=int, default=128)
    parser.add_argument("--grapple_out_dim", type=int, default=64)
    parser.add_argument("--grapple_layers", type=int, default=2)
    parser.add_argument("--grapple_proj_dim", type=int, default=64)
    parser.add_argument("--prototypes_per_class", type=int, default=2)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--lambda_clu", type=float, default=0.01)
    parser.add_argument("--lambda_etf", type=float, default=0.1)
    parser.add_argument("--lambda_bal", type=float, default=0.1)
    parser.add_argument("--lambda_cap", type=float, default=0.0)
    parser.add_argument("--geometry_warmup_epochs", type=int, default=10)
    parser.add_argument("--methods", default="mlp,gcn,gat,sage,grapple")
    args = parser.parse_args()
    args.output_root = Path(args.output_root)
    set_seed(args.seed)

    data, _, meta = load_dataset(
        "ogbn-arxiv",
        root=args.data_root,
        split="ogb",
        normalize_features=True,
        to_undirected=True,
    )
    stats = split_stats(data, int(meta["num_classes"]))
    stats_path = args.output_root / "debug_reports" / "ogbn_arxiv_split_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    result_path = args.output_root / "results" / "ogbn_arxiv_sanity_check.csv"
    report_path = args.output_root / "debug_reports" / "ogbn_arxiv_sanity_check.md"
    fieldnames = [
        "method",
        "status",
        "train_acc",
        "val_acc",
        "test_acc",
        "best_val_acc",
        "best_test_acc",
        "best_epoch",
        "unique_pred_classes",
        "trainable_params",
        "total_params",
        "kappa",
        "rho",
        "prediction_hist_json",
        "train_prediction_hist_json",
        "val_prediction_hist_json",
        "test_prediction_hist_json",
        "error",
    ]

    rows: list[dict[str, Any]] = []
    for method in [m.strip().lower() for m in args.methods.split(",") if m.strip()]:
        print(f"\n=== sanity method={method} ===", flush=True)
        try:
            if method == "grapple":
                metrics = train_grapple(args, data, meta)
            else:
                metrics = train_baseline(args, data, meta, method)
            row = row_for_csv(metrics, status="ok")
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(traceback.format_exc(), flush=True)
            row = {
                "method": method,
                "status": "failed",
                "train_acc": math.nan,
                "val_acc": math.nan,
                "test_acc": math.nan,
                "best_val_acc": math.nan,
                "best_test_acc": math.nan,
                "best_epoch": "",
                "unique_pred_classes": 0,
                "trainable_params": "",
                "total_params": "",
                "kappa": "",
                "rho": "",
                "prediction_hist_json": "[]",
                "train_prediction_hist_json": "[]",
                "val_prediction_hist_json": "[]",
                "test_prediction_hist_json": "[]",
                "error": repr(exc),
            }
        rows.append(row)
        append_csv(result_path, row, fieldnames)
        write_report(report_path, stats, rows)
    print(f"sanity report: {report_path}")
    print(f"sanity csv: {result_path}")


if __name__ == "__main__":
    main()
