#!/usr/bin/env python3
"""Same-backbone classifier-head comparison for GRAPPLE.

This runner keeps the encoder, split, optimizer, and training loop fixed while
swapping only the classifier head. It is intentionally serial and resumeable so
paper experiments remain easy to audit.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grapple.data import load_dataset
from grapple.models.gcn import GCNEncoder, SAGEEncoder
from grapple.models.grapple import GrappleModel, ModelConfig
from grapple.models.stereographic import clip_tangent, expmap0, geodesic_distance, max_geodesic_radius
from grapple.trainer import canonical_mask, masked_accuracy


HEADS = [
    "linear",
    "euc_proto",
    "euc_grouped",
    "geo_proto",
    "geo_grouped_no_reg",
    "grapple_full",
]

HEAD_LABELS = {
    "linear": "Linear",
    "euc_proto": "Euc-Proto",
    "euc_grouped": "Euc-Grouped",
    "geo_proto": "Geo-Proto",
    "geo_grouped_no_reg": "Geo-Grouped",
    "grapple_full": "GRAPPLE",
}


@dataclass(frozen=True)
class Job:
    dataset: str
    backbone: str
    head_variant: str
    seed: int


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text: str) -> None:
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self) -> None:
        for file in self.files:
            file.flush()


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.classifier(h), "kappa": None, "rho": None}

    def regularization(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        del out
        return self.classifier.weight.sum() * 0.0


class EuclideanPrototypeHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, prototypes_per_class: int, tau: float):
        super().__init__()
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.tau = float(tau)
        self.prototypes = nn.Parameter(torch.empty(self.num_classes * self.prototypes_per_class, in_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h_norm = F.normalize(h, p=2, dim=-1, eps=1e-8)
        proto = F.normalize(self.prototypes, p=2, dim=-1, eps=1e-8)
        dist2 = torch.cdist(h_norm, proto, p=2).pow(2)
        proto_logits = -dist2 / max(self.tau, 1e-8)
        if self.prototypes_per_class == 1:
            logits = proto_logits
        else:
            logits = torch.logsumexp(
                proto_logits.view(h.size(0), self.num_classes, self.prototypes_per_class),
                dim=-1,
            )
        return {"logits": logits, "kappa": None, "rho": None}

    def regularization(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        del out
        return self.prototypes.pow(2).mean()


class GeodesicPrototypeHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        prototypes_per_class: int,
        tau: float,
        kappa_max: float,
        radius_init: float,
        curvature_beta: float,
    ):
        super().__init__()
        if not (0.0 < radius_init < 1.0):
            raise ValueError("radius_init must lie in (0, 1).")
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.tau = float(tau)
        self.kappa_max = float(kappa_max)
        self.curvature_beta = float(curvature_beta)
        self.eta = nn.Parameter(torch.tensor(0.0))
        self.xi = nn.Parameter(torch.logit(torch.tensor(float(radius_init))))
        self.prototype_raw = nn.Parameter(torch.empty(self.num_classes * self.prototypes_per_class, in_dim))
        nn.init.xavier_uniform_(self.prototype_raw)

    def curvature(self) -> torch.Tensor:
        return self.kappa_max * torch.tanh(self.eta)

    def prototype_points(self, kappa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        directions = F.normalize(self.prototype_raw, p=2, dim=-1, eps=1e-8)
        limit = max_geodesic_radius(kappa, beta=self.curvature_beta, eps=1e-8, delta=1e-3)
        rho = limit * torch.sigmoid(self.xi)
        prototypes = expmap0(rho * directions, kappa=kappa, eps=1e-8)
        return prototypes, rho

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        kappa = self.curvature()
        v = F.normalize(h, p=2, dim=-1, eps=1e-8)
        v = clip_tangent(v, kappa=kappa, beta=self.curvature_beta, eps=1e-8, delta=1e-3)
        x_manifold = expmap0(v, kappa=kappa, eps=1e-8)
        prototypes, rho = self.prototype_points(kappa)
        dist = geodesic_distance(x_manifold, prototypes, kappa=kappa, eps=1e-8)
        proto_logits = -dist.pow(2) / max(self.tau, 1e-8)
        if self.prototypes_per_class == 1:
            logits = proto_logits
        else:
            logits = torch.logsumexp(
                proto_logits.view(h.size(0), self.num_classes, self.prototypes_per_class),
                dim=-1,
            )
        return {"logits": logits, "kappa": kappa, "rho": rho}

    def regularization(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        kappa = out["kappa"]
        return self.prototype_raw.pow(2).mean() + kappa.pow(2)


class HeadComparisonModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x, edge_index)
        out = self.head(h)
        out["h"] = h
        return out


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_options(dataset: str) -> dict[str, object]:
    name = dataset.lower().replace("_", "-")
    return {
        "split": "random",
        "train_ratio": 0.1,
        "val_ratio": 0.1,
        "test_ratio": 0.8,
        "normalize_features": True,
        "to_undirected": name in {"squirrel", "chameleon", "actor"},
    }


def build_encoder(backbone: str, in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float) -> nn.Module:
    name = backbone.lower()
    if name == "gcn":
        return GCNEncoder(in_dim, hidden_dim, out_dim, num_layers=layers, dropout=dropout)
    if name in {"sage", "graphsage"}:
        return SAGEEncoder(in_dim, hidden_dim, out_dim, num_layers=layers, dropout=dropout)
    raise ValueError(f"Unsupported backbone '{backbone}'.")


def build_model(args: argparse.Namespace, meta: dict[str, object], backbone: str, head_variant: str) -> nn.Module:
    in_dim = int(meta["num_features"])
    num_classes = int(meta["num_classes"])
    backbone_name = "sage" if backbone.lower() in {"sage", "graphsage"} else "gcn"
    if head_variant == "grapple_full":
        cfg = ModelConfig(
            in_dim=in_dim,
            num_classes=num_classes,
            encoder_type=backbone_name,
            gcn_hidden=args.hidden_dim,
            gcn_out=args.out_dim,
            gcn_layers=args.layers,
            proj_dim=args.out_dim,
            dropout=args.dropout,
            num_prototypes=num_classes * 2,
            tau=args.tau,
            kappa_max=args.kappa_max,
            radius_init=args.radius_init,
            prototype_init="simplex",
            geometry_logit_weight=1.0,
            euclidean_head_weight=0.0,
            curvature_beta=args.curvature_beta,
        )
        return GrappleModel(cfg)

    encoder = build_encoder(
        backbone=backbone_name,
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        layers=args.layers,
        dropout=args.dropout,
    )
    if head_variant == "linear":
        head = LinearHead(args.out_dim, num_classes)
    elif head_variant == "euc_proto":
        head = EuclideanPrototypeHead(args.out_dim, num_classes, prototypes_per_class=1, tau=args.tau)
    elif head_variant == "euc_grouped":
        head = EuclideanPrototypeHead(args.out_dim, num_classes, prototypes_per_class=2, tau=args.tau)
    elif head_variant == "geo_proto":
        head = GeodesicPrototypeHead(
            args.out_dim,
            num_classes,
            prototypes_per_class=1,
            tau=args.tau,
            kappa_max=args.kappa_max,
            radius_init=args.radius_init,
            curvature_beta=args.curvature_beta,
        )
    elif head_variant == "geo_grouped_no_reg":
        head = GeodesicPrototypeHead(
            args.out_dim,
            num_classes,
            prototypes_per_class=2,
            tau=args.tau,
            kappa_max=args.kappa_max,
            radius_init=args.radius_init,
            curvature_beta=args.curvature_beta,
        )
    else:
        raise ValueError(f"Unsupported head_variant '{head_variant}'.")
    return HeadComparisonModel(encoder, head)


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_one(args: argparse.Namespace, job: Job, log_path: Path) -> dict[str, object]:
    start = time.time()
    set_seed(job.seed)
    opts = dataset_options(job.dataset)
    data, _, meta = load_dataset(
        job.dataset,
        root=args.data_root,
        normalize_features=bool(opts["normalize_features"]),
        split=str(opts["split"]),
        train_ratio=float(opts["train_ratio"]),
        val_ratio=float(opts["val_ratio"]),
        test_ratio=float(opts["test_ratio"]),
        seed=job.seed,
        to_undirected=bool(opts["to_undirected"]),
    )

    device = torch.device(args.device)
    model = build_model(args, meta, job.backbone, job.head_variant).to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = canonical_mask(data.train_mask).to(device)
    val_mask = canonical_mask(data.val_mask).to(device)
    test_mask = canonical_mask(data.test_mask).to(device)
    optimizer = make_optimizer(model, args)

    best_val = -1.0
    best_test = -1.0
    best_epoch = 0
    best_kappa = math.nan
    best_rho = math.nan

    print(
        f"START dataset={job.dataset} backbone={job.backbone} head={job.head_variant} "
        f"seed={job.seed} nodes={data.num_nodes} classes={meta['num_classes']} device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        out = model(x, edge_index)
        loss_sup = F.cross_entropy(out["logits"][train_mask], y[train_mask])
        loss = loss_sup

        if job.head_variant == "grapple_full":
            geo = model.geometric_terms(out, cap_margin=args.cap_margin, y=y, capacity_mode="global")
            loss = (
                loss
                + args.lambda_clu * geo["l_clu"]
                + args.lambda_etf * geo["l_etf"]
                + args.lambda_bal * geo["l_bal"]
                + args.lambda_reg * geo["l_reg"]
                + args.lambda_kappa * geo["l_kappa"]
            )
        elif job.head_variant in {"geo_proto", "geo_grouped_no_reg", "euc_proto", "euc_grouped"}:
            loss = loss + args.lambda_reg * model.head.regularization(out)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.eval_interval == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                eval_out = model(x, edge_index)
                val_acc = masked_accuracy(eval_out["logits"], y, val_mask)
                test_acc = masked_accuracy(eval_out["logits"], y, test_mask)
            kappa_t = eval_out.get("kappa")
            rho_t = eval_out.get("rho")
            kappa = float(kappa_t.item()) if isinstance(kappa_t, torch.Tensor) else math.nan
            rho = float(rho_t.item()) if isinstance(rho_t, torch.Tensor) else math.nan
            print(
                f"epoch={epoch:04d} loss={loss.item():.5f} val={val_acc:.5f} "
                f"test={test_acc:.5f} kappa={kappa:.5f} rho={rho:.5f}"
            )
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_epoch = epoch
                best_kappa = kappa
                best_rho = rho

    duration = time.time() - start
    print(
        f"DONE best_epoch={best_epoch} best_val={best_val:.5f} "
        f"test_at_best_val={best_test:.5f} duration_sec={duration:.1f}"
    )
    return {
        "dataset": job.dataset,
        "backbone": "GraphSAGE" if job.backbone.lower() in {"sage", "graphsage"} else "GCN",
        "head_variant": job.head_variant,
        "seed": job.seed,
        "best_val_acc": best_val,
        "test_acc": best_test,
        "kappa": best_kappa,
        "rho": best_rho,
        "duration_sec": duration,
        "status": "ok",
        "command": " ".join(sys.argv),
        "log_path": str(log_path),
    }


def summary_fieldnames() -> list[str]:
    return [
        "dataset",
        "backbone",
        "head_variant",
        "seed",
        "best_val_acc",
        "test_acc",
        "kappa",
        "rho",
        "duration_sec",
        "status",
        "command",
        "log_path",
    ]


def read_done(summary_path: Path) -> set[tuple[str, str, str, int]]:
    if not summary_path.exists():
        return set()
    done = set()
    with summary_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add((row["dataset"], row["backbone"].lower(), row["head_variant"], int(row["seed"])))
    return done


def append_summary(summary_path: Path, row: dict[str, object]) -> None:
    exists = summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_failure(summary_path: Path, job: Job, log_path: Path, exc: BaseException) -> None:
    row = {
        "dataset": job.dataset,
        "backbone": "GraphSAGE" if job.backbone.lower() in {"sage", "graphsage"} else "GCN",
        "head_variant": job.head_variant,
        "seed": job.seed,
        "best_val_acc": "",
        "test_acc": "",
        "kappa": "",
        "rho": "",
        "duration_sec": "",
        "status": f"failed: {type(exc).__name__}: {exc}",
        "command": " ".join(sys.argv),
        "log_path": str(log_path),
    }
    append_summary(summary_path, row)


def load_ok_rows(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.exists():
        return []
    with summary_path.open() as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "ok"]


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def write_aggregate(output_dir: Path, summary_path: Path) -> None:
    rows = load_ok_rows(summary_path)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["dataset"], row["backbone"], row["head_variant"])
        groups.setdefault(key, []).append(row)

    agg_path = output_dir / "aggregate.csv"
    with agg_path.open("w", newline="") as f:
        fieldnames = [
            "dataset",
            "backbone",
            "head_variant",
            "n",
            "test_acc_mean",
            "test_acc_std",
            "best_val_acc_mean",
            "best_val_acc_std",
            "kappa_mean",
            "rho_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups):
            vals = groups[key]
            test_mean, test_std = mean_std(float(v["test_acc"]) for v in vals)
            val_mean, val_std = mean_std(float(v["best_val_acc"]) for v in vals)
            kappas = [float(v["kappa"]) for v in vals if v["kappa"] not in {"", "nan"} and not math.isnan(float(v["kappa"]))]
            rhos = [float(v["rho"]) for v in vals if v["rho"] not in {"", "nan"} and not math.isnan(float(v["rho"]))]
            kappa_mean, _ = mean_std(kappas)
            rho_mean, _ = mean_std(rhos)
            writer.writerow(
                {
                    "dataset": key[0],
                    "backbone": key[1],
                    "head_variant": key[2],
                    "n": len(vals),
                    "test_acc_mean": test_mean,
                    "test_acc_std": test_std,
                    "best_val_acc_mean": val_mean,
                    "best_val_acc_std": val_std,
                    "kappa_mean": kappa_mean,
                    "rho_mean": rho_mean,
                }
            )
    write_markdown_table(output_dir, agg_path)


def fmt_pct(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "N/A"
    return f"{100.0 * mean:.2f} ± {100.0 * std:.2f}"


def write_markdown_table(output_dir: Path, agg_path: Path) -> None:
    with agg_path.open() as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["dataset"], r["backbone"], r["head_variant"]): r for r in rows}
    datasets = sorted({r["dataset"] for r in rows})
    backbones = ["GraphSAGE", "GCN"]
    cols = ["linear", "euc_proto", "euc_grouped", "geo_proto", "geo_grouped_no_reg", "grapple_full"]
    lines = [
        "| Dataset | Backbone | Linear | Euc-Proto | Euc-Grouped | Geo-Proto | Geo-Grouped | GRAPPLE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        for backbone in backbones:
            cells = []
            for head in cols:
                r = by_key.get((dataset, backbone, head))
                if r is None:
                    cells.append("N/A")
                else:
                    cells.append(fmt_pct(float(r["test_acc_mean"]), float(r["test_acc_std"])))
            lines.append(f"| {dataset} | {backbone} | " + " | ".join(cells) + " |")
    (output_dir / "core_head_comparison_table.md").write_text("\n".join(lines) + "\n")


def make_jobs(args: argparse.Namespace) -> list[Job]:
    return [
        Job(dataset=d, backbone=b, head_variant=h, seed=s)
        for d in parse_list(args.datasets)
        for b in parse_list(args.backbones)
        for h in parse_list(args.heads)
        for s in parse_int_list(args.seeds)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="squirrel,coauthor-cs")
    parser.add_argument("--backbones", default="sage,gcn")
    parser.add_argument("--heads", default=",".join(HEADS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output_dir", type=Path, default=Path("runs/classifier_head_comparison"))
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--kappa_max", type=float, default=1.0)
    parser.add_argument("--radius_init", type=float, default=0.5)
    parser.add_argument("--curvature_beta", type=float, default=10.0)
    parser.add_argument("--lambda_clu", type=float, default=0.01)
    parser.add_argument("--lambda_etf", type=float, default=0.1)
    parser.add_argument("--lambda_bal", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)
    parser.add_argument("--lambda_kappa", type=float, default=1e-4)
    parser.add_argument("--cap_margin", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    unknown_heads = sorted(set(parse_list(args.heads)) - set(HEADS))
    if unknown_heads:
        raise ValueError(f"Unknown heads: {unknown_heads}. Valid heads: {HEADS}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    jobs = make_jobs(args)

    if args.dry_run:
        for job in jobs:
            print(job)
        print(f"Total jobs: {len(jobs)}")
        return

    done = read_done(summary_path) if args.resume else set()
    for idx, job in enumerate(jobs, start=1):
        canonical_backbone = "graphsage" if job.backbone.lower() in {"sage", "graphsage"} else "gcn"
        if (job.dataset, canonical_backbone, job.head_variant, job.seed) in done:
            print(f"[{idx}/{len(jobs)}] skip done {job}")
            continue
        log_path = log_dir / f"{job.dataset}_{canonical_backbone}_{job.head_variant}_seed{job.seed}.log"
        print(f"[{idx}/{len(jobs)}] running {job} -> {log_path}")
        with log_path.open("w") as log_f:
            tee = Tee(sys.stdout, log_f)
            try:
                with redirect_stdout(tee), redirect_stderr(tee):
                    row = train_one(args, job, log_path)
                append_summary(summary_path, row)
            except BaseException as exc:
                with redirect_stdout(tee), redirect_stderr(tee):
                    print(f"FAILED {type(exc).__name__}: {exc}")
                write_failure(summary_path, job, log_path, exc)
                raise
        write_aggregate(args.output_dir, summary_path)

    write_aggregate(args.output_dir, summary_path)
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.output_dir / 'aggregate.csv'}")
    print(f"Wrote {args.output_dir / 'core_head_comparison_table.md'}")


if __name__ == "__main__":
    main()
