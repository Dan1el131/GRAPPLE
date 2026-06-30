from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "main_experiments"


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    stage: str
    dataset: str
    split: str
    model: str
    seed: int
    script: str
    args: tuple[str, ...]
    notes: str


METRIC_PATTERNS = {
    "best_val_acc": [
        re.compile(r"Best validation accuracy:\s*([0-9.]+)"),
        re.compile(r"Best sampled validation accuracy:\s*([0-9.]+)"),
        re.compile(r"best_val=([0-9.]+)"),
    ],
    "test_acc": [
        re.compile(r"Prototype classification accuracy \(test\):\s*([0-9.]+)"),
        re.compile(r"Test accuracy at best validation:\s*([0-9.]+)"),
        re.compile(r"Sampled test accuracy at best validation:\s*([0-9.]+)"),
        re.compile(r"best_test=([0-9.]+)"),
    ],
    "kappa": [re.compile(r"Learned curvature kappa:\s*(-?[0-9.]+)")],
    "rho": [re.compile(r"Learned prototype radius rho:\s*([0-9.]+)")],
}


def parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in parse_list(raw)]


def metric_from_text(text: str, name: str) -> float | None:
    hits: list[str] = []
    for pattern in METRIC_PATTERNS[name]:
        hits.extend(pattern.findall(text))
    if not hits:
        return None
    return float(hits[-1])


def command_for(exp: Experiment) -> list[str]:
    return [sys.executable, "-u", exp.script, *exp.args]


def run_command(cmd: list[str], log_path: Path, timeout: int | None) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    chunks: list[str] = []
    start = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                chunks.append(line)
                fh.write(line)
                fh.flush()
                if timeout is not None and time.time() - start > timeout:
                    proc.kill()
                    chunks.append(f"\nTIMEOUT after {timeout} seconds\n")
                    fh.write(chunks[-1])
                    return 124, "".join(chunks)
        finally:
            proc.stdout.close()
    return proc.wait(), "".join(chunks)


def full_graph_args(
    dataset: str,
    split: str,
    seed: int,
    data_root: str,
    device: str,
    epochs: int,
    eval_interval: int,
    encoder_type: str,
    hidden: int,
    out_dim: int,
    layers: int,
    proj_dim: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[str, ...]:
    args = [
        "--dataset",
        dataset,
        "--data_root",
        data_root,
        "--split",
        split,
        "--device",
        device,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--eval_interval",
        str(eval_interval),
        "--encoder_type",
        encoder_type,
        "--gcn_hidden",
        str(hidden),
        "--gcn_out",
        str(out_dim),
        "--gcn_layers",
        str(layers),
        "--proj_dim",
        str(proj_dim),
        "--dropout",
        "0.2",
        "--lr",
        "1e-3",
        "--weight_decay",
        "5e-4",
        "--tau",
        "2.0",
        "--prototype_init",
        "simplex",
        "--prototypes_per_class",
        "2",
        "--lambda_sup",
        "1.0",
        "--lambda_clu",
        "0.01",
        "--lambda_etf",
        "0.1",
        "--lambda_bal",
        "0.1",
        "--lambda_cap",
        "0.0",
        "--lambda_reg",
        "1e-4",
        "--lambda_kappa",
        "1e-4",
    ]
    if split == "random":
        args.extend(
            [
                "--train_ratio",
                str(train_ratio),
                "--val_ratio",
                str(val_ratio),
                "--test_ratio",
                str(test_ratio),
            ]
        )
    if dataset == "ogbn-arxiv":
        args.append("--to_undirected")
    if dataset.startswith("amazon-"):
        args.append("--no_normalize_features")
    return tuple(args)


def baseline_args(
    dataset: str,
    split: str,
    seed: int,
    data_root: str,
    device: str,
    epochs: int,
    eval_interval: int,
    encoder_type: str,
    hidden: int,
    layers: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    gat_heads: int,
) -> tuple[str, ...]:
    args = [
        "--dataset",
        dataset,
        "--data_root",
        data_root,
        "--split",
        split,
        "--device",
        device,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--eval_interval",
        str(eval_interval),
        "--encoder_type",
        encoder_type,
        "--hidden_dim",
        str(hidden),
        "--num_layers",
        str(layers),
        "--gat_heads",
        str(gat_heads),
        "--dropout",
        "0.2",
        "--lr",
        "1e-3",
        "--weight_decay",
        "5e-4",
    ]
    if split == "random":
        args.extend(
            [
                "--train_ratio",
                str(train_ratio),
                "--val_ratio",
                str(val_ratio),
                "--test_ratio",
                str(test_ratio),
            ]
        )
    if dataset == "ogbn-arxiv":
        args.append("--to_undirected")
    if not dataset.startswith("amazon-"):
        args.append("--normalize_features")
    return tuple(args)


def sampled_products_baseline_args(args: argparse.Namespace, seed: int) -> tuple[str, ...]:
    return (
        "--dataset",
        "ogbn-products",
        "--data_root",
        args.data_root,
        "--ogb_source",
        args.ogb_source,
        "--no_normalize_features",
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--epochs",
        str(args.sampled_baseline_epochs),
        "--eval_interval",
        str(args.sampled_eval_interval),
        "--train_batches",
        str(args.train_batches),
        "--eval_batches",
        str(args.eval_batches),
        "--batch_size",
        str(args.batch_size),
        "--num_neighbors",
        *parse_list(args.num_neighbors),
        "--hidden_dim",
        str(args.hidden),
        "--num_layers",
        str(args.layers),
        "--dropout",
        "0.2",
        "--lr",
        "1e-3",
        "--weight_decay",
        "5e-4",
    )


def sampled_products_method_args(args: argparse.Namespace, seed: int) -> tuple[str, ...]:
    return (
        "--dataset",
        "ogbn-products",
        "--data_root",
        args.data_root,
        "--ogb_source",
        args.ogb_source,
        "--no_normalize_features",
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--epochs",
        str(args.sampled_epochs),
        "--eval_interval",
        str(args.sampled_eval_interval),
        "--train_batches",
        str(args.train_batches),
        "--eval_batches",
        str(args.eval_batches),
        "--batch_size",
        str(args.batch_size),
        "--num_neighbors",
        *parse_list(args.num_neighbors),
        "--encoder_type",
        "sage",
        "--gcn_hidden",
        str(args.hidden),
        "--gcn_out",
        str(args.out_dim),
        "--gcn_layers",
        str(args.layers),
        "--dropout",
        "0.2",
        "--proj_dim",
        str(args.proj_dim),
        "--prototypes_per_class",
        "2",
        "--tau",
        "2.0",
        "--prototype_init",
        "simplex",
        "--lr",
        "1e-3",
        "--weight_decay",
        "5e-4",
        "--lambda_clu",
        "0.01",
        "--lambda_etf",
        "0.1",
        "--lambda_bal",
        "0.1",
        "--lambda_cap",
        "0.0",
        "--lambda_reg",
        "1e-4",
        "--lambda_kappa",
        "1e-4",
    )


def make_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments: list[Experiment] = []
    datasets = parse_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    baselines = parse_list(args.baselines)
    split_by_dataset = {
        "cora": "public",
        "citeseer": "public",
        "pubmed": "public",
        "amazon-computers": "random",
        "amazon-photo": "random",
        "amazon-photos": "random",
        "coauthor-cs": "random",
        "coauthor-physics": "random",
        "wikics": "random",
        "webkb-cornell": "random",
        "webkb-texas": "random",
        "webkb-wisconsin": "random",
        "actor": "random",
        "film": "random",
        "chameleon": "random",
        "wiki-chameleon": "random",
        "squirrel": "random",
        "wiki-squirrel": "random",
        "ogbn-arxiv": "ogb",
    }
    for dataset in datasets:
        if dataset == "ogbn-products":
            for seed in seeds:
                experiments.append(
                    Experiment(
                        exp_id=f"{dataset}_seed{seed}_sage_linear_sampled",
                        stage="main_baseline",
                        dataset=dataset,
                        split="ogb",
                        model="GraphSAGE+Linear sampled",
                        seed=seed,
                        script="sampled_supervised_baseline.py",
                        args=sampled_products_baseline_args(args, seed),
                        notes="Large-graph sampled SAGE baseline for ogbn-products.",
                    )
                )
                experiments.append(
                    Experiment(
                        exp_id=f"{dataset}_seed{seed}_grapple_sampled",
                        stage="main_method",
                        dataset=dataset,
                        split="ogb",
                        model="GRAPPLE-SAGE sampled K=2C tau=2",
                        seed=seed,
                        script="sampled_ogb_experiment.py",
                        args=sampled_products_method_args(args, seed),
                        notes="Large-graph sampled main method for ogbn-products.",
                    )
                )
            continue
        split = split_by_dataset.get(dataset, "random")
        encoder = "sage" if dataset.startswith("amazon-") or dataset.startswith("coauthor-") else "gcn"
        for seed in seeds:
            for baseline in baselines:
                baseline = baseline.strip().lower()
                experiments.append(
                    Experiment(
                        exp_id=f"{dataset}_seed{seed}_{baseline}_linear",
                        stage="main_baseline",
                        dataset=dataset,
                        split=split,
                        model=f"{baseline.upper()}+Linear",
                        seed=seed,
                        script="full_supervised_baseline.py",
                        args=baseline_args(
                            dataset,
                            split,
                            seed,
                            args.data_root,
                            args.device,
                            args.baseline_epochs,
                            args.eval_interval,
                            baseline,
                            args.hidden,
                            args.layers,
                            args.train_ratio,
                            args.val_ratio,
                            args.test_ratio,
                            args.gat_heads,
                        ),
                        notes="Classic supervised baseline under the shared split/training budget.",
                    )
                )
            experiments.append(
                Experiment(
                    exp_id=f"{dataset}_seed{seed}_grapple",
                    stage="main_method",
                    dataset=dataset,
                    split=split,
                    model=f"GRAPPLE-{encoder.upper()} K=2C tau=2",
                    seed=seed,
                    script="main.py",
                    args=full_graph_args(
                        dataset,
                        split,
                        seed,
                        args.data_root,
                        args.device,
                        args.epochs,
                        args.eval_interval,
                        encoder,
                        args.hidden,
                        args.out_dim,
                        args.layers,
                        args.proj_dim,
                        args.train_ratio,
                        args.val_ratio,
                        args.test_ratio,
                    ),
                    notes="Recommended main method setting from the experiment template.",
                )
            )
    return experiments


def write_manifest(path: Path, experiments: list[Experiment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*asdict(experiments[0]).keys(), "command"])
        writer.writeheader()
        for exp in experiments:
            row = asdict(exp)
            row["args"] = " ".join(exp.args)
            row["command"] = " ".join(command_for(exp))
            writer.writerow(row)


def append_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "exp_id",
        "stage",
        "dataset",
        "split",
        "model",
        "seed",
        "status",
        "returncode",
        "duration_sec",
        "best_val_acc",
        "test_acc",
        "kappa",
        "rho",
        "log_path",
        "command",
        "notes",
    ]
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_markdown_summary(run_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    md_path = run_dir / "README.md"
    lines = [
        "# Main Experiment Run",
        "",
        f"- Created: {datetime.now().isoformat(timespec='seconds')}",
        f"- Total experiments: {len(summary_rows)}",
        "",
        "| Dataset | Model | Seed | Status | Val | Test | Kappa | Rho | Log |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        rel_log = Path(str(row["log_path"])).name
        lines.append(
            "| {dataset} | {model} | {seed} | {status} | {val} | {test} | {kappa} | {rho} | {log} |".format(
                dataset=row["dataset"],
                model=row["model"],
                seed=row["seed"],
                status=row["status"],
                val="" if row["best_val_acc"] is None else f"{float(row['best_val_acc']):.4f}",
                test="" if row["test_acc"] is None else f"{float(row['test_acc']):.4f}",
                kappa="" if row["kappa"] is None else f"{float(row['kappa']):.6f}",
                rho="" if row["rho"] is None else f"{float(row['rho']):.6f}",
                log=rel_log,
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and log GRAPPLE main experiments.")
    parser.add_argument("--datasets", default="amazon-computers,cora,citeseer,pubmed")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--baseline_epochs", type=int, default=200)
    parser.add_argument("--baselines", default="mlp,gcn,gat,sage")
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.8)
    parser.add_argument("--ogb_source", default="snap", choices=["snap", "official", "graphbolt", "dgl"])
    parser.add_argument("--sampled_epochs", type=int, default=30)
    parser.add_argument("--sampled_baseline_epochs", type=int, default=30)
    parser.add_argument("--sampled_eval_interval", type=int, default=1)
    parser.add_argument("--train_batches", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_neighbors", default="15,10")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--timeout", type=int, default=0, help="Per-experiment timeout in seconds; 0 disables it.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    log_dir = run_dir / "logs"
    summary_path = run_dir / "summary.csv"
    manifest_path = run_dir / "manifest.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    experiments = make_experiments(args)
    write_manifest(manifest_path, experiments)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")

    if args.dry_run:
        print(f"Wrote dry-run manifest: {manifest_path}")
        return

    rows: list[dict[str, object]] = []
    for index, exp in enumerate(experiments, start=1):
        cmd = command_for(exp)
        log_path = log_dir / f"{index:03d}_{exp.exp_id}.log"
        print(f"\n[{index}/{len(experiments)}] {exp.exp_id}")
        print("$ " + " ".join(cmd))
        start = time.time()
        returncode, text = run_command(cmd, log_path, timeout=args.timeout or None)
        duration = time.time() - start
        row: dict[str, object] = {
            "exp_id": exp.exp_id,
            "stage": exp.stage,
            "dataset": exp.dataset,
            "split": exp.split,
            "model": exp.model,
            "seed": exp.seed,
            "status": "ok" if returncode == 0 else "failed",
            "returncode": returncode,
            "duration_sec": round(duration, 2),
            "best_val_acc": metric_from_text(text, "best_val_acc"),
            "test_acc": metric_from_text(text, "test_acc"),
            "kappa": metric_from_text(text, "kappa"),
            "rho": metric_from_text(text, "rho"),
            "log_path": str(log_path),
            "command": " ".join(cmd),
            "notes": exp.notes,
        }
        rows.append(row)
        append_summary(summary_path, row)
        write_markdown_summary(run_dir, rows)
        print(
            "status={status} val={val} test={test} duration={duration:.1f}s".format(
                status=row["status"],
                val=row["best_val_acc"],
                test=row["test_acc"],
                duration=duration,
            )
        )

    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved markdown summary: {run_dir / 'README.md'}")


if __name__ == "__main__":
    main()
