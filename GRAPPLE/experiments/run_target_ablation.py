from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "target_ablation"


VARIANT_ORDER = ["full", "no_clu", "no_etf", "no_bal", "supervised_only", "no_cap"]
VARIANT_OVERRIDES: dict[str, dict[str, str]] = {
    "full": {},
    "no_clu": {"--lambda_clu": "0.0"},
    "no_etf": {"--lambda_etf": "0.0"},
    "no_bal": {"--lambda_bal": "0.0"},
    "supervised_only": {
        "--lambda_clu": "0.0",
        "--lambda_etf": "0.0",
        "--lambda_bal": "0.0",
        "--lambda_cap": "0.0",
        "--lambda_reg": "0.0",
        "--lambda_kappa": "0.0",
    },
    "no_cap": {"--lambda_cap": "0.0"},
}


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


@dataclass(frozen=True)
class AblationJob:
    exp_id: str
    dataset: str
    split: str
    variant: str
    seed: int
    script: str
    args: tuple[str, ...]


def parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in parse_list(raw)]


def metric_from_text(text: str, name: str) -> float | None:
    hits: list[str] = []
    for pattern in METRIC_PATTERNS[name]:
        hits.extend(pattern.findall(text))
    return None if not hits else float(hits[-1])


def apply_overrides(args: list[str], overrides: dict[str, str]) -> tuple[str, ...]:
    out = list(args)
    for flag, value in overrides.items():
        if flag in out:
            idx = out.index(flag)
            out[idx + 1] = value
        else:
            out.extend([flag, value])
    return tuple(out)


def products_args(args: argparse.Namespace, seed: int, variant: str) -> tuple[str, ...]:
    base = [
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
        str(args.lambda_cap),
        "--lambda_reg",
        "1e-4",
        "--lambda_kappa",
        "1e-4",
    ]
    return apply_overrides(base, VARIANT_OVERRIDES[variant])


def full_graph_args(dataset: str, args: argparse.Namespace, seed: int, variant: str) -> tuple[str, ...]:
    split = "random"
    encoder = "sage" if dataset.startswith(("amazon-", "coauthor-")) else "gcn"
    base = [
        "--dataset",
        dataset,
        "--data_root",
        args.data_root,
        "--split",
        split,
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--eval_interval",
        str(args.eval_interval),
        "--encoder_type",
        encoder,
        "--gcn_hidden",
        str(args.hidden),
        "--gcn_out",
        str(args.out_dim),
        "--gcn_layers",
        str(args.layers),
        "--proj_dim",
        str(args.proj_dim),
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
        str(args.lambda_cap),
        "--lambda_reg",
        "1e-4",
        "--lambda_kappa",
        "1e-4",
        "--train_ratio",
        "0.1",
        "--val_ratio",
        "0.1",
        "--test_ratio",
        "0.8",
    ]
    if dataset.startswith("amazon-"):
        base.append("--no_normalize_features")
    else:
        base.append("--normalize_features")
    if dataset in {"actor", "chameleon", "squirrel"}:
        base.append("--to_undirected")
    return apply_overrides(base, VARIANT_OVERRIDES[variant])


def make_jobs(args: argparse.Namespace) -> list[AblationJob]:
    datasets = parse_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    variants = [v for v in parse_list(args.variants) if v in VARIANT_OVERRIDES]
    jobs: list[AblationJob] = []
    for dataset in datasets:
        for seed in seeds:
            for variant in variants:
                if dataset == "ogbn-products":
                    jobs.append(
                        AblationJob(
                            exp_id=f"{dataset}_seed{seed}_{variant}",
                            dataset=dataset,
                            split="ogb",
                            variant=variant,
                            seed=seed,
                            script="sampled_ogb_experiment.py",
                            args=products_args(args, seed, variant),
                        )
                    )
                else:
                    jobs.append(
                        AblationJob(
                            exp_id=f"{dataset}_seed{seed}_{variant}",
                            dataset=dataset,
                            split="random",
                            variant=variant,
                            seed=seed,
                            script="main.py",
                            args=full_graph_args(dataset, args, seed, variant),
                        )
                    )
    return jobs


def run_job(job: AblationJob, log_path: Path, timeout: int | None) -> tuple[int, str, float]:
    cmd = [sys.executable, "-u", job.script, *job.args]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    start = time.time()
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            fh.write(line)
            fh.flush()
            if timeout is not None and time.time() - start > timeout:
                proc.kill()
                msg = f"\nTIMEOUT after {timeout} seconds\n"
                chunks.append(msg)
                fh.write(msg)
                return 124, "".join(chunks), time.time() - start
    return proc.wait(), "".join(chunks), time.time() - start


def write_manifest(run_dir: Path, jobs: list[AblationJob]) -> None:
    path = run_dir / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["exp_id", "dataset", "split", "variant", "seed", "script", "args", "command"],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "exp_id": job.exp_id,
                    "dataset": job.dataset,
                    "split": job.split,
                    "variant": job.variant,
                    "seed": job.seed,
                    "script": job.script,
                    "args": " ".join(job.args),
                    "command": " ".join([sys.executable, "-u", job.script, *job.args]),
                }
            )


def append_summary(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    fieldnames = [
        "exp_id",
        "dataset",
        "split",
        "variant",
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
    ]
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as fh:
        return {row["exp_id"] for row in csv.DictReader(fh) if row.get("status") == "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted GRAPPLE ablations on paper-relevant datasets.")
    parser.add_argument("--datasets", default="ogbn-products")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--variants", default="full,no_clu,no_etf,no_bal,supervised_only")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--ogb_source", default="snap", choices=["snap", "official", "graphbolt", "dgl"])
    parser.add_argument("--sampled_epochs", type=int, default=30)
    parser.add_argument("--sampled_eval_interval", type=int, default=1)
    parser.add_argument("--train_batches", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_neighbors", default="15,10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lambda_cap", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or f"target_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = DEFAULT_OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    jobs = make_jobs(args)
    write_manifest(run_dir, jobs)
    summary_path = run_dir / "summary.csv"
    done = completed(summary_path) if args.resume else set()

    for idx, job in enumerate(jobs, start=1):
        if job.exp_id in done:
            print(f"\n[{idx}/{len(jobs)}] skip {job.exp_id}")
            continue
        log_path = run_dir / "logs" / f"{idx:03d}_{job.exp_id}.log"
        print(f"\n[{idx}/{len(jobs)}] {job.dataset} seed={job.seed} variant={job.variant}")
        if args.dry_run:
            print("$ " + " ".join([sys.executable, "-u", job.script, *job.args]))
            continue
        code, text, duration = run_job(job, log_path, args.timeout)
        row = {
            "exp_id": job.exp_id,
            "dataset": job.dataset,
            "split": job.split,
            "variant": job.variant,
            "seed": job.seed,
            "status": "ok" if code == 0 else "failed",
            "returncode": code,
            "duration_sec": round(duration, 2),
            "best_val_acc": metric_from_text(text, "best_val_acc"),
            "test_acc": metric_from_text(text, "test_acc"),
            "kappa": metric_from_text(text, "kappa"),
            "rho": metric_from_text(text, "rho"),
            "log_path": str(log_path),
            "command": " ".join([sys.executable, "-u", job.script, *job.args]),
        }
        append_summary(summary_path, row)
        print(f"status={row['status']} val={row['best_val_acc']} test={row['test_acc']} log={log_path}")

    print(f"\nTarget ablation outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
