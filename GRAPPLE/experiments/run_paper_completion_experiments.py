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
DEFAULT_OUTPUT_ROOT = ROOT / "overnight_results"

METRIC_PATTERNS = {
    "best_val_acc": [
        re.compile(r"Best validation accuracy:\s*([0-9.]+)"),
        re.compile(r"best_val=([0-9.]+)"),
    ],
    "test_acc": [
        re.compile(r"Prototype classification accuracy \(test\):\s*([0-9.]+)"),
        re.compile(r"Test accuracy at best validation:\s*([0-9.]+)"),
        re.compile(r"best_test=([0-9.]+)"),
        re.compile(r'"test_acc":\s*([0-9.]+)'),
    ],
    "kappa": [re.compile(r"Learned curvature kappa:\s*(-?[0-9.]+)"), re.compile(r'"kappa":\s*(-?[0-9.]+)')],
    "rho": [re.compile(r"Learned prototype radius rho:\s*([0-9.]+)"), re.compile(r'"rho":\s*([0-9.]+)')],
}


@dataclass(frozen=True)
class Job:
    exp_id: str
    stage: str
    dataset: str
    seed: int
    setting: str
    epochs: int
    command: tuple[str, ...]


def parse_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in parse_list(raw)]


def parse_float_list(raw: str) -> list[float]:
    return [float(part) for part in parse_list(raw)]


def metric_from_text(text: str, name: str) -> float | None:
    hits: list[str] = []
    for pattern in METRIC_PATTERNS[name]:
        hits.extend(pattern.findall(text))
    return float(hits[-1]) if hits else None


def encoder_for(dataset: str) -> str:
    if dataset.startswith(("amazon-", "coauthor-")) or dataset in {"citeseer", "actor", "wikics", "wiki-cs"}:
        return "sage"
    return "gcn"


def split_args(dataset: str) -> tuple[str, ...]:
    if dataset == "ogbn-arxiv":
        return ("--split", "ogb", "--to_undirected")
    return ("--split", "random", "--train_ratio", "0.1", "--val_ratio", "0.1", "--test_ratio", "0.8")


def feature_norm_args(dataset: str) -> tuple[str, ...]:
    return ("--no_normalize_features",) if dataset.startswith("amazon-") else ("--normalize_features",)


def grapple_cmd(args, dataset: str, seed: int, epochs: int, *, tau: float, lambda_clu: float, ppc: int = 2) -> tuple[str, ...]:
    return (
        sys.executable,
        "-u",
        "main.py",
        "--dataset",
        dataset,
        "--data_root",
        args.data_root,
        *split_args(dataset),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--eval_interval",
        str(args.eval_interval),
        "--encoder_type",
        encoder_for(dataset),
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
        "--prototype_init",
        "simplex",
        "--prototypes_per_class",
        str(ppc),
        "--tau",
        str(tau),
        "--lambda_sup",
        "1.0",
        "--lambda_clu",
        str(lambda_clu),
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
        *feature_norm_args(dataset),
    )


def sanity_job(args) -> list[Job]:
    return [
        Job(
            "ogbn_arxiv_sanity",
            "sanity",
            "ogbn-arxiv",
            0,
            "split_label_prediction_hist",
            args.sanity_epochs,
            (
                sys.executable,
                "-u",
                "experiments/ogbn_arxiv_sanity_check.py",
                "--data_root",
                args.data_root,
                "--output_root",
                str(args.output_root),
                "--device",
                args.device,
                "--epochs",
                str(args.sanity_epochs),
                "--eval_interval",
                str(args.eval_interval),
            ),
        )
    ]


def sensitivity_jobs(args) -> list[Job]:
    jobs: list[Job] = []
    for dataset in parse_list(args.sensitivity_datasets):
        if dataset == "ogbn-products":
            continue
        for seed in parse_int_list(args.seeds):
            for tau in parse_float_list(args.tau_values):
                setting = f"tau={tau:g}_lambda_clu=0.01"
                jobs.append(
                    Job(
                        f"sensitivity_{dataset}_seed{seed}_tau{tau:g}",
                        "sensitivity",
                        dataset,
                        seed,
                        setting,
                        args.epochs,
                        grapple_cmd(args, dataset, seed, args.epochs, tau=tau, lambda_clu=0.01, ppc=2),
                    )
                )
            for clu in parse_float_list(args.lambda_clu_values):
                setting = f"tau=2_lambda_clu={clu:g}"
                jobs.append(
                    Job(
                        f"sensitivity_{dataset}_seed{seed}_clu{clu:g}",
                        "sensitivity",
                        dataset,
                        seed,
                        setting,
                        args.epochs,
                        grapple_cmd(args, dataset, seed, args.epochs, tau=2.0, lambda_clu=clu, ppc=2),
                    )
                )
    return jobs


def prototype_jobs(args) -> list[Job]:
    jobs: list[Job] = []
    for dataset in parse_list(args.prototype_datasets):
        if dataset == "ogbn-products":
            continue
        for seed in parse_int_list(args.seeds):
            for ppc in parse_int_list(args.prototype_multipliers):
                setting = f"K={ppc}C_tau=2_lambda_clu=0.01"
                jobs.append(
                    Job(
                        f"prototype_usage_{dataset}_seed{seed}_K{ppc}C",
                        "prototype_usage",
                        dataset,
                        seed,
                        setting,
                        args.prototype_epochs,
                        (
                            sys.executable,
                            "-u",
                            "experiments/prototype_usage_analysis.py",
                            "--dataset",
                            dataset,
                            "--data_root",
                            args.data_root,
                            "--output_root",
                            str(args.output_root),
                            "--device",
                            args.device,
                            "--seed",
                            str(seed),
                            "--epochs",
                            str(args.prototype_epochs),
                            "--eval_interval",
                            str(args.eval_interval),
                            "--hidden",
                            str(args.hidden),
                            "--out_dim",
                            str(args.out_dim),
                            "--proj_dim",
                            str(args.proj_dim),
                            "--layers",
                            str(args.layers),
                            "--prototypes_per_class",
                            str(ppc),
                            "--tau",
                            "2.0",
                            "--lambda_clu",
                            "0.01",
                        ),
                    )
                )
    return jobs


def build_jobs(args) -> list[Job]:
    jobs: list[Job] = []
    stages = set(parse_list(args.stages))
    if "sanity" in stages:
        jobs.extend(sanity_job(args))
    if "sensitivity" in stages:
        jobs.extend(sensitivity_jobs(args))
    if "prototype" in stages:
        jobs.extend(prototype_jobs(args))
    return jobs


def run_job(job: Job, log_path: Path, timeout: int | None) -> tuple[int, str, float]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    start = time.time()
    proc = subprocess.Popen(
        list(job.command),
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
        fh.write("$ " + " ".join(job.command) + "\n\n")
        fh.flush()
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if line:
                chunks.append(line)
                fh.write(line)
                fh.flush()
            if timeout is not None and time.time() - start > timeout:
                proc.kill()
                msg = f"\nTIMEOUT after {timeout} seconds\n"
                chunks.append(msg)
                fh.write(msg)
                return 124, "".join(chunks), time.time() - start
            if line == "" and proc.poll() is not None:
                break
    return proc.wait(), "".join(chunks), time.time() - start


def append_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "exp_id",
        "stage",
        "dataset",
        "seed",
        "setting",
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


def completed_exp_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as fh:
        return {row["exp_id"] for row in csv.DictReader(fh) if row.get("status") == "ok"}


def write_manifest(path: Path, jobs: list[Job]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["exp_id", "stage", "dataset", "seed", "setting", "epochs", "command"])
        writer.writeheader()
        for job in jobs:
            writer.writerow({**job.__dict__, "command": " ".join(job.command)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-completion follow-up experiments under overnight_results.")
    parser.add_argument("--stages", default="sanity,sensitivity,prototype")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--sensitivity_datasets", default="coauthor-cs,amazon-computers,citeseer,actor")
    parser.add_argument("--prototype_datasets", default="coauthor-cs,amazon-computers,citeseer,actor")
    parser.add_argument("--tau_values", default="0.5,1,2")
    parser.add_argument("--lambda_clu_values", default="0,0.0001,0.001,0.01")
    parser.add_argument("--prototype_multipliers", default="1,2,4")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--prototype_epochs", type=int, default=160)
    parser.add_argument("--sanity_epochs", type=int, default=80)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()
    args.output_root = Path(args.output_root)

    run_name = args.run_name or f"paper_completion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / "run_logs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["results", "figures", "debug_reports", "run_logs"]:
        (args.output_root / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({**vars(args), "output_root": str(args.output_root)}, indent=2) + "\n", encoding="utf-8")

    jobs = build_jobs(args)
    write_manifest(run_dir / "manifest.csv", jobs)
    summary_path = args.output_root / "results" / "paper_completion_queue_summary.csv"
    done = completed_exp_ids(summary_path)
    print(f"paper completion run dir: {run_dir}")
    print(f"total jobs: {len(jobs)}; already completed: {len(done)}")
    for idx, job in enumerate(jobs, start=1):
        if job.exp_id in done:
            print(f"[{idx}/{len(jobs)}] skip completed {job.exp_id}")
            continue
        print(f"\n[{idx}/{len(jobs)}] {job.stage} {job.dataset} seed={job.seed} {job.setting}", flush=True)
        log_path = run_dir / job.stage / f"{idx:04d}_{job.exp_id}.log"
        code, text, duration = run_job(job, log_path, args.timeout)
        row = {
            "exp_id": job.exp_id,
            "stage": job.stage,
            "dataset": job.dataset,
            "seed": job.seed,
            "setting": job.setting,
            "status": "ok" if code == 0 else "failed",
            "returncode": code,
            "duration_sec": round(duration, 2),
            "best_val_acc": metric_from_text(text, "best_val_acc"),
            "test_acc": metric_from_text(text, "test_acc"),
            "kappa": metric_from_text(text, "kappa"),
            "rho": metric_from_text(text, "rho"),
            "log_path": str(log_path),
            "command": " ".join(job.command),
        }
        append_summary(summary_path, row)
        print(f"status={row['status']} val={row['best_val_acc']} test={row['test_acc']} log={log_path}", flush=True)
    print(f"paper completion outputs saved to: {args.output_root}")


if __name__ == "__main__":
    main()
