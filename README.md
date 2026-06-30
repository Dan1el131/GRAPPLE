# GRAPPLE

Official code release for **GRAPPLE**, a prototype-curvature graph learning framework for node classification.

GRAPPLE combines graph neural encoders with grouped class prototypes in a stereographic geometry. The model learns node representations, prototype assignments, curvature, and radius parameters jointly, and is designed to support both full-graph training on common citation/coauthor/web graphs and mini-batch training on large OGB graphs.

This repository is a cleaned public release. It contains source code and reproducibility scripts only; local datasets, checkpoints, logs, generated figures, result archives, and paper draft files are intentionally excluded.

## Highlights

- Unified dataset loader for Planetoid, Amazon, Coauthor, WikiCS, WebKB, WikipediaNetwork, Actor, and OGB node-classification datasets.
- GRAPPLE model with GCN, GraphSAGE, GCNII, and APPNP encoder options.
- Learnable stereographic curvature and prototype radius.
- Grouped prototypes per class with geometry-aware classification.
- Reproducible scripts for main experiments, ablations, sensitivity analysis, rank diagnostics, and visualization.
- Full-graph and sampled large-graph training entry points.

## Repository Structure

```text
.
├── grapple/                         # Core Python package
│   ├── data.py                       # Unified dataset loading and splits
│   ├── trainer.py                    # Training loop, optimizer groups, metrics
│   ├── experiment_utils.py           # Helpers for experiment scripts
│   └── models/
│       ├── grapple.py                # Main GRAPPLE model
│       ├── gcn.py                    # GCN, GraphSAGE, GCNII, APPNP encoders
│       ├── stereographic.py          # Stereographic geometry utilities
│       └── hyperbolic.py             # Riemannian baseline components
├── main.py                           # Full-graph GRAPPLE training entry point
├── sampled_ogb_experiment.py         # Neighbor-sampled GRAPPLE for large graphs
├── full_supervised_baseline.py       # MLP/GCN/GAT/GraphSAGE full-graph baselines
├── sampled_supervised_baseline.py    # Neighbor-sampled supervised baselines
├── ablation_study.py                 # Single-dataset ablation study
├── sensitivity_analysis.py           # Hyperparameter sensitivity study
├── rank_collapse_experiment.py       # Rank-collapse diagnostics
├── experiments/                      # Paper-scale experiment drivers
├── scripts/                          # Figure and analysis helper scripts
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## Requirements

The code is tested with Python 3.10 and PyTorch/PyTorch Geometric. A CUDA GPU is recommended for larger datasets, but small datasets can run on CPU.

Core dependencies:

- `torch`
- `torch-geometric`
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`
- `tqdm`
- `geoopt`
- `ogb`

Some PyTorch Geometric extensions are platform-specific. If neighbor sampling fails on large datasets, install the PyG wheels matching your PyTorch and CUDA versions, especially `pyg-lib` or `torch-sparse`.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate grapple
```

Using pip:

```bash
pip install -r requirements.txt
```

If you install PyTorch manually, install the PyTorch build first, then install PyTorch Geometric following the official wheel instructions for your CUDA/CPU platform.

## Quick Start

Run GRAPPLE on Cora:

```bash
python main.py \
  --dataset cora \
  --split public \
  --device cpu \
  --epochs 200
```

Run GRAPPLE on Amazon Computers with a random split:

```bash
python main.py \
  --dataset amazon-computers \
  --split random \
  --train_ratio 0.1 \
  --val_ratio 0.1 \
  --test_ratio 0.8 \
  --seed 0 \
  --device cuda \
  --epochs 200
```

Run GRAPPLE on OGBN-Arxiv:

```bash
python main.py \
  --dataset ogbn-arxiv \
  --split ogb \
  --to_undirected \
  --encoder_type gcn \
  --gcn_hidden 128 \
  --gcn_out 64 \
  --proj_dim 64 \
  --device cuda
```

Run the sampled large-graph version on OGBN-Products:

```bash
python sampled_ogb_experiment.py \
  --dataset ogbn-products \
  --split ogb \
  --encoder_type sage \
  --batch_size 2048 \
  --num_neighbors 15 10 \
  --device cuda
```

Datasets are downloaded automatically by PyTorch Geometric or OGB on first use. The default cache path is `data/`, which is ignored by Git.

## Main Arguments

Common training arguments:

- `--dataset`: dataset name, for example `cora`, `citeseer`, `pubmed`, `amazon-computers`, `coauthor-cs`, `ogbn-arxiv`.
- `--split`: one of `public`, `random`, or `ogb`.
- `--encoder_type`: one of `gcn`, `sage`, `gcnii`, or `appnp`.
- `--prototypes_per_class`: number of grouped prototypes per class.
- `--tau`: assignment temperature.
- `--lambda_sup`: supervised cross-entropy weight.
- `--lambda_clu`: node-to-prototype compactness weight.
- `--lambda_etf`: simplex/ETF prototype regularization weight.
- `--lambda_bal`: balanced prototype usage weight.
- `--lambda_cap`: capacity-matching curvature loss weight.
- `--lambda_reg`: Mahalanobis regularization weight.

Inspect all available options with:

```bash
python main.py --help
```

## Supported Datasets

The unified loader supports the following dataset names:

| Family | Dataset names |
| --- | --- |
| Planetoid | `cora`, `citeseer`, `pubmed` |
| Amazon | `amazon-computers`, `amazon-photos` |
| Coauthor | `coauthor-cs`, `coauthor-physics` |
| WikiCS | `wikics` |
| WebKB | `webkb-cornell`, `webkb-texas`, `webkb-wisconsin` |
| WikipediaNetwork | `chameleon`, `squirrel` |
| Actor | `actor` |
| OGB | `ogbn-arxiv`, `ogbn-products` |

Use `--split public` for datasets with official public masks, `--split random` for randomly generated node splits, and `--split ogb` for OGB official splits.

## Reproducing Experiments

Generate a dry-run manifest for the main experiment queue:

```bash
python experiments/run_main_experiments.py --dry_run
```

Run a compact multi-seed main experiment:

```bash
python experiments/run_main_experiments.py \
  --datasets cora,citeseer,pubmed,amazon-computers \
  --seeds 0,1,2 \
  --device cuda \
  --epochs 200
```

Run targeted ablations:

```bash
python experiments/run_target_ablation.py \
  --datasets cora,citeseer,pubmed \
  --seeds 0,1,2 \
  --device cuda
```

Run a single-dataset ablation:

```bash
python ablation_study.py \
  --dataset cora \
  --split public \
  --epochs 200 \
  --device cuda
```

Experiment outputs are written under `runs/`, `results/`, or the output directory specified by each script. These directories are ignored by Git.

## Baselines

Full-graph supervised baselines:

```bash
python full_supervised_baseline.py \
  --dataset cora \
  --split public \
  --encoder_type gcn \
  --epochs 200
```

Sampled supervised baselines for large graphs:

```bash
python sampled_supervised_baseline.py \
  --dataset ogbn-products \
  --split ogb \
  --encoder_type sage \
  --batch_size 2048 \
  --num_neighbors 15 10
```

Additional SSL and Riemannian baseline scripts are included for comparison studies. Some optional baseline scripts may require extra third-party packages beyond the core GRAPPLE dependencies.

## Output Files

Training scripts may create:

- best-checkpoint files such as `best_checkpoint_<dataset>.pt`;
- CSV summaries and manifests;
- log files;
- PDF/PNG figures for analysis scripts.

These files are generated artifacts and are excluded from the release by `.gitignore`.

## Citation

If you use this code, please cite the paper. 

## License

This project is released under the MIT License. See `LICENSE` for details.

## Acknowledgements

This implementation builds on PyTorch, PyTorch Geometric, OGB, scikit-learn, and the broader open-source graph learning ecosystem.
