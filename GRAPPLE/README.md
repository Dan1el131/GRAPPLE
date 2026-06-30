# GRAPPLE

Reference implementation of **GRAPPLE**, a graph representation learning model built around grouped prototypes, stereographic geometry, and learnable curvature.

This release is a cleaned GitHub-ready code package. It intentionally excludes local datasets, checkpoints, logs, generated figures, compressed result bundles, and paper draft files.

## Repository Layout

```text
.
├── grapple/                         # Core package
│   ├── data.py                       # Unified PyG/OGB dataset loader
│   ├── trainer.py                    # Training loop and metrics
│   └── models/
│       ├── grapple.py                # GRAPPLE model
│       ├── gcn.py                    # GCN/SAGE/GCNII/APPNP encoders
│       ├── hyperbolic.py             # Riemannian baseline components
│       └── stereographic.py          # Geometry utilities
├── main.py                           # Main GRAPPLE training entry point
├── full_supervised_baseline.py       # MLP/GCN/GAT/GraphSAGE full-graph baselines
├── sampled_ogb_experiment.py         # Neighbor-sampled GRAPPLE for large OGB graphs
├── sampled_supervised_baseline.py    # Neighbor-sampled supervised baselines
├── ablation_study.py                 # Single-dataset ablations
├── sensitivity_analysis.py           # Hyperparameter sensitivity experiments
├── rank_collapse_experiment.py       # Rank-collapse diagnostics
├── experiments/                      # Paper-level experiment drivers
└── scripts/                          # Figure/table helper scripts
```

## Installation

Create an environment with Python 3.10 or later.

```bash
conda env create -f environment.yml
conda activate grapple
```

Alternatively, install from `requirements.txt` after installing the PyTorch build that matches your CUDA/CPU platform:

```bash
pip install -r requirements.txt
```

For PyTorch Geometric compiled extensions, use the wheel selector that matches your local PyTorch and CUDA versions. Large-graph neighbor sampling may require `pyg-lib` or `torch-sparse`.

## Quick Start

Run GRAPPLE on Cora:

```bash
python main.py --dataset cora --split public --device cpu --epochs 200
```

Run on Amazon Computers with a random split:

```bash
python main.py --dataset amazon-computers --split random \
  --train_ratio 0.1 --val_ratio 0.1 --test_ratio 0.8 \
  --seed 0 --epochs 200
```

Run on OGBN-Arxiv:

```bash
python main.py --dataset ogbn-arxiv --split ogb --to_undirected \
  --encoder_type gcn --gcn_hidden 128 --gcn_out 64 --proj_dim 64
```

Large OGBN-Products runs use the sampled entry point:

```bash
python sampled_ogb_experiment.py --dataset ogbn-products --split ogb \
  --encoder_type sage --batch_size 2048 --num_neighbors 15 10
```

Datasets are downloaded by PyTorch Geometric or OGB into `--data_root` on first use. The default cache directory is `data/`, which is ignored by Git.

## Reproducing Experiments

Create a dry-run manifest for the main paper experiments:

```bash
python experiments/run_main_experiments.py --dry_run
```

Run a smaller main experiment queue:

```bash
python experiments/run_main_experiments.py \
  --datasets cora,citeseer,pubmed,amazon-computers \
  --seeds 0,1,2 --device cuda --epochs 200
```

Run ablations:

```bash
python ablation_study.py --dataset cora --split public --epochs 200
python experiments/run_target_ablation.py --datasets cora,citeseer,pubmed --seeds 0,1,2
```

Generated outputs are written under `runs/`, `results/`, or explicit output directories and are ignored by Git.

## Supported Datasets

The unified loader supports:

- Planetoid: `cora`, `citeseer`, `pubmed`
- Amazon: `amazon-computers`, `amazon-photos`
- Coauthor: `coauthor-cs`, `coauthor-physics`
- WikiCS: `wikics`
- WebKB: `webkb-cornell`, `webkb-texas`, `webkb-wisconsin`
- WikipediaNetwork: `chameleon`, `squirrel`
- Actor: `actor`
- OGB: `ogbn-arxiv`, `ogbn-products`

Use `--split public`, `--split random`, or `--split ogb` as appropriate for the dataset.

## Notes for Public Release

- No local checkpoints, logs, generated plots, cached datasets, or compressed result archives are included.
- The package namespace is `grapple`, and method labels use `GRAPPLE`.
- Add the final paper citation once the manuscript metadata is public.

## License

This cleaned release includes an MIT license file. Adjust it before publishing if the project should use a different license.
