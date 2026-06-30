# GRAPPLE

Minimal reference implementation of **GRAPPLE**, a prototype-curvature graph learning model for node classification.

This repository keeps only the core code needed to run the method:

- the GRAPPLE model;
- graph encoders and stereographic geometry utilities;
- dataset loading and train/validation/test split handling;
- full-graph training entry point;
- sampled large-graph training entry point for OGB-style datasets.

Experiment orchestration, paper figure drawing, ablation plotting, sensitivity analysis, rank diagnostics, logs, checkpoints, cached datasets, generated figures, and result archives are intentionally not included.

## Structure

```text
.
├── grapple/
│   ├── data.py                       # Dataset loading and split handling
│   ├── trainer.py                    # Training loop and metrics
│   ├── utils/seed.py                 # Reproducibility helper
│   └── models/
│       ├── grapple.py                # GRAPPLE model
│       ├── gcn.py                    # GCN, GraphSAGE, GCNII, APPNP encoders
│       └── stereographic.py          # Geometry utilities
├── main.py                           # Full-graph GRAPPLE training
├── sampled_ogb_experiment.py         # Neighbor-sampled GRAPPLE training
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── LICENSE
```

## Installation

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate grapple
```

Or install with pip after installing the PyTorch build that matches your machine:

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, install wheels that match your PyTorch and CUDA/CPU versions. Large-graph neighbor sampling may also require `pyg-lib` or `torch-sparse`.

## Quick Start

Run GRAPPLE on Cora:

```bash
python main.py --dataset cora --split public --device cpu --epochs 200
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

Run the sampled version for OGBN-Products:

```bash
python sampled_ogb_experiment.py \
  --dataset ogbn-products \
  --split ogb \
  --encoder_type sage \
  --batch_size 2048 \
  --num_neighbors 15 10 \
  --device cuda
```

Datasets are downloaded automatically by PyTorch Geometric or OGB on first use. By default they are cached under `data/`, which is ignored by Git.

## Supported Datasets

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

Use `--split public` for datasets with public masks, `--split random` for generated splits, and `--split ogb` for OGB official splits.

## Main Options

```bash
python main.py --help
```

Common options:

- `--dataset`: dataset name.
- `--split`: `public`, `random`, or `ogb`.
- `--encoder_type`: `gcn`, `sage`, `gcnii`, or `appnp`.
- `--prototypes_per_class`: grouped prototypes per class.
- `--tau`: assignment temperature.
- `--lambda_sup`: supervised loss weight.
- `--lambda_clu`: node-to-prototype compactness weight.
- `--lambda_etf`: prototype simplex regularization weight.
- `--lambda_bal`: balanced prototype usage weight.
- `--lambda_cap`: capacity-matching curvature loss weight.
- `--lambda_reg`: Mahalanobis regularization weight.

## Outputs

Training may write checkpoint files such as `best_checkpoint_<dataset>.pt`. Generated checkpoints, datasets, logs, and result folders are ignored by `.gitignore`.

## Citation

Please wait the final decision on paper.

## License

This project is released under the MIT License. See `LICENSE` for details.
