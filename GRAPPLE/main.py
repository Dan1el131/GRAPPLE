from __future__ import annotations

import argparse

import torch

from grapple.data import load_dataset
from grapple.models.grapple import GrappleModel, ModelConfig
from grapple.trainer import TrainConfig, masked_accuracy, train
from grapple.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help=(
            "Dataset name (case-insensitive). Supported: "
            "cora/citeseer/pubmed, amazon-computers/amazon-photos, "
            "coauthor-cs/coauthor-physics, wikics, "
            "webkb-cornell/webkb-texas/webkb-wisconsin, "
            "ogbn-arxiv/ogbn-products."
        ),
    )
    p.add_argument("--data_root", type=str, default="data", help="Root directory for datasets")
    p.add_argument("--split", type=str, default="public", choices=["public", "random", "ogb"])
    p.add_argument("--train_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.8)

    norm_group = p.add_mutually_exclusive_group()
    norm_group.add_argument("--normalize_features", dest="normalize_features", action="store_true")
    norm_group.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    p.set_defaults(normalize_features=True)

    p.add_argument("--to_undirected", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--encoder_lr_mult", type=float, default=1.0)
    p.add_argument("--projector_lr_mult", type=float, default=1.0)
    p.add_argument("--prototype_lr_mult", type=float, default=1.0)
    p.add_argument("--curvature_lr_mult", type=float, default=1.0)
    p.add_argument("--mahalanobis_lr_mult", type=float, default=1.0)
    p.add_argument("--euclidean_lr_mult", type=float, default=1.0)
    p.add_argument("--other_lr_mult", type=float, default=1.0)
    p.add_argument("--prototype_freeze_epochs", type=int, default=0)
    p.add_argument("--curvature_freeze_epochs", type=int, default=0)
    p.add_argument("--mahalanobis_freeze_epochs", type=int, default=0)
    p.add_argument("--eval_interval", type=int, default=20)

    p.add_argument("--gcn_hidden", type=int, default=256)
    p.add_argument("--gcn_out", type=int, default=128)
    p.add_argument("--gcn_layers", type=int, default=2)
    p.add_argument("--encoder_type", type=str, default="gcn", choices=["gcn", "sage", "gcnii", "appnp"])
    p.add_argument("--gcnii_alpha", type=float, default=0.1)
    p.add_argument("--gcnii_theta", type=float, default=0.5)
    p.add_argument("--appnp_k", type=int, default=10)
    p.add_argument("--appnp_alpha", type=float, default=0.1)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--num_prototypes", type=int, default=None)
    p.add_argument("--prototypes_per_class", type=int, default=1)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--kappa_max", type=float, default=1.0)
    p.add_argument("--init_kappa", type=float, default=0.0)
    p.add_argument("--radius_init", type=float, default=0.5)
    p.add_argument("--prototype_init", type=str, default="random", choices=["random", "simplex"])
    p.add_argument("--geometry_logit_weight", type=float, default=1.0)
    p.add_argument("--euclidean_head_weight", type=float, default=0.0)
    p.add_argument("--curvature_beta", type=float, default=10.0)
    p.add_argument("--clip_eps", type=float, default=1e-8)
    p.add_argument("--clip_delta", type=float, default=1e-3)

    p.add_argument("--lambda_clu", type=float, default=1.0)
    p.add_argument("--lambda_sup", type=float, default=1.0)
    p.add_argument("--lambda_etf", type=float, default=1.0)
    p.add_argument("--lambda_bal", type=float, default=1.0)
    p.add_argument("--lambda_cap", type=float, default=1.0)
    p.add_argument("--lambda_reg", type=float, default=1e-4)
    p.add_argument("--lambda_kappa", type=float, default=0.0)
    p.add_argument("--cap_margin", type=float, default=0.1)
    p.add_argument("--capacity_mode", type=str, default="global", choices=["global", "pairwise"])
    p.add_argument("--confusion_weight", type=float, default=0.0)
    p.add_argument("--geometry_warmup_epochs", type=int, default=0)
    p.add_argument("--geometry_warmup_start", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    data, _, meta = load_dataset(
        name=args.dataset,
        root=args.data_root,
        normalize_features=bool(args.normalize_features),
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        to_undirected=args.to_undirected,
    )

    if args.prototypes_per_class < 1:
        raise ValueError("prototypes_per_class must be at least 1.")
    default_num_prototypes = int(meta["num_classes"]) * int(args.prototypes_per_class)
    num_prototypes = default_num_prototypes if args.num_prototypes is None else int(args.num_prototypes)
    if num_prototypes < meta["num_classes"] or num_prototypes % meta["num_classes"] != 0:
        raise ValueError(
            "The exploratory grouped-prototype implementation requires num_prototypes to be "
            "a positive multiple of num_classes. "
            f"Got num_prototypes={num_prototypes}, num_classes={meta['num_classes']}."
        )

    model_cfg = ModelConfig(
        in_dim=meta["num_features"],
        num_classes=meta["num_classes"],
        encoder_type=args.encoder_type,
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
        gcn_layers=args.gcn_layers,
        gcnii_alpha=args.gcnii_alpha,
        gcnii_theta=args.gcnii_theta,
        appnp_k=args.appnp_k,
        appnp_alpha=args.appnp_alpha,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        num_prototypes=num_prototypes,
        tau=args.tau,
        kappa_max=args.kappa_max,
        init_kappa=args.init_kappa,
        radius_init=args.radius_init,
        prototype_init=args.prototype_init,
        geometry_logit_weight=args.geometry_logit_weight,
        euclidean_head_weight=args.euclidean_head_weight,
        curvature_beta=args.curvature_beta,
        clip_eps=args.clip_eps,
        clip_delta=args.clip_delta,
    )
    model = GrappleModel(model_cfg)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of total parameters: {total_params}")

    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        encoder_lr_mult=args.encoder_lr_mult,
        projector_lr_mult=args.projector_lr_mult,
        prototype_lr_mult=args.prototype_lr_mult,
        curvature_lr_mult=args.curvature_lr_mult,
        mahalanobis_lr_mult=args.mahalanobis_lr_mult,
        euclidean_lr_mult=args.euclidean_lr_mult,
        other_lr_mult=args.other_lr_mult,
        prototype_freeze_epochs=args.prototype_freeze_epochs,
        curvature_freeze_epochs=args.curvature_freeze_epochs,
        mahalanobis_freeze_epochs=args.mahalanobis_freeze_epochs,
        lambda_sup=args.lambda_sup,
        lambda_clu=args.lambda_clu,
        lambda_etf=args.lambda_etf,
        lambda_bal=args.lambda_bal,
        lambda_cap=args.lambda_cap,
        lambda_reg=args.lambda_reg,
        lambda_kappa=args.lambda_kappa,
        cap_margin=args.cap_margin,
        capacity_mode=args.capacity_mode,
        confusion_weight=args.confusion_weight,
        geometry_warmup_epochs=args.geometry_warmup_epochs,
        geometry_warmup_start=args.geometry_warmup_start,
        eval_interval=args.eval_interval,
        checkpoint_path=f"best_checkpoint_{args.dataset}.pt",
    )

    best_val_acc = train(model, data, device=device, cfg=train_cfg)

    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        test_acc = masked_accuracy(out["logits"], data.y.to(device), data.test_mask.to(device))

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Prototype classification accuracy (test): {test_acc:.4f}")
    print(f"Learned curvature kappa: {out['kappa'].item():.6f}")
    print(f"Learned prototype radius rho: {out['rho'].item():.6f}")


if __name__ == "__main__":
    main()
