from __future__ import annotations

from typing import Any
import copy

import torch

from .data import load_dataset
from .models.grapple import GrappleModel, ModelConfig
from .trainer import TrainConfig, masked_accuracy, train


DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "encoder_type": "gcn",
    "gcn_hidden": 256,
    "gcn_out": 128,
    "gcn_layers": 2,
    "gcnii_alpha": 0.1,
    "gcnii_theta": 0.5,
    "gcnii_shared_weights": True,
    "appnp_k": 10,
    "appnp_alpha": 0.1,
    "proj_dim": 128,
    "dropout": 0.2,
    "tau": 0.2,
    "kappa_max": 1.0,
    "init_kappa": 0.0,
    "radius_init": 0.5,
    "geometry_logit_weight": 1.0,
    "euclidean_head_weight": 0.0,
    "curvature_beta": 10.0,
    "clip_eps": 1e-8,
    "clip_delta": 1e-3,
}


DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "epochs": 200,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "encoder_lr_mult": 1.0,
    "projector_lr_mult": 1.0,
    "prototype_lr_mult": 1.0,
    "curvature_lr_mult": 1.0,
    "mahalanobis_lr_mult": 1.0,
    "euclidean_lr_mult": 1.0,
    "other_lr_mult": 1.0,
    "prototype_freeze_epochs": 0,
    "curvature_freeze_epochs": 0,
    "mahalanobis_freeze_epochs": 0,
    "lambda_clu": 1.0,
    "lambda_sup": 1.0,
    "lambda_etf": 1.0,
    "lambda_bal": 1.0,
    "lambda_cap": 1.0,
    "lambda_reg": 1e-4,
    "lambda_kappa": 0.0,
    "cap_margin": 0.1,
    "capacity_mode": "global",
    "confusion_weight": 0.0,
    "geometry_warmup_epochs": 0,
    "geometry_warmup_start": 0.0,
    "eval_interval": 20,
}


def load_experiment_dataset(
    dataset_name: str,
    data_root: str,
    split: str,
    seed: int,
    normalize_features: bool = True,
    to_undirected: bool = False,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.8,
):
    return load_dataset(
        name=dataset_name,
        root=data_root,
        normalize_features=normalize_features,
        split=split,
        seed=seed,
        to_undirected=to_undirected,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def build_model(meta: dict[str, Any], model_overrides: dict[str, Any] | None = None) -> GrappleModel:
    model_cfg = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    if model_overrides:
        model_cfg.update(model_overrides)
    if "num_prototypes" not in model_cfg or model_cfg["num_prototypes"] is None:
        model_cfg["num_prototypes"] = int(meta["num_classes"])
    model_cfg["in_dim"] = int(meta["num_features"])
    model_cfg["num_classes"] = int(meta["num_classes"])
    return GrappleModel(ModelConfig(**model_cfg))


def build_train_config(
    train_overrides: dict[str, Any] | None = None,
    checkpoint_path: str = "best_checkpoint.pt",
) -> TrainConfig:
    train_cfg = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
    if train_overrides:
        train_cfg.update(train_overrides)
    cfg = TrainConfig(**train_cfg)
    cfg.checkpoint_path = checkpoint_path
    return cfg


def evaluate_model(model: GrappleModel, data, device: torch.device) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        y = data.y.to(device)
        metrics = {
            "val_acc": masked_accuracy(out["logits"], y, data.val_mask.to(device)),
            "test_acc": masked_accuracy(out["logits"], y, data.test_mask.to(device)),
            "train_acc": masked_accuracy(out["logits"], y, data.train_mask.to(device)),
            "kappa": float(out["kappa"].item()),
            "rho": float(out["rho"].item()),
            "out": out,
        }
    return metrics


def train_and_evaluate(
    dataset_name: str,
    data_root: str,
    device: torch.device,
    seed: int,
    split: str,
    model_overrides: dict[str, Any] | None = None,
    train_overrides: dict[str, Any] | None = None,
    checkpoint_path: str = "best_checkpoint.pt",
    normalize_features: bool = True,
    to_undirected: bool = False,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.8,
):
    data, dataset, meta = load_experiment_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        seed=seed,
        normalize_features=normalize_features,
        to_undirected=to_undirected,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    model = build_model(meta, model_overrides=model_overrides).to(device)
    train_cfg = build_train_config(train_overrides=train_overrides, checkpoint_path=checkpoint_path)
    best_val = train(model, data, device=device, cfg=train_cfg)
    metrics = evaluate_model(model, data, device=device)
    metrics["best_val_during_train"] = float(best_val)
    return model, data, meta, metrics


def get_representation(out: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    mapping = {
        "h": out["h"],
        "z": out["z"],
        "v": out["v"],
        "v_clipped": out["v_clipped"],
        "x_manifold": out["x_manifold"],
        "logits": out["logits"],
    }
    if name not in mapping:
        raise ValueError(f"Unknown representation '{name}'. Choices: {sorted(mapping)}")
    return mapping[name]
