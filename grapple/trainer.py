from __future__ import annotations

from dataclasses import dataclass
import copy

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from .models.grapple import GrappleModel


@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    encoder_lr_mult: float = 1.0
    projector_lr_mult: float = 1.0
    prototype_lr_mult: float = 1.0
    curvature_lr_mult: float = 1.0
    mahalanobis_lr_mult: float = 1.0
    euclidean_lr_mult: float = 1.0
    other_lr_mult: float = 1.0
    prototype_freeze_epochs: int = 0
    curvature_freeze_epochs: int = 0
    mahalanobis_freeze_epochs: int = 0
    lambda_sup: float = 1.0
    lambda_clu: float = 1.0
    lambda_etf: float = 1.0
    lambda_bal: float = 1.0
    lambda_cap: float = 1.0
    lambda_reg: float = 1e-4
    lambda_kappa: float = 0.0
    cap_margin: float = 0.1
    capacity_mode: str = "global"
    confusion_weight: float = 0.0
    geometry_warmup_epochs: int = 0
    geometry_warmup_start: float = 0.0
    eval_interval: int = 20
    checkpoint_path: str = "best_checkpoint.pt"
    # Legacy fields kept for script compatibility; they are not used by the
    # revised methodology trainer.
    tau: float = 0.2
    lambda_cv: float = 0.5
    gamma: float = 0.1
    augment: object | None = None
    reg: object | None = None


def build_optimizer(model: GrappleModel, cfg: TrainConfig) -> Adam:
    groups: dict[str, dict[str, object]] = {
        "encoder": {"params": [], "lr_mult": cfg.encoder_lr_mult, "freeze_epochs": 0},
        "projector": {"params": [], "lr_mult": cfg.projector_lr_mult, "freeze_epochs": 0},
        "prototype": {"params": [], "lr_mult": cfg.prototype_lr_mult, "freeze_epochs": cfg.prototype_freeze_epochs},
        "curvature": {"params": [], "lr_mult": cfg.curvature_lr_mult, "freeze_epochs": cfg.curvature_freeze_epochs},
        "mahalanobis": {"params": [], "lr_mult": cfg.mahalanobis_lr_mult, "freeze_epochs": cfg.mahalanobis_freeze_epochs},
        "euclidean": {"params": [], "lr_mult": cfg.euclidean_lr_mult, "freeze_epochs": 0},
        "other": {"params": [], "lr_mult": cfg.other_lr_mult, "freeze_epochs": 0},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            key = "encoder"
        elif name.startswith("projector."):
            key = "projector"
        elif name == "prototype_raw":
            key = "prototype"
        elif name in {"eta", "xi"}:
            key = "curvature"
        elif name.startswith("mahalanobis."):
            key = "mahalanobis"
        elif name.startswith("euclidean_classifier."):
            key = "euclidean"
        else:
            key = "other"
        groups[key]["params"].append(param)

    param_groups = []
    for key, group in groups.items():
        params = group["params"]
        if not params:
            continue
        base_lr = float(cfg.lr) * float(group["lr_mult"])
        param_groups.append(
            {
                "params": params,
                "lr": base_lr,
                "base_lr": base_lr,
                "freeze_epochs": int(group["freeze_epochs"]),
                "name": key,
            }
        )
    return Adam(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def update_optimizer_lrs(optimizer: Adam, epoch: int) -> None:
    for group in optimizer.param_groups:
        freeze_epochs = int(group.get("freeze_epochs", 0))
        base_lr = float(group.get("base_lr", group["lr"]))
        group["lr"] = 0.0 if epoch <= freeze_epochs else base_lr


def canonical_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() > 1:
        mask = mask[:, 0]
    return mask.bool().view(-1)


def masked_accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    mask = canonical_mask(mask)
    if y.dim() > 1 and y.size(-1) == 1:
        y = y.squeeze(-1)
    if int(mask.sum().item()) == 0:
        return 0.0
    pred = logits[mask].argmax(dim=-1)
    acc = (pred == y[mask]).float().mean()
    return float(acc.item())


def train(model: GrappleModel, data, device: torch.device, cfg: TrainConfig) -> float:
    model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = canonical_mask(data.train_mask).to(device)
    val_mask = canonical_mask(data.val_mask).to(device)
    test_mask = canonical_mask(data.test_mask).to(device)

    optimizer = build_optimizer(model, cfg)

    best_val_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    def geometry_scale(epoch: int) -> float:
        warmup_epochs = int(cfg.geometry_warmup_epochs)
        if warmup_epochs <= 0:
            return 1.0
        start = float(cfg.geometry_warmup_start)
        progress = min(max(epoch - 1, 0) / float(warmup_epochs), 1.0)
        return start + (1.0 - start) * progress

    for epoch in tqdm(range(1, cfg.epochs + 1), desc="Training"):
        update_optimizer_lrs(optimizer, epoch)
        model.train()
        out = model(x, edge_index)
        geo = model.geometric_terms(
            out,
            cap_margin=cfg.cap_margin,
            y=y,
            capacity_mode=cfg.capacity_mode,
            confusion_weight=cfg.confusion_weight,
        )
        geo_scale = geometry_scale(epoch)

        loss_sup = F.cross_entropy(out["logits"][train_mask], y[train_mask])
        loss = (
            cfg.lambda_sup * loss_sup
            + geo_scale * (
                cfg.lambda_clu * geo["l_clu"]
                + cfg.lambda_etf * geo["l_etf"]
                + cfg.lambda_bal * geo["l_bal"]
                + cfg.lambda_cap * geo["l_cap"]
                + cfg.lambda_reg * geo["l_reg"]
                + cfg.lambda_kappa * geo["l_kappa"]
            )
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            tqdm.write(
                "Epoch "
                f"{epoch:03d} | loss={loss.item():.4f} | "
                f"Lsup={loss_sup.item():.4f} | "
                f"Lclu={geo['l_clu'].item():.4f} | "
                f"Letf={geo['l_etf'].item():.4f} | "
                f"Lbal={geo['l_bal'].item():.4f} | "
                f"Lcap={geo['l_cap'].item():.4f} | "
                f"geo_scale={geo_scale:.3f} | "
                f"kappa={out['kappa'].item():.4f} | "
                f"rho={out['rho'].item():.4f}"
            )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                eval_out = model(x, edge_index)
                val_acc = masked_accuracy(eval_out["logits"], y, val_mask)
                test_acc = masked_accuracy(eval_out["logits"], y, test_mask)

            tqdm.write(
                f"Eval @ epoch {epoch:03d} | val_acc={val_acc:.4f} | "
                f"test_acc={test_acc:.4f} | kappa={eval_out['kappa'].item():.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                try:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": best_state,
                            "val_acc": val_acc,
                            "test_acc": test_acc,
                        },
                        cfg.checkpoint_path,
                    )
                except Exception as exc:
                    tqdm.write(f"Failed to save checkpoint: {exc}")

    if best_state is not None:
        model.load_state_dict(best_state)
        tqdm.write(f"Loaded best checkpoint from epoch {best_epoch:03d} (val_acc={best_val_acc:.4f}).")

    return float(best_val_acc)
