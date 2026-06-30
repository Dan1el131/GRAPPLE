from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from grapple.data import load_dataset
from grapple.models.grapple import GrappleModel, ModelConfig
from grapple.trainer import TrainConfig, build_optimizer, canonical_mask, masked_accuracy, update_optimizer_lrs
from grapple.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ogbn-products")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--ogb_source", type=str, default="snap", choices=["snap", "official", "graphbolt", "dgl"])
    p.add_argument("--no_normalize_features", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--train_batches", type=int, default=50)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 10])

    p.add_argument("--encoder_type", type=str, default="gcn", choices=["gcn", "sage", "gcnii", "appnp"])
    p.add_argument("--gcn_hidden", type=int, default=128)
    p.add_argument("--gcn_out", type=int, default=64)
    p.add_argument("--gcn_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--prototypes_per_class", type=int, default=2)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--kappa_max", type=float, default=1.0)
    p.add_argument("--init_kappa", type=float, default=0.0)
    p.add_argument("--radius_init", type=float, default=0.5)
    p.add_argument("--prototype_init", type=str, default="random", choices=["random", "simplex"])
    p.add_argument("--geometry_logit_weight", type=float, default=1.0)
    p.add_argument("--euclidean_head_weight", type=float, default=0.0)

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
    p.add_argument("--class_weight", type=str, default="none", choices=["none", "inverse", "sqrt_inv", "effective"])
    p.add_argument("--effective_beta", type=float, default=0.9999)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--lambda_clu", type=float, default=0.0)
    p.add_argument("--lambda_etf", type=float, default=0.1)
    p.add_argument("--lambda_bal", type=float, default=0.1)
    p.add_argument("--lambda_cap", type=float, default=0.0)
    p.add_argument("--lambda_reg", type=float, default=1e-4)
    p.add_argument("--lambda_kappa", type=float, default=1e-4)
    p.add_argument("--cap_margin", type=float, default=0.1)
    p.add_argument("--capacity_mode", type=str, default="global", choices=["global", "pairwise"])
    p.add_argument("--confusion_weight", type=float, default=0.0)
    return p.parse_args()


def make_class_weight(data, num_classes: int, args, device: torch.device) -> torch.Tensor | None:
    mode = args.class_weight.strip().lower()
    if mode == "none":
        return None
    y_train = data.y[canonical_mask(data.train_mask)].view(-1)
    counts = torch.bincount(y_train, minlength=num_classes).float().clamp_min(1.0)
    if mode == "inverse":
        weight = 1.0 / counts
    elif mode == "sqrt_inv":
        weight = 1.0 / torch.sqrt(counts)
    elif mode == "effective":
        beta = float(args.effective_beta)
        if not (0.0 < beta < 1.0):
            raise ValueError("effective_beta must lie in (0, 1).")
        weight = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), counts))
    else:
        raise ValueError(f"Unknown class_weight mode: {args.class_weight}")
    weight = weight / weight.mean().clamp_min(1e-12)
    return weight.to(device)


def make_loader(data, mask: torch.Tensor, args, shuffle: bool) -> NeighborLoader:
    return NeighborLoader(
        data,
        input_nodes=canonical_mask(mask),
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


@torch.no_grad()
def sampled_accuracy(model: GrappleModel, loader: NeighborLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, logit_node_count=batch.batch_size)
        logits = out["logits"]
        y = batch.y[: batch.batch_size]
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct / max(total, 1))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    data, _, meta = load_dataset(
        args.dataset,
        root=args.data_root,
        split="ogb",
        normalize_features=not args.no_normalize_features,
        ogb_source=args.ogb_source,
    )
    num_classes = int(meta["num_classes"])
    num_prototypes = num_classes * int(args.prototypes_per_class)
    class_weight = make_class_weight(data, num_classes, args, device)

    model = GrappleModel(
        ModelConfig(
            in_dim=int(meta["num_features"]),
            num_classes=num_classes,
            encoder_type=args.encoder_type,
            gcn_hidden=args.gcn_hidden,
            gcn_out=args.gcn_out,
            gcn_layers=args.gcn_layers,
            dropout=args.dropout,
            proj_dim=args.proj_dim,
            num_prototypes=num_prototypes,
            tau=args.tau,
            kappa_max=args.kappa_max,
            init_kappa=args.init_kappa,
            radius_init=args.radius_init,
            prototype_init=args.prototype_init,
            geometry_logit_weight=args.geometry_logit_weight,
            euclidean_head_weight=args.euclidean_head_weight,
        )
    ).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of total parameters: {total_params}")
    optimizer = build_optimizer(
        model,
        TrainConfig(
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
        ),
    )

    train_loader = make_loader(data, data.train_mask, args, shuffle=True)
    val_loader = make_loader(data, data.val_mask, args, shuffle=False)
    test_loader = make_loader(data, data.test_mask, args, shuffle=False)
    best_val = -1.0
    best_test = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        update_optimizer_lrs(optimizer, epoch)
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        seen_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.train_batches:
                break
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, logit_node_count=batch.batch_size)
            logits = out["logits"]
            y = batch.y[: batch.batch_size]
            geo = model.geometric_terms(
                out,
                cap_margin=args.cap_margin,
                y=y,
                capacity_mode=args.capacity_mode,
                confusion_weight=args.confusion_weight,
            )
            loss_sup = F.cross_entropy(
                logits,
                y,
                weight=class_weight,
                label_smoothing=float(args.label_smoothing),
            )
            loss = (
                loss_sup
                + args.lambda_clu * geo["l_clu"]
                + args.lambda_etf * geo["l_etf"]
                + args.lambda_bal * geo["l_bal"]
                + args.lambda_cap * geo["l_cap"]
                + args.lambda_reg * geo["l_reg"]
                + args.lambda_kappa * geo["l_kappa"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_acc += masked_accuracy(logits, y, torch.ones_like(y, dtype=torch.bool))
            seen_batches += 1

        should_eval = epoch % max(int(args.eval_interval), 1) == 0 or epoch == args.epochs
        if should_eval:
            val_acc = sampled_accuracy(model, val_loader, device, args.eval_batches)
            test_acc = sampled_accuracy(model, test_loader, device, args.eval_batches)
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_epoch = epoch
            print(
                f"Epoch {epoch:03d} | loss={total_loss / max(seen_batches, 1):.4f} | "
                f"train_acc={total_acc / max(seen_batches, 1):.4f} | "
                f"sampled_val_acc={val_acc:.4f} | sampled_test_acc={test_acc:.4f} | "
                f"best_epoch={best_epoch:03d} | best_val={best_val:.4f} | best_test={best_test:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | loss={total_loss / max(seen_batches, 1):.4f} | "
                f"train_acc={total_acc / max(seen_batches, 1):.4f}"
            )

    print(f"Best sampled validation accuracy: {best_val:.4f}")
    print(f"Sampled test accuracy at best validation: {best_test:.4f}")
    model.eval()
    with torch.no_grad():
        kappa = model.curvature().item()
        rho = model.prototype_radius(model.curvature()).item()
    print(f"Learned curvature kappa: {kappa:.6f}")
    print(f"Learned prototype radius rho: {rho:.6f}")


if __name__ == "__main__":
    main()
