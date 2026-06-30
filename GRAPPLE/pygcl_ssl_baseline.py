from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from GCL.augmentors import Compose, EdgeRemoving, FeatureMasking, Identity
from GCL.losses import BarlowTwins, BootstrapLatent, InfoNCE, JSD
from GCL.models import BootstrapContrast, DualBranchContrast, SingleBranchContrast

from grapple.data import load_dataset
from grapple.models.gcn import GCNEncoder, SAGEEncoder
from grapple.trainer import canonical_mask, masked_accuracy
from grapple.utils.seed import set_seed


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float, encoder: str):
        super().__init__()
        if encoder == "sage":
            self.encoder = SAGEEncoder(in_dim, hidden_dim, out_dim, num_layers=layers, dropout=dropout)
        elif encoder == "gcn":
            self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim, num_layers=layers, dropout=dropout)
        else:
            raise ValueError("encoder must be gcn or sage")
        self.projector = nn.Sequential(nn.Linear(out_dim, out_dim), nn.PReLU(), nn.Linear(out_dim, out_dim))
        self.predictor = nn.Sequential(nn.Linear(out_dim, out_dim), nn.PReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, project: bool = False, predict: bool = False) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        if project:
            z = self.projector(z)
        if predict:
            z = self.predictor(z)
        return z


@torch.no_grad()
def update_momentum(online: nn.Module, target: nn.Module, momentum: float) -> None:
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)


@torch.no_grad()
def evaluate_probe(z: torch.Tensor, y: torch.Tensor, train_mask, val_mask, test_mask, epochs: int, lr: float) -> tuple[float, float]:
    z = z.detach()
    clf = nn.Linear(z.size(1), int(y.max().item()) + 1).to(z.device)
    opt = Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    best_val = -1.0
    best_test = -1.0
    best_state = None
    for _ in range(epochs):
        clf.train()
        logits = clf(z)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        clf.eval()
        logits = clf(z)
        val = masked_accuracy(logits, y, val_mask)
        test = masked_accuracy(logits, y, test_mask)
        if val > best_val:
            best_val = val
            best_test = test
            best_state = copy.deepcopy(clf.state_dict())
    if best_state is not None:
        clf.load_state_dict(best_state)
    return float(best_val), float(best_test)


def graph_summary(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z.mean(dim=0, keepdim=True))


def apply_aug(aug, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = aug(x, edge_index)
    if len(out) == 3:
        x_aug, edge_aug, _ = out
    else:
        _, x_aug, edge_aug, _ = out
    return x_aug, edge_aug


def parse_args():
    p = argparse.ArgumentParser(description="PyGCL-backed SSL node classification baselines.")
    p.add_argument("--method", required=True, choices=["dgi", "grace", "bgrl", "cca-ssg"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--split", default="random", choices=["random", "public", "ogb"])
    p.add_argument("--train_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.8)
    p.add_argument("--normalize_features", action="store_true")
    p.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    p.set_defaults(normalize_features=True)
    p.add_argument("--to_undirected", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", default="gcn", choices=["gcn", "sage"])
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--edge_drop", type=float, default=0.2)
    p.add_argument("--feat_mask", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--ema_momentum", type=float, default=0.99)
    p.add_argument("--probe_epochs", type=int, default=300)
    p.add_argument("--probe_lr", type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    data, _, meta = load_dataset(
        args.dataset,
        root=args.data_root,
        split=args.split,
        normalize_features=args.normalize_features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        to_undirected=args.to_undirected,
    )
    data = data.to(device)
    y = data.y
    train_mask = canonical_mask(data.train_mask).to(device)
    val_mask = canonical_mask(data.val_mask).to(device)
    test_mask = canonical_mask(data.test_mask).to(device)

    model = Encoder(int(meta["num_features"]), args.hidden_dim, args.out_dim, args.layers, args.dropout, args.encoder).to(device)
    target = copy.deepcopy(model).to(device) if args.method == "bgrl" else None
    if target is not None:
        target.eval()
        for param in target.parameters():
            param.requires_grad = False

    aug1 = Compose([EdgeRemoving(pe=args.edge_drop), FeatureMasking(pf=args.feat_mask)])
    aug2 = Compose([EdgeRemoving(pe=args.edge_drop), FeatureMasking(pf=args.feat_mask)])
    identity = Identity()
    if args.method == "dgi":
        contrast = SingleBranchContrast(loss=JSD(), mode="G2L").to(device)
    elif args.method == "grace":
        contrast = DualBranchContrast(loss=InfoNCE(tau=args.tau), mode="L2L").to(device)
    elif args.method == "bgrl":
        contrast = BootstrapContrast(loss=BootstrapLatent(), mode="L2L").to(device)
    else:
        contrast = DualBranchContrast(loss=BarlowTwins(lambda_=5e-3), mode="L2L").to(device)
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.method == "dgi":
            x1, e1 = apply_aug(identity, data.x, data.edge_index)
            z = model(x1, e1, project=True)
            zn = model(x1[torch.randperm(x1.size(0), device=device)], e1, project=True)
            loss = contrast(h=z, g=graph_summary(z), hn=zn)
        elif args.method == "bgrl":
            x1, e1 = apply_aug(aug1, data.x, data.edge_index)
            x2, e2 = apply_aug(aug2, data.x, data.edge_index)
            z1_pred = model(x1, e1, project=True, predict=True)
            z2_pred = model(x2, e2, project=True, predict=True)
            assert target is not None
            with torch.no_grad():
                z1_target = target(x1, e1, project=True)
                z2_target = target(x2, e2, project=True)
            loss = contrast(h1_pred=z1_pred, h2_pred=z2_pred, h1_target=z1_target, h2_target=z2_target)
        else:
            x1, e1 = apply_aug(aug1, data.x, data.edge_index)
            x2, e2 = apply_aug(aug2, data.x, data.edge_index)
            z1 = model(x1, e1, project=True)
            z2 = model(x2, e2, project=True)
            loss = contrast(h1=z1, h2=z2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if target is not None:
            update_momentum(model, target, args.ema_momentum)
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | ssl_loss={loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index, project=False)
    best_val, best_test = evaluate_probe(z, y, train_mask, val_mask, test_mask, args.probe_epochs, args.probe_lr)
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Linear probe test accuracy: {best_test:.4f}")
    print(f"PyGCL note: {args.method} uses PyGCL losses/augmentors with the project dataset split.")


if __name__ == "__main__":
    main()
