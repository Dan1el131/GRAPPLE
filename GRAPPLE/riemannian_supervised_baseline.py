from __future__ import annotations

import argparse
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv

from grapple.data import load_dataset
from grapple.models.hyperbolic import HGCNEncoder, hyp_to_tangent0
from grapple.models.stereographic import expmap0, geodesic_distance, project_stereographic
from grapple.trainer import canonical_mask, masked_accuracy
from grapple.utils.seed import set_seed


def _curvature_from_raw(raw: torch.Tensor, max_abs: float, mode: str) -> torch.Tensor:
    c = max_abs * torch.tanh(raw)
    if mode == "negative":
        return -F.softplus(c.abs()) / (1.0 + F.softplus(c.abs()))
    if mode == "positive":
        return F.softplus(c.abs()) / (1.0 + F.softplus(c.abs()))
    return c


class KappaGCNClassifier(nn.Module):
    """Lightweight κ-GCN-style classifier in the stereographic model.

    This is a faithful local reproduction of constant-curvature GCN behavior:
    GCN message passing is performed in tangent coordinates, embeddings are
    mapped to a shared constant-curvature space, and logits are negative
    geodesic distances to learnable class prototypes.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + ([hidden_dim] * max(num_layers - 1, 0))
        self.layers = nn.ModuleList([GCNConv(dims[i], dims[i + 1], cached=False, normalize=True) for i in range(len(dims) - 1)])
        self.dropout = dropout
        self.raw_kappa = nn.Parameter(torch.tensor(-0.25))
        self.prototypes_tangent = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes_tangent)

    def curvature(self) -> torch.Tensor:
        return _curvature_from_raw(self.raw_kappa, max_abs=1.0, mode="signed")

    def encode_tangent(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode_tangent(x, edge_index)
        kappa = self.curvature()
        z = expmap0(h, kappa=kappa, eps=1e-6)
        p = expmap0(self.prototypes_tangent, kappa=kappa, eps=1e-6)
        logits = -geodesic_distance(z, p, kappa=kappa, eps=1e-6)
        return logits, kappa


class HGCNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = HGCNEncoder(in_dim, hidden_dim, hidden_dim, num_layers=max(num_layers, 1), c=1.0)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_h = self.encoder(x, edge_index)
        z = hyp_to_tangent0(z_h, c=1.0)
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.classifier(z), torch.as_tensor(-1.0, device=x.device)


class LorentzGCNClassifier(nn.Module):
    """Lorentzian-style GCN with hyperboloid lift and Lorentzian prototypes."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + ([hidden_dim] * max(num_layers - 1, 0))
        self.layers = nn.ModuleList([GCNConv(dims[i], dims[i + 1], cached=False, normalize=True) for i in range(len(dims) - 1)])
        self.dropout = dropout
        self.prototypes = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)

    @staticmethod
    def lift(x: torch.Tensor) -> torch.Tensor:
        space = x
        time = torch.sqrt(1.0 + (space * space).sum(dim=-1, keepdim=True).clamp_min(0.0))
        return torch.cat([time, space], dim=-1)

    @staticmethod
    def lorentz_dist(x_l: torch.Tensor, y_l: torch.Tensor) -> torch.Tensor:
        prod = -x_l[:, None, :1] * y_l[None, :, :1] + (x_l[:, None, 1:] * y_l[None, :, 1:]).sum(dim=-1, keepdim=True)
        arg = (-prod.squeeze(-1)).clamp_min(1.0 + 1e-6)
        return torch.acosh(arg)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        z_l = self.lift(h)
        p_l = self.lift(self.prototypes)
        return -self.lorentz_dist(z_l, p_l), torch.as_tensor(-1.0, device=x.device)


def build_model(args, in_dim: int, num_classes: int) -> nn.Module:
    method = args.method.lower()
    if method == "hgcn":
        return HGCNClassifier(in_dim, args.hidden_dim, num_classes, args.num_layers, args.dropout)
    if method in {"k-gcn", "kgcn", "κ-gcn"}:
        return KappaGCNClassifier(in_dim, args.hidden_dim, num_classes, args.num_layers, args.dropout)
    if method == "lgcn":
        return LorentzGCNClassifier(in_dim, args.hidden_dim, num_classes, args.num_layers, args.dropout)
    raise ValueError("method must be one of: hgcn, k-gcn, lgcn")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, required=True, choices=["hgcn", "k-gcn", "lgcn"])
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--split", type=str, default="random", choices=["random", "ogb"])
    p.add_argument("--ogb_source", type=str, default="graphbolt", choices=["snap", "official", "graphbolt", "dgl"])
    p.add_argument("--normalize_features", action="store_true")
    p.add_argument("--no_normalize_features", action="store_true")
    p.add_argument("--to_undirected", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--train_batches", type=int, default=120)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 10])
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    return p.parse_args()


@torch.no_grad()
def evaluate_full(model, x, edge_index, y, train_mask, val_mask, test_mask):
    model.eval()
    logits, kappa = model(x, edge_index)
    return {
        "train_acc": masked_accuracy(logits, y, train_mask),
        "val_acc": masked_accuracy(logits, y, val_mask),
        "test_acc": masked_accuracy(logits, y, test_mask),
        "kappa": float(kappa.detach().cpu().item()),
    }


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
def sampled_accuracy(model, loader: NeighborLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index)
        logits = logits[: batch.batch_size]
        y = batch.y[: batch.batch_size]
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += int(y.numel())
    return float(correct / max(total, 1))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    normalize = bool(args.normalize_features) and not bool(args.no_normalize_features)
    data, _, meta = load_dataset(
        args.dataset,
        root=args.data_root,
        split=args.split,
        normalize_features=normalize,
        to_undirected=bool(args.to_undirected),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        ogb_source=args.ogb_source,
    )
    model = build_model(args, int(meta["num_features"]), int(meta["num_classes"])).to(device)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Number of total parameters: {sum(p.numel() for p in model.parameters())}")
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.split == "ogb":
        train_loader = make_loader(data, data.train_mask, args, shuffle=True)
        val_loader = make_loader(data, data.val_mask, args, shuffle=False)
        test_loader = make_loader(data, data.test_mask, args, shuffle=False)
        best_val = -1.0
        best_test = -1.0
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            seen = 0
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= args.train_batches:
                    break
                batch = batch.to(device)
                logits, _ = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(logits[: batch.batch_size], batch.y[: batch.batch_size])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                seen += 1
            if epoch % max(args.eval_interval, 1) == 0 or epoch == args.epochs:
                val = sampled_accuracy(model, val_loader, device, args.eval_batches)
                test = sampled_accuracy(model, test_loader, device, args.eval_batches)
                if val > best_val:
                    best_val, best_test, best_epoch = val, test, epoch
                print(f"Epoch {epoch:03d} | loss={total_loss/max(seen,1):.4f} | sampled_val_acc={val:.4f} | sampled_test_acc={test:.4f} | best_epoch={best_epoch:03d} | best_val={best_val:.4f} | best_test={best_test:.4f}")
        print(f"Best sampled validation accuracy: {best_val:.4f}")
        print(f"Sampled test accuracy at best validation: {best_test:.4f}")
        return

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = canonical_mask(data.train_mask).to(device)
    val_mask = canonical_mask(data.val_mask).to(device)
    test_mask = canonical_mask(data.test_mask).to(device)
    best_val = -1.0
    best_test = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits, _ = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % max(args.eval_interval, 1) == 0 or epoch == args.epochs:
            metrics = evaluate_full(model, x, edge_index, y, train_mask, val_mask, test_mask)
            if metrics["val_acc"] > best_val:
                best_val = metrics["val_acc"]
                best_test = metrics["test_acc"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_acc={metrics['val_acc']:.4f} | test_acc={metrics['test_acc']:.4f} | best_epoch={best_epoch:03d} | best_val={best_val:.4f} | best_test={best_test:.4f} | kappa={metrics['kappa']:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Test accuracy at best validation: {best_test:.4f}")


if __name__ == "__main__":
    main()
