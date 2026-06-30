from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from grapple.data import load_dataset
from grapple.trainer import canonical_mask, masked_accuracy
from grapple.utils.seed import set_seed


class SampledClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        dropout: float,
        encoder_type: str,
        gat_heads: int,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.dropout = dropout
        self.encoder_type = encoder_type.strip().lower()
        dims = [in_dim] + ([hidden_dim] * max(num_layers - 1, 0)) + [num_classes]
        if self.encoder_type == "mlp":
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        elif self.encoder_type == "sage":
            self.layers = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        elif self.encoder_type == "gcn":
            self.layers = nn.ModuleList(
                [GCNConv(dims[i], dims[i + 1], cached=False, normalize=True) for i in range(len(dims) - 1)]
            )
        elif self.encoder_type == "gat":
            layers = []
            in_channels = in_dim
            for layer_idx in range(num_layers):
                is_last = layer_idx == num_layers - 1
                out_channels = num_classes if is_last else hidden_dim
                heads = 1 if is_last else int(gat_heads)
                concat = not is_last
                layers.append(GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout))
                in_channels = out_channels * heads if concat else out_channels
            self.layers = nn.ModuleList(layers)
        else:
            raise ValueError("encoder_type must be one of: mlp, gcn, sage, gat.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            if self.encoder_type == "mlp":
                h = layer(h)
            else:
                h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ogbn-products")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--ogb_source", type=str, default="snap", choices=["snap", "official", "graphbolt", "dgl"])
    p.add_argument("--no_normalize_features", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--train_batches", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 10])
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--encoder_type", type=str, default="sage", choices=["mlp", "gcn", "sage", "gat"])
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    return p.parse_args()


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
def sampled_accuracy(model: nn.Module, loader: NeighborLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)[: batch.batch_size]
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
    model = SampledClassifier(
        in_dim=int(meta["num_features"]),
        hidden_dim=args.hidden_dim,
        num_classes=int(meta["num_classes"]),
        num_layers=args.num_layers,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
        gat_heads=args.gat_heads,
    ).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of total parameters: {total_params}")
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(data, data.train_mask, args, shuffle=True)
    val_loader = make_loader(data, data.val_mask, args, shuffle=False)
    test_loader = make_loader(data, data.test_mask, args, shuffle=False)
    best_val = -1.0
    best_test = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        seen_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.train_batches:
                break
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)[: batch.batch_size]
            y = batch.y[: batch.batch_size]
            loss = F.cross_entropy(logits, y)
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


if __name__ == "__main__":
    main()
