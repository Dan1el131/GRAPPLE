from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from grapple.data import load_dataset
from grapple.trainer import canonical_mask, masked_accuracy
from grapple.utils.seed import set_seed


class FullGraphClassifier(nn.Module):
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
        encoder_type = encoder_type.strip().lower()
        self.encoder_type = encoder_type
        dims = [in_dim] + ([hidden_dim] * max(num_layers - 1, 0)) + [num_classes]
        if encoder_type == "mlp":
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.layers = nn.ModuleList(layers)
        elif encoder_type == "sage":
            conv_cls = SAGEConv
            self.layers = nn.ModuleList([conv_cls(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        elif encoder_type == "gcn":
            conv_cls = GCNConv
            self.layers = nn.ModuleList(
                [conv_cls(dims[i], dims[i + 1], cached=False, normalize=True) for i in range(len(dims) - 1)]
            )
        elif encoder_type == "gat":
            if gat_heads < 1:
                raise ValueError("gat_heads must be at least 1.")
            self.layers = nn.ModuleList()
            if len(dims) == 2:
                self.layers.append(GATConv(dims[0], dims[1], heads=1, concat=False, dropout=dropout))
            else:
                self.layers.append(GATConv(dims[0], hidden_dim, heads=gat_heads, concat=True, dropout=dropout))
                for _ in range(max(num_layers - 2, 0)):
                    self.layers.append(
                        GATConv(hidden_dim * gat_heads, hidden_dim, heads=gat_heads, concat=True, dropout=dropout)
                    )
                self.layers.append(GATConv(hidden_dim * gat_heads, num_classes, heads=1, concat=False, dropout=dropout))
        else:
            raise ValueError("encoder_type must be one of: mlp, sage, gcn, gat.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h) if self.encoder_type == "mlp" else layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ogbn-arxiv")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--split", type=str, default="ogb", choices=["public", "random", "ogb"])
    p.add_argument("--normalize_features", action="store_true")
    p.add_argument("--to_undirected", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=10)
    p.add_argument("--encoder_type", type=str, default="sage", choices=["mlp", "sage", "gcn", "gat"])
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, x, edge_index, y, train_mask, val_mask, test_mask):
    model.eval()
    logits = model(x, edge_index)
    return {
        "train_acc": masked_accuracy(logits, y, train_mask),
        "val_acc": masked_accuracy(logits, y, val_mask),
        "test_acc": masked_accuracy(logits, y, test_mask),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    data, _, meta = load_dataset(
        args.dataset,
        root=args.data_root,
        split=args.split,
        normalize_features=bool(args.normalize_features),
        to_undirected=bool(args.to_undirected),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = canonical_mask(data.train_mask).to(device)
    val_mask = canonical_mask(data.val_mask).to(device)
    test_mask = canonical_mask(data.test_mask).to(device)

    model = FullGraphClassifier(
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

    best_val = -1.0
    best_test = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % max(int(args.eval_interval), 1) == 0 or epoch == args.epochs:
            metrics = evaluate(model, x, edge_index, y, train_mask, val_mask, test_mask)
            if metrics["val_acc"] > best_val:
                best_val = metrics["val_acc"]
                best_test = metrics["test_acc"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"train_acc={metrics['train_acc']:.4f} | val_acc={metrics['val_acc']:.4f} | "
                f"test_acc={metrics['test_acc']:.4f} | best_epoch={best_epoch:03d} | "
                f"best_val={best_val:.4f} | best_test={best_test:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Test accuracy at best validation: {best_test:.4f}")


if __name__ == "__main__":
    main()
