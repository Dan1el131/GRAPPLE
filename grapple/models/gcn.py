from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import APPNP, GCN2Conv, GCNConv, SAGEConv


class GCNEncoder(nn.Module):
    """L-layer GCN encoder (Euclidean branch)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        dims = [in_dim] + ([hidden_dim] * (num_layers - 1)) + [out_dim]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i+1], cached=False, normalize=True) for i in range(num_layers)])
        self.acts = nn.ModuleList([nn.PReLU() for _ in range(num_layers - 1)])

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                h = self.acts[i](h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return h


class SAGEEncoder(nn.Module):
    """GraphSAGE encoder, better suited to sampled large-graph training."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        dims = [in_dim] + ([hidden_dim] * (num_layers - 1)) + [out_dim]
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.acts = nn.ModuleList([nn.PReLU() for _ in range(num_layers - 1)])

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        del edge_weight
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.acts[i](h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return h


class GCNIIEncoder(nn.Module):
    """GCNII encoder with an input stem and output projection."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 8,
        dropout: float = 0.0,
        alpha: float = 0.1,
        theta: float = 0.5,
        shared_weights: bool = True,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("GCNIIEncoder requires num_layers >= 2.")
        self.dropout = dropout
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_dim)
        self.input_act = nn.PReLU()
        self.hidden_act = nn.PReLU()
        self.convs = nn.ModuleList(
            [
                GCN2Conv(
                    channels=hidden_dim,
                    alpha=alpha,
                    theta=theta,
                    layer=layer_idx + 1,
                    shared_weights=shared_weights,
                    cached=False,
                    normalize=True,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        del edge_weight
        h0 = self.lin_in(x)
        h0 = self.input_act(h0)
        h0 = nn.functional.dropout(h0, p=self.dropout, training=self.training)
        h = h0
        for conv in self.convs:
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, h0, edge_index)
            h = self.hidden_act(h)
        h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return self.lin_out(h)


class APPNPEncoder(nn.Module):
    """MLP encoder followed by personalized PageRank propagation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        appnp_k: int = 10,
        appnp_alpha: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("APPNPEncoder requires num_layers >= 1.")
        self.dropout = dropout
        dims = [in_dim] + ([hidden_dim] * max(num_layers - 1, 0)) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.acts = nn.ModuleList([nn.PReLU() for _ in range(len(self.layers) - 1)])
        self.propagation = APPNP(K=appnp_k, alpha=appnp_alpha, dropout=dropout, cached=False)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.acts[i](h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        return self.propagation(h, edge_index, edge_weight=edge_weight)
