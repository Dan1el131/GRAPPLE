from __future__ import annotations

import argparse
import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling, dropout_edge

from grapple.data import load_dataset
from grapple.models.gcn import GCNEncoder, SAGEEncoder
from grapple.trainer import canonical_mask, masked_accuracy
from grapple.utils.seed import set_seed


METHODS = {
    "dgi": "dgi",
    "grace": "contrastive",
    "bgrl": "bgrl",
    "cca-ssg": "cca",
    "graphmae": "feature",
    "graphmae2": "feature_cosine",
    "aug-mae": "aug_feature",
    "maskgae": "edge",
    "s2gae": "edge",
    "nedm": "dual",
    "dgmae": "dual_cosine",
    "graphacl": "contrastive",
    "bandana": "contrastive_balanced",
}


class SSLAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float, encoder: str):
        super().__init__()
        if encoder == "sage":
            self.encoder = SAGEEncoder(in_dim, hidden_dim, out_dim, num_layers=layers, dropout=dropout)
        elif encoder == "gcn":
            self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim, num_layers=layers, dropout=dropout)
        else:
            raise ValueError("encoder must be gcn or sage")
        self.feature_decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.discriminator = nn.Bilinear(out_dim, out_dim, 1)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def reconstruct_x(self, z: torch.Tensor) -> torch.Tensor:
        return self.feature_decoder(z)


def mask_features(x: torch.Tensor, rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    if rate <= 0:
        return x, torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    mask = torch.rand(x.size(0), device=x.device) < rate
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def feature_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, cosine: bool) -> torch.Tensor:
    if mask.sum() == 0:
        mask = torch.ones_like(mask)
    if cosine:
        pred_n = F.normalize(pred[mask], dim=-1)
        target_n = F.normalize(target[mask], dim=-1)
        return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return F.mse_loss(pred[mask], target[mask])


def edge_loss(z: torch.Tensor, edge_index: torch.Tensor, num_neg: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return z.new_tensor(0.0)
    pos = edge_index
    if pos.size(1) > num_neg:
        idx = torch.randperm(pos.size(1), device=pos.device)[:num_neg]
        pos = pos[:, idx]
    neg = negative_sampling(
        edge_index=edge_index,
        num_nodes=z.size(0),
        num_neg_samples=pos.size(1),
        method="sparse",
    )
    pos_score = (z[pos[0]] * z[pos[1]]).sum(dim=-1)
    neg_score = (z[neg[0]] * z[neg[1]]).sum(dim=-1)
    logits = torch.cat([pos_score, neg_score], dim=0)
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float, max_nodes: int) -> torch.Tensor:
    n = min(z1.size(0), max_nodes)
    if z1.size(0) > n:
        idx = torch.randperm(z1.size(0), device=z1.device)[:n]
        z1, z2 = z1[idx], z2[idx]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def dgi_loss(model: SSLAdapter, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    z_pos = model.embed(x, edge_index)
    perm = torch.randperm(x.size(0), device=x.device)
    z_neg = model.embed(x[perm], edge_index)
    summary = torch.sigmoid(z_pos.mean(dim=0, keepdim=True)).expand_as(z_pos)
    pos_logits = model.discriminator(z_pos, summary).squeeze(-1)
    neg_logits = model.discriminator(z_neg, summary).squeeze(-1)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def cca_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 1e-3) -> torch.Tensor:
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
    n = z1.size(0)
    c = (z1.T @ z2) / n
    invariance = -torch.diagonal(c).sum()
    off_diag = c - torch.diag(torch.diagonal(c))
    decorrelation = off_diag.pow(2).sum()
    return invariance + lambd * decorrelation


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=-1)
    z = F.normalize(z.detach(), dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


@torch.no_grad()
def update_target_encoder(online: SSLAdapter, target: SSLAdapter, momentum: float) -> None:
    for param_q, param_k in zip(online.encoder.parameters(), target.encoder.parameters()):
        param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)


def bgrl_loss(online: SSLAdapter, target: SSLAdapter, x: torch.Tensor, edge_index: torch.Tensor, args) -> torch.Tensor:
    e1, _ = dropout_edge(edge_index, p=args.edge_drop, training=True)
    e2, _ = dropout_edge(edge_index, p=args.edge_drop, training=True)
    x1, _ = mask_features(x, args.mask_rate)
    x2, _ = mask_features(x, args.mask_rate)
    z1 = online.embed(x1, e1)
    z2 = online.embed(x2, e2)
    with torch.no_grad():
        t1 = target.embed(x1, e1)
        t2 = target.embed(x2, e2)
    p1 = online.predictor(z1)
    p2 = online.predictor(z2)
    return 0.5 * (byol_loss(p1, t2) + byol_loss(p2, t1))


def ssl_loss(model: SSLAdapter, method: str, x: torch.Tensor, edge_index: torch.Tensor, args) -> torch.Tensor:
    objective = METHODS[method]
    cosine = objective in {"feature_cosine", "dual_cosine"}
    if objective == "dgi":
        return dgi_loss(model, x, edge_index)
    if objective == "cca":
        e1, _ = dropout_edge(edge_index, p=args.edge_drop, training=True)
        e2, _ = dropout_edge(edge_index, p=args.edge_drop, training=True)
        x1, _ = mask_features(x, args.mask_rate)
        x2, _ = mask_features(x, args.mask_rate)
        z1 = model.projector(model.embed(x1, e1))
        z2 = model.projector(model.embed(x2, e2))
        return cca_loss(z1, z2, args.cca_lambda)
    if objective in {"feature", "feature_cosine", "aug_feature"}:
        use_edges = edge_index
        if objective == "aug_feature":
            use_edges, _ = dropout_edge(edge_index, p=args.edge_drop, training=True)
        x_masked, mask = mask_features(x, args.mask_rate)
        z = model.embed(x_masked, use_edges)
        return feature_loss(model.reconstruct_x(z), x, mask, cosine=cosine)
    if objective in {"edge"}:
        z = model.embed(x, edge_index)
        return edge_loss(z, edge_index, args.edge_samples)
    if objective in {"dual", "dual_cosine"}:
        x_masked, mask = mask_features(x, args.mask_rate)
        z = model.embed(x_masked, edge_index)
        l_x = feature_loss(model.reconstruct_x(z), x, mask, cosine=cosine)
        l_e = edge_loss(z, edge_index, args.edge_samples)
        return l_x + args.edge_loss_weight * l_e
    if objective in {"contrastive", "contrastive_balanced"}:
        e1, _ = dropout_edge(edge_index, p=args.edge_drop, training=True)
        e2, _ = dropout_edge(edge_index, p=args.edge_drop, training=True)
        x1, _ = mask_features(x, args.mask_rate)
        x2, _ = mask_features(x, args.mask_rate)
        z1 = model.projector(model.embed(x1, e1))
        z2 = model.projector(model.embed(x2, e2))
        loss = contrastive_loss(z1, z2, args.tau, args.contrast_nodes)
        if objective == "contrastive_balanced":
            loss = loss + 0.01 * (z1.std(dim=0).mean() - 1.0).abs()
        return loss
    raise ValueError(f"Unsupported method: {method}")


@torch.no_grad()
def evaluate_probe(clf: nn.Module, z: torch.Tensor, y: torch.Tensor, train_mask, val_mask, test_mask) -> dict[str, float]:
    clf.eval()
    logits = clf(z)
    return {
        "train_acc": masked_accuracy(logits, y, train_mask),
        "val_acc": masked_accuracy(logits, y, val_mask),
        "test_acc": masked_accuracy(logits, y, test_mask),
    }


def linear_probe(z: torch.Tensor, y: torch.Tensor, train_mask, val_mask, test_mask, epochs: int, lr: float) -> tuple[float, float]:
    z = z.detach()
    clf = nn.Linear(z.size(1), int(y.max().item()) + 1).to(z.device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
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
        metrics = evaluate_probe(clf, z, y, train_mask, val_mask, test_mask)
        if metrics["val_acc"] > best_val:
            best_val = metrics["val_acc"]
            best_test = metrics["test_acc"]
            best_state = copy.deepcopy(clf.state_dict())
    if best_state is not None:
        clf.load_state_dict(best_state)
    return float(best_val), float(best_test)


def full_graph_run(args) -> tuple[float, float]:
    data, _, meta = load_dataset(
        args.dataset,
        root=args.data_root,
        split=args.split,
        normalize_features=not args.no_normalize_features,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        to_undirected=args.to_undirected,
    )
    device = torch.device(args.device)
    data = data.to(device)
    model = SSLAdapter(
        int(meta["num_features"]),
        args.hidden_dim,
        args.out_dim,
        args.layers,
        args.dropout,
        args.encoder,
    ).to(device)
    target_model = None
    if METHODS[args.method] == "bgrl":
        target_model = copy.deepcopy(model).to(device)
        target_model.eval()
        for param in target_model.parameters():
            param.requires_grad = False
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if METHODS[args.method] == "bgrl":
            assert target_model is not None
            loss = bgrl_loss(model, target_model, data.x, data.edge_index, args)
        else:
            loss = ssl_loss(model, args.method, data.x, data.edge_index, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if target_model is not None:
            update_target_encoder(model, target_model, args.ema_momentum)
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | ssl_loss={loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        z = model.embed(data.x, data.edge_index)
    train_mask = canonical_mask(data.train_mask).to(device)
    val_mask = canonical_mask(data.val_mask).to(device)
    test_mask = canonical_mask(data.test_mask).to(device)
    y = data.y.to(device)
    return linear_probe(z, y, train_mask, val_mask, test_mask, args.probe_epochs, args.probe_lr)


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
def collect_embeddings(model: SSLAdapter, loader: NeighborLoader, device: torch.device, max_batches: int):
    model.eval()
    chunks, labels = [], []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = batch.to(device)
        z = model.embed(batch.x, batch.edge_index)[: batch.batch_size]
        chunks.append(z.cpu())
        labels.append(batch.y[: batch.batch_size].cpu())
    return torch.cat(chunks, dim=0), torch.cat(labels, dim=0)


def sampled_run(args) -> tuple[float, float]:
    data, _, meta = load_dataset(
        args.dataset,
        root=args.data_root,
        split="ogb",
        normalize_features=not args.no_normalize_features,
        ogb_source=args.ogb_source,
    )
    device = torch.device(args.device)
    model = SSLAdapter(
        int(meta["num_features"]),
        args.hidden_dim,
        args.out_dim,
        args.layers,
        args.dropout,
        "sage",
    ).to(device)
    target_model = None
    if METHODS[args.method] == "bgrl":
        target_model = copy.deepcopy(model).to(device)
        target_model.eval()
        for param in target_model.parameters():
            param.requires_grad = False
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(data, data.train_mask, args, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.train_batches:
                break
            batch = batch.to(device)
            if METHODS[args.method] == "bgrl":
                assert target_model is not None
                loss = bgrl_loss(model, target_model, batch.x, batch.edge_index, args)
            else:
                loss = ssl_loss(model, args.method, batch.x, batch.edge_index, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if target_model is not None:
                update_target_encoder(model, target_model, args.ema_momentum)
            total_loss += float(loss.item())
            seen += 1
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | ssl_loss={total_loss / max(seen, 1):.4f}", flush=True)
    train_loader = make_loader(data, data.train_mask, args, shuffle=False)
    val_loader = make_loader(data, data.val_mask, args, shuffle=False)
    test_loader = make_loader(data, data.test_mask, args, shuffle=False)
    z_train, y_train = collect_embeddings(model, train_loader, device, args.probe_train_batches)
    z_val, y_val = collect_embeddings(model, val_loader, device, args.eval_batches)
    z_test, y_test = collect_embeddings(model, test_loader, device, args.eval_batches)
    z = torch.cat([z_train, z_val, z_test], dim=0).to(device)
    y = torch.cat([y_train, y_val, y_test], dim=0).to(device)
    n_train, n_val = z_train.size(0), z_val.size(0)
    train_mask = torch.zeros(z.size(0), dtype=torch.bool, device=device)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[:n_train] = True
    val_mask[n_train : n_train + n_val] = True
    test_mask[n_train + n_val :] = True
    return linear_probe(z, y, train_mask, val_mask, test_mask, args.probe_epochs, args.probe_lr)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, required=True, choices=sorted(METHODS))
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--split", type=str, default="random", choices=["public", "random", "ogb"])
    p.add_argument("--train_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.8)
    p.add_argument("--to_undirected", action="store_true")
    p.add_argument("--no_normalize_features", action="store_true")
    p.add_argument("--ogb_source", type=str, default="graphbolt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", type=str, default="gcn", choices=["gcn", "sage"])
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--out_dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--mask_rate", type=float, default=0.3)
    p.add_argument("--edge_drop", type=float, default=0.2)
    p.add_argument("--edge_samples", type=int, default=20000)
    p.add_argument("--edge_loss_weight", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--cca_lambda", type=float, default=1e-3)
    p.add_argument("--ema_momentum", type=float, default=0.99)
    p.add_argument("--contrast_nodes", type=int, default=4096)
    p.add_argument("--probe_epochs", type=int, default=300)
    p.add_argument("--probe_lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 10])
    p.add_argument("--train_batches", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--probe_train_batches", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.dataset == "ogbn-products":
        best_val, test = sampled_run(args)
    else:
        best_val, test = full_graph_run(args)
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Linear probe test accuracy: {test:.4f}")
    print(f"Adapter note: {args.method} is run through the unified PyG SSL adapter protocol.")


if __name__ == "__main__":
    main()
