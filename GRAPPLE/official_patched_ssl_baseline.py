from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from grapple.data import load_dataset
from grapple.trainer import canonical_mask, masked_accuracy
from grapple.utils.seed import set_seed


ROOT = Path(__file__).resolve().parent
REPOS = ROOT / "repos_external_ssl"


def add_repo(name: str) -> None:
    path = str(REPOS / name)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_pyg(args):
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
    return data, meta


def linear_probe(z, y, train_mask, val_mask, test_mask, epochs: int, lr: float) -> tuple[float, float]:
    z = z.detach()
    clf = nn.Linear(z.size(-1), int(y.max().item()) + 1).to(z.device)
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
        clf.eval()
        with torch.no_grad():
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


def mask_features(x: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0:
        return x
    drop_mask = torch.empty(x.size(1), device=x.device).uniform_(0, 1) < rate
    out = x.clone()
    out[:, drop_mask] = 0
    return out


def run_grace(args) -> tuple[float, float]:
    add_repo("GRACE")
    from model import Encoder, Model, drop_feature  # type: ignore
    from torch_geometric.nn import GCNConv

    data, meta = load_pyg(args)
    device = torch.device(args.device)
    data = data.to(device)
    activation = nn.PReLU() if args.activation == "prelu" else F.relu
    encoder = Encoder(int(meta["num_features"]), args.hidden_dim, activation, base_model=GCNConv, k=args.layers).to(device)
    model = Model(encoder, args.hidden_dim, args.out_dim, args.tau).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        model.train()
        edge_index_1 = dropout_edge(data.edge_index, p=args.edge_drop_1)[0]
        edge_index_2 = dropout_edge(data.edge_index, p=args.edge_drop_2)[0]
        x_1 = drop_feature(data.x, args.feat_drop_1)
        x_2 = drop_feature(data.x, args.feat_drop_2)
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = model.loss(z1, z2, batch_size=0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | official_grace_loss={loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    return linear_probe(
        z,
        data.y,
        canonical_mask(data.train_mask).to(device),
        canonical_mask(data.val_mask).to(device),
        canonical_mask(data.test_mask).to(device),
        args.probe_epochs,
        args.probe_lr,
    )


def pyg_to_sparse_adj(data) -> sp.spmatrix:
    edge = data.edge_index.cpu().numpy()
    values = np.ones(edge.shape[1], dtype=np.float32)
    adj = sp.coo_matrix((values, (edge[0], edge[1])), shape=(data.num_nodes, data.num_nodes))
    return adj


def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def run_dgi(args) -> tuple[float, float]:
    add_repo("DGI")
    from models import DGI  # type: ignore

    data, meta = load_pyg(args)
    features = data.x.cpu().numpy()
    adj = normalize_adj(pyg_to_sparse_adj(data) + sp.eye(data.num_nodes))
    sp_adj = sparse_mx_to_torch_sparse_tensor(adj)
    device = torch.device(args.device)
    features_t = torch.FloatTensor(features[np.newaxis]).to(device)
    sp_adj = sp_adj.to(device)
    model = DGI(int(meta["num_features"]), args.hidden_dim, "prelu").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss()
    best = float("inf")
    wait = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = np.random.permutation(data.num_nodes)
        shuf = features_t[:, idx, :]
        lbl = torch.cat(
            [torch.ones(1, data.num_nodes, device=device), torch.zeros(1, data.num_nodes, device=device)],
            dim=1,
        )
        logits = model(features_t, shuf, sp_adj, True, None, None, None)
        loss = b_xent(logits, lbl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if float(loss.item()) < best:
            best = float(loss.item())
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | official_dgi_loss={loss.item():.4f} | wait={wait}", flush=True)
        if wait >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeds, _ = model.embed(features_t, sp_adj, True, None)
        z = embeds[0]
    y = data.y.to(device)
    return linear_probe(
        z,
        y,
        canonical_mask(data.train_mask).to(device),
        canonical_mask(data.val_mask).to(device),
        canonical_mask(data.test_mask).to(device),
        args.probe_epochs,
        args.probe_lr,
    )


def run_cca(args) -> tuple[float, float]:
    add_repo("CCA-SSG")
    import dgl  # type: ignore
    from aug import random_aug  # type: ignore
    from model import CCA_SSG  # type: ignore

    data, meta = load_pyg(args)
    src, dst = data.edge_index
    graph = dgl.graph((src.cpu(), dst.cpu()), num_nodes=data.num_nodes)
    feat = data.x
    device = torch.device(args.device)
    if args.device != "cpu":
        graph = graph.to(device)
    feat = feat.to(device)
    model = CCA_SSG(int(meta["num_features"]), args.hidden_dim, args.out_dim, args.layers, args.use_mlp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = graph.number_of_nodes()
    for epoch in range(1, args.epochs + 1):
        model.train()
        g1, x1 = random_aug(graph, feat, args.feat_drop_1, args.edge_drop_1)
        g2, x2 = random_aug(graph, feat, args.feat_drop_2, args.edge_drop_2)
        if args.device != "cpu":
            g1, g2 = g1.to(device), g2.to(device)
        z1, z2 = model(g1.add_self_loop(), x1, g2.add_self_loop(), x2)
        c = (z1.T @ z2) / n
        c1 = (z1.T @ z1) / n
        c2 = (z2.T @ z2) / n
        eye = torch.eye(c.shape[0], device=device)
        loss = -torch.diagonal(c).sum() + args.cca_lambda * ((eye - c1).pow(2).sum() + (eye - c2).pow(2).sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | official_cca_loss={loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        g = graph.remove_self_loop().add_self_loop()
        z = model.get_embedding(g, feat)
    y = data.y.to(device)
    return linear_probe(
        z,
        y,
        canonical_mask(data.train_mask).to(device),
        canonical_mask(data.val_mask).to(device),
        canonical_mask(data.test_mask).to(device),
        args.probe_epochs,
        args.probe_lr,
    )


def run_graphmae(args) -> tuple[float, float]:
    add_repo("GraphMAE")
    import dgl  # type: ignore
    from graphmae.models import build_model  # type: ignore
    from graphmae.evaluation import node_classification_evaluation  # type: ignore
    from graphmae.utils import create_optimizer  # type: ignore

    data, meta = load_pyg(args)
    src, dst = data.edge_index
    graph = dgl.graph((src.cpu(), dst.cpu()), num_nodes=data.num_nodes)
    graph = graph.remove_self_loop().add_self_loop()
    graph.ndata["feat"] = data.x.cpu()
    graph.ndata["label"] = data.y.cpu()
    graph.ndata["train_mask"] = canonical_mask(data.train_mask).cpu()
    graph.ndata["val_mask"] = canonical_mask(data.val_mask).cpu()
    graph.ndata["test_mask"] = canonical_mask(data.test_mask).cpu()

    # GraphMAE official code is DGL-based. The current server has CPU DGL, so
    # keep this method on CPU unless a CUDA-enabled DGL build is installed.
    device = "cpu" if args.device != "cuda_dgl" else "cuda"
    class Obj:
        pass

    gargs = Obj()
    gargs.num_features = int(meta["num_features"])
    gargs.num_heads = 4
    gargs.num_out_heads = 1
    gargs.num_hidden = args.hidden_dim
    gargs.num_layers = args.layers
    gargs.residual = False
    gargs.attn_drop = 0.1
    gargs.in_drop = 0.2
    gargs.norm = None
    gargs.negative_slope = 0.2
    gargs.encoder = args.graphmae_encoder
    gargs.decoder = args.graphmae_decoder
    gargs.mask_rate = args.mask_rate
    gargs.drop_edge_rate = args.edge_drop_1
    gargs.replace_rate = args.replace_rate
    gargs.activation = "prelu"
    gargs.loss_fn = "sce"
    gargs.alpha_l = 2
    gargs.concat_hidden = False

    model = build_model(gargs).to(device)
    opt = create_optimizer("adam", model, args.lr, args.weight_decay)
    graph = graph.to(device)
    feat = graph.ndata["feat"].to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss, loss_dict = model(graph, feat)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | official_graphmae_loss={loss.item():.4f}", flush=True)

    final_acc, estp_acc = node_classification_evaluation(
        model,
        graph,
        feat,
        int(meta["num_classes"]),
        args.probe_lr,
        1e-4,
        args.probe_epochs,
        device,
        True,
        mute=True,
    )
    return float(estp_acc), float(estp_acc)


def run_bgrl(args) -> tuple[float, float]:
    add_repo("BGRL")
    from bgrl import BGRL, GCN, MLP_Predictor, compute_representations, get_graph_drop_transform  # type: ignore

    data, meta = load_pyg(args)
    device = torch.device(args.device)
    data = data.to(device)
    dataset = [data]
    encoder = GCN([int(meta["num_features"]), args.hidden_dim, args.out_dim], batchnorm=True)
    predictor = MLP_Predictor(args.out_dim, args.out_dim, hidden_size=args.hidden_dim)
    model = BGRL(encoder, predictor).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t1 = get_graph_drop_transform(drop_edge_p=args.edge_drop_1, drop_feat_p=args.feat_drop_1)
    t2 = get_graph_drop_transform(drop_edge_p=args.edge_drop_2, drop_feat_p=args.feat_drop_2)
    for epoch in range(1, args.epochs + 1):
        model.train()
        x1, x2 = t1(data), t2(data)
        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)
        loss = 2 - F.cosine_similarity(q1, y2.detach(), dim=-1).mean() - F.cosine_similarity(q2, y1.detach(), dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.update_target_network(args.ema_momentum)
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d} | official_bgrl_loss={loss.item():.4f}", flush=True)
    tmp = copy.deepcopy(model.online_encoder).eval()
    z, labels = compute_representations(tmp, dataset, device)
    y = labels.to(device)
    return linear_probe(
        z.to(device),
        y,
        canonical_mask(data.train_mask).to(device),
        canonical_mask(data.val_mask).to(device),
        canonical_mask(data.test_mask).to(device),
        args.probe_epochs,
        args.probe_lr,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["dgi", "grace", "bgrl", "cca-ssg", "graphmae"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--split", default="random", choices=["random"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.8)
    p.add_argument("--to_undirected", action="store_true")
    p.add_argument("--no_normalize_features", action="store_true")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--activation", default="prelu")
    p.add_argument("--use_mlp", action="store_true")
    p.add_argument("--graphmae_encoder", default="gat")
    p.add_argument("--graphmae_decoder", default="gat")
    p.add_argument("--mask_rate", type=float, default=0.5)
    p.add_argument("--replace_rate", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--edge_drop_1", type=float, default=0.2)
    p.add_argument("--edge_drop_2", type=float, default=0.3)
    p.add_argument("--feat_drop_1", type=float, default=0.2)
    p.add_argument("--feat_drop_2", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--cca_lambda", type=float, default=1e-3)
    p.add_argument("--ema_momentum", type=float, default=0.99)
    p.add_argument("--probe_epochs", type=int, default=300)
    p.add_argument("--probe_lr", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.method == "dgi":
        best_val, test = run_dgi(args)
    elif args.method == "grace":
        best_val, test = run_grace(args)
    elif args.method == "bgrl":
        best_val, test = run_bgrl(args)
    elif args.method == "cca-ssg":
        best_val, test = run_cca(args)
    elif args.method == "graphmae":
        best_val, test = run_graphmae(args)
    else:
        raise ValueError(args.method)
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Linear probe test accuracy: {test:.4f}")
    print("Official patched note: official model/training code with GRAPPLE dataset-loader bridge.")


if __name__ == "__main__":
    main()
