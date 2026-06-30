from __future__ import annotations

from typing import Any, Dict, Tuple
from contextlib import contextmanager
from pathlib import Path
import builtins
import zipfile

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import coalesce, to_undirected


def _canonical_name(name: str) -> str:
    """Normalize dataset name (case-insensitive, hyphen/underscore tolerant)."""
    n = name.strip().lower()
    n = n.replace("_", "-")
    while "--" in n:
        n = n.replace("--", "-")
    return n


def _ensure_y_node_level(data: Data) -> Data:
    """Ensure data.y is a 1D LongTensor of shape [num_nodes]."""
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Loaded dataset has no labels (data.y is missing).")
    y = data.y
    if isinstance(y, (list, tuple)):
        y = torch.tensor(y)
    if y.dim() > 1:
        y = y.view(-1)
    if y.dtype != torch.long:
        y = y.to(torch.long)
    data.y = y
    return data


def _random_split_masks(
    num_nodes: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio+val_ratio+test_ratio must sum to 1.0, got "
            f"{train_ratio}+{val_ratio}+{test_ratio}"
        )

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=g)

    n_train = int(round(train_ratio * num_nodes))
    n_val = int(round(val_ratio * num_nodes))
    n_train = min(n_train, num_nodes)
    n_val = min(n_val, num_nodes - n_train)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _idx_to_mask(num_nodes: int, idx: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask


def _ensure_1d_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() > 1:
        mask = mask[:, 0]
    return mask.to(torch.bool).view(-1)


@contextmanager
def _torch_load_weights_only_compat():
    """Temporarily restore torch.load(weights_only=False) for trusted OGB assets.

    OGB's processed PyG files store Data objects, which are blocked by the
    newer torch.load(weights_only=True) default in recent PyTorch versions.
    """
    original_torch_load = torch.load

    def compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _postprocess_graph(
    data: Data,
    *,
    normalize_features: bool,
    to_undirected_flag: bool,
) -> Data:
    if normalize_features:
        data = NormalizeFeatures()(data)

    if data.edge_index is None:
        raise ValueError("Loaded dataset has no edge_index.")

    if not bool(getattr(data, "_skip_edge_postprocess", False)):
        data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)

        if to_undirected_flag:
            data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
            data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)
    elif to_undirected_flag:
        raise ValueError("to_undirected is not supported when edge postprocessing is skipped.")

    return data


class _GraphBoltProductsDataset:
    def __init__(self, split_idx: Dict[str, torch.Tensor]):
        self.num_classes = 47
        self._split_idx = split_idx

    def get_idx_split(self) -> Dict[str, torch.Tensor]:
        return self._split_idx


def _read_edge_csv(edge_path: Path, chunksize: int = 5_000_000) -> torch.Tensor:
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("pandas is required to load GraphBolt ogbn-products edges.") from e

    chunks = []
    for chunk in pd.read_csv(edge_path, header=None, dtype=np.int64, chunksize=chunksize):
        if chunk.shape[1] != 2:
            raise ValueError(f"Expected two columns in edge CSV, got shape={chunk.shape}.")
        chunks.append(torch.from_numpy(chunk.to_numpy(copy=True)).t().contiguous())
    if not chunks:
        raise ValueError(f"No edges found in {edge_path}.")
    return torch.cat(chunks, dim=1)


def _load_graphbolt_ogbn_products(root: str) -> Tuple[Data, _GraphBoltProductsDataset]:
    root_path = Path(root)
    graphbolt_dir = root_path / "ogbn-products-seeds"
    zip_path = root_path / "graphbolt-ogbn-products-seeds.zip"
    processed_path = root_path / "ogbn-products-graphbolt-pyg.pt"

    if processed_path.exists():
        with _torch_load_weights_only_compat():
            payload = torch.load(processed_path, map_location="cpu")
        return payload["data"], _GraphBoltProductsDataset(payload["split_idx"])

    if not graphbolt_dir.exists():
        if not zip_path.exists():
            raise FileNotFoundError(
                "GraphBolt ogbn-products mirror archive not found. Expected either "
                f"{graphbolt_dir} or {zip_path}."
            )
        root_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root_path)

    x = torch.from_numpy(np.load(graphbolt_dir / "data" / "node-feat.npy")).float()
    y = torch.from_numpy(np.load(graphbolt_dir / "data" / "node-label.npy")).view(-1).long()
    edge_index = _read_edge_csv(graphbolt_dir / "edges" / "bi_edge.csv")

    train_idx = torch.from_numpy(np.load(graphbolt_dir / "set" / "train_node.npy")).view(-1).long()
    valid_idx = torch.from_numpy(np.load(graphbolt_dir / "set" / "valid_node.npy")).view(-1).long()
    test_idx = torch.from_numpy(np.load(graphbolt_dir / "set" / "test_node.npy")).view(-1).long()

    # Sanity-check split labels without using validation/test labels for training.
    for split_name, idx_name, label_name in [
        ("train", train_idx, "train_label.npy"),
        ("valid", valid_idx, "valid_label.npy"),
        ("test", test_idx, "test_label.npy"),
    ]:
        split_labels = torch.from_numpy(np.load(graphbolt_dir / "set" / label_name)).view(-1).long()
        if not torch.equal(y[idx_name], split_labels):
            raise ValueError(f"GraphBolt {split_name} labels do not match node-label.npy.")

    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=x.size(0))
    data._skip_edge_postprocess = True
    split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    torch.save({"data": data, "split_idx": split_idx}, processed_path)
    return data, _GraphBoltProductsDataset(split_idx)


@contextmanager
def _auto_confirm_ogb_downloads():
    """Make OGB dataset downloads non-interactive in batch experiments."""
    original_input = builtins.input

    def yes_input(prompt: str = "") -> str:
        if prompt:
            print(prompt, end="")
        return "y"

    builtins.input = yes_input
    try:
        yield
    finally:
        builtins.input = original_input


def load_dataset(
    name: str,
    root: str,
    normalize_features: bool = True,
    split: str = "public",
    **kwargs: Any,
) -> Tuple[Data, Any, Dict[str, Any]]:
    """Unified dataset loader.

    Args:
        name: Dataset name (case-insensitive). Supported (aliases):
            - Planetoid: cora, citeseer, pubmed
            - Amazon: amazon-computers, amazon-photos
            - Coauthor: coauthor-cs, coauthor-physics
            - WikiCS: wikics
            - WebKB: webkb-cornell, webkb-texas, webkb-wisconsin
            - WikipediaNetwork: chameleon, squirrel
            - Actor: actor
            - OGB: ogbn-arxiv, ogbn-products
        root: Root directory for datasets.
        normalize_features: Whether to apply NormalizeFeatures(). Default True.
        split: One of {"public", "random", "ogb"}.
        **kwargs:
            - train_ratio, val_ratio, test_ratio, seed (for split="random")
            - to_undirected (bool)

    Returns:
        data: Data with required fields:
            x, edge_index, y, train_mask, val_mask, test_mask, num_nodes
        dataset: underlying dataset object
        meta: dict with at least:
            num_features, num_classes, is_undirected, has_edge_attr, name, split, task_type
    """
    n = _canonical_name(name)
    split = split.strip().lower()
    if split not in {"public", "random", "ogb"}:
        raise ValueError(f"Unsupported split='{split}'. Choose from public/random/ogb.")

    to_undirected_flag = bool(kwargs.pop("to_undirected", False))


    planetoid_map = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}

    amazon_map = {
        "amazon-computers": "Computers",
        "amazon-photos": "Photo",
        "amazon-photo": "Photo",
        "amazon-photographs": "Photo",
    }

    coauthor_map = {
        "coauthor-cs": "CS",
        "coauthor-physics": "Physics",
        "coauthor-phys": "Physics",
    }

    wikics_set = {"wikics", "wiki-cs"}

    webkb_map = {
        "webkb-cornell": "Cornell",
        "webkb-texas": "Texas",
        "webkb-wisconsin": "Wisconsin",
        "cornell": "Cornell",
        "texas": "Texas",
        "wisconsin": "Wisconsin",
    }

    wikipedia_map = {
        "chameleon": "chameleon",
        "wiki-chameleon": "chameleon",
        "squirrel": "squirrel",
        "wiki-squirrel": "squirrel",
    }

    actor_set = {"actor", "film"}

    ogb_set = {"ogbn-arxiv", "ogbn-products"}
    ogb_source = str(kwargs.pop("ogb_source", "snap")).strip().lower()


    if n in planetoid_map:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root=root, name=planetoid_map[n])
        data = dataset[0]
    elif n in amazon_map:
        from torch_geometric.datasets import Amazon

        dataset = Amazon(root=root, name=amazon_map[n])
        data = dataset[0]
    elif n in coauthor_map:
        from torch_geometric.datasets import Coauthor

        dataset = Coauthor(root=root, name=coauthor_map[n])
        data = dataset[0]
    elif n in wikics_set:
        from torch_geometric.datasets import WikiCS

        dataset = WikiCS(root=root)
        data = dataset[0]
    elif n in webkb_map:
        from torch_geometric.datasets import WebKB

        dataset = WebKB(root=root, name=webkb_map[n])
        data = dataset[0]
    elif n in wikipedia_map:
        from torch_geometric.datasets import WikipediaNetwork

        dataset = WikipediaNetwork(root=root, name=wikipedia_map[n])
        data = dataset[0]
    elif n in actor_set:
        from torch_geometric.datasets import Actor

        dataset = Actor(root=root)
        data = dataset[0]
    elif n in ogb_set:
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except Exception as e:
            raise ImportError(
                "OGB is required for ogbn-* datasets. Install with: pip install ogb"
            ) from e

        if n == "ogbn-products" and ogb_source in {"graphbolt", "dgl"}:
            data, dataset = _load_graphbolt_ogbn_products(root)
        else:
            if ogb_source not in {"snap", "official"}:
                raise ValueError(
                    "Unsupported ogb_source. Use 'snap'/'official' or "
                    "'graphbolt'/'dgl' for ogbn-products."
                )
            with _auto_confirm_ogb_downloads(), _torch_load_weights_only_compat():
                dataset = PygNodePropPredDataset(name=n, root=root)
            data = dataset[0]
            if hasattr(data, "y") and data.y is not None:
                data.y = data.y.view(-1)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: "
            "cora/citeseer/pubmed, amazon-computers/amazon-photos, "
            "coauthor-cs/coauthor-physics, wikics, "
            "webkb-cornell/webkb-texas/webkb-wisconsin, "
            "chameleon/squirrel, actor, ogbn-arxiv/ogbn-products."
        )


    if getattr(data, "x", None) is None:
        raise ValueError("Loaded dataset has no node features (data.x is missing).")
    if getattr(data, "num_nodes", None) is None:
        data.num_nodes = data.x.size(0)

    data = _ensure_y_node_level(data)


    if split == "public":
        for k in ["train_mask", "val_mask", "test_mask"]:
            if not hasattr(data, k) or getattr(data, k) is None:
                raise ValueError(
                    f"split='public' requires data.{k} to exist for dataset '{name}'. "
                    "Use --split random (or ogb for ogbn-*) instead."
                )
    elif split == "random":
        train_ratio = float(kwargs.pop("train_ratio", 0.1))
        val_ratio = float(kwargs.pop("val_ratio", 0.1))
        test_ratio = float(kwargs.pop("test_ratio", 0.8))
        seed = int(kwargs.pop("seed", 42))
        tr, va, te = _random_split_masks(
            data.num_nodes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        data.train_mask, data.val_mask, data.test_mask = tr, va, te
    elif split == "ogb":
        if n not in ogb_set:
            raise ValueError("split='ogb' is only valid for ogbn-* datasets.")
        idx = dataset.get_idx_split()
        train_idx = idx["train"]
        valid_idx = idx.get("valid", idx.get("val"))
        test_idx = idx["test"]
        data.train_mask = _idx_to_mask(data.num_nodes, train_idx)
        data.val_mask = _idx_to_mask(data.num_nodes, valid_idx)
        data.test_mask = _idx_to_mask(data.num_nodes, test_idx)


    data = _postprocess_graph(
        data,
        normalize_features=normalize_features,
        to_undirected_flag=to_undirected_flag,
    )

    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(data, mask_name) and getattr(data, mask_name) is not None:
            setattr(data, mask_name, _ensure_1d_mask(getattr(data, mask_name)))

    is_undirected = bool(getattr(data, "is_undirected", lambda: False)()) if hasattr(data, "is_undirected") else False
    has_edge_attr = getattr(data, "edge_attr", None) is not None
    num_classes = int(getattr(dataset, "num_classes", int(data.y.max().item() + 1)))

    meta: Dict[str, Any] = {
        "name": n,
        "split": split,
        "task_type": "node_classification",
        "num_features": int(data.x.size(-1)),
        "num_classes": num_classes,
        "is_undirected": is_undirected,
        "has_edge_attr": bool(has_edge_attr),
    }

    return data, dataset, meta
