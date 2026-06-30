from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .gcn import APPNPEncoder, GCNEncoder, SAGEEncoder
from .gcn import GCNIIEncoder
from .stereographic import (
    clip_tangent,
    expmap0,
    geodesic_distance,
    max_geodesic_radius,
    simplex_edge_length,
)


@dataclass
class ModelConfig:
    in_dim: int
    num_classes: int
    encoder_type: str = "gcn"
    gcn_hidden: int = 256
    gcn_out: int = 128
    gcn_layers: int = 2
    gcnii_alpha: float = 0.1
    gcnii_theta: float = 0.5
    gcnii_shared_weights: bool = True
    appnp_k: int = 10
    appnp_alpha: float = 0.1
    proj_dim: int = 128
    dropout: float = 0.2
    num_prototypes: int = 0
    assign_tau: float | None = None
    tau: float = 0.2
    kappa_max: float = 1.0
    init_kappa: float = 0.0
    radius_init: float = 0.5
    prototype_init: str = "random"
    geometry_logit_weight: float = 1.0
    euclidean_head_weight: float = 0.0
    curvature_beta: float = 10.0
    clip_eps: float = 1e-8
    clip_delta: float = 1e-3
    # Legacy fields kept so older scripts can still instantiate ModelConfig.
    hgcn_hidden: int = 256
    hgcn_out: int = 128
    hgcn_layers: int = 2
    ema: float = 0.99
    hyp_c: float = 1.0
    diff_clusters: int = 10
    diff_noise: float = 0.02
    prompting: object | None = None


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MahalanobisTransform(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.L = nn.Parameter(torch.eye(dim))

    def forward(self, z: Tensor) -> Tensor:
        return z @ self.L.transpose(0, 1)

    def frobenius_sq(self) -> Tensor:
        return self.L.pow(2).sum()


class GrappleModel(nn.Module):
    """Core model rewritten to match the revised prototype-curvature methodology."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.num_classes < 2:
            raise ValueError("ModelConfig.num_classes must be at least 2.")
        if cfg.num_prototypes < 2:
            raise ValueError("ModelConfig.num_prototypes must be at least 2.")
        if cfg.num_prototypes < cfg.num_classes:
            raise ValueError(
                "num_prototypes must be at least num_classes. "
                f"Got num_prototypes={cfg.num_prototypes}, num_classes={cfg.num_classes}."
            )
        if cfg.num_prototypes % cfg.num_classes != 0:
            raise ValueError(
                "num_prototypes must be an integer multiple of num_classes for grouped supervision. "
                f"Got num_prototypes={cfg.num_prototypes}, num_classes={cfg.num_classes}."
            )
        if cfg.kappa_max <= 0.0:
            raise ValueError("kappa_max must be positive.")
        tau = float(cfg.assign_tau) if cfg.assign_tau is not None else float(cfg.tau)
        if tau <= 0.0:
            raise ValueError("Assignment temperature tau must be positive.")
        if not (0.0 < float(cfg.radius_init) < 1.0):
            raise ValueError("radius_init must lie strictly between 0 and 1.")
        if abs(float(cfg.init_kappa)) >= float(cfg.kappa_max):
            raise ValueError("init_kappa must satisfy |init_kappa| < kappa_max.")
        self.cfg = cfg
        self.prototypes_per_class = cfg.num_prototypes // cfg.num_classes
        encoder_type = cfg.encoder_type.strip().lower()
        if encoder_type == "gcn":
            self.encoder = GCNEncoder(
                cfg.in_dim,
                cfg.gcn_hidden,
                cfg.gcn_out,
                num_layers=cfg.gcn_layers,
                dropout=cfg.dropout,
            )
        elif encoder_type == "sage":
            self.encoder = SAGEEncoder(
                cfg.in_dim,
                cfg.gcn_hidden,
                cfg.gcn_out,
                num_layers=cfg.gcn_layers,
                dropout=cfg.dropout,
            )
        elif encoder_type == "gcnii":
            self.encoder = GCNIIEncoder(
                cfg.in_dim,
                cfg.gcn_hidden,
                cfg.gcn_out,
                num_layers=cfg.gcn_layers,
                dropout=cfg.dropout,
                alpha=cfg.gcnii_alpha,
                theta=cfg.gcnii_theta,
                shared_weights=cfg.gcnii_shared_weights,
            )
        elif encoder_type == "appnp":
            self.encoder = APPNPEncoder(
                cfg.in_dim,
                cfg.gcn_hidden,
                cfg.gcn_out,
                num_layers=cfg.gcn_layers,
                dropout=cfg.dropout,
                appnp_k=cfg.appnp_k,
                appnp_alpha=cfg.appnp_alpha,
            )
        else:
            raise ValueError(
                f"Unsupported encoder_type='{cfg.encoder_type}'. Choose from ['gcn', 'sage', 'gcnii', 'appnp']."
            )
        self.projector = ProjectionHead(cfg.gcn_out, cfg.proj_dim)
        self.mahalanobis = MahalanobisTransform(cfg.proj_dim)
        self.euclidean_classifier = nn.Linear(cfg.proj_dim, cfg.num_classes)

        self.xi = nn.Parameter(torch.logit(torch.tensor(float(cfg.radius_init), dtype=torch.float32)))

        init_ratio = float(cfg.init_kappa) / float(cfg.kappa_max)
        self.eta = nn.Parameter(torch.tensor(math.atanh(init_ratio), dtype=torch.float32))
        self.prototype_raw = nn.Parameter(self._init_prototypes(cfg))
        prototype_to_class = torch.arange(cfg.num_prototypes, dtype=torch.long) // self.prototypes_per_class
        self.register_buffer("prototype_to_class", prototype_to_class, persistent=False)

    @staticmethod
    def _exact_simplex(num_prototypes: int, proj_dim: int) -> Tensor:
        gram = torch.full(
            (num_prototypes, num_prototypes),
            -1.0 / float(num_prototypes - 1),
            dtype=torch.float32,
        )
        gram.fill_diagonal_(1.0)
        evals, evecs = torch.linalg.eigh(gram)
        keep = evals > 1e-8
        basis = evecs[:, keep] * torch.sqrt(evals[keep]).unsqueeze(0)
        simplex = torch.zeros(num_prototypes, proj_dim, dtype=torch.float32)
        use_dim = min(basis.size(1), proj_dim)
        simplex[:, :use_dim] = basis[:, :use_dim]
        return simplex

    def _init_prototypes(self, cfg: ModelConfig) -> Tensor:
        init_mode = cfg.prototype_init.strip().lower()
        if init_mode == "random":
            return torch.randn(cfg.num_prototypes, cfg.proj_dim, dtype=torch.float32) * 0.02
        if init_mode == "simplex":
            return self._exact_simplex(cfg.num_prototypes, cfg.proj_dim)
        raise ValueError(f"Unsupported prototype_init='{cfg.prototype_init}'. Choose from ['random', 'simplex'].")

    @property
    def assignment_temperature(self) -> float:
        if self.cfg.assign_tau is not None:
            return float(self.cfg.assign_tau)
        return float(self.cfg.tau)

    def curvature(self) -> Tensor:
        return self.cfg.kappa_max * torch.tanh(self.eta)

    def normalize_projected(self, z: Tensor) -> Tensor:
        return F.normalize(z, p=2, dim=-1, eps=self.cfg.clip_eps)

    def prototype_directions(self) -> Tensor:
        return F.normalize(self.prototype_raw, dim=-1, eps=self.cfg.clip_eps)

    def grouped_prototype_directions(self, directions: Tensor) -> Tensor:
        return directions.view(self.cfg.num_classes, self.prototypes_per_class, -1)

    def class_representative_directions(self, directions: Tensor) -> Tensor:
        grouped = self.grouped_prototype_directions(directions)
        return F.normalize(grouped.mean(dim=1), dim=-1, eps=self.cfg.clip_eps)

    def prototype_radius(self, kappa: Tensor) -> Tensor:
        radius_limit = max_geodesic_radius(
            kappa,
            beta=self.cfg.curvature_beta,
            eps=self.cfg.clip_eps,
            delta=self.cfg.clip_delta,
        )
        return radius_limit * torch.sigmoid(self.xi)

    def prototype_points(self, kappa: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        directions = self.prototype_directions()
        rho = self.prototype_radius(kappa)
        prototypes = expmap0(rho * directions, kappa=kappa, eps=self.cfg.clip_eps)
        return prototypes, directions, rho

    def class_logits_from_prototype_logits(self, prototype_logits: Tensor) -> Tensor:
        if self.prototypes_per_class == 1:
            return prototype_logits
        num_nodes = prototype_logits.size(0)
        grouped = prototype_logits.view(num_nodes, self.cfg.num_classes, self.prototypes_per_class)
        return torch.logsumexp(grouped, dim=-1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        logit_node_count: int | None = None,
    ) -> dict[str, Tensor]:
        h = self.encoder(x, edge_index, edge_weight=edge_weight)
        z = self.projector(h)
        z_unit = self.normalize_projected(z)
        v = self.mahalanobis(z_unit)

        kappa = self.curvature()
        v_clipped = clip_tangent(
            v,
            kappa=kappa,
            beta=self.cfg.curvature_beta,
            eps=self.cfg.clip_eps,
            delta=self.cfg.clip_delta,
        )
        x_manifold = expmap0(v_clipped, kappa=kappa, eps=self.cfg.clip_eps)
        x_for_logits = x_manifold
        z_for_logits = z_unit
        if logit_node_count is not None:
            node_count = int(logit_node_count)
            x_for_logits = x_manifold[:node_count]
            z_for_logits = z_unit[:node_count]

        prototypes, directions, rho = self.prototype_points(kappa)
        dist = geodesic_distance(x_for_logits, prototypes, kappa=kappa, eps=self.cfg.clip_eps)
        prototype_logits = -(dist.pow(2)) / max(self.assignment_temperature, self.cfg.clip_eps)
        alpha = torch.softmax(prototype_logits, dim=-1)
        geom_class_logits = self.class_logits_from_prototype_logits(prototype_logits)
        euclidean_logits = self.euclidean_classifier(z_for_logits)
        class_logits = (
            float(self.cfg.geometry_logit_weight) * geom_class_logits
            + float(self.cfg.euclidean_head_weight) * euclidean_logits
        )
        class_probs = torch.softmax(class_logits, dim=-1)
        geom_class_probs = torch.softmax(geom_class_logits, dim=-1)

        return {
            "h": h,
            "z": z,
            "z_unit": z_unit,
            "v": v,
            "v_clipped": v_clipped,
            "x_manifold": x_manifold,
            "x_for_logits": x_for_logits,
            "kappa": kappa,
            "rho": rho,
            "prototypes": prototypes,
            "prototype_directions": directions,
            "dist": dist,
            "prototype_logits": prototype_logits,
            "geom_logits": geom_class_logits,
            "euclidean_logits": euclidean_logits,
            "logits": class_logits,
            "class_probs": class_probs,
            "geom_class_probs": geom_class_probs,
            "alpha": alpha,
            "prototype_to_class": self.prototype_to_class,
        }

    def forward_views(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_index_drop: Tensor | None = None,
        x_node_masked: Tensor | None = None,
        edge_weight: Tensor | None = None,
        seed: int = 0,
    ) -> dict[str, Tensor]:
        del edge_index_drop, x_node_masked, seed
        return self.forward(x=x, edge_index=edge_index, edge_weight=edge_weight)

    def _pairwise_capacity_loss(
        self,
        out: dict[str, Tensor],
        class_dirs: Tensor,
        cap_margin: float,
        y: Tensor | None,
        confusion_weight: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        kappa = out["kappa"]
        rho = out["rho"]
        class_points = expmap0(rho * class_dirs, kappa=kappa, eps=self.cfg.clip_eps)
        class_dist = geodesic_distance(
            out["x_for_logits"],
            class_points,
            kappa=kappa,
            eps=self.cfg.clip_eps,
        )

        if y is not None:
            y = y.view(-1).to(torch.long)
            one_hot = F.one_hot(y, num_classes=self.cfg.num_classes).to(class_dist.dtype)
            counts = one_hot.sum(dim=0)
            r_c = (one_hot * class_dist).sum(dim=0) / counts.clamp_min(1.0)
            valid = counts > 0
            probs = out["class_probs"].detach()
            confusion = one_hot.transpose(0, 1) @ probs
            confusion = confusion / counts.clamp_min(1.0).unsqueeze(1)
            pair_weights = 1.0 + float(confusion_weight) * 0.5 * (confusion + confusion.transpose(0, 1))
            valid_pairs = valid[:, None] & valid[None, :]
        else:
            probs = out["geom_class_probs"].detach()
            mass = probs.sum(dim=0)
            r_c = (probs * class_dist).sum(dim=0) / mass.clamp_min(self.cfg.clip_eps)
            valid_pairs = mass[:, None].gt(0.0) & mass[None, :].gt(0.0)
            pair_weights = torch.ones(
                self.cfg.num_classes,
                self.cfg.num_classes,
                device=class_dist.device,
                dtype=class_dist.dtype,
            )

        class_pair_dist = geodesic_distance(class_points, class_points, kappa=kappa, eps=self.cfg.clip_eps)
        margin = float(cap_margin) + r_c[:, None] + r_c[None, :]
        eye = torch.eye(self.cfg.num_classes, device=class_pair_dist.device, dtype=torch.bool)
        pair_mask = valid_pairs & ~eye
        violations = F.relu(margin - class_pair_dist).pow(2) * pair_weights
        if pair_mask.any():
            l_cap = violations[pair_mask].mean()
            d_cap = class_pair_dist[pair_mask].mean()
        else:
            l_cap = violations.sum() * 0.0
            d_cap = class_pair_dist.mean()
        return l_cap, r_c[valid_pairs.any(dim=1)].mean(), d_cap

    def geometric_terms(
        self,
        out: dict[str, Tensor],
        cap_margin: float,
        y: Tensor | None = None,
        capacity_mode: str = "global",
        confusion_weight: float = 0.0,
    ) -> dict[str, Tensor]:
        alpha = out["alpha"]
        dist = out["dist"]
        directions = out["prototype_directions"]
        kappa = out["kappa"]
        rho = out["rho"]

        l_clu = (alpha * dist.pow(2)).sum(dim=-1).mean()

        if self.prototypes_per_class == 1:
            gram = directions @ directions.transpose(0, 1)
            etf_target = torch.full_like(gram, -1.0 / float(self.cfg.num_prototypes - 1))
            etf_target.fill_diagonal_(1.0)
            l_etf = (gram - etf_target).pow(2).sum()

            alpha_bar = alpha.mean(dim=0)
            uniform = 1.0 / float(self.cfg.num_prototypes)
            l_bal = (alpha_bar - uniform).pow(2).sum()
            class_dirs = directions
            within_class_usage = alpha.mean(dim=0).view(self.cfg.num_classes, self.prototypes_per_class)
        else:
            class_dirs = self.class_representative_directions(directions)
            class_gram = class_dirs @ class_dirs.transpose(0, 1)
            etf_target = torch.full_like(class_gram, -1.0 / float(self.cfg.num_classes - 1))
            etf_target.fill_diagonal_(1.0)
            l_etf = (class_gram - etf_target).pow(2).sum()

            proto_usage = alpha.mean(dim=0).view(self.cfg.num_classes, self.prototypes_per_class)
            class_mass = proto_usage.sum(dim=-1, keepdim=True).clamp_min(self.cfg.clip_eps)
            within_class_usage = proto_usage / class_mass
            uniform = torch.full_like(within_class_usage, 1.0 / float(self.prototypes_per_class))
            l_bal = ((within_class_usage - uniform).pow(2) * class_mass).sum()

        capacity_mode = capacity_mode.strip().lower()
        if capacity_mode == "global":
            r_emp = (alpha * dist).sum(dim=-1).mean()
            d_cap = simplex_edge_length(
                kappa=kappa,
                rho=rho,
                num_prototypes=self.cfg.num_prototypes,
                eps=self.cfg.clip_eps,
            )
            l_cap = F.relu(cap_margin + 2.0 * r_emp - d_cap).pow(2)
        elif capacity_mode == "pairwise":
            l_cap, r_emp, d_cap = self._pairwise_capacity_loss(
                out,
                class_dirs=class_dirs,
                cap_margin=cap_margin,
                y=y,
                confusion_weight=confusion_weight,
            )
        else:
            raise ValueError("capacity_mode must be either 'global' or 'pairwise'.")

        return {
            "l_clu": l_clu,
            "l_etf": l_etf,
            "l_bal": l_bal,
            "l_cap": l_cap,
            "l_reg": self.mahalanobis.frobenius_sq(),
            "l_kappa": kappa.pow(2),
            "r_emp": r_emp,
            "d_cap": d_cap,
            "class_directions": class_dirs,
            "within_class_usage": within_class_usage,
        }
