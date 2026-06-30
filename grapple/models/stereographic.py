from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-12) -> Tensor:
    return x.norm(dim=dim, keepdim=keepdim).clamp_min(eps)


def kappa_positive_part(kappa: Tensor, beta: float) -> Tensor:
    return F.softplus(beta * kappa) / beta


def max_geodesic_radius(kappa: Tensor, beta: float, eps: float, delta: float) -> Tensor:
    kappa_pos = kappa_positive_part(kappa, beta=beta)
    radius = math.pi / (2.0 * torch.sqrt(kappa_pos + eps)) - delta
    return radius.clamp_min(eps)


def _tan_kappa(r: Tensor, kappa: Tensor, eps: float) -> Tensor:
    kappa_val = float(kappa.detach().cpu().item())
    if abs(kappa_val) < eps:
        return r
    if kappa_val > 0.0:
        sqrt_kappa = torch.sqrt(kappa.clamp_min(eps))
        return torch.tan(sqrt_kappa * r) / sqrt_kappa
    sqrt_neg_kappa = torch.sqrt((-kappa).clamp_min(eps))
    return torch.tanh(sqrt_neg_kappa * r) / sqrt_neg_kappa


def _arctan_kappa(r: Tensor, kappa: Tensor, eps: float) -> Tensor:
    kappa_val = float(kappa.detach().cpu().item())
    if abs(kappa_val) < eps:
        return r
    if kappa_val > 0.0:
        sqrt_kappa = torch.sqrt(kappa.clamp_min(eps))
        return torch.atan(sqrt_kappa * r) / sqrt_kappa
    sqrt_neg_kappa = torch.sqrt((-kappa).clamp_min(eps))
    bounded = (sqrt_neg_kappa * r).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    return torch.atanh(bounded) / sqrt_neg_kappa


def clip_tangent(v: Tensor, kappa: Tensor, beta: float, eps: float, delta: float) -> Tensor:
    radius = max_geodesic_radius(kappa, beta=beta, eps=eps, delta=delta)
    v_norm = safe_norm(v, keepdim=True, eps=eps)
    scale = radius * torch.tanh(v_norm / radius) / v_norm
    return scale * v


def project_stereographic(x: Tensor, kappa: Tensor, eps: float, margin: float = 1e-5) -> Tensor:
    kappa_val = float(kappa.detach().cpu().item())
    if kappa_val >= 0.0:
        return x
    max_norm = (1.0 - margin) / torch.sqrt((-kappa).clamp_min(eps))
    x_norm = safe_norm(x, keepdim=True, eps=eps)
    scale = torch.clamp(max_norm / x_norm, max=1.0)
    return x * scale


def expmap0(v: Tensor, kappa: Tensor, eps: float) -> Tensor:
    v_norm = safe_norm(v, keepdim=True, eps=eps)
    scale = _tan_kappa(v_norm, kappa=kappa, eps=eps) / v_norm
    x = scale * v
    return project_stereographic(x, kappa=kappa, eps=eps)


def mobius_add(x: Tensor, y: Tensor, kappa: Tensor, eps: float) -> Tensor:
    xy = (x * y).sum(dim=-1, keepdim=True)
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    numerator = (1.0 - 2.0 * kappa * xy - kappa * y_sq) * x + (1.0 + kappa * x_sq) * y
    denominator = 1.0 - 2.0 * kappa * xy + (kappa * kappa) * x_sq * y_sq
    denominator = denominator.clamp_min(eps)
    out = numerator / denominator
    return project_stereographic(out, kappa=kappa, eps=eps)


def geodesic_distance(x: Tensor, y: Tensor, kappa: Tensor, eps: float = 1e-8) -> Tensor:
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("geodesic_distance expects 2D tensors [N, d] and [K, d].")
    if x.size(-1) != y.size(-1):
        raise ValueError("geodesic_distance requires matching embedding dimensions.")

    kappa_val = float(kappa.detach().cpu().item())
    if abs(kappa_val) < eps:
        return torch.cdist(x, y, p=2)
    mobius = mobius_add(-x[:, None, :], y[None, :, :], kappa=kappa, eps=eps)
    mobius_norm = safe_norm(mobius, dim=-1, keepdim=False, eps=eps)
    return 2.0 * _arctan_kappa(mobius_norm, kappa=kappa, eps=eps)


def simplex_edge_length(kappa: Tensor, rho: Tensor, num_prototypes: int, eps: float = 1e-8) -> Tensor:
    if num_prototypes < 2:
        raise ValueError("num_prototypes must be at least 2.")

    cos_theta = -1.0 / float(num_prototypes - 1)
    kappa_val = float(kappa.detach().cpu().item())

    if abs(kappa_val) < eps:
        return rho * math.sqrt(2.0 * num_prototypes / float(num_prototypes - 1))

    if kappa_val > 0.0:
        sqrt_kappa = torch.sqrt(kappa.clamp_min(eps))
        angle = sqrt_kappa * rho
        arg = torch.cos(angle).pow(2) + torch.sin(angle).pow(2) * cos_theta
        arg = arg.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return torch.acos(arg) / sqrt_kappa

    sqrt_neg_kappa = torch.sqrt((-kappa).clamp_min(eps))
    angle = sqrt_neg_kappa * rho
    arg = torch.cosh(angle).pow(2) - torch.sinh(angle).pow(2) * cos_theta
    arg = arg.clamp_min(1.0 + 1e-7)
    return torch.acosh(arg) / sqrt_neg_kappa
