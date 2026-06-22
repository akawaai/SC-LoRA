from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RankController(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BudgetController(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def sparse_softmax(logits: torch.Tensor, topk: int | None = None, temperature: float = 1.0) -> torch.Tensor:
    # Guard against non-finite controller outputs so one unstable step does not poison training.
    logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
    temp = max(float(temperature), 1e-6)

    weights = F.softmax(logits / temp, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    if topk is None or topk >= weights.shape[-1]:
        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    if topk <= 0:
        return torch.zeros_like(weights)

    top_values, top_idx = torch.topk(weights, k=topk, dim=-1)
    sparse = torch.zeros_like(weights)
    sparse.scatter_(dim=-1, index=top_idx, src=top_values)
    sparse = sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)
    return sparse


def monotonic_rank_mask(rank_logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
    rank_logits = torch.nan_to_num(rank_logits, nan=0.0, posinf=20.0, neginf=-20.0)
    alpha = torch.sigmoid(rank_logits)
    alpha = torch.clamp(alpha, min=1e-4, max=1.0)
    mask = torch.cumprod(alpha, dim=-1)
    mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    if hard:
        return (mask > 0.5).float()
    return mask


def reweight_gate_with_similarity(
    gate_weights: torch.Tensor,
    z: torch.Tensor,
    prototypes: torch.Tensor | None,
    temperature: float = 1.0,
) -> torch.Tensor:
    if prototypes is None:
        return gate_weights
    if prototypes.ndim == 1:
        return gate_weights

    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    prototypes = torch.nan_to_num(prototypes, nan=0.0, posinf=0.0, neginf=0.0)

    z_norm = F.normalize(z, p=2, dim=-1, eps=1e-6)
    p_norm = F.normalize(prototypes, p=2, dim=-1, eps=1e-6)
    sim = torch.matmul(z_norm, p_norm.T)
    sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)

    temp = max(float(temperature), 1e-6)
    sim = F.softmax(sim / temp, dim=-1)
    sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)

    if sim.shape[-1] != gate_weights.shape[-1]:
        return gate_weights

    out = gate_weights * sim
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = out / (out.sum(dim=-1, keepdim=True) + 1e-8)
    return out
