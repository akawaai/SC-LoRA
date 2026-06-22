from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs and isinstance(outputs["logits"], torch.Tensor):
        return outputs["logits"]
    raise TypeError("Could not extract logits tensor from model outputs.")


def compute_task_loss(outputs: Any, targets: torch.Tensor | None) -> torch.Tensor:
    if hasattr(outputs, "loss") and isinstance(outputs.loss, torch.Tensor):
        return outputs.loss

    if isinstance(outputs, dict) and "loss" in outputs and isinstance(outputs["loss"], torch.Tensor):
        return outputs["loss"]

    logits = _extract_logits(outputs)

    if targets is None:
        return torch.tensor(0.0, device=logits.device)

    if logits.ndim == targets.ndim + 1:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    if logits.shape == targets.shape:
        return F.mse_loss(logits, targets)

    raise ValueError(
        f"Unsupported logits/targets shapes for task loss: logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
    )


def preservation_loss(
    z_ft: dict[str, torch.Tensor],
    z_base: dict[str, torch.Tensor],
    general_mask: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = next(iter(z_ft.values())).device
    total = torch.tensor(0.0, device=device)
    for layer, zf in z_ft.items():
        zb = z_base[layer]
        mask = general_mask[layer].to(device)
        total = total + (((zf - zb) * mask) ** 2).mean()
    return total


def steering_loss(
    z_ft: dict[str, torch.Tensor],
    domain_mask_for_domain: dict[str, torch.Tensor],
    target_prototypes_for_domain: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = next(iter(z_ft.values())).device
    total = torch.tensor(0.0, device=device)
    for layer, zf in z_ft.items():
        if layer not in domain_mask_for_domain:
            continue
        if layer not in target_prototypes_for_domain:
            continue
        mask = domain_mask_for_domain[layer].to(device)
        proto = target_prototypes_for_domain[layer].to(device)
        total = total + (((zf - proto) * mask) ** 2).mean()
    return total


def gate_sparsity_loss(adapter_stats: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    if not adapter_stats:
        return torch.tensor(0.0)

    device = next(iter(adapter_stats.values()))["gate"].device
    total = torch.tensor(0.0, device=device)
    num_terms = 0

    for stats in adapter_stats.values():
        gate = stats["gate"]
        gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-8)

        active_k = int(gate.shape[-1])
        active_mask = stats.get("active_mask")
        if isinstance(active_mask, torch.Tensor) and active_mask.numel() == gate.shape[-1]:
            active_k = int((active_mask > 0.5).sum().item())
        active_k = max(1, active_k)

        if active_k <= 1:
            continue

        entropy = -(gate * (gate + 1e-8).log()).sum(dim=-1)
        entropy_floor = 0.20 * math.log(float(active_k))
        entropy_penalty = F.relu(gate.new_tensor(entropy_floor) - entropy).mean()

        gate_peak = gate.max(dim=-1).values
        peak_penalty = F.relu(gate_peak - 0.90).mean()

        total = total + entropy_penalty + 0.5 * peak_penalty
        num_terms += 1

    return total / max(1, num_terms)


def rank_budget_loss(adapter_stats: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    if not adapter_stats:
        return torch.tensor(0.0)

    device = next(iter(adapter_stats.values()))["rank_mask"].device
    total = torch.tensor(0.0, device=device)

    for stats in adapter_stats.values():
        rank_mask = stats["rank_mask"]
        total = total + rank_mask.mean()

    return total / max(1, len(adapter_stats))


def load_balance_loss(adapter_stats: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    if not adapter_stats:
        return torch.tensor(0.0)

    device = next(iter(adapter_stats.values()))["gate"].device
    total = torch.tensor(0.0, device=device)
    num_terms = 0

    for stats in adapter_stats.values():
        gate = stats["gate"]

        active_mask = stats.get("active_mask")
        if isinstance(active_mask, torch.Tensor) and active_mask.numel() == gate.shape[-1]:
            active_idx = (active_mask > 0.5).nonzero(as_tuple=False).reshape(-1)
            if active_idx.numel() >= 2:
                gate = gate[:, active_idx]

        usage = gate.mean(dim=0)
        uniform = torch.full_like(usage, 1.0 / usage.shape[0])
        total = total + ((usage - uniform) ** 2).mean()
        num_terms += 1

    return total / max(1, num_terms)


def specialist_diversity_loss(model_theta: Any) -> torch.Tensor:
    def _safe_normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        valid = (norms.squeeze(-1) > eps)
        x_norm = x / norms.clamp_min(eps)
        x_norm = torch.where(valid.unsqueeze(-1), x_norm, torch.zeros_like(x_norm))
        return x_norm, valid

    penalties: list[torch.Tensor] = []

    for wrapper in model_theta.wrappers.values():
        bank = wrapper.bank
        a = bank.a.reshape(bank.num_specialists, -1)
        b = bank.b.reshape(bank.num_specialists, -1)

        active_mask = getattr(bank, "active_mask", None)
        if isinstance(active_mask, torch.Tensor) and active_mask.numel() == bank.num_specialists:
            active_idx = (active_mask > 0.5).nonzero(as_tuple=False).reshape(-1)
            if active_idx.numel() < 2:
                continue
            a = a.index_select(0, active_idx)
            b = b.index_select(0, active_idx)

        a, valid_a = _safe_normalize_rows(a)
        b, valid_b = _safe_normalize_rows(b)

        if int(valid_a.sum().item()) >= 2:
            a_valid = a[valid_a]
            sim_a = a_valid @ a_valid.T
            identity_a = torch.eye(sim_a.shape[0], device=sim_a.device)
            penalties.append(((sim_a - identity_a) ** 2).mean())

        if int(valid_b.sum().item()) >= 2:
            b_valid = b[valid_b]
            sim_b = b_valid @ b_valid.T
            identity_b = torch.eye(sim_b.shape[0], device=sim_b.device)
            penalties.append(((sim_b - identity_b) ** 2).mean())

    if not penalties:
        return torch.tensor(0.0)

    return torch.stack(penalties).mean()


def summarize_usage(adapter_stats: dict[str, dict[str, torch.Tensor]]) -> tuple[float, float]:
    if not adapter_stats:
        return 0.0, 0.0

    active_specialists: list[float] = []
    mean_rank: list[float] = []

    for stats in adapter_stats.values():
        gate = stats["gate"]
        rank_mask = stats["rank_mask"]
        active_specialists.append(float((gate > 0.05).float().sum(dim=-1).float().mean().item()))
        mean_rank.append(float(rank_mask.sum(dim=-1).mean().item()))

    return sum(active_specialists) / len(active_specialists), sum(mean_rank) / len(mean_rank)
