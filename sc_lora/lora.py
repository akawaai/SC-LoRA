from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SCLoRAConfig
from .controllers import (
    BudgetController,
    GateNetwork,
    RankController,
    monotonic_rank_mask,
    reweight_gate_with_similarity,
    sparse_softmax,
)
from .utils import resolve_parent


@dataclass
class SCForwardContext:
    features_by_layer: dict[str, torch.Tensor]
    prototypes_by_layer: dict[str, torch.Tensor] | None = None
    domain_id: str | None = None


def _align_conditioning(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(-1, x.shape[-1])

    if z.ndim == x.ndim and z.shape[:-1] == x.shape[:-1]:
        return z.reshape(-1, z.shape[-1])

    if z.ndim == 2 and z.shape[0] == x_flat.shape[0]:
        return z

    if z.ndim == 2 and x.ndim >= 3 and z.shape[0] == x.shape[0]:
        expanded = z.unsqueeze(1).expand(x.shape[0], x.shape[1], z.shape[-1])
        return expanded.reshape(-1, z.shape[-1])

    if z.ndim == 1:
        return z.unsqueeze(0).expand(x_flat.shape[0], z.shape[0])

    pooled = z.reshape(-1, z.shape[-1]).mean(dim=0, keepdim=True)
    return pooled.expand(x_flat.shape[0], pooled.shape[-1])


def build_delta_w(
    gate_weights: torch.Tensor,
    rank_mask: torch.Tensor,
    a_params: torch.Tensor,
    b_params: torch.Tensor,
) -> torch.Tensor:
    n = gate_weights.shape[0]
    d_out = b_params.shape[1]
    d_in = a_params.shape[2]

    delta = torch.zeros(n, d_out, d_in, device=gate_weights.device, dtype=gate_weights.dtype)
    for i in range(n):
        m = rank_mask[i]
        weighted = torch.zeros(d_out, d_in, device=gate_weights.device, dtype=gate_weights.dtype)
        for k in range(gate_weights.shape[1]):
            a_mask = a_params[k] * m.unsqueeze(-1)
            b_mask = b_params[k] * m.unsqueeze(0)
            weighted = weighted + gate_weights[i, k] * (b_mask @ a_mask)
        delta[i] = weighted
    return delta


class SCLoRAAdapterBank(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        sae_feature_dim: int,
        cfg: SCLoRAConfig,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_specialists = cfg.num_specialists
        self.max_rank = cfg.max_rank
        self.topk = cfg.controller_topk
        self.alpha = cfg.lora_alpha

        self.memory_ema_decay = float(min(max(cfg.memory_ema_decay, 0.0), 0.9999))
        self.spawn_confidence_threshold = float(cfg.spawn_confidence_threshold)
        self.merge_similarity_threshold = float(cfg.merge_similarity_threshold)
        self.prune_usage_threshold = float(cfg.prune_usage_threshold)
        self.min_active_specialists = max(1, int(cfg.min_active_specialists))
        self.memory_policy_interval = max(1, int(cfg.memory_policy_interval))
        self.memory_compaction_warmup_steps = max(0, int(getattr(cfg, "memory_compaction_warmup_steps", 0)))
        self.retrieval_topk = max(1, int(cfg.retrieval_topk))
        self.enable_memory_policy = bool(getattr(cfg, "enable_memory_policy", True))

        fixed_rank = getattr(cfg, "fixed_rank", None)
        self.fixed_rank = None
        if fixed_rank is not None and int(fixed_rank) > 0:
            self.fixed_rank = min(self.max_rank, int(fixed_rank))

        self.a = nn.Parameter(torch.empty(cfg.num_specialists, cfg.max_rank, d_in))
        self.b = nn.Parameter(torch.empty(cfg.num_specialists, d_out, cfg.max_rank))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        nn.init.zeros_(self.b)

        self.gate = GateNetwork(sae_feature_dim, cfg.num_specialists, hidden_dim=cfg.gate_hidden_dim)
        self.rank = RankController(sae_feature_dim, cfg.max_rank, hidden_dim=cfg.rank_hidden_dim)
        self.budget = BudgetController(sae_feature_dim, hidden_dim=cfg.rank_hidden_dim) if cfg.use_budget_controller else None

        if self.fixed_rank is not None:
            for param in self.rank.parameters():
                param.requires_grad = False
            if self.budget is not None:
                for param in self.budget.parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(cfg.lora_dropout)

        self.register_buffer("active_mask", torch.zeros(self.num_specialists, dtype=torch.bool))
        self.register_buffer("usage_ema", torch.zeros(self.num_specialists))
        self.register_buffer("prototype_memory", torch.zeros(self.num_specialists, sae_feature_dim))
        self.register_buffer("prototype_count", torch.zeros(self.num_specialists))

        active = min(
            self.num_specialists,
            max(1, int(cfg.initial_active_specialists), self._min_active_floor()),
        )
        self.active_mask[:active] = True

        self._last_z_for_memory: torch.Tensor | None = None
        self._last_gate_for_memory: torch.Tensor | None = None

    def _active_indices(self) -> torch.Tensor:
        return self.active_mask.nonzero(as_tuple=False).reshape(-1)

    def _min_active_floor(self) -> int:
        floor = max(1, int(self.min_active_specialists))
        if self.topk is not None:
            floor = max(floor, min(self.num_specialists, max(1, int(self.topk))))
        return min(self.num_specialists, floor)

    def _param_ref(self) -> torch.Tensor:
        return self.a

    def _match_tensor(self, t: torch.Tensor) -> torch.Tensor:
        ref = self._param_ref()
        if t.device != ref.device or t.dtype != ref.dtype:
            return t.to(device=ref.device, dtype=ref.dtype)
        return t

    def _mask_inactive_logits(self, gate_logits: torch.Tensor) -> torch.Tensor:
        masked = gate_logits
        inactive = ~self.active_mask
        if inactive.any():
            masked = gate_logits.clone()
            masked[:, inactive] = -1e9
        return masked

    def _sync_memory_stats(self, z_flat: torch.Tensor, gate_weights: torch.Tensor) -> None:
        if z_flat.numel() == 0 or gate_weights.numel() == 0:
            return

        decay = self.memory_ema_decay
        with torch.no_grad():
            batch_usage = gate_weights.mean(dim=0)
            self.usage_ema.mul_(decay).add_((1.0 - decay) * batch_usage)
            self.usage_ema.mul_(self.active_mask.float())

            active_idx = self._active_indices()
            for idx_t in active_idx:
                idx = int(idx_t.item())
                weights = gate_weights[:, idx]
                denom = float(weights.sum().item())
                if denom <= 1e-8:
                    continue

                proto_batch = (weights.unsqueeze(-1) * z_flat).sum(dim=0) / (weights.sum() + 1e-8)
                self.prototype_memory[idx] = decay * self.prototype_memory[idx] + (1.0 - decay) * proto_batch.detach()
                self.prototype_count[idx] = self.prototype_count[idx] + 1.0

    def _reset_specialist(self, idx: int) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.a[idx : idx + 1], a=5**0.5)
            self.b[idx].zero_()
            self.usage_ema[idx] = 0.0
            self.prototype_count[idx] = 0.0
            self.prototype_memory[idx].zero_()

    def _spawn_specialist(self, z_flat: torch.Tensor, gate_weights: torch.Tensor) -> int | None:
        confidence = float(gate_weights.max(dim=-1).values.mean().item())
        if confidence >= self.spawn_confidence_threshold:
            return None

        inactive_idx = (~self.active_mask).nonzero(as_tuple=False).reshape(-1)
        if inactive_idx.numel() == 0:
            return None

        idx = int(inactive_idx[0].item())
        self._reset_specialist(idx)

        with torch.no_grad():
            self.active_mask[idx] = True
            self.prototype_memory[idx] = z_flat.mean(dim=0).detach()
            self.prototype_count[idx] = 1.0

        return idx

    def _deactivate_specialist(self, idx: int) -> None:
        if int(self.active_mask.sum().item()) <= self._min_active_floor():
            return

        with torch.no_grad():
            self.active_mask[idx] = False
            self.usage_ema[idx] = 0.0
            self.prototype_count[idx] = 0.0
            self.prototype_memory[idx].zero_()
            self.b[idx].zero_()

    def _merge_specialists(self) -> tuple[int, int] | None:
        active_idx = self._active_indices()
        if active_idx.numel() < 2:
            return None

        valid_mask = self.prototype_count[active_idx] > 0
        valid_idx = active_idx[valid_mask]
        if valid_idx.numel() < 2:
            return None

        protos = F.normalize(self.prototype_memory[valid_idx], p=2, dim=-1)
        sim = protos @ protos.T
        sim.fill_diagonal_(-1.0)

        max_val, flat_idx = torch.max(sim.reshape(-1), dim=0)
        if float(max_val.item()) < self.merge_similarity_threshold:
            return None

        n = sim.shape[0]
        row = int(flat_idx.item() // n)
        col = int(flat_idx.item() % n)
        keep = int(valid_idx[row].item())
        drop = int(valid_idx[col].item())

        if int(self.active_mask.sum().item()) <= self._min_active_floor():
            return None

        with torch.no_grad():
            w_keep = float(max(self.usage_ema[keep].item(), 1e-6))
            w_drop = float(max(self.usage_ema[drop].item(), 1e-6))
            w_sum = w_keep + w_drop

            self.a[keep] = (self.a[keep] * w_keep + self.a[drop] * w_drop) / w_sum
            self.b[keep] = (self.b[keep] * w_keep + self.b[drop] * w_drop) / w_sum
            self.prototype_memory[keep] = (self.prototype_memory[keep] * w_keep + self.prototype_memory[drop] * w_drop) / w_sum
            self.prototype_count[keep] = self.prototype_count[keep] + self.prototype_count[drop]
            self.usage_ema[keep] = max(self.usage_ema[keep], self.usage_ema[drop])

        self._deactivate_specialist(drop)
        return keep, drop

    def _prune_low_usage(self) -> int | None:
        if int(self.active_mask.sum().item()) <= self._min_active_floor():
            return None

        active_idx = self._active_indices()
        usage = self.usage_ema[active_idx]
        min_usage, local_idx = torch.min(usage, dim=0)

        if float(min_usage.item()) >= self.prune_usage_threshold:
            return None

        idx = int(active_idx[int(local_idx.item())].item())
        self._deactivate_specialist(idx)
        return idx

    def _retrieve_topk(self, z_flat: torch.Tensor, gate_weights: torch.Tensor, topk: int) -> torch.Tensor:
        active_idx = self._active_indices()
        if active_idx.numel() == 0:
            return torch.zeros(z_flat.shape[0], 1, dtype=torch.long, device=z_flat.device)

        valid_proto_mask = self.prototype_count[active_idx] > 0
        valid_idx = active_idx[valid_proto_mask]

        k = min(topk, int(active_idx.numel()))
        if valid_idx.numel() == 0:
            return torch.topk(gate_weights, k=k, dim=-1).indices

        z_norm = F.normalize(z_flat, p=2, dim=-1)
        p_norm = F.normalize(self.prototype_memory[valid_idx], p=2, dim=-1)
        sim = torch.matmul(z_norm, p_norm.T)
        k = min(k, int(valid_idx.numel()))
        top_local = torch.topk(sim, k=k, dim=-1).indices
        return valid_idx[top_local]

    def step_memory_policy(self, step: int | None = None) -> dict[str, float]:
        events = {"spawned": 0.0, "merged": 0.0, "pruned": 0.0}

        if self._last_z_for_memory is None or self._last_gate_for_memory is None:
            events["active_specialists"] = float(self.active_mask.sum().item())
            return events

        z_flat = self._last_z_for_memory
        gate_weights = self._last_gate_for_memory

        self._sync_memory_stats(z_flat, gate_weights)

        if self.enable_memory_policy:
            spawned = self._spawn_specialist(z_flat, gate_weights)
            if spawned is not None:
                events["spawned"] = 1.0

            should_compact = (
                step is not None
                and step > 0
                and step >= self.memory_compaction_warmup_steps
                and step % self.memory_policy_interval == 0
            )
            if should_compact:
                merged = self._merge_specialists()
                if merged is not None:
                    events["merged"] = 1.0

                pruned = self._prune_low_usage()
                if pruned is not None:
                    events["pruned"] = 1.0

        events["active_specialists"] = float(self.active_mask.sum().item())

        self._last_z_for_memory = None
        self._last_gate_for_memory = None
        return events

    def controller_forward(
        self,
        z: torch.Tensor,
        prototypes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z = self._match_tensor(z)
        if prototypes is not None:
            prototypes = self._match_tensor(prototypes)

        gate_logits = self.gate(z)
        gate_logits = self._mask_inactive_logits(gate_logits)
        gate_weights = sparse_softmax(gate_logits, topk=self.topk)

        if self.fixed_rank is not None:
            rank_mask = torch.zeros(z.shape[0], self.max_rank, device=z.device, dtype=z.dtype)
            if self.fixed_rank > 0:
                rank_mask[:, : self.fixed_rank] = 1.0
            budget = None
        else:
            rank_logits = self.rank(z)
            rank_mask = monotonic_rank_mask(rank_logits, hard=False)
            budget = self.budget(z) if self.budget is not None else None

        if prototypes is not None:
            gate_weights = reweight_gate_with_similarity(gate_weights, z, prototypes)

        return gate_weights, rank_mask, budget

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        prototypes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_in = self._match_tensor(self.dropout(x))
        x_flat = x_in.reshape(-1, x_in.shape[-1])
        z_flat = _align_conditioning(x_in, z)

        gate_weights, rank_mask, budget = self.controller_forward(z_flat, prototypes=prototypes)

        x_a = torch.einsum("nd,krd->nkr", x_flat, self.a)
        x_a = x_a * rank_mask.unsqueeze(1)
        out_per_specialist = torch.einsum("nkr,kdr->nkd", x_a, self.b)
        delta = torch.einsum("nk,nkd->nd", gate_weights, out_per_specialist)
        delta = delta * (self.alpha / float(self.max_rank))

        delta = delta.reshape(*x.shape[:-1], self.d_out)

        retrieved_topk = self._retrieve_topk(z_flat, gate_weights, topk=self.retrieval_topk)

        if self.training:
            self._last_z_for_memory = z_flat.detach()
            self._last_gate_for_memory = gate_weights.detach()

        stats = {
            "gate": gate_weights.detach(),
            "rank_mask": rank_mask.detach(),
            "retrieved_topk": retrieved_topk.detach(),
            "active_mask": self.active_mask.detach().float(),
            "usage_ema": self.usage_ema.detach(),
        }
        if budget is not None:
            stats["budget"] = budget.detach()

        return delta, stats


class SCLoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        bank: SCLoRAAdapterBank,
        layer_name: str,
        context_getter: Any,
    ):
        super().__init__()
        self.base = base_linear
        self.bank = bank
        self.layer_name = layer_name
        self._context_getter = context_getter
        self.last_stats: dict[str, torch.Tensor] = {}

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_ref = self.base.weight
        if x.device != base_ref.device or x.dtype != base_ref.dtype:
            x_base = x.to(device=base_ref.device, dtype=base_ref.dtype)
        else:
            x_base = x

        y = self.base(x_base)
        context: SCForwardContext | None = self._context_getter()
        if context is None:
            return y

        if self.layer_name not in context.features_by_layer:
            return y

        z = context.features_by_layer[self.layer_name].to(x_base.device)
        prototype = None
        if context.prototypes_by_layer is not None and self.layer_name in context.prototypes_by_layer:
            prototype = context.prototypes_by_layer[self.layer_name].to(x_base.device)

        delta, stats = self.bank(x_base, z, prototypes=prototype)
        self.last_stats = stats
        if delta.dtype != y.dtype or delta.device != y.device:
            delta = delta.to(device=y.device, dtype=y.dtype)
        return y + delta


class SCLoRAModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.context: SCForwardContext | None = None
        self.wrappers: dict[str, SCLoRALinear] = {}

    def set_context(
        self,
        features_by_layer: dict[str, torch.Tensor],
        prototypes_by_layer: dict[str, torch.Tensor] | None = None,
        domain_id: str | None = None,
    ) -> None:
        self.context = SCForwardContext(
            features_by_layer=features_by_layer,
            prototypes_by_layer=prototypes_by_layer,
            domain_id=domain_id,
        )

    def clear_context(self) -> None:
        self.context = None

    def _get_context(self) -> SCForwardContext | None:
        return self.context

    def init_sc_lora_bank(
        self,
        hook_layers: list[str],
        target_modules_per_layer: dict[str, list[str]],
        sae_feature_dims: dict[str, int],
        cfg: SCLoRAConfig,
    ) -> None:
        for layer in hook_layers:
            modules = target_modules_per_layer.get(layer, [])
            feature_dim = sae_feature_dims[layer]

            for module_path in modules:
                parent, child_name = resolve_parent(self.backbone, module_path)
                original = getattr(parent, child_name)
                if not isinstance(original, nn.Linear):
                    raise TypeError(
                        f"SC-LoRA wrapper currently supports nn.Linear only. Got {type(original)} for '{module_path}'."
                    )

                bank = SCLoRAAdapterBank(
                    d_in=original.in_features,
                    d_out=original.out_features,
                    sae_feature_dim=feature_dim,
                    cfg=cfg,
                )
                wrapper = SCLoRALinear(
                    base_linear=original,
                    bank=bank,
                    layer_name=layer,
                    context_getter=self._get_context,
                )

                setattr(parent, child_name, wrapper)
                self.wrappers[module_path] = wrapper

        for param in self.backbone.parameters():
            param.requires_grad = False

        for wrapper in self.wrappers.values():
            for param in wrapper.bank.parameters():
                param.requires_grad = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for wrapper in self.wrappers.values():
            params.extend([p for p in wrapper.bank.parameters() if p.requires_grad])
        return params

    def infer_controllers(self, features_by_layer: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        gates: dict[str, torch.Tensor] = {}
        ranks: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for module_path, wrapper in self.wrappers.items():
                z = features_by_layer[wrapper.layer_name]
                z_flat = z.reshape(-1, z.shape[-1])
                gate, rank, _ = wrapper.bank.controller_forward(z_flat)
                gates[module_path] = gate
                ranks[module_path] = rank

        return gates, ranks

    def apply_memory_policies(self, step: int | None = None) -> dict[str, float]:
        events = {"spawned": 0.0, "merged": 0.0, "pruned": 0.0, "active_specialists": 0.0}
        if not self.wrappers:
            return events

        active_counts: list[float] = []
        for wrapper in self.wrappers.values():
            bank_events = wrapper.bank.step_memory_policy(step=step)
            events["spawned"] += float(bank_events.get("spawned", 0.0))
            events["merged"] += float(bank_events.get("merged", 0.0))
            events["pruned"] += float(bank_events.get("pruned", 0.0))
            active_counts.append(float(bank_events.get("active_specialists", 0.0)))

        events["active_specialists"] = sum(active_counts) / len(active_counts)
        return events

    def memory_snapshot(self) -> dict[str, dict[str, torch.Tensor]]:
        snap: dict[str, dict[str, torch.Tensor]] = {}
        for module_path, wrapper in self.wrappers.items():
            bank = wrapper.bank
            snap[module_path] = {
                "active_mask": bank.active_mask.detach().clone(),
                "usage_ema": bank.usage_ema.detach().clone(),
                "prototype_count": bank.prototype_count.detach().clone(),
            }
        return snap

    def collect_adapter_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        return {name: wrapper.last_stats for name, wrapper in self.wrappers.items() if wrapper.last_stats}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone(*args, **kwargs)


