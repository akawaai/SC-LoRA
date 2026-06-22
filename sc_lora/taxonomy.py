from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .data import split_batch
from .hooks import forward_with_hooks
from .sae import SparseAutoencoder
from .utils import ensure_dir, flatten_tokens_with_attention, get_attention_mask, to_device_tree


@dataclass
class FeatureUsageAccumulator:
    feature_dim: int
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.activation_sum = torch.zeros(self.feature_dim)
        self.active_count = torch.zeros(self.feature_dim)
        self.token_count = 0

    def accumulate(self, z: torch.Tensor) -> None:
        z_cpu = z.detach().abs().cpu()
        self.activation_sum += z_cpu.sum(dim=0)
        self.active_count += (z_cpu > self.eps).float().sum(dim=0)
        self.token_count += z_cpu.shape[0]

    @property
    def mean_activation(self) -> torch.Tensor:
        return self.activation_sum / max(1, self.token_count)

    @property
    def frequency(self) -> torch.Tensor:
        return self.active_count / max(1, self.token_count)


@dataclass
class BaseVsInstAccumulator:
    feature_dim: int

    def __post_init__(self) -> None:
        self.diff_sum = torch.zeros(self.feature_dim)
        self.count = 0

    def accumulate(self, z0: torch.Tensor, z1: torch.Tensor) -> None:
        diff = (z0 - z1).abs().detach().cpu().mean(dim=0)
        self.diff_sum += diff
        self.count += 1

    @property
    def mean_drift(self) -> torch.Tensor:
        return self.diff_sum / max(1, self.count)


def _encode_layer(
    sae: SparseAutoencoder,
    h: torch.Tensor,
    device: torch.device | str,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    x, _ = flatten_tokens_with_attention(h, attention_mask=attention_mask, drop_padding=True)
    x = x.to(device)
    with torch.no_grad():
        z = sae.encode(x)
    return z


def _select_general_mask(
    general_stats: FeatureUsageAccumulator,
    domain_stats: dict[str, FeatureUsageAccumulator],
    base_vs_inst: BaseVsInstAccumulator | None,
    min_keep: int = 64,
    freq_quantile: float = 0.7,
    max_domain_cv: float = 0.8,
    max_drift_quantile: float = 0.7,
) -> torch.Tensor:
    freq = general_stats.frequency
    freq_threshold = torch.quantile(freq, q=freq_quantile)
    keep_freq = freq >= freq_threshold

    if domain_stats:
        means = torch.stack([domain_stats[d].mean_activation for d in sorted(domain_stats.keys())], dim=0)
        mean = means.mean(dim=0)
        std = means.std(dim=0)
        cv = std / (mean + 1e-6)
        keep_domain = cv <= max_domain_cv
    else:
        keep_domain = torch.ones_like(keep_freq, dtype=torch.bool)

    if base_vs_inst is not None:
        drift = base_vs_inst.mean_drift
        drift_threshold = torch.quantile(drift, q=max_drift_quantile)
        keep_stable = drift <= drift_threshold
    else:
        keep_stable = torch.ones_like(keep_freq, dtype=torch.bool)

    mask = keep_freq & keep_domain & keep_stable
    if int(mask.sum().item()) < min_keep:
        topk = min(min_keep, freq.numel())
        idx = torch.topk(freq, k=topk).indices
        fallback = torch.zeros_like(mask)
        fallback[idx] = True
        mask = mask | fallback

    return mask.float()


def _select_domain_mask(
    domain_id: str,
    domain_stats: dict[str, FeatureUsageAccumulator],
    general_mask: torch.Tensor,
    min_keep: int = 64,
    contrast_quantile: float = 0.8,
) -> torch.Tensor:
    current = domain_stats[domain_id].mean_activation
    others = [v.mean_activation for k, v in domain_stats.items() if k != domain_id]

    if others:
        other_mean = torch.stack(others, dim=0).mean(dim=0)
    else:
        other_mean = torch.zeros_like(current)

    contrast = current - other_mean
    threshold = torch.quantile(contrast, q=contrast_quantile)
    keep_contrast = contrast >= threshold

    keep_not_general = general_mask < 0.5
    mask = keep_contrast & keep_not_general

    if int(mask.sum().item()) < min_keep:
        topk = min(min_keep, contrast.numel())
        idx = torch.topk(contrast, k=topk).indices
        fallback = torch.zeros_like(mask)
        fallback[idx] = True
        mask = mask | fallback

    return mask.float()


def build_feature_taxonomy(
    base_model: torch.nn.Module,
    instruction_model: torch.nn.Module | None,
    saes: dict[str, SparseAutoencoder],
    d_general: Any,
    d_domains: dict[str, Any],
    hook_layers: list[str],
    save_path: str,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    base_model.eval()
    if instruction_model is not None:
        instruction_model.eval()

    general_stats: dict[str, FeatureUsageAccumulator] = {}
    domain_stats: dict[str, dict[str, FeatureUsageAccumulator]] = {}
    base_vs_inst_stats: dict[str, BaseVsInstAccumulator] = {}

    for layer in hook_layers:
        fdim = saes[layer].feature_dim
        general_stats[layer] = FeatureUsageAccumulator(fdim)
        domain_stats[layer] = {dom: FeatureUsageAccumulator(fdim) for dom in d_domains.keys()}
        base_vs_inst_stats[layer] = BaseVsInstAccumulator(fdim)

    with torch.no_grad():
        for batch in d_general:
            inputs, _ = split_batch(batch)
            inputs = to_device_tree(inputs, device)
            attention_mask = get_attention_mask(inputs)
            out = forward_with_hooks(base_model, inputs, hook_layers)
            for layer in hook_layers:
                z = _encode_layer(
                    saes[layer],
                    out.activations[layer],
                    device=device,
                    attention_mask=attention_mask,
                )
                general_stats[layer].accumulate(z)

        for domain_id, loader in d_domains.items():
            for batch in loader:
                inputs, _ = split_batch(batch)
                inputs = to_device_tree(inputs, device)
                attention_mask = get_attention_mask(inputs)
                out = forward_with_hooks(base_model, inputs, hook_layers)
                for layer in hook_layers:
                    z = _encode_layer(
                        saes[layer],
                        out.activations[layer],
                        device=device,
                        attention_mask=attention_mask,
                    )
                    domain_stats[layer][domain_id].accumulate(z)

        if instruction_model is not None:
            for batch in d_general:
                inputs, _ = split_batch(batch)
                inputs = to_device_tree(inputs, device)
                attention_mask = get_attention_mask(inputs)
                out_base = forward_with_hooks(base_model, inputs, hook_layers)
                out_inst = forward_with_hooks(instruction_model, inputs, hook_layers)
                for layer in hook_layers:
                    z0 = _encode_layer(
                        saes[layer],
                        out_base.activations[layer],
                        device=device,
                        attention_mask=attention_mask,
                    )
                    z1 = _encode_layer(
                        saes[layer],
                        out_inst.activations[layer],
                        device=device,
                        attention_mask=attention_mask,
                    )
                    base_vs_inst_stats[layer].accumulate(z0, z1)

    general_mask: dict[str, torch.Tensor] = {}
    domain_mask: dict[str, dict[str, torch.Tensor]] = {}

    for layer in hook_layers:
        drift_stats = base_vs_inst_stats[layer] if instruction_model is not None else None
        general_mask[layer] = _select_general_mask(
            general_stats=general_stats[layer],
            domain_stats=domain_stats[layer],
            base_vs_inst=drift_stats,
        )

        domain_mask[layer] = {}
        for domain_id in d_domains.keys():
            domain_mask[layer][domain_id] = _select_domain_mask(
                domain_id=domain_id,
                domain_stats=domain_stats[layer],
                general_mask=general_mask[layer],
            )

    ensure_dir(save_path)
    torch.save(general_mask, f"{save_path}/general_mask.pt")
    torch.save(domain_mask, f"{save_path}/domain_mask.pt")

    return general_mask, domain_mask
