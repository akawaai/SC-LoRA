from __future__ import annotations

from typing import Any

import torch

from .data import split_batch
from .hooks import forward_with_hooks
from .sae import SparseAutoencoder
from .utils import ensure_dir, flatten_tokens_with_attention, get_attention_mask, to_device_tree


def build_steering_prototypes(
    base_model: torch.nn.Module,
    saes: dict[str, SparseAutoencoder],
    domain_masks: dict[str, dict[str, torch.Tensor]],
    d_domains: dict[str, Any],
    hook_layers: list[str],
    save_path: str,
    device: torch.device | str = "cpu",
    full_covariance: bool = False,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    base_model.eval()

    prototype: dict[str, dict[str, torch.Tensor]] = {domain_id: {} for domain_id in d_domains.keys()}
    covariance: dict[str, dict[str, torch.Tensor]] = {domain_id: {} for domain_id in d_domains.keys()}

    with torch.no_grad():
        for domain_id, loader in d_domains.items():
            z_collect: dict[str, list[torch.Tensor]] = {layer: [] for layer in hook_layers}

            for batch in loader:
                inputs, _ = split_batch(batch)
                inputs = to_device_tree(inputs, device)
                attention_mask = get_attention_mask(inputs)
                out = forward_with_hooks(base_model, inputs, hook_layers)

                for layer in hook_layers:
                    h, _ = flatten_tokens_with_attention(
                        out.activations[layer],
                        attention_mask=attention_mask,
                        drop_padding=True,
                    )
                    h = h.to(device)
                    z = saes[layer].encode(h)
                    mask = domain_masks[layer][domain_id].to(device)
                    z_dom = z * mask
                    z_collect[layer].append(z_dom.detach().cpu())

            for layer in hook_layers:
                z_all = torch.cat(z_collect[layer], dim=0)
                prototype[domain_id][layer] = z_all.mean(dim=0)

                if full_covariance:
                    covariance[domain_id][layer] = torch.cov(z_all.T)
                else:
                    covariance[domain_id][layer] = z_all.var(dim=0, unbiased=False)

    ensure_dir(save_path)
    torch.save(prototype, f"{save_path}/prototypes.pt")
    torch.save(covariance, f"{save_path}/covariances.pt")

    return prototype, covariance
