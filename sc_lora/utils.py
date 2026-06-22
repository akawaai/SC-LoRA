from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def get_module_dict(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


def resolve_module(model: nn.Module, path: str) -> nn.Module:
    modules = get_module_dict(model)
    if path not in modules:
        raise KeyError(f"Module path '{path}' not found in model.")
    return modules[path]


def resolve_parent(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    if "." not in path:
        return model, path
    parent_path, child_name = path.rsplit(".", 1)
    parent = resolve_module(model, parent_path)
    return parent, child_name


def to_device_tree(obj: Any, device: torch.device | str) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: to_device_tree(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device_tree(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device_tree(v, device) for v in obj)
    return obj


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"Expected tensor with >=2 dims, got shape {tuple(x.shape)}")
    if x.ndim == 2:
        return x
    return x.reshape(-1, x.shape[-1])


def get_attention_mask(inputs: Any) -> torch.Tensor | None:
    if isinstance(inputs, Mapping):
        mask = inputs.get("attention_mask")
        if isinstance(mask, torch.Tensor):
            return mask
    return None


def token_mask_from_attention(x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    if attention_mask is None or x.ndim < 2:
        return None

    mask = attention_mask
    if not isinstance(mask, torch.Tensor):
        return None

    if x.ndim >= 3 and mask.ndim == 2 and mask.shape[0] == x.shape[0] and mask.shape[1] == x.shape[1]:
        return mask.reshape(-1) > 0

    x_flat = flatten_tokens(x)
    if mask.ndim == 1 and mask.shape[0] == x_flat.shape[0]:
        return mask > 0

    return None


def flatten_tokens_with_attention(
    x: torch.Tensor,
    attention_mask: torch.Tensor | None,
    drop_padding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x_flat = flatten_tokens(x)
    token_mask = token_mask_from_attention(x, attention_mask)
    if token_mask is None:
        return x_flat, None

    token_mask = token_mask.to(device=x_flat.device, dtype=torch.bool)

    if drop_padding and bool(token_mask.any().item()):
        return x_flat[token_mask], token_mask

    x_masked = x_flat * token_mask.to(dtype=x_flat.dtype).unsqueeze(-1)
    return x_masked, token_mask


def extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    raise TypeError(f"Could not extract tensor from output type: {type(output)}")


def maybe_stack(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("Empty tensor list.")
    return torch.cat(values, dim=0)


def ensure_dir(path: str) -> None:
    import os

    os.makedirs(path, exist_ok=True)
