from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import random

import torch

from .data import batch_size_of_inputs, split_batch


def _detach_cpu_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_cpu_tree(v) for v in obj)
    return obj


def _pad_value_for(path: str, dtype: torch.dtype) -> float | int | bool:
    key = path.split(".")[-1]
    if key == "labels":
        return -100
    if key in {"attention_mask", "token_type_ids", "input_ids", "position_ids"}:
        return 0
    if dtype == torch.bool:
        return False
    return 0


def _right_pad_to_match(t: torch.Tensor, target_shape: torch.Size, pad_value: float | int | bool) -> torch.Tensor:
    if list(t.shape[1:]) == list(target_shape[1:]):
        return t

    out = torch.full(
        size=(t.shape[0], *target_shape[1:]),
        fill_value=pad_value,
        dtype=t.dtype,
        device=t.device,
    )
    slices = (slice(None),) + tuple(slice(0, dim) for dim in t.shape[1:])
    out[slices] = t
    return out


def _cat_tree(a: Any, b: Any, path: str = "") -> Any:
    if a is None:
        return b
    if b is None:
        return a

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.ndim != b.ndim:
            raise ValueError(f"Tensor rank mismatch at '{path}': {a.shape} vs {b.shape}")

        if a.ndim <= 1:
            return torch.cat([a, b], dim=0)

        target_shape = torch.Size([0] + [max(a.shape[d], b.shape[d]) for d in range(1, a.ndim)])
        pad_value = _pad_value_for(path, a.dtype)
        a_pad = _right_pad_to_match(a, target_shape, pad_value)
        b_pad = _right_pad_to_match(b, target_shape, pad_value)
        return torch.cat([a_pad, b_pad], dim=0)

    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for key in a.keys() | b.keys():
            next_path = f"{path}.{key}" if path else str(key)
            out[key] = _cat_tree(a.get(key), b.get(key), path=next_path)
        return out

    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return tuple(_cat_tree(x, y, path=f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))

    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_cat_tree(x, y, path=f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))]

    raise TypeError(f"Unsupported batch merge type: {type(a)} vs {type(b)} at path '{path}'")


def _slice_tree(obj: Any, n: int) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj[:n]
    if isinstance(obj, dict):
        return {k: _slice_tree(v, n) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_slice_tree(v, n) for v in obj)
    if isinstance(obj, list):
        return [_slice_tree(v, n) for v in obj]
    return obj


class BalancedReplayBuffer:
    def __init__(self, max_batches_per_domain: int = 128):
        self.max_batches_per_domain = max_batches_per_domain
        self.data: dict[str, deque[Any]] = defaultdict(lambda: deque(maxlen=max_batches_per_domain))

    def add(self, batch: Any, domain_id: str) -> None:
        self.data[domain_id].append(_detach_cpu_tree(batch))

    def sample(self, num_batches: int = 1) -> Any | None:
        domains = [d for d, items in self.data.items() if len(items) > 0]
        if not domains:
            return None

        mixed = None
        for _ in range(num_batches):
            domain = random.choice(domains)
            item = random.choice(list(self.data[domain]))
            mixed = _cat_tree(mixed, item)

        return mixed


def merge_batches(current_batch: Any, replay_batch: Any | None, alpha: float) -> Any:
    if replay_batch is None or alpha <= 0.0:
        return current_batch

    cur_inputs, _ = split_batch(current_batch)
    batch_size = batch_size_of_inputs(cur_inputs)
    replay_needed = max(1, int(batch_size * alpha))
    replay_batch = _slice_tree(replay_batch, replay_needed)

    if isinstance(current_batch, dict) and isinstance(replay_batch, dict):
        return _cat_tree(current_batch, replay_batch)

    if isinstance(current_batch, (tuple, list)) and isinstance(replay_batch, (tuple, list)):
        return _cat_tree(tuple(current_batch), tuple(replay_batch))

    if isinstance(current_batch, torch.Tensor) and isinstance(replay_batch, torch.Tensor):
        return torch.cat([current_batch, replay_batch], dim=0)

    raise TypeError(
        f"Unsupported current/replay batch types: {type(current_batch)} and {type(replay_batch)}"
    )
