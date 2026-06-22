from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from .utils import extract_tensor, resolve_module


@dataclass
class HookResult:
    outputs: Any
    activations: dict[str, torch.Tensor]


class ActivationStore:
    def __init__(self, model: nn.Module, layers: list[str], detach: bool = True):
        self.model = model
        self.layers = layers
        self.detach = detach
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _make_hook(self, name: str) -> Callable:
        def hook_fn(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = extract_tensor(output)
            self.activations[name] = tensor.detach() if self.detach else tensor

        return hook_fn

    def register(self) -> None:
        self.clear()
        for layer in self.layers:
            module = resolve_module(self.model, layer)
            handle = module.register_forward_hook(self._make_hook(layer))
            self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def clear(self) -> None:
        self.activations = {}

    def __enter__(self) -> "ActivationStore":
        self.register()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.remove()


def run_model(model: nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, Mapping):
        return model(**dict(inputs))
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


def forward_with_hooks(model: nn.Module, inputs: Any, layers: list[str], detach: bool = True) -> HookResult:
    with ActivationStore(model, layers=layers, detach=detach) as store:
        outputs = run_model(model, inputs)
        activations = {k: v for k, v in store.activations.items()}
    return HookResult(outputs=outputs, activations=activations)
