from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .data import split_batch
from .hooks import forward_with_hooks
from .losses import compute_task_loss
from .lora import SCLoRAModel
from .sae import SparseAutoencoder
from .utils import flatten_tokens_with_attention, get_attention_mask, to_device_tree


def _sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _to_finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _encode_layerwise(
    saes: dict[str, SparseAutoencoder],
    activations: dict[str, torch.Tensor],
    hook_layers: list[str],
    device: torch.device | str,
    attention_mask: torch.Tensor | None = None,
    drop_padding: bool = False,
) -> dict[str, torch.Tensor]:
    z: dict[str, torch.Tensor] = {}
    for layer in hook_layers:
        h, token_mask = flatten_tokens_with_attention(
            activations[layer],
            attention_mask=attention_mask,
            drop_padding=drop_padding,
        )
        h = h.to(device)
        token_mask_dev: torch.Tensor | None = None
        if token_mask is not None:
            token_mask_dev = token_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
        with torch.no_grad():
            z_layer = _sanitize_tensor(saes[layer].encode(h))
        if token_mask_dev is not None and not drop_padding:
            z_layer = _sanitize_tensor(z_layer * token_mask_dev)
        z[layer] = z_layer
    return z


def _extract_logits(outputs: Any) -> torch.Tensor | None:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits") and isinstance(outputs.logits, torch.Tensor):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs and isinstance(outputs["logits"], torch.Tensor):
        return outputs["logits"]
    return None


def _default_task_metric(outputs: Any, targets: torch.Tensor | None) -> float:
    logits = _extract_logits(outputs)
    if logits is not None and targets is not None and logits.ndim == targets.ndim + 1:
        pred = logits.argmax(dim=-1)
        return _to_finite_float((pred == targets).float().mean().item())

    if targets is not None:
        loss = compute_task_loss(outputs, targets)
        return _to_finite_float((-loss).item())

    return 0.0


def _aggregate(values: list[float]) -> float:
    finite = [_to_finite_float(v, default=float("nan")) for v in values]
    finite = [v for v in finite if math.isfinite(v)]
    if not finite:
        return 0.0
    return float(sum(finite) / len(finite))


def _entropy_from_probs(probs: torch.Tensor) -> float:
    if probs.numel() == 0:
        return 0.0
    probs = _sanitize_tensor(probs)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    eps = 1e-8
    entropy = (-(probs * (probs + eps).log()).sum(dim=-1).mean()).item()
    return _to_finite_float(entropy)


def _concat_retrieved_topk(tensors: list[torch.Tensor], pad_value: int = -1) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot concatenate empty retrieved_topk tensor list.")

    normalized: list[torch.Tensor] = []
    max_width = 1

    for tensor in tensors:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        elif tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        if tensor.shape[-1] > max_width:
            max_width = int(tensor.shape[-1])
        normalized.append(tensor)

    padded: list[torch.Tensor] = []
    for tensor in normalized:
        width = int(tensor.shape[-1])
        if width < max_width:
            fill = torch.full(
                (tensor.shape[0], max_width - width),
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([tensor, fill], dim=-1)
        padded.append(tensor)

    return torch.cat(padded, dim=0)


def _module_to_layer(module_path: str, hook_layers: list[str]) -> str | None:
    for layer in hook_layers:
        if module_path.startswith(layer + "."):
            return layer
    return None


def _mean_stacked_tokens(
    tensors: list[torch.Tensor],
    expected_tokens: int | None = None,
) -> torch.Tensor | None:
    if not tensors:
        return None

    min_tokens = min(int(t.shape[0]) for t in tensors)
    if expected_tokens is not None:
        min_tokens = min(min_tokens, int(expected_tokens))
    if min_tokens <= 0:
        return None

    trimmed = [t[:min_tokens] for t in tensors]
    return _sanitize_tensor(torch.stack(trimmed, dim=0).mean(dim=0))


def _group_adapter_stats_by_layer(
    adapter_stats: dict[str, dict[str, torch.Tensor]],
    hook_layers: list[str],
) -> dict[str, dict[str, list[torch.Tensor]]]:
    grouped: dict[str, dict[str, list[torch.Tensor]]] = {
        layer: {
            "gate": [],
            "rank_mask": [],
            "retrieved_topk": [],
            "active_mask": [],
            "usage_ema": [],
        }
        for layer in hook_layers
    }

    for module_path, stats in adapter_stats.items():
        layer = _module_to_layer(module_path, hook_layers)
        if layer is None:
            continue

        gate = stats.get("gate")
        if isinstance(gate, torch.Tensor):
            grouped[layer]["gate"].append(_sanitize_tensor(gate))

        rank_mask = stats.get("rank_mask")
        if isinstance(rank_mask, torch.Tensor):
            grouped[layer]["rank_mask"].append(_sanitize_tensor(rank_mask))

        retrieved_topk = stats.get("retrieved_topk")
        if isinstance(retrieved_topk, torch.Tensor):
            grouped[layer]["retrieved_topk"].append(retrieved_topk)

        active_mask = stats.get("active_mask")
        if isinstance(active_mask, torch.Tensor):
            grouped[layer]["active_mask"].append(_sanitize_tensor(active_mask))

        usage_ema = stats.get("usage_ema")
        if isinstance(usage_ema, torch.Tensor):
            grouped[layer]["usage_ema"].append(_sanitize_tensor(usage_ema))

    return grouped

def evaluate_contextual_specialization(
    model_theta: SCLoRAModel,
    saes: dict[str, SparseAutoencoder],
    eval_sets: dict[str, Any],
    hook_layers: list[str],
    frozen_backbone: torch.nn.Module,
    device: torch.device | str = "cpu",
    task_metric: Callable[[Any, torch.Tensor | None], float] | None = None,
    domain_to_specialist: dict[str, int] | None = None,
    prototypes: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, dict[str, float]]:
    metric_fn = task_metric or _default_task_metric

    model_theta.eval()
    frozen_backbone.eval()

    results: dict[str, dict[str, list[float]]] = {}

    with torch.no_grad():
        for domain_id, loader in eval_sets.items():
            results[domain_id] = {
                "task_score": [],
                "gate_usage": [],
                "rank_usage": [],
                "routing_accuracy": [],
                "retrieval_consistency": [],
                "active_specialists": [],
                "gate_entropy": [],
            }

            for batch in loader:
                inputs, targets = split_batch(batch)
                inputs = to_device_tree(inputs, device)
                attention_mask = get_attention_mask(inputs)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(device)

                out_base = forward_with_hooks(frozen_backbone, inputs, hook_layers, detach=False)
                z = _encode_layerwise(
                    saes,
                    out_base.activations,
                    hook_layers,
                    device=device,
                    attention_mask=attention_mask,
                    drop_padding=False,
                )

                proto_layer = prototypes.get(domain_id, None) if prototypes is not None else None
                model_theta.set_context(z, prototypes_by_layer=proto_layer, domain_id=domain_id)
                out_tuned = model_theta(**inputs) if isinstance(inputs, dict) else model_theta(inputs)
                adapter_stats = model_theta.collect_adapter_stats()
                model_theta.clear_context()

                results[domain_id]["task_score"].append(_to_finite_float(metric_fn(out_tuned, targets)))

                if adapter_stats:
                    gate_all = _sanitize_tensor(torch.cat([s["gate"] for s in adapter_stats.values()], dim=0))
                    rank_all = _sanitize_tensor(torch.cat([s["rank_mask"] for s in adapter_stats.values()], dim=0))
                    results[domain_id]["gate_usage"].append(_to_finite_float(gate_all.mean().item()))
                    results[domain_id]["rank_usage"].append(_to_finite_float(rank_all.sum(dim=-1).mean().item()))
                    results[domain_id]["gate_entropy"].append(_entropy_from_probs(gate_all))

                    active_counts: list[float] = []
                    for stats in adapter_stats.values():
                        active_mask = stats.get("active_mask")
                        if active_mask is not None:
                            active_counts.append(_to_finite_float(active_mask.sum().item()))
                    if active_counts:
                        results[domain_id]["active_specialists"].append(sum(active_counts) / len(active_counts))

                    retrieved_list: list[torch.Tensor] = []
                    for stats in adapter_stats.values():
                        retrieved = stats.get("retrieved_topk")
                        if isinstance(retrieved, torch.Tensor):
                            retrieved_list.append(retrieved)
                    if retrieved_list:
                        retrieved_all = _concat_retrieved_topk(retrieved_list)
                        top1 = gate_all.argmax(dim=-1, keepdim=True)
                        consistency = (retrieved_all == top1).any(dim=-1).float().mean().item()
                        results[domain_id]["retrieval_consistency"].append(_to_finite_float(consistency))

                    if domain_to_specialist is not None and domain_id in domain_to_specialist:
                        expected = domain_to_specialist[domain_id]
                        pred = gate_all.argmax(dim=-1)
                        acc = (pred == expected).float().mean().item()
                        results[domain_id]["routing_accuracy"].append(_to_finite_float(acc))

    aggregated: dict[str, dict[str, float]] = {}
    for domain_id, metrics in results.items():
        aggregated[domain_id] = {k: _aggregate(v) for k, v in metrics.items()}

    return aggregated


def evaluate_memory_routing(
    model_theta: SCLoRAModel,
    saes: dict[str, SparseAutoencoder],
    eval_sets: dict[str, Any],
    hook_layers: list[str],
    frozen_backbone: torch.nn.Module,
    device: torch.device | str = "cpu",
    domain_to_specialist: dict[str, int] | None = None,
    prototypes: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, dict[str, float]]:
    model_theta.eval()
    frozen_backbone.eval()

    results: dict[str, dict[str, list[float]]] = {}

    with torch.no_grad():
        for domain_id, loader in eval_sets.items():
            results[domain_id] = {
                "retrieval_hit_at_k": [],
                "retrieval_consistency": [],
                "routing_top1_accuracy": [],
                "active_specialists": [],
                "mean_usage_ema": [],
            }

            for batch in loader:
                inputs, _ = split_batch(batch)
                inputs = to_device_tree(inputs, device)
                attention_mask = get_attention_mask(inputs)

                out_base = forward_with_hooks(frozen_backbone, inputs, hook_layers, detach=False)
                z = _encode_layerwise(
                    saes,
                    out_base.activations,
                    hook_layers,
                    device=device,
                    attention_mask=attention_mask,
                    drop_padding=False,
                )

                proto_layer = prototypes.get(domain_id, None) if prototypes is not None else None
                model_theta.set_context(z, prototypes_by_layer=proto_layer, domain_id=domain_id)
                _ = model_theta(**inputs) if isinstance(inputs, dict) else model_theta(inputs)
                adapter_stats = model_theta.collect_adapter_stats()
                model_theta.clear_context()

                if not adapter_stats:
                    continue

                gate_all = _sanitize_tensor(torch.cat([s["gate"] for s in adapter_stats.values()], dim=0))
                top1 = gate_all.argmax(dim=-1)

                retrieved_tensors = [s["retrieved_topk"] for s in adapter_stats.values() if "retrieved_topk" in s]
                if retrieved_tensors:
                    retrieved_all = _concat_retrieved_topk(retrieved_tensors)
                    consistency = (retrieved_all == top1.unsqueeze(-1)).any(dim=-1).float().mean().item()
                    results[domain_id]["retrieval_consistency"].append(_to_finite_float(consistency))

                    if domain_to_specialist is not None and domain_id in domain_to_specialist:
                        expected = int(domain_to_specialist[domain_id])
                        hit_at_k = (retrieved_all == expected).any(dim=-1).float().mean().item()
                        results[domain_id]["retrieval_hit_at_k"].append(_to_finite_float(hit_at_k))

                if domain_to_specialist is not None and domain_id in domain_to_specialist:
                    expected = int(domain_to_specialist[domain_id])
                    routing_acc = (top1 == expected).float().mean().item()
                    results[domain_id]["routing_top1_accuracy"].append(_to_finite_float(routing_acc))

                active_counts: list[float] = []
                usage_values: list[float] = []
                for stats in adapter_stats.values():
                    if "active_mask" in stats:
                        active_counts.append(_to_finite_float(stats["active_mask"].sum().item()))
                    if "usage_ema" in stats:
                        usage_values.append(_to_finite_float(_sanitize_tensor(stats["usage_ema"]).mean().item()))

                if active_counts:
                    results[domain_id]["active_specialists"].append(sum(active_counts) / len(active_counts))
                if usage_values:
                    results[domain_id]["mean_usage_ema"].append(sum(usage_values) / len(usage_values))

    aggregated: dict[str, dict[str, float]] = {}
    for domain_id, metrics in results.items():
        aggregated[domain_id] = {k: _aggregate(v) for k, v in metrics.items()}

    return aggregated


def _normalize_histogram(hist: torch.Tensor) -> list[float]:
    hist = _sanitize_tensor(hist.reshape(-1).float())
    total = float(hist.sum().item())
    if total <= 1e-12:
        if hist.numel() == 0:
            return []
        uniform = 1.0 / float(hist.numel())
        return [uniform for _ in range(int(hist.numel()))]
    return [
        _to_finite_float(v)
        for v in (hist / total).detach().cpu().tolist()
    ]


def evaluate_routing_diagnostics(
    model_theta: SCLoRAModel,
    saes: dict[str, SparseAutoencoder],
    eval_sets: dict[str, Any],
    hook_layers: list[str],
    frozen_backbone: torch.nn.Module,
    device: torch.device | str = "cpu",
    domain_to_specialist: dict[str, int] | None = None,
    prototypes: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    model_theta.eval()
    frozen_backbone.eval()

    raw: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    with torch.no_grad():
        for domain_id, loader in eval_sets.items():
            raw[domain_id] = {layer: {} for layer in hook_layers}

            for batch in loader:
                inputs, _ = split_batch(batch)
                inputs = to_device_tree(inputs, device)
                attention_mask = get_attention_mask(inputs)

                out_base = forward_with_hooks(frozen_backbone, inputs, hook_layers, detach=False)
                z = _encode_layerwise(
                    saes,
                    out_base.activations,
                    hook_layers,
                    device=device,
                    attention_mask=attention_mask,
                    drop_padding=False,
                )

                proto_layer = prototypes.get(domain_id, None) if prototypes is not None else None
                model_theta.set_context(z, prototypes_by_layer=proto_layer, domain_id=domain_id)
                _ = model_theta(**inputs) if isinstance(inputs, dict) else model_theta(inputs)
                adapter_stats = model_theta.collect_adapter_stats()
                model_theta.clear_context()

                for module_path, module_stats in adapter_stats.items():
                    layer = _module_to_layer(module_path, hook_layers)
                    if layer is None:
                        continue

                    gate = module_stats.get("gate")
                    rank_mask = module_stats.get("rank_mask")
                    if not isinstance(gate, torch.Tensor) or not isinstance(rank_mask, torch.Tensor):
                        continue

                    if gate.ndim == 1:
                        gate = gate.unsqueeze(-1)
                    elif gate.ndim > 2:
                        gate = gate.reshape(-1, gate.shape[-1])

                    if rank_mask.ndim == 1:
                        rank_mask = rank_mask.unsqueeze(-1)
                    elif rank_mask.ndim > 2:
                        rank_mask = rank_mask.reshape(-1, rank_mask.shape[-1])

                    gate = _sanitize_tensor(gate)
                    rank_mask = _sanitize_tensor(rank_mask)

                    n_tokens = min(int(gate.shape[0]), int(rank_mask.shape[0]))
                    if n_tokens <= 0:
                        continue

                    gate = gate[:n_tokens]
                    rank_mask = rank_mask[:n_tokens]

                    top1 = gate.argmax(dim=-1)
                    gate_max = _sanitize_tensor(gate.max(dim=-1).values)
                    rank_usage = _sanitize_tensor(rank_mask.sum(dim=-1))
                    usage_vec = _sanitize_tensor(gate.mean(dim=0))
                    hist = torch.bincount(top1, minlength=int(gate.shape[-1])).float()

                    expected_acc = None
                    if domain_to_specialist is not None and domain_id in domain_to_specialist:
                        expected = int(domain_to_specialist[domain_id])
                        expected_acc = _to_finite_float((top1 == expected).float().mean().item())

                    retrieval_consistency = None
                    retrieved = module_stats.get("retrieved_topk")
                    if isinstance(retrieved, torch.Tensor):
                        if retrieved.ndim == 1:
                            retrieved = retrieved.unsqueeze(-1)
                        elif retrieved.ndim > 2:
                            retrieved = retrieved.reshape(-1, retrieved.shape[-1])
                        retrieved = retrieved[:n_tokens]
                        retrieval_consistency = _to_finite_float(
                            (retrieved == top1.unsqueeze(-1)).any(dim=-1).float().mean().item()
                        )

                    active_specialists = None
                    active_mask = module_stats.get("active_mask")
                    if isinstance(active_mask, torch.Tensor):
                        active_specialists = _to_finite_float(active_mask.sum().item())

                    usage_ema_mean = None
                    usage_ema = module_stats.get("usage_ema")
                    if isinstance(usage_ema, torch.Tensor):
                        usage_ema_mean = _to_finite_float(_sanitize_tensor(usage_ema).mean().item())

                    mod_state = raw[domain_id][layer].setdefault(
                        module_path,
                        {
                            "gate_entropy": [],
                            "gate_max_mean": [],
                            "gate_max_std": [],
                            "rank_usage_mean": [],
                            "retrieval_consistency": [],
                            "routing_top1_accuracy": [],
                            "active_specialists": [],
                            "usage_ema_mean": [],
                            "usage_vectors": [],
                            "top1_hist": [],
                        },
                    )

                    mod_state["gate_entropy"].append(_entropy_from_probs(gate))
                    mod_state["gate_max_mean"].append(_to_finite_float(gate_max.mean().item()))
                    mod_state["gate_max_std"].append(_to_finite_float(gate_max.std(unbiased=False).item()))
                    mod_state["rank_usage_mean"].append(_to_finite_float(rank_usage.mean().item()))
                    if retrieval_consistency is not None:
                        mod_state["retrieval_consistency"].append(retrieval_consistency)
                    if expected_acc is not None:
                        mod_state["routing_top1_accuracy"].append(expected_acc)
                    if active_specialists is not None:
                        mod_state["active_specialists"].append(active_specialists)
                    if usage_ema_mean is not None:
                        mod_state["usage_ema_mean"].append(usage_ema_mean)
                    mod_state["usage_vectors"].append(usage_vec.detach().cpu())
                    mod_state["top1_hist"].append(hist.detach().cpu())

    aggregated: dict[str, Any] = {}
    for domain_id, per_layer in raw.items():
        layer_out: dict[str, Any] = {}
        global_metrics = {
            "gate_entropy": [],
            "gate_max_mean": [],
            "rank_usage_mean": [],
            "retrieval_consistency": [],
            "routing_top1_accuracy": [],
        }

        for layer in hook_layers:
            per_module = per_layer.get(layer, {})
            module_metrics: dict[str, Any] = {}

            layer_gate_entropy: list[float] = []
            layer_gate_max_mean: list[float] = []
            layer_gate_max_std: list[float] = []
            layer_rank_usage_mean: list[float] = []
            layer_retrieval_consistency: list[float] = []
            layer_top1_acc: list[float] = []
            layer_active_specialists: list[float] = []
            layer_usage_ema_mean: list[float] = []
            layer_usage_vectors: list[torch.Tensor] = []
            layer_top1_hists: list[torch.Tensor] = []

            for module_path in sorted(per_module.keys()):
                mod_state = per_module[module_path]
                usage_vectors = mod_state.get("usage_vectors", [])
                top1_hists = mod_state.get("top1_hist", [])

                if usage_vectors:
                    usage_tensor = _sanitize_tensor(torch.stack(usage_vectors, dim=0).mean(dim=0))
                else:
                    usage_tensor = torch.tensor([], dtype=torch.float32)

                if top1_hists:
                    hist_tensor = _sanitize_tensor(torch.stack(top1_hists, dim=0).sum(dim=0))
                else:
                    hist_tensor = torch.tensor([], dtype=torch.float32)

                mod_result = {
                    "gate_entropy": _aggregate(mod_state.get("gate_entropy", [])),
                    "gate_max_mean": _aggregate(mod_state.get("gate_max_mean", [])),
                    "gate_max_std": _aggregate(mod_state.get("gate_max_std", [])),
                    "rank_usage_mean": _aggregate(mod_state.get("rank_usage_mean", [])),
                    "retrieval_consistency": _aggregate(mod_state.get("retrieval_consistency", [])),
                    "routing_top1_accuracy": _aggregate(mod_state.get("routing_top1_accuracy", [])),
                    "active_specialists": _aggregate(mod_state.get("active_specialists", [])),
                    "usage_ema_mean": _aggregate(mod_state.get("usage_ema_mean", [])),
                    "specialist_usage": [_to_finite_float(v) for v in usage_tensor.tolist()],
                    "top1_specialist_hist": _normalize_histogram(hist_tensor),
                }
                module_metrics[module_path] = mod_result

                layer_gate_entropy.append(mod_result["gate_entropy"])
                layer_gate_max_mean.append(mod_result["gate_max_mean"])
                layer_gate_max_std.append(mod_result["gate_max_std"])
                layer_rank_usage_mean.append(mod_result["rank_usage_mean"])
                layer_retrieval_consistency.append(mod_result["retrieval_consistency"])
                layer_top1_acc.append(mod_result["routing_top1_accuracy"])
                layer_active_specialists.append(mod_result["active_specialists"])
                layer_usage_ema_mean.append(mod_result["usage_ema_mean"])

                if usage_tensor.numel() > 0:
                    layer_usage_vectors.append(usage_tensor)
                if hist_tensor.numel() > 0:
                    layer_top1_hists.append(hist_tensor)

            if layer_usage_vectors:
                layer_usage = _sanitize_tensor(torch.stack(layer_usage_vectors, dim=0).mean(dim=0))
            else:
                layer_usage = torch.tensor([], dtype=torch.float32)

            if layer_top1_hists:
                layer_hist = _sanitize_tensor(torch.stack(layer_top1_hists, dim=0).sum(dim=0))
            else:
                layer_hist = torch.tensor([], dtype=torch.float32)

            layer_result = {
                "num_modules": len(module_metrics),
                "gate_entropy": _aggregate(layer_gate_entropy),
                "gate_max_mean": _aggregate(layer_gate_max_mean),
                "gate_max_std": _aggregate(layer_gate_max_std),
                "rank_usage_mean": _aggregate(layer_rank_usage_mean),
                "retrieval_consistency": _aggregate(layer_retrieval_consistency),
                "routing_top1_accuracy": _aggregate(layer_top1_acc),
                "active_specialists": _aggregate(layer_active_specialists),
                "usage_ema_mean": _aggregate(layer_usage_ema_mean),
                "specialist_usage": [_to_finite_float(v) for v in layer_usage.tolist()],
                "top1_specialist_hist": _normalize_histogram(layer_hist),
                "module_metrics": module_metrics,
            }

            layer_out[layer] = layer_result

            if layer_result["num_modules"] > 0:
                global_metrics["gate_entropy"].append(layer_result["gate_entropy"])
                global_metrics["gate_max_mean"].append(layer_result["gate_max_mean"])
                global_metrics["rank_usage_mean"].append(layer_result["rank_usage_mean"])
                global_metrics["retrieval_consistency"].append(layer_result["retrieval_consistency"])
                global_metrics["routing_top1_accuracy"].append(layer_result["routing_top1_accuracy"])

        aggregated[domain_id] = {
            "layers": layer_out,
            "global": {k: _aggregate(v) for k, v in global_metrics.items()},
        }

    return aggregated


def evaluate_sequential_forgetting(
    checkpoints: list[Any],
    eval_sets: dict[str, Any],
    evaluator: Callable[[Any, Any], float],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    domains = list(eval_sets.keys())

    score: dict[str, list[float]] = {domain: [] for domain in domains}
    forgetting: dict[str, list[float]] = {domain: [] for domain in domains}

    for checkpoint in checkpoints:
        for domain in domains:
            value = evaluator(checkpoint, eval_sets[domain])
            score[domain].append(value)

        for domain in domains:
            series = score[domain]
            best_before = max(series)
            forgetting[domain].append(best_before - series[-1])

    return score, forgetting


def _topk_overlap(a: torch.Tensor, b_mask: torch.Tensor, k: int = 64) -> float:
    a = _sanitize_tensor(a.reshape(-1))
    b_mask = _sanitize_tensor(b_mask.reshape(-1))
    k = min(k, a.numel(), b_mask.numel())
    if k <= 0:
        return 0.0
    idx_a = torch.topk(a, k=k).indices
    idx_b = torch.topk(b_mask, k=k).indices
    inter = len(set(idx_a.tolist()).intersection(set(idx_b.tolist())))
    return inter / float(k)


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = _sanitize_tensor(x.reshape(-1))
    y = _sanitize_tensor(y.reshape(-1))
    n = min(int(x.numel()), int(y.numel()))
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = (x_centered.std(unbiased=False) * y_centered.std(unbiased=False)) + 1e-8
    corr = ((x_centered * y_centered).mean() / denom).item()
    return _to_finite_float(corr)


def evaluate_representation_interference(
    model_theta: SCLoRAModel,
    frozen_base: torch.nn.Module,
    saes: dict[str, SparseAutoencoder],
    general_mask: dict[str, torch.Tensor],
    eval_sets: dict[str, Any],
    hook_layers: list[str],
    device: torch.device | str = "cpu",
    prototypes: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    model_theta.eval()
    frozen_base.eval()

    domain_layer_deltas: dict[str, dict[str, list[torch.Tensor]]] = {}

    with torch.no_grad():
        for domain_id, loader in eval_sets.items():
            domain_layer_deltas[domain_id] = {layer: [] for layer in hook_layers}

            for batch in loader:
                inputs, _ = split_batch(batch)
                inputs = to_device_tree(inputs, device)
                attention_mask = get_attention_mask(inputs)

                out_base = forward_with_hooks(frozen_base, inputs, hook_layers, detach=False)
                z_base = _encode_layerwise(
                    saes,
                    out_base.activations,
                    hook_layers,
                    device=device,
                    attention_mask=attention_mask,
                    drop_padding=False,
                )

                proto_layer = prototypes.get(domain_id, None) if prototypes is not None else None
                model_theta.set_context(z_base, prototypes_by_layer=proto_layer, domain_id=domain_id)
                out_ft = forward_with_hooks(model_theta.backbone, inputs, hook_layers, detach=False)
                z_ft = _encode_layerwise(
                    saes,
                    out_ft.activations,
                    hook_layers,
                    device=device,
                    attention_mask=attention_mask,
                    drop_padding=False,
                )
                model_theta.clear_context()

                for layer in hook_layers:
                    zf = _sanitize_tensor(z_ft[layer])
                    zb = _sanitize_tensor(z_base[layer])
                    gm = _sanitize_tensor(general_mask[layer].to(device))
                    delta = ((zf - zb) * gm).mean(dim=0).detach().cpu()
                    domain_layer_deltas[domain_id][layer].append(delta)

    domain_layer_mean: dict[str, dict[str, torch.Tensor]] = {}
    for domain_id, layer_values in domain_layer_deltas.items():
        domain_layer_mean[domain_id] = {}
        for layer, deltas in layer_values.items():
            if deltas:
                domain_layer_mean[domain_id][layer] = torch.stack(deltas, dim=0).mean(dim=0)
            else:
                feature_dim = int(general_mask[layer].shape[0])
                domain_layer_mean[domain_id][layer] = torch.zeros(feature_dim)

    pairwise: dict[str, dict[str, float]] = {}
    pairwise_abs: list[float] = []

    domains = sorted(domain_layer_mean.keys())
    for da, db in combinations(domains, 2):
        key = f"{da}__{db}"
        pairwise[key] = {}
        for layer in hook_layers:
            va = domain_layer_mean[da][layer]
            vb = domain_layer_mean[db][layer]
            va = _sanitize_tensor(va)
            vb = _sanitize_tensor(vb)

            denom = float(va.norm(p=2).item() * vb.norm(p=2).item())
            if denom <= 1e-12:
                cos = 0.0
            else:
                cos = _to_finite_float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0), dim=-1).item())
            pairwise[key][layer] = cos
            pairwise_abs.append(abs(cos))

    mean_norms: dict[str, float] = {}
    for domain_id, layer_map in domain_layer_mean.items():
        norms = [_to_finite_float(_sanitize_tensor(vec).norm(p=2).item()) for vec in layer_map.values()]
        mean_norms[domain_id] = _aggregate(norms)

    return {
        "pairwise_cosine_by_layer": pairwise,
        "mean_abs_pairwise_cosine": _aggregate(pairwise_abs),
        "mean_general_delta_norm_by_domain": mean_norms,
    }


def evaluate_mechanistic_alignment(
    model_theta: SCLoRAModel,
    frozen_base: torch.nn.Module,
    saes: dict[str, SparseAutoencoder],
    general_mask: dict[str, torch.Tensor],
    domain_mask: dict[str, dict[str, torch.Tensor]],
    eval_sets: dict[str, Any],
    hook_layers: list[str],
    device: torch.device | str = "cpu",
    prototypes: dict[str, dict[str, torch.Tensor]] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    model_theta.eval()
    frozen_base.eval()

    stats: dict[str, dict[str, dict[str, list[float]]]] = {}

    with torch.no_grad():
        for domain_id, loader in eval_sets.items():
            stats[domain_id] = {}

            for layer in hook_layers:
                stats[domain_id][layer] = {
                    "drift_general": [],
                    "drift_domain": [],
                    "topk_overlap": [],
                    "gate_rank_corr": [],
                    "gate_feature_corr": [],
                }

            for batch in loader:
                inputs, _ = split_batch(batch)
                inputs = to_device_tree(inputs, device)
                attention_mask = get_attention_mask(inputs)

                out_base = forward_with_hooks(frozen_base, inputs, hook_layers, detach=False)
                z_base = _encode_layerwise(
                    saes,
                    out_base.activations,
                    hook_layers,
                    device=device,
                    attention_mask=attention_mask,
                    drop_padding=False,
                )

                proto_layer = prototypes.get(domain_id, None) if prototypes is not None else None
                model_theta.set_context(z_base, prototypes_by_layer=proto_layer, domain_id=domain_id)
                out_ft = forward_with_hooks(model_theta.backbone, inputs, hook_layers, detach=False)
                z_ft = _encode_layerwise(
                    saes,
                    out_ft.activations,
                    hook_layers,
                    device=device,
                    attention_mask=attention_mask,
                    drop_padding=False,
                )
                adapter_stats = model_theta.collect_adapter_stats()
                grouped_by_layer = _group_adapter_stats_by_layer(adapter_stats, hook_layers)
                model_theta.clear_context()

                for layer in hook_layers:
                    zb = _sanitize_tensor(z_base[layer])
                    zf = _sanitize_tensor(z_ft[layer])
                    gm = _sanitize_tensor(general_mask[layer].to(device))
                    dm = _sanitize_tensor(domain_mask[layer][domain_id].to(device))

                    drift_general = _to_finite_float(torch.norm((zf - zb) * gm, p=2, dim=-1).mean().item())
                    drift_domain = _to_finite_float(torch.norm((zf - zb) * dm, p=2, dim=-1).mean().item())

                    topk_overlap = _topk_overlap(zf.mean(dim=0), dm, k=64)

                    layer_gate = _mean_stacked_tokens(
                        grouped_by_layer[layer]["gate"],
                        expected_tokens=int(zb.shape[0]),
                    )
                    layer_rank = _mean_stacked_tokens(
                        grouped_by_layer[layer]["rank_mask"],
                        expected_tokens=int(zb.shape[0]),
                    )

                    if layer_gate is None:
                        gate_focus = torch.zeros(int(zb.shape[0]), device=zb.device, dtype=zb.dtype)
                    else:
                        gate_focus = _sanitize_tensor(layer_gate.max(dim=-1).values)

                    if layer_rank is None:
                        rank_usage = torch.zeros_like(gate_focus)
                    else:
                        rank_usage = _sanitize_tensor(layer_rank.sum(dim=-1))

                    feature_intensity = _sanitize_tensor((dm * zb).abs().sum(dim=-1))
                    gate_rank_corr = _safe_corr(gate_focus, rank_usage)
                    gate_feature_corr = _safe_corr(gate_focus, feature_intensity)

                    stats[domain_id][layer]["drift_general"].append(_to_finite_float(drift_general))
                    stats[domain_id][layer]["drift_domain"].append(_to_finite_float(drift_domain))
                    stats[domain_id][layer]["topk_overlap"].append(_to_finite_float(topk_overlap))
                    stats[domain_id][layer]["gate_rank_corr"].append(_to_finite_float(gate_rank_corr))
                    stats[domain_id][layer]["gate_feature_corr"].append(_to_finite_float(gate_feature_corr))

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for domain_id, layer_stats in stats.items():
        aggregated[domain_id] = {}
        for layer, values in layer_stats.items():
            aggregated[domain_id][layer] = {k: _aggregate(v) for k, v in values.items()}

    return aggregated
