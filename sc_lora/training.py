from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any

import torch
import torch.nn as nn

from .config import LossWeights, SAEConfig, TrainingSchedule
from .data import split_batch
from .hooks import forward_with_hooks
from .losses import (
    compute_task_loss,
    gate_sparsity_loss,
    load_balance_loss,
    preservation_loss,
    rank_budget_loss,
    specialist_diversity_loss,
    steering_loss,
    summarize_usage,
)
from .lora import SCLoRAModel
from .replay import BalancedReplayBuffer, merge_batches
from .sae import IdentitySparseAutoencoder, RandomProjectionSparseAutoencoder, SparseAutoencoder, train_single_sae
from .utils import ensure_dir, flatten_tokens_with_attention, freeze_module, get_attention_mask, to_device_tree


def _safe_len(obj: Any) -> int | None:
    try:
        return int(len(obj))
    except (TypeError, AttributeError):
        return None


def _planned_steps(obj: Any, max_steps: int | None) -> int | None:
    total = _safe_len(obj)
    if total is None:
        return max_steps
    if max_steps is None:
        return total
    return min(total, max_steps)


def _should_log_step(step_idx: int, total_steps: int | None, log_every_steps: int) -> bool:
    if step_idx == 1:
        return True
    if step_idx % max(1, log_every_steps) == 0:
        return True
    if total_steps is not None and step_idx >= total_steps:
        return True
    return False


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _avg_metric(logs: list[dict[str, float]], key: str) -> float:
    if not logs:
        return 0.0
    values = [float(entry.get(key, 0.0)) for entry in logs]
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _tensor_is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _safe_item(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _mean_active_specialists(model_theta: SCLoRAModel) -> float:
    if not model_theta.wrappers:
        return 0.0
    values = [float(wrapper.bank.active_mask.sum().item()) for wrapper in model_theta.wrappers.values()]
    return sum(values) / max(1, len(values))


def _sanitize_trainable_parameters(model_theta: SCLoRAModel, clamp_value: float = 1e3) -> int:
    repaired = 0
    with torch.no_grad():
        for param in model_theta.trainable_parameters():
            finite_mask = torch.isfinite(param)
            if not bool(finite_mask.all().item()):
                repaired += int((~finite_mask).sum().item())
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
            param.data.clamp_(min=-clamp_value, max=clamp_value)
    return repaired


def _sanitize_optimizer_state(optimizer: torch.optim.Optimizer, clamp_value: float = 1e3) -> int:
    repaired = 0
    with torch.no_grad():
        for state in optimizer.state.values():
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue

                finite_mask = torch.isfinite(value)
                if not bool(finite_mask.all().item()):
                    repaired += int((~finite_mask).sum().item())
                    cleaned = torch.nan_to_num(value, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
                    state[key] = cleaned
                    value = cleaned

                if value.dtype.is_floating_point:
                    value.clamp_(min=-clamp_value, max=clamp_value)
    return repaired


def _sanitize_gradients(model_theta: SCLoRAModel, clamp_value: float = 1e3) -> tuple[int, int]:
    repaired_values = 0
    repaired_tensors = 0
    with torch.no_grad():
        for param in model_theta.trainable_parameters():
            grad = param.grad
            if grad is None:
                continue
            finite_mask = torch.isfinite(grad)
            if not bool(finite_mask.all().item()):
                repaired_values += int((~finite_mask).sum().item())
                grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
                repaired_tensors += 1
            if grad.dtype.is_floating_point:
                grad.data.clamp_(min=-clamp_value, max=clamp_value)
    return repaired_values, repaired_tensors


def _sanitize_adapter_stats(
    adapter_stats: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    cleaned: dict[str, dict[str, torch.Tensor]] = {}
    for key, stats in adapter_stats.items():
        gate = stats.get("gate")
        rank_mask = stats.get("rank_mask")
        if gate is None or rank_mask is None:
            continue

        gate_clean = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)
        gate_clean = gate_clean.clamp_min(0.0)
        gate_clean = gate_clean / (gate_clean.sum(dim=-1, keepdim=True) + 1e-8)

        rank_clean = torch.nan_to_num(rank_mask, nan=0.0, posinf=1.0, neginf=0.0)
        rank_clean = rank_clean.clamp(min=0.0, max=1.0)

        stats_clean = dict(stats)
        stats_clean["gate"] = gate_clean
        stats_clean["rank_mask"] = rank_clean
        cleaned[key] = stats_clean

    return cleaned


def collect_activations(
    base_model: nn.Module,
    d_probe: Any,
    hook_layers: list[str],
    save_path: str,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
    log_every_steps: int = 50,
) -> str:
    base_model.eval()
    freeze_module(base_model)

    store: dict[str, list[torch.Tensor]] = {layer: [] for layer in hook_layers}
    total_batches = _planned_steps(d_probe, max_batches)
    start_time = time.time()
    print(
        f"[ACT] Collecting activations | layers={len(hook_layers)} | batches={total_batches if total_batches is not None else 'unknown'}",
        flush=True,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(d_probe, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            inputs, _ = split_batch(batch)
            inputs = to_device_tree(inputs, device)
            attention_mask = get_attention_mask(inputs)
            out = forward_with_hooks(base_model, inputs, layers=hook_layers, detach=True)

            for layer in hook_layers:
                h_l, _ = flatten_tokens_with_attention(
                    out.activations[layer],
                    attention_mask=attention_mask,
                    drop_padding=True,
                )
                h_l = h_l.cpu()
                store[layer].append(h_l)

            if _should_log_step(batch_idx, total_batches, log_every_steps):
                elapsed = time.time() - start_time
                speed = batch_idx / max(1e-6, elapsed)
                if total_batches is not None:
                    remaining = max(0, total_batches - batch_idx)
                    eta = _format_duration(remaining / max(1e-6, speed))
                    print(
                        f"[ACT] batch={batch_idx}/{total_batches} | speed={speed:.2f} b/s | elapsed={_format_duration(elapsed)} | eta={eta}",
                        flush=True,
                    )
                else:
                    print(
                        f"[ACT] batch={batch_idx} | speed={speed:.2f} b/s | elapsed={_format_duration(elapsed)}",
                        flush=True,
                    )

    ensure_dir(save_path)
    for layer in hook_layers:
        tensor = torch.cat(store[layer], dim=0)
        torch.save(tensor, f"{save_path}/layer_{layer}.pt")

    print(f"[ACT] Activations saved at {save_path}", flush=True)

    return save_path


def train_layerwise_saes(
    activation_path: str,
    hook_layers: list[str],
    sae_config: SAEConfig,
    save_path: str,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, SparseAutoencoder], dict[str, list[dict[str, float]]]]:
    ensure_dir(save_path)

    saes: dict[str, SparseAutoencoder] = {}
    logs: dict[str, list[dict[str, float]]] = {}
    print(f"[SAE] Training layerwise SAEs for {len(hook_layers)} layers", flush=True)

    for layer_idx, layer in enumerate(hook_layers, start=1):
        x = torch.load(f"{activation_path}/layer_{layer}.pt", map_location="cpu")
        print(
            f"[SAE] layer {layer_idx}/{len(hook_layers)} | name={layer} | samples={x.shape[0]} | hidden_dim={x.shape[-1]}",
            flush=True,
        )
        sae = SparseAutoencoder(
            input_dim=x.shape[-1],
            expansion_factor=sae_config.expansion_factor,
            normalize_inputs=sae_config.normalize_inputs,
        )

        train_logs = train_single_sae(
            x=x,
            sae=sae,
            epochs=sae_config.epochs,
            batch_size=sae_config.batch_size,
            lr=sae_config.lr,
            lambda_sparse=sae_config.lambda_sparse,
            lambda_aux=sae_config.lambda_aux,
            sparse_mode=sae_config.sparse_mode,
            dead_threshold=sae_config.dead_feature_threshold,
            decoder_norm_weight=sae_config.decoder_norm_penalty,
            device=device,
        )

        saes[layer] = sae.to(device)
        logs[layer] = [
            {
                "epoch": float(entry.epoch),
                "l_rec": entry.l_rec,
                "l_sparse": entry.l_sparse,
                "dead_feature_rate": entry.dead_feature_rate,
            }
            for entry in train_logs
        ]

        torch.save(
            {
                "sae_mode": "trained",
                "state_dict": sae.state_dict(),
                "input_dim": sae.input_dim,
                "feature_dim": sae.feature_dim,
                "expansion_factor": sae_config.expansion_factor,
                "normalize_inputs": sae_config.normalize_inputs,
                "logs": logs[layer],
            },
            f"{save_path}/sae_layer_{layer}.pt",
        )
        last_log = logs[layer][-1] if logs[layer] else {}
        print(
            "[SAE] done "
            f"{layer} | l_rec={float(last_log.get('l_rec', 0.0)):.6f} | "
            f"l_sparse={float(last_log.get('l_sparse', 0.0)):.6f} | "
            f"dead_rate={float(last_log.get('dead_feature_rate', 0.0)):.4f}",
            flush=True,
        )

    torch.save(logs, f"{save_path}/sae_training_logs.pt")
    print(f"[SAE] Logs and checkpoints saved at {save_path}", flush=True)
    return saes, logs



def build_untrained_saes(
    activation_path: str,
    hook_layers: list[str],
    sae_config: SAEConfig,
    save_path: str,
    mode: str,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[dict[str, SparseAutoencoder], dict[str, list[dict[str, float]]]]:
    mode = mode.strip().lower()
    if mode not in {"identity", "random_projection"}:
        raise ValueError(f"Unsupported untrained SAE mode: {mode}")

    ensure_dir(save_path)
    saes: dict[str, SparseAutoencoder] = {}
    logs: dict[str, list[dict[str, float]]] = {}

    print(f"[SAE] Building untrained SAE bank | mode={mode} | layers={len(hook_layers)}", flush=True)

    for layer_idx, layer in enumerate(hook_layers, start=1):
        x = torch.load(f"{activation_path}/layer_{layer}.pt", map_location="cpu")
        input_dim = int(x.shape[-1])

        if mode == "identity":
            sae = IdentitySparseAutoencoder(
                input_dim=input_dim,
                normalize_inputs=sae_config.normalize_inputs,
            )
        else:
            feature_dim = max(1, int(input_dim * max(1, int(sae_config.expansion_factor))))
            sae = RandomProjectionSparseAutoencoder(
                input_dim=input_dim,
                feature_dim=feature_dim,
                normalize_inputs=sae_config.normalize_inputs,
                seed=int(seed) + layer_idx * 9973,
            )

        sae = sae.to(device)
        sae.eval()
        for param in sae.parameters():
            param.requires_grad = False

        saes[layer] = sae
        logs[layer] = [
            {
                "epoch": 0.0,
                "l_rec": 0.0,
                "l_sparse": 0.0,
                "dead_feature_rate": 0.0,
            }
        ]

        payload = {
            "sae_mode": mode,
            "input_dim": int(getattr(sae, "input_dim")),
            "feature_dim": int(getattr(sae, "feature_dim")),
            "expansion_factor": float(getattr(sae, "feature_dim")) / float(max(1, getattr(sae, "input_dim"))),
            "normalize_inputs": bool(getattr(sae, "normalize_inputs", True)),
            "seed": int(seed),
            "logs": logs[layer],
        }
        if mode == "random_projection":
            payload["state_dict"] = sae.state_dict()

        torch.save(payload, f"{save_path}/sae_layer_{layer}.pt")
        print(
            f"[SAE] layer {layer_idx}/{len(hook_layers)} | name={layer} | mode={mode} | input_dim={input_dim} | feature_dim={int(getattr(sae, 'feature_dim'))}",
            flush=True,
        )

    torch.save(logs, f"{save_path}/sae_training_logs.pt")
    print(f"[SAE] Logs and checkpoints saved at {save_path}", flush=True)
    return saes, logs

def _encode_layerwise(
    saes: dict[str, SparseAutoencoder],
    activations: dict[str, torch.Tensor],
    hook_layers: list[str],
    device: torch.device | str,
    track_grad: bool,
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
        if track_grad:
            z_layer = saes[layer].encode(h)
        else:
            with torch.no_grad():
                z_layer = saes[layer].encode(h)
        if token_mask_dev is not None and not drop_padding:
            z_layer = z_layer * token_mask_dev
        z[layer] = z_layer
    return z


def _zero_like_device(reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, device=reference.device)


def _domain_mask_for(domain_mask: dict[str, dict[str, torch.Tensor]], domain_id: str) -> dict[str, torch.Tensor]:
    return {layer: layer_map[domain_id] for layer, layer_map in domain_mask.items() if domain_id in layer_map}


def init_sc_lora_bank(
    model: nn.Module,
    hook_layers: list[str],
    target_modules_per_layer: dict[str, list[str]],
    saes: dict[str, SparseAutoencoder],
    sc_cfg: Any,
) -> SCLoRAModel:
    m_theta = SCLoRAModel(deepcopy(model))
    sae_feature_dims = {layer: saes[layer].feature_dim for layer in hook_layers}
    m_theta.init_sc_lora_bank(
        hook_layers=hook_layers,
        target_modules_per_layer=target_modules_per_layer,
        sae_feature_dims=sae_feature_dims,
        cfg=sc_cfg,
    )
    return m_theta


def _train_step(
    model_theta: SCLoRAModel,
    frozen_base: nn.Module,
    saes: dict[str, SparseAutoencoder],
    batch: Any,
    hook_layers: list[str],
    general_mask: dict[str, torch.Tensor],
    domain_mask: dict[str, dict[str, torch.Tensor]],
    prototypes: dict[str, dict[str, torch.Tensor]],
    phase_domain: str | None,
    loss_weights: LossWeights,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    global_step: int | None = None,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    inputs, targets = split_batch(batch)
    inputs = to_device_tree(inputs, device)
    attention_mask = get_attention_mask(inputs)
    if isinstance(targets, torch.Tensor):
        targets = targets.to(device)

    with torch.no_grad():
        base_out = forward_with_hooks(frozen_base, inputs, hook_layers, detach=False)
    z_base = _encode_layerwise(
        saes,
        base_out.activations,
        hook_layers,
        device=device,
        track_grad=False,
        attention_mask=attention_mask,
        drop_padding=False,
    )

    prototypes_by_layer = prototypes.get(phase_domain, None) if phase_domain is not None else None
    model_theta.set_context(
        features_by_layer=z_base,
        prototypes_by_layer=prototypes_by_layer,
        domain_id=phase_domain,
    )

    tuned_out = forward_with_hooks(model_theta.backbone, inputs, hook_layers, detach=False)
    z_ft = _encode_layerwise(
        saes,
        tuned_out.activations,
        hook_layers,
        device=device,
        track_grad=True,
        attention_mask=attention_mask,
        drop_padding=False,
    )

    adapter_stats_raw = model_theta.collect_adapter_stats()
    adapter_stats = _sanitize_adapter_stats(adapter_stats_raw)

    l_task = compute_task_loss(tuned_out.outputs, targets)
    l_pres = preservation_loss(z_ft=z_ft, z_base=z_base, general_mask=general_mask)

    if phase_domain is not None and phase_domain in prototypes:
        domain_mask_phase = _domain_mask_for(domain_mask, phase_domain)
        proto_phase = prototypes[phase_domain]
        l_steer = steering_loss(z_ft, domain_mask_phase, proto_phase)
    else:
        l_steer = _zero_like_device(l_task)

    l_sparse_gate = gate_sparsity_loss(adapter_stats).to(l_task.device)
    l_rank_budget = rank_budget_loss(adapter_stats).to(l_task.device)
    l_load_balance = load_balance_loss(adapter_stats).to(l_task.device)
    l_specialist_div = specialist_diversity_loss(model_theta).to(l_task.device)

    l_total = (
        loss_weights.lambda_task * l_task
        + loss_weights.lambda_pres * l_pres
        + loss_weights.lambda_steer * l_steer
        + loss_weights.lambda_sparse_gate * l_sparse_gate
        + loss_weights.lambda_rank_budget * l_rank_budget
        + loss_weights.lambda_load_balance * l_load_balance
        + loss_weights.lambda_specialist_div * l_specialist_div
    )

    loss_tensors = [l_task, l_pres, l_steer, l_sparse_gate, l_rank_budget, l_load_balance, l_specialist_div, l_total]
    losses_finite = all(_tensor_is_finite(t) for t in loss_tensors)

    grad_norm_value = 0.0
    grads_repaired = 0
    grad_tensors_repaired = 0
    params_repaired = 0
    optimizer_state_repaired = 0
    nonfinite_step = 0.0

    if losses_finite:
        optimizer.zero_grad(set_to_none=True)
        l_total.backward()
        grads_repaired, grad_tensors_repaired = _sanitize_gradients(model_theta)

        if max_grad_norm is not None and max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model_theta.trainable_parameters(),
                max_norm=float(max_grad_norm),
                error_if_nonfinite=False,
            )
            grad_norm_value = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        if math.isfinite(grad_norm_value):
            optimizer.step()
        else:
            nonfinite_step = 1.0
            optimizer.zero_grad(set_to_none=True)
            params_repaired = _sanitize_trainable_parameters(model_theta)
            optimizer_state_repaired = _sanitize_optimizer_state(optimizer)

    else:
        nonfinite_step = 1.0
        optimizer.zero_grad(set_to_none=True)
        params_repaired = _sanitize_trainable_parameters(model_theta)
        optimizer_state_repaired = _sanitize_optimizer_state(optimizer)

    if nonfinite_step > 0.0:
        memory_events = {
            "spawned": 0.0,
            "merged": 0.0,
            "pruned": 0.0,
            "active_specialists": _mean_active_specialists(model_theta),
        }
    else:
        memory_events = model_theta.apply_memory_policies(step=global_step)

    model_theta.clear_context()

    avg_active, avg_rank = summarize_usage(adapter_stats)

    return {
        "l_total": _safe_item(l_total),
        "l_task": _safe_item(l_task),
        "l_pres": _safe_item(l_pres),
        "l_steer": _safe_item(l_steer),
        "l_sparse_gate": _safe_item(l_sparse_gate),
        "l_rank_budget": _safe_item(l_rank_budget),
        "l_load_balance": _safe_item(l_load_balance),
        "l_specialist_div": _safe_item(l_specialist_div),
        "avg_active_specialists": avg_active,
        "avg_rank": avg_rank,
        "memory_spawned": float(memory_events.get("spawned", 0.0)),
        "memory_merged": float(memory_events.get("merged", 0.0)),
        "memory_pruned": float(memory_events.get("pruned", 0.0)),
        "memory_active_specialists": float(memory_events.get("active_specialists", 0.0)),
        "grad_norm": float(grad_norm_value),
        "grads_repaired": float(grads_repaired),
        "grad_tensors_repaired": float(grad_tensors_repaired),
        "nonfinite_step": float(nonfinite_step),
        "params_repaired": float(params_repaired),
        "optimizer_state_repaired": float(optimizer_state_repaired),
    }



def train_sc_lora(
    model_theta: SCLoRAModel,
    frozen_base: nn.Module,
    saes: dict[str, SparseAutoencoder],
    general_mask: dict[str, torch.Tensor],
    domain_mask: dict[str, dict[str, torch.Tensor]],
    prototypes: dict[str, dict[str, torch.Tensor]],
    training_schedule: TrainingSchedule,
    hook_layers: list[str],
    loss_weights: LossWeights,
    save_path: str,
    device: torch.device | str = "cpu",
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    nonfinite_patience: int = 20,
    adam_eps: float = 1e-6,
    max_steps_per_phase: int | None = None,
    log_every_steps: int = 50,
) -> tuple[SCLoRAModel, dict[str, Any]]:
    ensure_dir(save_path)

    frozen_base.eval()
    freeze_module(frozen_base)

    for sae in saes.values():
        sae.eval()
        for param in sae.parameters():
            param.requires_grad = False

    model_theta.to(device)
    frozen_base.to(device)

    optimizer = torch.optim.AdamW(
        model_theta.trainable_parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=adam_eps,
    )

    logs: dict[str, Any] = {"phases": {}}

    global_step = 0

    for phase in training_schedule.phases:
        model_theta.train()
        phase_logs: list[dict[str, float]] = []
        phase_total_steps = _planned_steps(phase.dataloader, max_steps_per_phase)
        phase_start = time.time()
        consecutive_nonfinite = 0
        print(
            f"[TRAIN] Phase '{phase.name}' started | domain={phase.domain_id} | steps={phase_total_steps if phase_total_steps is not None else 'unknown'}",
            flush=True,
        )

        for step_idx, batch in enumerate(phase.dataloader, start=1):
            if max_steps_per_phase is not None and step_idx > max_steps_per_phase:
                break
            global_step += 1

            step_log = _train_step(
                model_theta=model_theta,
                frozen_base=frozen_base,
                saes=saes,
                batch=batch,
                hook_layers=hook_layers,
                general_mask=general_mask,
                domain_mask=domain_mask,
                prototypes=prototypes,
                phase_domain=phase.domain_id,
                loss_weights=loss_weights,
                optimizer=optimizer,
                device=device,
                global_step=global_step,
                max_grad_norm=max_grad_norm,
            )
            step_log["step"] = float(step_idx)
            step_log["global_step"] = float(global_step)
            phase_logs.append(step_log)

            if float(step_log.get("nonfinite_step", 0.0)) > 0.0:
                consecutive_nonfinite += 1
            else:
                consecutive_nonfinite = 0

            if not math.isfinite(step_log["l_total"]):
                print(
                    f"[WARN] Non-finite loss detected | phase={phase.name} | step={step_idx} | l_total={step_log['l_total']} | repaired={int(step_log.get('params_repaired', 0.0))} | opt_repaired={int(step_log.get('optimizer_state_repaired', 0.0))}",
                    flush=True,
                )

            if consecutive_nonfinite >= max(1, int(nonfinite_patience)):
                print(
                    f"[ERROR] Aborting phase '{phase.name}' after {consecutive_nonfinite} consecutive non-finite steps.",
                    flush=True,
                )
                break

            if _should_log_step(step_idx, phase_total_steps, log_every_steps):
                elapsed = time.time() - phase_start
                speed = step_idx / max(1e-6, elapsed)
                if phase_total_steps is not None:
                    remaining = max(0, phase_total_steps - step_idx)
                    eta = _format_duration(remaining / max(1e-6, speed))
                    step_str = f"{step_idx}/{phase_total_steps}"
                else:
                    eta = "?"
                    step_str = str(step_idx)
                print(
                    "[TRAIN] "
                    f"{phase.name} step={step_str} "
                    f"| l_total={step_log['l_total']:.4f} "
                    f"| l_task={step_log['l_task']:.4f} "
                    f"| l_pres={step_log['l_pres']:.4f} "
                    f"| active={step_log['avg_active_specialists']:.2f} "
                    f"| rank={step_log['avg_rank']:.2f} "
                    f"| grad_norm={step_log.get('grad_norm', 0.0):.2f} "
                    f"| grad_fix={int(step_log.get('grads_repaired', 0.0))} "
                    f"| speed={speed:.2f} step/s "
                    f"| elapsed={_format_duration(elapsed)} "
                    f"| eta={eta}",
                    flush=True,
                )

        logs["phases"][phase.name] = phase_logs

        checkpoint = {
            "model_state": model_theta.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "phase": phase.name,
            "domain_id": phase.domain_id,
            "logs": phase_logs,
        }
        torch.save(checkpoint, f"{save_path}/checkpoint_{phase.name}.pt")
        phase_elapsed = time.time() - phase_start
        print(
            f"[TRAIN] Phase '{phase.name}' finished | steps={len(phase_logs)} | avg_l_total={_avg_metric(phase_logs, 'l_total'):.4f} | elapsed={_format_duration(phase_elapsed)}",
            flush=True,
        )

    torch.save(logs, f"{save_path}/training_logs.pt")
    print(f"[TRAIN] Training logs saved at {save_path}/training_logs.pt", flush=True)
    return model_theta, logs



def train_sc_lora_with_replay(
    model_theta: SCLoRAModel,
    frozen_base: nn.Module,
    saes: dict[str, SparseAutoencoder],
    general_mask: dict[str, torch.Tensor],
    domain_mask: dict[str, dict[str, torch.Tensor]],
    prototypes: dict[str, dict[str, torch.Tensor]],
    training_schedule: TrainingSchedule,
    hook_layers: list[str],
    loss_weights: LossWeights,
    replay_buffer: BalancedReplayBuffer,
    replay_ratio: float,
    save_path: str,
    device: torch.device | str = "cpu",
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    nonfinite_patience: int = 20,
    adam_eps: float = 1e-6,
    max_steps_per_phase: int | None = None,
    log_every_steps: int = 50,
) -> tuple[SCLoRAModel, BalancedReplayBuffer, dict[str, Any]]:
    ensure_dir(save_path)

    frozen_base.eval()
    freeze_module(frozen_base)

    for sae in saes.values():
        sae.eval()
        for param in sae.parameters():
            param.requires_grad = False

    model_theta.to(device)
    frozen_base.to(device)

    optimizer = torch.optim.AdamW(
        model_theta.trainable_parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=adam_eps,
    )

    logs: dict[str, Any] = {"phases": {}}

    global_step = 0

    for phase in training_schedule.phases:
        model_theta.train()
        phase_logs: list[dict[str, float]] = []
        phase_total_steps = _planned_steps(phase.dataloader, max_steps_per_phase)
        phase_start = time.time()
        consecutive_nonfinite = 0
        print(
            f"[TRAIN] Phase '{phase.name}' (replay) started | domain={phase.domain_id} | steps={phase_total_steps if phase_total_steps is not None else 'unknown'}",
            flush=True,
        )

        for step_idx, batch_cur in enumerate(phase.dataloader, start=1):
            if max_steps_per_phase is not None and step_idx > max_steps_per_phase:
                break
            global_step += 1

            replay_batch = replay_buffer.sample(num_batches=max(1, int(2 * replay_ratio)))
            batch_mix = merge_batches(batch_cur, replay_batch, alpha=replay_ratio)

            step_log = _train_step(
                model_theta=model_theta,
                frozen_base=frozen_base,
                saes=saes,
                batch=batch_mix,
                hook_layers=hook_layers,
                general_mask=general_mask,
                domain_mask=domain_mask,
                prototypes=prototypes,
                phase_domain=phase.domain_id,
                loss_weights=loss_weights,
                optimizer=optimizer,
                device=device,
                global_step=global_step,
                max_grad_norm=max_grad_norm,
            )
            step_log["step"] = float(step_idx)
            step_log["global_step"] = float(global_step)
            phase_logs.append(step_log)

            replay_buffer.add(batch_cur, domain_id=phase.domain_id or "default")

            if float(step_log.get("nonfinite_step", 0.0)) > 0.0:
                consecutive_nonfinite += 1
            else:
                consecutive_nonfinite = 0

            if not math.isfinite(step_log["l_total"]):
                print(
                    f"[WARN] Non-finite loss detected | phase={phase.name} | step={step_idx} | l_total={step_log['l_total']} | repaired={int(step_log.get('params_repaired', 0.0))} | opt_repaired={int(step_log.get('optimizer_state_repaired', 0.0))}",
                    flush=True,
                )

            if consecutive_nonfinite >= max(1, int(nonfinite_patience)):
                print(
                    f"[ERROR] Aborting phase '{phase.name}' after {consecutive_nonfinite} consecutive non-finite steps.",
                    flush=True,
                )
                break

            if _should_log_step(step_idx, phase_total_steps, log_every_steps):
                elapsed = time.time() - phase_start
                speed = step_idx / max(1e-6, elapsed)
                if phase_total_steps is not None:
                    remaining = max(0, phase_total_steps - step_idx)
                    eta = _format_duration(remaining / max(1e-6, speed))
                    step_str = f"{step_idx}/{phase_total_steps}"
                else:
                    eta = "?"
                    step_str = str(step_idx)

                replay_domains = len(replay_buffer.data)
                replay_batches = sum(len(v) for v in replay_buffer.data.values())
                print(
                    "[TRAIN] "
                    f"{phase.name} step={step_str} "
                    f"| l_total={step_log['l_total']:.4f} "
                    f"| l_task={step_log['l_task']:.4f} "
                    f"| l_pres={step_log['l_pres']:.4f} "
                    f"| active={step_log['avg_active_specialists']:.2f} "
                    f"| rank={step_log['avg_rank']:.2f} "
                    f"| grad_norm={step_log.get('grad_norm', 0.0):.2f} "
                    f"| grad_fix={int(step_log.get('grads_repaired', 0.0))} "
                    f"| replay_domains={replay_domains} "
                    f"| replay_batches={replay_batches} "
                    f"| speed={speed:.2f} step/s "
                    f"| elapsed={_format_duration(elapsed)} "
                    f"| eta={eta}",
                    flush=True,
                )

        logs["phases"][phase.name] = phase_logs

        checkpoint = {
            "model_state": model_theta.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "phase": phase.name,
            "domain_id": phase.domain_id,
            "logs": phase_logs,
        }
        torch.save(checkpoint, f"{save_path}/checkpoint_{phase.name}.pt")
        phase_elapsed = time.time() - phase_start
        print(
            f"[TRAIN] Phase '{phase.name}' finished | steps={len(phase_logs)} | avg_l_total={_avg_metric(phase_logs, 'l_total'):.4f} | elapsed={_format_duration(phase_elapsed)}",
            flush=True,
        )

    torch.save(logs, f"{save_path}/training_logs_with_replay.pt")
    print(f"[TRAIN] Training logs saved at {save_path}/training_logs_with_replay.pt", flush=True)
    return model_theta, replay_buffer, logs
