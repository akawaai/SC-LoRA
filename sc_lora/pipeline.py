from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from .config import LossWeights, SAEConfig, SCLoRAConfig, TrainingSchedule
from .evaluation import (
    evaluate_contextual_specialization,
    evaluate_memory_routing,
    evaluate_mechanistic_alignment,
    evaluate_representation_interference,
    evaluate_routing_diagnostics,
    evaluate_sequential_forgetting,
)
from .prototypes import build_steering_prototypes
from .replay import BalancedReplayBuffer
from .taxonomy import build_feature_taxonomy
from .sae import PermutedSparseAutoencoder
from .training import (
    build_untrained_saes,
    collect_activations,
    init_sc_lora_bank,
    train_layerwise_saes,
    train_sc_lora,
    train_sc_lora_with_replay,
)


def _clone_general_mask(
    general_mask: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {layer: mask.detach().clone() for layer, mask in general_mask.items()}


def _clone_domain_mask(
    domain_mask: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        layer: {
            domain_id: mask.detach().clone() for domain_id, mask in per_domain.items()
        }
        for layer, per_domain in domain_mask.items()
    }


def _random_mask_like(
    reference: torch.Tensor,
    keep_count: int,
    generator: torch.Generator,
    restrict_to: torch.Tensor | None = None,
) -> torch.Tensor:
    flat = torch.zeros(reference.numel(), dtype=torch.float32)
    if keep_count <= 0:
        return flat.reshape(reference.shape)

    if restrict_to is not None:
        candidates = (restrict_to.reshape(-1) > 0.5).nonzero(as_tuple=False).reshape(-1)
    else:
        candidates = torch.arange(reference.numel(), dtype=torch.long)

    if candidates.numel() == 0:
        candidates = torch.arange(reference.numel(), dtype=torch.long)

    k = min(int(keep_count), int(candidates.numel()))
    if k > 0:
        perm = torch.randperm(int(candidates.numel()), generator=generator)
        selected = candidates[perm[:k]]
        flat[selected] = 1.0

    return flat.reshape(reference.shape)


def _apply_taxonomy_ablation(
    general_mask: dict[str, torch.Tensor],
    domain_mask: dict[str, dict[str, torch.Tensor]],
    mode: str,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    mode = mode.strip().lower()
    if mode == "learned":
        return general_mask, domain_mask

    valid = {"learned", "dense", "random_partition"}
    if mode not in valid:
        raise ValueError(
            f"Unknown taxonomy ablation mode '{mode}'. Valid: {sorted(valid)}"
        )

    general = _clone_general_mask(general_mask)
    domain = _clone_domain_mask(domain_mask)

    if mode == "dense":
        for layer in general.keys():
            general[layer] = torch.ones_like(general[layer], dtype=torch.float32)
            for domain_id in domain[layer].keys():
                domain[layer][domain_id] = torch.ones_like(
                    domain[layer][domain_id], dtype=torch.float32
                )
        return general, domain

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1337)

    for layer in general.keys():
        g_ref = general[layer]
        g_keep = int((g_ref.reshape(-1) > 0.5).sum().item())
        g_new = _random_mask_like(g_ref, keep_count=g_keep, generator=generator)
        general[layer] = g_new

        available = (g_new < 0.5).float()
        for domain_id, d_ref in domain[layer].items():
            d_keep = int((d_ref.reshape(-1) > 0.5).sum().item())
            d_new = _random_mask_like(
                d_ref,
                keep_count=d_keep,
                generator=generator,
                restrict_to=available,
            )
            domain[layer][domain_id] = d_new

    return general, domain


def _clone_nested_tensor_map(
    values: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        outer_k: {
            inner_k: tensor.detach().clone() for inner_k, tensor in inner_map.items()
        }
        for outer_k, inner_map in values.items()
    }


def _apply_prototype_ablation(
    prototypes: dict[str, dict[str, torch.Tensor]],
    mode: str,
) -> dict[str, dict[str, torch.Tensor]]:
    mode = mode.strip().lower()
    if mode == "learned":
        return prototypes

    valid = {"learned", "none", "shuffled_domains"}
    if mode not in valid:
        raise ValueError(
            f"Unknown prototype ablation mode '{mode}'. Valid: {sorted(valid)}"
        )

    if mode == "none":
        return {}

    domains = sorted(prototypes.keys())
    if len(domains) <= 1:
        return _clone_nested_tensor_map(prototypes)

    rotated = domains[1:] + domains[:1]
    shuffled: dict[str, dict[str, torch.Tensor]] = {}
    for target, source in zip(domains, rotated):
        shuffled[target] = {
            layer: tensor.detach().clone()
            for layer, tensor in prototypes[source].items()
        }
    return shuffled


def _build_saes_for_mode(
    mode: str,
    activation_path: str,
    hook_layers: list[str],
    sae_config: SAEConfig,
    save_path: str,
    device: torch.device | str,
    ablation_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = mode.strip().lower()
    if mode == "trained":
        return train_layerwise_saes(
            activation_path=activation_path,
            hook_layers=hook_layers,
            sae_config=sae_config,
            save_path=save_path,
            device=device,
        )

    if mode in {"identity", "random_projection"}:
        return build_untrained_saes(
            activation_path=activation_path,
            hook_layers=hook_layers,
            sae_config=sae_config,
            save_path=save_path,
            mode=mode,
            device=device,
            seed=ablation_seed,
        )

    if mode == "trained_permuted":
        saes, sae_logs = train_layerwise_saes(
            activation_path=activation_path,
            hook_layers=hook_layers,
            sae_config=sae_config,
            save_path=save_path,
            device=device,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(ablation_seed) + 4242)

        wrapped: dict[str, Any] = {}
        permutations: dict[str, torch.Tensor] = {}
        for layer, sae in saes.items():
            permutation = torch.randperm(
                int(getattr(sae, "feature_dim")), generator=generator
            )
            permutations[layer] = permutation.cpu()
            wrapped[layer] = PermutedSparseAutoencoder(
                base_sae=sae, permutation=permutation
            ).to(device)

        torch.save(permutations, f"{save_path}/sae_feature_permutations.pt")
        return wrapped, sae_logs

    valid = ["trained", "identity", "random_projection", "trained_permuted"]
    raise ValueError(f"Unknown SAE mode '{mode}'. Valid: {valid}")


def run_poster_level_pipeline(
    base_model: torch.nn.Module,
    d_probe: Any,
    d_general: Any,
    d_domains: dict[str, Any],
    training_schedule: TrainingSchedule,
    eval_sets: dict[str, Any],
    hook_layers: list[str],
    target_modules_per_layer: dict[str, list[str]],
    output_dir: str,
    sae_config: SAEConfig,
    sc_config: SCLoRAConfig,
    loss_weights: LossWeights,
    instruction_model: torch.nn.Module | None = None,
    device: torch.device | str = "cpu",
    use_replay: bool = False,
    replay_ratio: float = 0.2,
    replay_buffer_size: int = 128,
    max_probe_batches: int | None = None,
    max_steps_per_phase: int | None = None,
    forgetting_checkpoints: list[Any] | None = None,
    forgetting_evaluator: Callable[[Any, Any], float] | None = None,
    task_metric: Callable[[Any, torch.Tensor | None], float] | None = None,
    domain_to_specialist: dict[str, int] | None = None,
    log_every_steps: int = 50,
    sae_mode: str = "trained",
    taxonomy_mode: str = "learned",
    prototype_mode: str = "learned",
    ablation_seed: int = 0,
    train_lr: float = 1e-4,
    train_weight_decay: float = 0.0,
    train_max_grad_norm: float = 1.0,
    train_nonfinite_patience: int = 20,
    train_adam_eps: float = 1e-6,
) -> dict[str, Any]:
    out = Path(output_dir)
    activation_path = str(out / "activations")
    sae_path = str(out / "sae")
    taxonomy_path = str(out / "taxonomy")
    prototypes_path = str(out / "prototypes")
    checkpoints_path = str(out / "checkpoints")
    figures_path = str(out / "figures")

    out.mkdir(parents=True, exist_ok=True)
    base_model.to(device)
    if instruction_model is not None:
        instruction_model.to(device)

    print("[PIPELINE] Step 1/8: collecting probe activations...", flush=True)
    collect_activations(
        base_model=base_model,
        d_probe=d_probe,
        hook_layers=hook_layers,
        save_path=activation_path,
        device=device,
        max_batches=max_probe_batches,
        log_every_steps=log_every_steps,
    )

    print(f"[PIPELINE] Step 2/8: building SAE bank (mode={sae_mode})...", flush=True)
    saes, sae_logs = _build_saes_for_mode(
        mode=sae_mode,
        activation_path=activation_path,
        hook_layers=hook_layers,
        sae_config=sae_config,
        save_path=sae_path,
        device=device,
        ablation_seed=ablation_seed,
    )

    print(
        f"[PIPELINE] Step 3/8: building feature taxonomy (mode={taxonomy_mode})...",
        flush=True,
    )
    general_mask, domain_mask = build_feature_taxonomy(
        base_model=base_model,
        instruction_model=instruction_model,
        saes=saes,
        d_general=d_general,
        d_domains=d_domains,
        hook_layers=hook_layers,
        save_path=taxonomy_path,
        device=device,
    )
    general_mask, domain_mask = _apply_taxonomy_ablation(
        general_mask=general_mask,
        domain_mask=domain_mask,
        mode=taxonomy_mode,
        seed=ablation_seed,
    )
    torch.save(general_mask, f"{taxonomy_path}/general_mask_used.pt")
    torch.save(domain_mask, f"{taxonomy_path}/domain_mask_used.pt")

    print(
        f"[PIPELINE] Step 4/8: building steering prototypes (mode={prototype_mode})...",
        flush=True,
    )
    prototypes, covariances = build_steering_prototypes(
        base_model=base_model,
        saes=saes,
        domain_masks=domain_mask,
        d_domains=d_domains,
        hook_layers=hook_layers,
        save_path=prototypes_path,
        device=device,
        full_covariance=False,
    )
    prototypes = _apply_prototype_ablation(prototypes=prototypes, mode=prototype_mode)
    torch.save(prototypes, f"{prototypes_path}/prototypes_used.pt")
    torch.save(covariances, f"{prototypes_path}/covariances_used.pt")

    print("[PIPELINE] Step 5/8: initializing SC-LoRA banks...", flush=True)
    model_theta = init_sc_lora_bank(
        model=base_model,
        hook_layers=hook_layers,
        target_modules_per_layer=target_modules_per_layer,
        saes=saes,
        sc_cfg=sc_config,
    )

    print("[PIPELINE] Step 6/8: training continual SC-LoRA...", flush=True)
    if use_replay:
        replay_buffer = BalancedReplayBuffer(max_batches_per_domain=replay_buffer_size)
        model_theta, replay_buffer, train_logs = train_sc_lora_with_replay(
            model_theta=model_theta,
            frozen_base=base_model,
            saes=saes,
            general_mask=general_mask,
            domain_mask=domain_mask,
            prototypes=prototypes,
            training_schedule=training_schedule,
            hook_layers=hook_layers,
            loss_weights=loss_weights,
            replay_buffer=replay_buffer,
            replay_ratio=replay_ratio,
            save_path=checkpoints_path,
            device=device,
            lr=train_lr,
            weight_decay=train_weight_decay,
            max_grad_norm=train_max_grad_norm,
            nonfinite_patience=train_nonfinite_patience,
            adam_eps=train_adam_eps,
            max_steps_per_phase=max_steps_per_phase,
            log_every_steps=log_every_steps,
        )
    else:
        model_theta, train_logs = train_sc_lora(
            model_theta=model_theta,
            frozen_base=base_model,
            saes=saes,
            general_mask=general_mask,
            domain_mask=domain_mask,
            prototypes=prototypes,
            training_schedule=training_schedule,
            hook_layers=hook_layers,
            loss_weights=loss_weights,
            save_path=checkpoints_path,
            device=device,
            lr=train_lr,
            weight_decay=train_weight_decay,
            max_grad_norm=train_max_grad_norm,
            nonfinite_patience=train_nonfinite_patience,
            adam_eps=train_adam_eps,
            max_steps_per_phase=max_steps_per_phase,
            log_every_steps=log_every_steps,
        )
        replay_buffer = None

    print("[PIPELINE] Step 7/8: running evaluations...", flush=True)
    eval_context = evaluate_contextual_specialization(
        model_theta=model_theta,
        saes=saes,
        eval_sets=eval_sets,
        hook_layers=hook_layers,
        frozen_backbone=base_model,
        device=device,
        task_metric=task_metric,
        domain_to_specialist=domain_to_specialist,
        prototypes=prototypes,
    )

    if forgetting_checkpoints is not None and forgetting_evaluator is not None:
        score, forgetting = evaluate_sequential_forgetting(
            checkpoints=forgetting_checkpoints,
            eval_sets=eval_sets,
            evaluator=forgetting_evaluator,
        )
    else:
        score, forgetting = {}, {}

    eval_mech = evaluate_mechanistic_alignment(
        model_theta=model_theta,
        frozen_base=base_model,
        saes=saes,
        general_mask=general_mask,
        domain_mask=domain_mask,
        eval_sets=eval_sets,
        hook_layers=hook_layers,
        device=device,
        prototypes=prototypes,
    )

    eval_memory = evaluate_memory_routing(
        model_theta=model_theta,
        saes=saes,
        eval_sets=eval_sets,
        hook_layers=hook_layers,
        frozen_backbone=base_model,
        device=device,
        domain_to_specialist=domain_to_specialist,
        prototypes=prototypes,
    )

    eval_routing = evaluate_routing_diagnostics(
        model_theta=model_theta,
        saes=saes,
        eval_sets=eval_sets,
        hook_layers=hook_layers,
        frozen_backbone=base_model,
        device=device,
        domain_to_specialist=domain_to_specialist,
        prototypes=prototypes,
    )

    eval_interference = evaluate_representation_interference(
        model_theta=model_theta,
        frozen_base=base_model,
        saes=saes,
        general_mask=general_mask,
        eval_sets=eval_sets,
        hook_layers=hook_layers,
        device=device,
        prototypes=prototypes,
    )

    eval_logs = {
        "contextual": eval_context,
        "forgetting": {"score": score, "forgetting": forgetting},
        "memory_routing": eval_memory,
        "routing_diagnostics": eval_routing,
        "interference": eval_interference,
    }

    print("[PIPELINE] Step 8/8: generating figures...", flush=True)
    try:
        from .figures import generate_poster_figures
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            print(
                "[PIPELINE] Skipping figure generation (matplotlib is not installed).",
                flush=True,
            )
        else:
            raise
    else:
        generate_poster_figures(
            logs_training=train_logs,
            logs_eval=eval_logs,
            logs_mechanistic=eval_mech,
            save_path=figures_path,
        )
    print("[PIPELINE] Completed.", flush=True)

    return {
        "model_theta": model_theta,
        "saes": saes,
        "masks": {"general": general_mask, "domain": domain_mask},
        "prototypes": prototypes,
        "covariances": covariances,
        "ablation": {
            "sae_mode": sae_mode,
            "taxonomy_mode": taxonomy_mode,
            "prototype_mode": prototype_mode,
            "ablation_seed": int(ablation_seed),
        },
        "optimization": {
            "lr": float(train_lr),
            "weight_decay": float(train_weight_decay),
            "max_grad_norm": float(train_max_grad_norm),
            "nonfinite_patience": int(train_nonfinite_patience),
            "adam_eps": float(train_adam_eps),
        },
        "logs": {
            "sae": sae_logs,
            "training": train_logs,
            "eval_context": eval_context,
            "eval_forgetting": {"score": score, "forgetting": forgetting},
            "eval_mechanistic": eval_mech,
            "eval_memory_routing": eval_memory,
            "eval_routing_diagnostics": eval_routing,
            "eval_interference": eval_interference,
        },
        "paths": {
            "output_dir": str(out),
            "activation_path": activation_path,
            "sae_path": sae_path,
            "taxonomy_path": taxonomy_path,
            "prototypes_path": prototypes_path,
            "checkpoints_path": checkpoints_path,
            "figures_path": figures_path,
        },
        "replay_buffer": replay_buffer,
    }


def train_hybrid_bridge(
    hybrid_model: torch.nn.Module,
    frozen_hybrid: torch.nn.Module,
    transformer_submodules: list[str],
    hybrid_specific_submodules: list[str],
    mode: str,
    d_probe: Any,
    d_general: Any,
    d_domains: dict[str, Any],
    training_schedule: TrainingSchedule,
    eval_sets: dict[str, Any],
    target_modules_per_layer: dict[str, list[str]],
    output_dir: str,
    sae_config: SAEConfig,
    sc_config: SCLoRAConfig,
    loss_weights: LossWeights,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    if mode == "bridge_only":
        active_hooks = transformer_submodules
    elif mode == "full_hybrid":
        active_hooks = list(
            dict.fromkeys(transformer_submodules + hybrid_specific_submodules)
        )
    else:
        raise ValueError("mode must be either 'bridge_only' or 'full_hybrid'")

    _ = hybrid_model

    return run_poster_level_pipeline(
        base_model=frozen_hybrid,
        d_probe=d_probe,
        d_general=d_general,
        d_domains=d_domains,
        training_schedule=training_schedule,
        eval_sets=eval_sets,
        hook_layers=active_hooks,
        target_modules_per_layer=target_modules_per_layer,
        output_dir=output_dir,
        sae_config=sae_config,
        sc_config=sc_config,
        loss_weights=loss_weights,
        instruction_model=None,
        device=device,
    )
