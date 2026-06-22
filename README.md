# SC-LoRA

Official implementation of **Steering-Conditioned LoRA (SC-LoRA)**, published at [The 2nd Workshop on Connecting Low-rank Representations in AI (CoLorAI)](https://colorai-workshop.github.io/) (ICML 2026).

**Authors:** [David O'Neil Campos Ferreira](https://github.com/davidoneilai), Diogo Fernandes Costa Silva, Arlindo Rodrigues Galvão Filho

SC-LoRA combines sparse autoencoder (SAE) features, steering prototypes, and a conditional LoRA adapter bank with specialist routing and dynamic rank control. The method targets specialization on new domains while limiting catastrophic forgetting on prior domains.

## Overview

SC-LoRA pipeline:

1. Collect layer activations from a frozen base model.
2. Train layerwise sparse autoencoders (SAEs).
3. Build a feature taxonomy and steering prototypes per domain.
4. Train a conditional LoRA adapter bank with routing, rank control, and optional replay.
5. Evaluate forgetting, interference, routing, and mechanistic alignment.

This repository contains the core `sc_lora` Python package.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Smoke check:

```bash
python -m compileall sc_lora
python -c "import sc_lora; print('ok')"
```

## Quick start

```python
from sc_lora import (
    SAEConfig,
    SCLoRAConfig,
    LossWeights,
    TrainingSchedule,
    TrainingPhase,
    run_poster_level_pipeline,
    resolve_model_source,
)

# Load your base model and dataloaders, then run the end-to-end pipeline.
resolution = resolve_model_source("Qwen/Qwen2.5-3B")
# summary = run_poster_level_pipeline(
#     base_model=...,
#     d_probe=...,
#     d_general=...,
#     d_domains={"math": ...},
#     training_schedule=TrainingSchedule(phases=[...]),
#     eval_sets={...},
#     hook_layers=[...],
#     target_modules_per_layer={...},
#     output_dir="outputs/my_run",
#     sae_config=SAEConfig(...),
#     sc_config=SCLoRAConfig(...),
#     loss_weights=LossWeights(...),
#     device="cuda",
#     use_replay=True,
# )
```

You provide the model, dataloaders, and hook configuration. The pipeline handles activation collection, SAE training, prototype construction, SC-LoRA training, evaluation, and diagnostic figure generation under `output_dir`.

## Repository structure

```text
SC-LoRa/
├── sc_lora/          # Core library
├── LICENSE
├── pyproject.toml
└── README.md
```

### Core modules

| Module | Role |
|--------|------|
| `pipeline.py` | End-to-end orchestration |
| `training.py` | Activation collection, SAE training, SC-LoRA training |
| `lora.py` | Conditional adapter bank, routing, memory policy |
| `sae.py` | Sparse autoencoder variants |
| `taxonomy.py` | General vs domain feature masks |
| `prototypes.py` | Steering prototype construction |
| `controllers.py` | Gate, rank, and budget controllers |
| `losses.py` | Task, preservation, steering, and regularization losses |
| `evaluation.py` | Forgetting, interference, routing, and mechanistic metrics |
| `replay.py` | Balanced replay buffer for continual training |
| `model_source.py` | Resolve Hugging Face IDs, local paths, and PEFT adapters |
| `config.py` | Dataclasses for SAE, SC-LoRA, losses, and training schedule |

## Programmatic building blocks

Train individual stages instead of the full pipeline:

```python
from sc_lora import (
    collect_activations,
    train_layerwise_saes,
    init_sc_lora_bank,
    train_sc_lora_with_replay,
    evaluate_sequential_forgetting,
    evaluate_memory_routing,
)
```

Configuration dataclasses live in `sc_lora.config`:

- `SAEConfig` — SAE architecture and training
- `SCLoRAConfig` — adapter bank, specialists, rank, memory policy
- `LossWeights` — multitask loss coefficients
- `TrainingSchedule` / `TrainingPhase` — sequential domain training phases

## Acknowledgements

This work has been funded by the project *Research and Development of Gênese Digital: Scaling Interactive and Culturally Adapted Digital Humans with Generative AI*, supported by the Advanced Knowledge Center in Immersive Technologies (AKCIT), with financial resources from the PPI IoT/Manufatura 4.0 / PPI HardwareBR of the MCTI, grant number 057/2023, signed with EMBRAPII, and supported by P&D CEMIG/ANEEL PD-04950-D0677/2023.

This codebase builds on open-source tooling from PyTorch, Hugging Face Transformers, PEFT, and the broader mechanistic interpretability / sparse autoencoder ecosystem.

## License

MIT — see [LICENSE](LICENSE).
