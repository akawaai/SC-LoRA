from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SAEConfig:
    expansion_factor: int = 4
    epochs: int = 5
    batch_size: int = 1024
    lr: float = 1e-3
    lambda_sparse: float = 1e-3
    lambda_aux: float = 1e-3
    sparse_mode: str = "l1"
    dead_feature_threshold: float = 1e-4
    normalize_inputs: bool = True
    decoder_norm_penalty: float = 1e-3


@dataclass
class SCLoRAConfig:
    num_specialists: int = 4
    initial_active_specialists: int = 2
    max_rank: int = 16
    gate_hidden_dim: int = 256
    rank_hidden_dim: int = 256
    controller_topk: Optional[int] = 2
    use_budget_controller: bool = True
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    memory_ema_decay: float = 0.95
    spawn_confidence_threshold: float = 0.45
    merge_similarity_threshold: float = 0.92
    prune_usage_threshold: float = 0.01
    min_active_specialists: int = 2
    memory_policy_interval: int = 50
    memory_compaction_warmup_steps: int = 400
    retrieval_topk: int = 2
    enable_memory_policy: bool = True
    fixed_rank: Optional[int] = None


@dataclass
class LossWeights:
    lambda_task: float = 1.0
    lambda_pres: float = 1.0
    lambda_steer: float = 1.0
    lambda_sparse_gate: float = 1e-2
    lambda_rank_budget: float = 1e-3
    lambda_load_balance: float = 1e-2
    lambda_specialist_div: float = 1e-3


@dataclass
class TrainingPhase:
    name: str
    dataloader: Any
    domain_id: Optional[str] = None


@dataclass
class TrainingSchedule:
    phases: list[TrainingPhase] = field(default_factory=list)
