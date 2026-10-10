"""Low-rank adaptation (LoRA) for frozen base weights."""
from .adapter import LoRALinear, LoRAConfig, inject_lora
from .decompose import low_rank_init, compute_lora_params

__all__ = ["LoRALinear", "LoRAConfig", "inject_lora", "low_rank_init", "compute_lora_params"]
