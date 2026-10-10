"""Low-rank initialization utilities and parameter accounting."""
from .adapter import LoRAConfig


def low_rank_init(in_dim, out_dim, rank, seed=""):
    from .adapter import LoRALinear
    from ..core.nn import Linear
    return LoRALinear(Linear(in_dim, out_dim, seed), in_dim, out_dim, LoRAConfig(rank=rank), seed)


def compute_lora_params(in_dim, out_dim, rank):
    """Trainable LoRA params = in_dim*rank + out_dim*rank (A + B)."""
    return in_dim * rank + out_dim * rank


def trainable_fraction(full_params, lora_params):
    return lora_params / full_params if full_params else 0.0
