"""Rank-r LoRA linear adapter: W' = W + (alpha/r) * A @ B.

Freeze the base weight W and only train the low-rank factors A (in x r) and
B (r x out). For the DMAD story, a rank-128 adapter over a 33B backbone is
~1.4GB -- the only part that gets trained during distillation.
"""
from dataclasses import dataclass

from ..core.tensor import matmul, add, scale


@dataclass
class LoRAConfig:
    rank: int = 128
    alpha: float = 128.0
    dropout: float = 0.0

    @property
    def scaling(self):
        return self.alpha / self.rank if self.rank else 1.0


class LoRALinear:
    def __init__(self, base, in_dim, out_dim, cfg=None, seed=""):
        import math
        import random
        self.base = base
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cfg = cfg or LoRAConfig()
        r = random.Random(abs(hash(seed)) % (2 ** 31) if seed else 7)
        k = 1.0 / math.sqrt(in_dim)
        self.A = [[r.uniform(-k, k) for _ in range(in_dim)] for _ in range(self.cfg.rank)]
        self.B = [[r.uniform(-k, k) for _ in range(self.cfg.rank)] for _ in range(out_dim)]

    def lora_delta(self, x):
        # x (m x in) @ A^T (in x r) @ B^T (r x out)
        at = [list(col) for col in zip(*self.A)]
        bt = [list(col) for col in zip(*self.B)]
        xa = matmul(x, at)
        return matmul(xa, bt)

    def __call__(self, x):
        base_out = self.base(x)
        delta = scale(self.lora_delta(x), self.cfg.scaling)
        return add(base_out, delta)


def inject_lora(model_dict, in_dim, out_dim, rank=128, seed=""):
    """Wrap every linear layer in model_dict with a LoRA adapter."""
    out = {}
    for name, layer in model_dict.items():
        if hasattr(layer, "w"):
            out[name] = LoRALinear(layer, in_dim, out_dim, LoRAConfig(rank=rank), seed=seed + name)
        else:
            out[name] = layer
    return out
