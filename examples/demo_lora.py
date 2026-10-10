"""Demo: LoRA parameter accounting for a 33B backbone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmadforge.lora.decompose import compute_lora_params, trainable_fraction
from dmadforge.lora.adapter import LoRALinear, LoRAConfig
from dmadforge.core.nn import Linear


def main():
    in_dim = out_dim = 4096
    rank = 128
    lora = compute_lora_params(in_dim, out_dim, rank)
    full = 33_000_000_000
    print("LoRA trainable params: %.2fM" % (lora / 1e6))
    print("trainable fraction  : %.5f" % trainable_fraction(full, lora))
    base = Linear(8, 8, seed="b")
    wrapped = LoRALinear(base, 8, 8, LoRAConfig(rank=4), seed="l")
    out = wrapped([[0.1] * 8])
    print("LoRALinear output shape:", len(out[0]))


if __name__ == "__main__":
    main()
