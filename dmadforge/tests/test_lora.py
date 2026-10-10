import unittest

from dmadforge.lora.adapter import LoRALinear, LoRAConfig, inject_lora
from dmadforge.lora.decompose import compute_lora_params, trainable_fraction
from dmadforge.core.nn import Linear


class TestLoRA(unittest.TestCase):
    def test_lora_output_shape(self):
        base = Linear(8, 8, seed="b")
        cfg = LoRAConfig(rank=4)
        lora = LoRALinear(base, 8, 8, cfg, seed="l")
        out = lora([[0.1] * 8])
        self.assertEqual(len(out[0]), 8)

    def test_rank128_params(self):
        p = compute_lora_params(4096, 4096, 128)
        self.assertEqual(p, 4096 * 128 * 2)

    def test_trainable_fraction(self):
        full = 33_000_000_000
        lora = 1_400_000_000
        self.assertLess(trainable_fraction(full, lora), 0.05)

    def test_inject_lora(self):
        layers = {"fc": Linear(8, 8, seed="x")}
        out = inject_lora(layers, 8, 8, rank=4, seed="i")
        self.assertIsInstance(out["fc"], LoRALinear)


if __name__ == "__main__":
    unittest.main()
