import unittest

from dmadforge.core.tensor import matmul, add, relu, softmax, layernorm, normalize, dot
from dmadforge.core.nn import Linear, MLP, Attention


class TestTensor(unittest.TestCase):
    def test_matmul(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        out = matmul(a, b)
        self.assertAlmostEqual(out[0][0], 19)
        self.assertAlmostEqual(out[1][1], 50)

    def test_add_broadcast_scalar(self):
        out = add([[1.0, 2.0]], [[3.0]])
        self.assertAlmostEqual(out[0][0], 4.0)
        self.assertAlmostEqual(out[0][1], 5.0)

    def test_softmax_sums_to_one(self):
        out = softmax([[1.0, 2.0, 3.0]])
        self.assertAlmostEqual(sum(out[0]), 1.0, places=5)

    def test_layernorm_zero_mean(self):
        out = layernorm([[1.0, 3.0, 5.0, 7.0]])
        self.assertAlmostEqual(sum(out[0]), 0.0, places=5)

    def test_normalize_unit_norm(self):
        v = normalize([3.0, 4.0])
        self.assertAlmostEqual((v[0] ** 2 + v[1] ** 2) ** 0.5, 1.0, places=5)


class TestNN(unittest.TestCase):
    def test_linear_shape(self):
        lin = Linear(4, 8, seed="t")
        out = lin([[0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 8)

    def test_mlp_shape(self):
        mlp = MLP(8, 16, seed="t")
        out = mlp([[0.1] * 8])
        self.assertEqual(len(out[0]), 8)

    def test_attention_shape(self):
        attn = Attention(8, seed="t")
        out = attn([[0.1] * 8, [0.2] * 8])
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 8)


if __name__ == "__main__":
    unittest.main()
