"""Tiny neural layers for the reference DiT / discriminator."""
import math

from .tensor import matmul, add, mul, scale, relu, gelu, layernorm, softmax, normalize


class Linear:
    def __init__(self, in_dim, out_dim, seed="", bias=True):
        import random
        r = random.Random(abs(hash(seed)) % (2 ** 31) if seed else 42)
        k = 1.0 / math.sqrt(in_dim)
        self.w = [[r.uniform(-k, k) for _ in range(in_dim)] for _ in range(out_dim)]
        self.b = [r.uniform(-k, k) for _ in range(out_dim)] if bias else [0.0] * out_dim

    def __call__(self, x):
        y = matmul(x, self._wt())
        return [[y[i][j] + self.b[j] for j in range(len(self.b))] for i in range(len(y))]

    def _wt(self):
        return [list(col) for col in zip(*self.w)]


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps

    def __call__(self, x):
        if x and isinstance(x[0], (list, tuple)):
            return [self._norm(row) for row in x]
        return self._norm(x)

    def _norm(self, row):
        n = len(row)
        mu = sum(row) / n
        var = sum((v - mu) ** 2 for v in row) / n
        std = (var + self.eps) ** 0.5
        return [(v - mu) / std for v in row]


class MLP:
    def __init__(self, dim, hidden, seed=""):
        self.fc1 = Linear(dim, hidden, seed + "fc1")
        self.fc2 = Linear(hidden, dim, seed + "fc2")

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Attention:
    """Single-head scaled dot-product attention (reference)."""

    def __init__(self, dim, seed=""):
        self.dim = dim
        self.q = Linear(dim, dim, seed + "q")
        self.k = Linear(dim, dim, seed + "k")
        self.v = Linear(dim, dim, seed + "v")
        self.o = Linear(dim, dim, seed + "o")

    def __call__(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        kt = [list(col) for col in zip(*k)]
        scores = matmul(q, kt)
        sc = 1.0 / math.sqrt(self.dim)
        scores = [[s * sc for s in row] for row in scores]
        attn = softmax(scores)
        out = matmul(attn, v)
        return self.o(out)
