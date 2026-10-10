"""Minimal pure-Python tensor helpers (nested lists).

Reference implementation: correctness over speed. Use small dims in doctor and
examples; the deterministic mock backend avoids heavy tensor math entirely.
"""


def _rows(x):
    return x if isinstance(x, list) else [[x]]


def matmul(a, b):
    """Matrix multiply a (m x k) @ b (k x n) -> (m x n)."""
    a = _rows(a)
    b = _rows(b)
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    return [[sum(a[i][kk] * b[kk][j] for kk in range(k)) for j in range(n)] for i in range(m)]


def _binop(a, b, op):
    a = _rows(a)
    b = _rows(b)
    if len(b) == 1 and len(b[0]) == 1:
        s = b[0][0]
        return [[op(row[j], s) for j in range(len(row))] for row in a]
    if len(a) == 1 and len(a[0]) == 1:
        s = a[0][0]
        return [[op(s, b[i][j]) for j in range(len(b[i]))] for i in range(len(b))]
    if len(b) == 1 and len(b[0]) == len(a[0]):
        brow = b[0]
        return [[op(a[i][j], brow[j]) for j in range(len(a[i]))] for i in range(len(a))]
    if len(a) == 1 and len(a[0]) == len(b[0]):
        arow = a[0]
        return [[op(arow[j], b[i][j]) for j in range(len(b[i]))] for i in range(len(b))]
    return [[op(a[i][j], b[i][j]) for j in range(len(a[i]))] for i in range(len(a))]


def add(a, b):
    return _binop(a, b, lambda x, y: x + y)


def sub(a, b):
    return _binop(a, b, lambda x, y: x - y)


def mul(a, b):
    return _binop(a, b, lambda x, y: x * y)


def scale(a, s):
    a = _rows(a)
    return [[a[i][j] * s for j in range(len(a[i]))] for i in range(len(a))]


def relu(x):
    x = _rows(x)
    return [[max(0.0, v) for v in row] for row in x]


def gelu(x):
    import math
    x = _rows(x)
    return [[0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3))) for v in row] for row in x]


def _max_row(row):
    return max(row)


def softmax(x):
    x = _rows(x)
    out = []
    for row in x:
        m = max(row)
        ex = [math_exp(v - m) for v in row]
        s = sum(ex)
        out.append([e / s for e in ex])
    return out


def math_exp(v):
    import math
    return math.exp(v)


def layernorm(x, eps=1e-5):
    x = _rows(x)
    out = []
    for row in x:
        n = len(row)
        mu = sum(row) / n
        var = sum((v - mu) ** 2 for v in row) / n
        std = (var + eps) ** 0.5
        out.append([(v - mu) / std for v in row])
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def normalize(v):
    n = (sum(x * x for x in v) + 1e-12) ** 0.5
    return [x / n for x in v]


def mean(x):
    x = _rows(x)
    total = 0.0
    n = 0
    for row in x:
        total += sum(row)
        n += len(row)
    return total / n if n else 0.0
