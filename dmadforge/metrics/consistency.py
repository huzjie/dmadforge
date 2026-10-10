"""Similarity helpers."""


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = (sum(x * x for x in a) + 1e-12) ** 0.5
    nb = (sum(y * y for y in b) + 1e-12) ** 0.5
    return dot / (na * nb)
