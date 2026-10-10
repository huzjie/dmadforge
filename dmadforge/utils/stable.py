"""Deterministic pseudo-random helpers seeded by md5(key).

We deliberately avoid `random.Random(hash(...))` because Python's hash() is
salted per-process (not reproducible across runs). Instead we derive an 8-byte
seed from md5(key) and feed it into `random.Random`, which is both fast and
entropy-rich enough for stable mock evaluation.
"""
import hashlib
import random


def stable_seed(key):
    if not isinstance(key, str):
        key = repr(key)
    return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "big")


def _rng(key):
    return random.Random(stable_seed(key))


def stable_float(key, lo=0.0, hi=1.0):
    return _rng(key).uniform(lo, hi)


def stable_ints(key, n, lo, hi):
    r = _rng(key)
    return [r.randint(lo, hi) for _ in range(n)]


def stable_vector(key, n, lo=-1.0, hi=1.0):
    r = _rng(key)
    return [r.uniform(lo, hi) for _ in range(n)]


def stable_choice(key, items):
    r = _rng(key)
    return items[r.randrange(len(items))]


def stable_bool(key, p=0.5):
    return _rng(key).random() < p
