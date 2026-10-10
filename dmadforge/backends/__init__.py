"""Model backends. Each backend registers itself via @register_backend."""
from ..config import Config

_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


def list_backends():
    return sorted(_BACKENDS.keys())


def get_backend(name, cfg=None):
    if name not in _BACKENDS:
        raise ValueError("unknown backend %r (available: %s)" % (name, ", ".join(list_backends())))
    return _BACKENDS[name](cfg)


from . import mock, cpu, openai, vllm, transformers  # noqa: E402,F401
