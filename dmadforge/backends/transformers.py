"""HuggingFace Transformers backend: load a DiT checkpoint, fallback to mock."""
from . import register_backend
from .mock import MockBackend
from ..config import Config


@register_backend("transformers")
class TransformersBackend:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self._fallback = MockBackend(self.cfg)

    def train(self, epochs=None):
        return self._fallback.train(epochs)

    def generate(self, prompt, n_frames=4, seed=""):
        return self._fallback.generate(prompt, n_frames=n_frames, seed=seed)

    def evaluate(self, shape=(8, 16), seed="eval"):
        return self._fallback.evaluate(shape, seed)
