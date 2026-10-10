"""CPU backend: run the actual small diffusion engine on CPU."""
from . import register_backend
from ..config import Config
from ..model.engine import DmadEngine


@register_backend("cpu")
class CpuBackend:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.engine = DmadEngine(self.cfg)

    def train(self, epochs=None):
        return self.engine.train(epochs)

    def generate(self, prompt, n_frames=4, seed=""):
        r = self.engine.generate(prompt, n_frames=n_frames, seed=seed)
        return r["video_latent"]

    def evaluate(self, shape=(8, 16), seed="eval"):
        return self.engine.evaluate(shape, seed)
