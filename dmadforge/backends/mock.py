"""Deterministic trainable mock backend.

The mock models the distillation dynamics directly: it holds a scalar `skill`
that scales how close the student's latents are to the teacher's. Training
raises `skill` monotonically, so every observable metric (MMD/FID fall, disc
accuracy falls to chance, av-consistency rises) moves in the right direction.
"""
from . import register_backend
from ..config import Config


@register_backend("mock")
class MockBackend:
    def __init__(self, cfg=None):
        cfg = cfg or Config()
        self.cfg = cfg
        self.skill = cfg.train.skill_start

    def train(self, epochs=None):
        lr = self.cfg.train.lr
        epochs = epochs or self.cfg.train.epochs
        for _ in range(epochs):
            self.skill = min(1.0, self.skill + lr)
        return self.skill

    def generate(self, prompt, n_frames=4, seed=""):
        from ..utils.stable import stable_vector
        m = self.cfg.model
        shape = (n_frames, m.latent_dim)
        gold = [stable_vector("gold:" + seed + ":" + str(i), m.latent_dim) for i in range(n_frames)]
        out = []
        for i in range(n_frames):
            noise = stable_vector("noise:" + seed + ":" + str(i), m.latent_dim)
            out.append([gold[i][j] * self.skill + noise[j] * (1.0 - self.skill) for j in range(m.latent_dim)])
        return out

    def evaluate(self, shape=(8, 16), seed="eval"):
        from ..distillation.losses import mmd2, wasserstein1, reconstruction_loss
        from ..utils.stable import stable_vector
        n, c = shape
        gold = [stable_vector("gold:" + seed + ":" + str(i), c) for i in range(n)]
        stu = []
        for i in range(n):
            noise = stable_vector("noise:" + seed + ":" + str(i), c)
            stu.append([gold[i][j] * self.skill + noise[j] * (1.0 - self.skill) for j in range(c)])
        return {
            "skill": self.skill,
            "mmd2": mmd2(stu, gold, self.cfg.distillation.mmd_sigma),
            "wasserstein1": wasserstein1(stu, gold),
            "recon_mse": reconstruction_loss(stu, gold),
        }
