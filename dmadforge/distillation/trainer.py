"""DMAD training loop: pull the student distribution onto the teacher's.

The deterministic mock uses a scalar `skill` in [0, 1] to control how close the
student's latents are to the teacher's. Training uses a constant-positive delta
so `skill` climbs monotonically to 1.0 (never gets stuck at 0), which drives
the observable metrics: MMD/FID fall, discriminator accuracy falls to chance.
"""
from ..utils.stable import stable_vector


class DMADTrainer:
    def __init__(self, cfg, discriminator=None):
        self.cfg = cfg
        self.skill = cfg.train.skill_start
        self.discriminator = discriminator
        self.history = []

    def generate_teacher(self, shape, seed=""):
        """Deterministic gold latents (target distribution P)."""
        n, c = shape
        return [stable_vector("gold:" + seed + ":" + str(i), c) for i in range(n)]

    def generate_student(self, shape, seed=""):
        """student_latent = skill * teacher_latent + (1 - skill) * noise."""
        gold = self.generate_teacher(shape, seed=seed)
        n, c = shape
        out = []
        for i in range(n):
            noise = stable_vector("noise:" + seed + ":" + str(i), c)
            out.append([gold[i][j] * self.skill + noise[j] * (1.0 - self.skill) for j in range(c)])
        return out

    def generate_av(self, shape, seed=""):
        """Generate video + audio latents from a SHARED gold (joint generation).

        Both are blends of the same gold latent with *different* noise, so as
        skill -> 1 they converge to the identical gold and av-consistency -> 1.
        """
        gold = self.generate_teacher(shape, seed=seed)
        n, c = shape
        video = []
        audio = []
        for i in range(n):
            nv = stable_vector("noise_v:" + seed + ":" + str(i), c)
            na = stable_vector("noise_a:" + seed + ":" + str(i), c)
            video.append([gold[i][j] * self.skill + nv[j] * (1.0 - self.skill) for j in range(c)])
            audio.append([gold[i][j] * self.skill + na[j] * (1.0 - self.skill) for j in range(c)])
        return video, audio

    def step(self):
        """One training step. Constant-positive delta keeps skill monotonic."""
        delta = self.cfg.train.lr
        self.skill = min(1.0, self.skill + delta)
        self.history.append(self.skill)
        return self.skill

    def train(self, epochs=None):
        epochs = epochs or self.cfg.train.epochs
        for _ in range(epochs):
            self.step()
        return self.skill

    def evaluate(self, shape=(8, 16), seed="eval"):
        """Return a dict of observable metrics at the current skill."""
        from .losses import mmd2, wasserstein1, reconstruction_loss
        gold = self.generate_teacher(shape, seed=seed)
        stu = self.generate_student(shape, seed=seed)
        return {
            "skill": self.skill,
            "mmd2": mmd2(stu, gold, self.cfg.distillation.mmd_sigma),
            "wasserstein1": wasserstein1(stu, gold),
            "recon_mse": reconstruction_loss(stu, gold),
            "disc_acc": self.discriminator.accuracy(gold, stu) if self.discriminator else 0.5,
        }
