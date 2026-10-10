"""Deterministic discriminator: nearest-neighbor "realness" against the teacher.

This is a deterministic proxy for the GAN decision boundary. The discriminator
fits the teacher's empirical distribution (its latents) and scores a latent by
how close it is to the nearest teacher sample:

    score(z) = 1 / (1 + min_distance(z, teacher_latents))

Teacher latents have distance 0 -> score 1 (real). A noisy student latent at
low skill is far from any teacher sample -> score ~0 (fake). As the student's
distribution converges to the teacher's (skill -> 1), its latents land on the
teacher samples, the discriminator can no longer separate them, and its
accuracy collapses to 0.5 (chance) -- the classic adversarial-convergence
signal.
"""


class Discriminator:
    def __init__(self, dim=64, seed="disc"):
        self.dim = dim
        self._ref = None
        self._ndim = dim

    def fit(self, teacher_latents):
        self._ref = [list(r) for r in teacher_latents]
        if self._ref:
            self._ndim = len(self._ref[0])

    def score(self, latent):
        if self._ref is None:
            return [[0.5] for _ in latent]
        ndim = self._ndim
        out = []
        for r in latent:
            d = min(sum((r[j] - q[j]) ** 2 for j in range(ndim)) for q in self._ref) ** 0.5
            out.append([1.0 / (1.0 + d)])
        return out

    def predict(self, latent, threshold=0.5):
        s = self.score(latent)
        return [1.0 if x[0] > threshold else 0.0 for x in s]

    def accuracy(self, teacher_z, student_z):
        self.fit(teacher_z)
        tp = self.predict(teacher_z)
        sp = self.predict(student_z)
        correct = sum(1 for x in tp if x > 0.5) + sum(1 for x in sp if x <= 0.5)
        return correct / max(1, len(tp) + len(sp))
