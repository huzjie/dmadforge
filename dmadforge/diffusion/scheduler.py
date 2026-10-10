"""DDIM / DDPM samplers.

The teacher runs a full 50-step DDIM chain; the student runs a 4-step chain
where each step uses a LoRA-adjusted denoiser. The scheduler only decides the
noise schedule and the per-step update rule -- the network is injected.
"""
from ..core.noise import LinearSchedule


class DDPMStep:
    """One denoising step for a DDPM-style forward process."""

    def step(self, x, t, alpha_bar, predicted_noise, prev_alpha_bar):
        import math
        at = alpha_bar ** 0.5
        pn = (1.0 - alpha_bar) ** 0.5
        x0 = [[(x[i][j] - pn * predicted_noise[i][j]) / max(at, 1e-6) for j in range(len(x[i]))] for i in range(len(x))]
        pat = prev_alpha_bar ** 0.5
        ppn = (1.0 - prev_alpha_bar) ** 0.5
        return [[x0[i][j] * pat + predicted_noise[i][j] * ppn for j in range(len(x0[i]))] for i in range(len(x0))]


class DDIMScheduler:
    def __init__(self, steps=50):
        self.schedule = LinearSchedule(steps=steps)
        self.steps = steps

    def timesteps(self):
        return list(range(self.steps - 1, -1, -1))

    def alpha_bar(self, t):
        return self.schedule.alpha_bar_at(t)

    def sample(self, denoiser, shape, seed="", eta=0.0):
        """Deterministic DDIM sample from pure noise using denoiser(x, t, ab)."""
        from ..utils.stable import stable_vector
        import math
        n, c = shape
        x = [stable_vector("noise:" + seed, c) for _ in range(n)]
        ts = self.timesteps()
        for i, t in enumerate(ts):
            ab = self.alpha_bar(t)
            noise = denoiser(x, t, ab)
            at = ab ** 0.5
            pn = (1.0 - ab) ** 0.5
            x0 = [[(x[r][j] - pn * noise[r][j]) / max(at, 1e-6) for j in range(c)] for r in range(n)]
            if i + 1 < len(ts):
                nt = ts[i + 1]
                nab = self.alpha_bar(nt)
                pat = nab ** 0.5
                ppn = (1.0 - nab) ** 0.5
                x = [[x0[r][j] * pat + noise[r][j] * ppn for j in range(c)] for r in range(n)]
            else:
                x = x0
        return x
