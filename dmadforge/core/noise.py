"""Diffusion noise schedules (linear and cosine) and noising helpers."""


class LinearSchedule:
    def __init__(self, beta_start=1e-4, beta_end=0.02, steps=50):
        self.steps = steps
        self.betas = [beta_start + (beta_end - beta_start) * t / max(steps - 1, 1) for t in range(steps)]
        self.alphas = [1.0 - b for b in self.betas]
        self.alpha_bar = []
        acc = 1.0
        for a in self.alphas:
            acc *= a
            self.alpha_bar.append(acc)

    def alpha_bar_at(self, t):
        return self.alpha_bar[max(0, min(t, self.steps - 1))]

    def sigma_at(self, t):
        return (1.0 - self.alpha_bar_at(t)) ** 0.5


class CosineSchedule:
    def __init__(self, steps=50, s=0.008):
        import math
        self.steps = steps
        self.alpha_bar = []
        for t in range(steps):
            f = math.cos(((t / steps + s) / (1.0 + s)) * math.pi / 2.0) ** 2
            self.alpha_bar.append(min(1.0, f))


def add_noise(x, noise, alpha_bar):
    """x = sqrt(alpha_bar) * x + sqrt(1 - alpha_bar) * noise."""
    sa = alpha_bar ** 0.5
    sn = (1.0 - alpha_bar) ** 0.5
    return [[x[i][j] * sa + noise[i][j] * sn for j in range(len(x[i]))] for i in range(len(x))]
