"""Distribution-matching and adversarial losses.

The core of DMAD: match *distributions*, not pointwise samples.

- mmd2          : Maximum Mean Discrepancy^2 with an RBF kernel -- a proper
                  integral-probability-metric that vanishes iff the two
                  distributions are equal.
- wasserstein1  : 1-Wasserstein distance (via sorted quantiles) -- a robust
                  transport cost.
- adversarial_loss : non-saturating GAN loss for the student (fool the disc).
- reconstruction_loss : optional MSE anchor so the student stays in-basin.
"""
import math


def _rbf(x, y, sigma=1.0):
    return math.exp(-sum((a - b) ** 2 for a, b in zip(x, y)) / (2.0 * sigma * sigma))


def mmd2(x, y, sigma=1.0):
    """Unbiased MMD^2 estimator with RBF kernel."""
    n = len(x)
    m = len(y)
    xx = sum(_rbf(x[i], x[j], sigma) for i in range(n) for j in range(n) if i != j) / max(1, n * (n - 1))
    yy = sum(_rbf(y[i], y[j], sigma) for i in range(m) for j in range(m) if i != j) / max(1, m * (m - 1))
    xy = sum(_rbf(x[i], y[j], sigma) for i in range(n) for j in range(m)) / max(1, n * m)
    return max(0.0, xx + yy - 2.0 * xy)


def wasserstein1(x, y):
    """1-Wasserstein via sorted scalar projections (cheap, deterministic)."""
    xs = sorted(sum(r) for r in x)
    ys = sorted(sum(r) for r in y)
    n = max(1, len(xs))
    return sum(abs(xs[i] - ys[i]) for i in range(min(len(xs), len(ys)))) / n


def adversarial_loss(disc_scores_student, target_real=1.0):
    """Non-saturating GAN loss: student wants disc to call it real."""
    if not disc_scores_student:
        return 0.0
    return -sum(math.log(max(1e-6, min(1.0, s[0])) + 1e-6) for s in disc_scores_student) / len(disc_scores_student)


def reconstruction_loss(x, y):
    n = max(1, len(x))
    return sum(sum((a - b) ** 2 for a, b in zip(x[i], y[i])) for i in range(len(x))) / n
