"""Frechet Inception Distance (FID) proxy on latent statistics.

True FID compares Inception-v3 activations of real and generated images. Here
we compare the Gaussian statistics (mean + covariance) of two latent sets, which
is a valid distribution distance on the latent space -- exactly the DMAD target.
"""
import math


def _mean_cov(rows):
    n = len(rows)
    d = len(rows[0]) if rows else 0
    mu = [sum(r[j] for r in rows) / n for j in range(d)]
    cov = [[0.0] * d for _ in range(d)]
    for r in rows:
        for i in range(d):
            for j in range(d):
                cov[i][j] += (r[i] - mu[i]) * (r[j] - mu[j])
    for i in range(d):
        for j in range(d):
            cov[i][j] /= max(1, n - 1)
    return mu, cov


def frechet_distance(mu1, cov1, mu2, cov2):
    d = len(mu1)
    diff = sum((mu1[i] - mu2[i]) ** 2 for i in range(d))
    # trace(cov1 + cov2 - 2*sqrt(cov1*cov2)) approximated by a scalar trace
    tr1 = sum(cov1[i][i] for i in range(d))
    tr2 = sum(cov2[i][i] for i in range(d))
    cross = sum((cov1[i][i] * cov2[i][i]) ** 0.5 for i in range(d))
    return diff + tr1 + tr2 - 2 * cross


def fid_proxy(real, generated):
    """FID-like distance between two latent sets (lower is better)."""
    mu1, cov1 = _mean_cov(real)
    mu2, cov2 = _mean_cov(generated)
    return max(0.0, frechet_distance(mu1, cov1, mu2, cov2))
