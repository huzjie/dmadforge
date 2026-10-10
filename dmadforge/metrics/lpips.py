"""LPIPS proxy: perceptual distance via feature-space MSE.

True LPIPS uses a pretrained network's activations; here we approximate it by
projecting latents through a fixed random linear map then computing MSE -- a
deterministic, dependency-free stand-in for perceptual similarity.
"""
from ..utils.stable import stable_vector


def lpips_proxy(a, b, feat_dim=32, seed="lpips"):
    proj = [stable_vector("proj:" + seed + ":" + str(i), len(a[0])) for i in range(feat_dim)]
    fa = [[sum(a[i][j] * proj[k][j] for j in range(len(a[i]))) for k in range(feat_dim)] for i in range(len(a))]
    fb = [[sum(b[i][j] * proj[k][j] for j in range(len(b[i]))) for k in range(feat_dim)] for i in range(len(b))]
    n = max(1, len(a))
    return sum(sum((fa[i][k] - fb[i][k]) ** 2 for k in range(feat_dim)) for i in range(n)) / n
