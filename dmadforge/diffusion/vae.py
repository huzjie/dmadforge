"""Latent VAE: compress video frames to a low-dim latent and decode back.

We keep a tiny deterministic encoder/decoder. The latent has `latent_dim`
channels; audio shares the same latent width so video and audio fuse in one
space (see audiovideo.fusion).
"""
from ..core.nn import Linear
from ..core.tensor import matmul


class LatentVAE:
    def __init__(self, in_dim=48, latent_dim=4, seed=""):
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.enc = Linear(in_dim, latent_dim, seed + "enc")
        self.dec = Linear(latent_dim, in_dim, seed + "dec")

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)
