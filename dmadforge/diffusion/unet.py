"""Video U-Net / DiT-style denoiser (reference, small dims).

Predicts the noise residual given a noisy latent, a timestep, and an optional
audio conditioning. Latents live in `latent_dim` space; the model projects them
up to a hidden width `dim`, processes, and projects back. In the DMAD story the
student shares this architecture but injects Rank-128 LoRA adapters on the
input/output projections.
"""
from ..core.nn import MLP, Linear, Attention
from ..core.tensor import add, mul, scale


class VideoUNet:
    def __init__(self, latent_dim=4, dim=64, n_frames=4, seed="", lora_cfg=None):
        self.latent_dim = latent_dim
        self.dim = dim
        self.n_frames = n_frames
        self.in_proj = Linear(latent_dim, dim, seed + "in")
        self.blocks = [MLP(dim, dim * 2, seed + "b%d" % i) for i in range(3)]
        self.attn = Attention(dim, seed + "attn")
        self.time_emb = Linear(1, dim, seed + "time")
        self.out_proj = Linear(dim, latent_dim, seed + "out")
        if lora_cfg is not None:
            from ..lora.adapter import LoRALinear
            self.in_proj = LoRALinear(self.in_proj, latent_dim, dim, lora_cfg, seed + "lora_in")
            self.out_proj = LoRALinear(self.out_proj, dim, latent_dim, lora_cfg, seed + "lora_out")

    def __call__(self, x, t, ab):
        h = self.in_proj(x)
        te = self.time_emb([[float(t)]])
        h = add(h, te)
        for blk in self.blocks:
            h = add(h, blk(h))
        h = self.attn(h)
        return self.out_proj(h)
