"""Teacher: a full 50-step DDIM diffusion model producing high-quality latents.

The teacher is the "gold" generator. Its latents define the target distribution
that the few-step student must match. In a real 33B setting the teacher is the
frozen pretrained model; here we implement a deterministic reference teacher.
"""
from ..diffusion.scheduler import DDIMScheduler
from ..diffusion.unet import VideoUNet


class TeacherDiffusion:
    def __init__(self, latent_dim=4, dim=64, steps=50, seed="teacher"):
        self.latent_dim = latent_dim
        self.dim = dim
        self.steps = steps
        self.scheduler = DDIMScheduler(steps=steps)
        self.unet = VideoUNet(latent_dim=latent_dim, dim=dim, seed=seed + "unet")

    def _denoiser(self, x, t, ab):
        return self.unet(x, t, ab)

    def sample(self, shape, seed=""):
        """Generate gold latents via the full multi-step chain."""
        return self.scheduler.sample(self._denoiser, shape, seed=seed)
