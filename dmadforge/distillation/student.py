"""Student: a 4-step generator with Rank-128 LoRA over the teacher backbone.

The student reuses the teacher's weights (frozen) plus a low-rank adapter.
Its few-step sampler applies a learned per-step correction. The student's
output distribution is what we pull toward the teacher's via adversarial +
distribution-matching losses.
"""
from ..diffusion.scheduler import DDIMScheduler
from ..diffusion.unet import VideoUNet
from ..lora.adapter import LoRAConfig
from ..core.tensor import matmul


class StudentGenerator:
    def __init__(self, latent_dim=4, dim=64, teacher_steps=50, student_steps=4, lora_rank=128, seed="student"):
        self.latent_dim = latent_dim
        self.dim = dim
        self.teacher_steps = teacher_steps
        self.student_steps = student_steps
        self.lora_rank = lora_rank
        self.scheduler = DDIMScheduler(steps=student_steps)
        cfg = LoRAConfig(rank=lora_rank)
        self.unet = VideoUNet(latent_dim=latent_dim, dim=dim, lora_cfg=cfg, seed=seed + "unet")

    def _denoiser(self, x, t, ab):
        return self.unet(x, t, ab)

    def sample(self, shape, seed=""):
        return self.scheduler.sample(self._denoiser, shape, seed=seed)

    def step_ratio(self):
        return self.teacher_steps / self.student_steps
