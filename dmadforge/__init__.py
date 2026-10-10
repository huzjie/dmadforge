"""dmadforge: distribution-matching adversarial distillation for video generation.

A zero-dependency, fully runnable reference implementation of the engineering
ideas behind ByteDance x TAMU's DMAD (Distribution Matching as Adversarial
Distillation), which compresses a 33B audio-video diffusion model from a
50-step denoising chain down to just 4 steps using a 1.4GB Rank-128 LoRA --
no base-model reconstruction required.

The core insight: instead of minimizing point-wise MSE between teacher and
student denoised samples (which blurs and collapses modes), DMAD treats
distillation as a *distribution matching* problem. A GAN-style discriminator
estimates the divergence between the teacher distribution and the student
distribution, and the student is trained to minimize that divergence.

This repo distills the reusable engineering ideas into one framework:

1. Few-step distillation -- teacher runs a 50-step DDIM chain, student runs a
   4-step chain, and the student's per-step correction is learned.
2. Distribution matching -- adversarial loss + MMD / Wasserstein objectives
   keep the student output *distribution* aligned with the teacher, not just
   pointwise close.
3. Rank-128 LoRA adapters -- freeze the base weights and inject low-rank A*B
   adapters (rank 128) so only ~1.4GB of parameters are trained.
4. Audio-video joint latents -- a shared latent space fuses video frames and
   32kHz audio so both are generated in one forward pass.
5. Five backends -- deterministic trainable mock / cpu / openai / vllm /
   transformers -- plus an OpenAI-compatible stdlib HTTP server.

Highlights
----------
- Zero-dependency pure-Python core (diffusion + distillation + benchmark)
- Deterministic trainable mock backend (student quality 0.5 -> 1.0)
- Step-reduction benchmark (50 -> 4 steps, ~12.5x fewer denoising steps)
- FID/MMD distribution-matching benchmark with known ground truth
- CLI / Docker / K8s / Helm / CI + prompt library + recipes + docs
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
