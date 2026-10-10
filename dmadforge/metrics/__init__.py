"""Generation-quality metrics."""
from .fid import fid_proxy, frechet_distance
from .psnr import psnr, mse
from .lpips import lpips_proxy
from .consistency import cosine_sim

__all__ = ["fid_proxy", "frechet_distance", "psnr", "mse", "lpips_proxy", "cosine_sim"]
