import unittest

from dmadforge.diffusion.scheduler import DDIMScheduler
from dmadforge.diffusion.unet import VideoUNet
from dmadforge.diffusion.vae import LatentVAE
from dmadforge.core.noise import LinearSchedule


class TestDiffusion(unittest.TestCase):
    def test_schedule_shape(self):
        s = LinearSchedule(steps=50)
        self.assertEqual(len(s.betas), 50)
        self.assertAlmostEqual(s.alpha_bar_at(0), 1.0 - 1e-4, places=5)

    def test_unet_shape(self):
        unet = VideoUNet(latent_dim=8, dim=8, n_frames=2, seed="u")
        x = [[0.1] * 8 for _ in range(2)]
        out = unet(x, 1, 0.5)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 8)

    def test_vae_roundtrip(self):
        vae = LatentVAE(in_dim=8, latent_dim=4, seed="v")
        x = [[0.1] * 8]
        z = vae.encode(x)
        self.assertEqual(len(z[0]), 4)
        x2 = vae.decode(z)
        self.assertEqual(len(x2[0]), 8)

    def test_ddim_sample_shape(self):
        sched = DDIMScheduler(steps=4)
        unet = VideoUNet(latent_dim=8, dim=8, n_frames=2, seed="d")

        def denoiser(x, t, ab):
            return unet(x, t, ab)

        out = sched.sample(denoiser, (2, 8), seed="s")
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 8)


if __name__ == "__main__":
    unittest.main()
