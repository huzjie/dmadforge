import unittest

from dmadforge.metrics.psnr import mse, psnr
from dmadforge.metrics.fid import fid_proxy
from dmadforge.metrics.lpips import lpips_proxy
from dmadforge.metrics.consistency import cosine_sim


class TestMetrics(unittest.TestCase):
    def test_mse_zero(self):
        x = [[1.0, 2.0], [3.0, 4.0]]
        self.assertAlmostEqual(mse(x, x), 0.0, places=5)

    def test_psnr_infinite(self):
        x = [[1.0, 2.0]]
        self.assertEqual(psnr(x, x), float("inf"))

    def test_fid_zero(self):
        x = [[0.1, 0.2] for _ in range(5)]
        self.assertAlmostEqual(fid_proxy(x, x), 0.0, places=5)

    def test_lpips_zero(self):
        x = [[0.1, 0.2] for _ in range(3)]
        self.assertAlmostEqual(lpips_proxy(x, x), 0.0, places=5)

    def test_cosine_sim(self):
        self.assertAlmostEqual(cosine_sim([1, 0], [1, 0]), 1.0, places=5)
        self.assertAlmostEqual(cosine_sim([1, 0], [0, 1]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
