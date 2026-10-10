import unittest

from dmadforge.config import Config
from dmadforge.model.engine import DmadEngine


class TestEngine(unittest.TestCase):
    def test_generate(self):
        cfg = Config()
        engine = DmadEngine(cfg)
        r = engine.generate("a cat", n_frames=3, seed="e")
        self.assertEqual(len(r["video_latent"]), 3)
        self.assertAlmostEqual(r["step_ratio"], 12.5)

    def test_distill_report(self):
        cfg = Config()
        engine = DmadEngine(cfg)
        r = engine.distill_report(epochs=10)
        self.assertLess(r["after"]["mmd2"], r["before"]["mmd2"])


if __name__ == "__main__":
    unittest.main()
