import unittest

from dmadforge.config import Config
from dmadforge.pipeline.distill_pipeline import DistillPipeline


class TestPipeline(unittest.TestCase):
    def test_run(self):
        cfg = Config()
        p = DistillPipeline(cfg)
        report = p.run(epochs=10, shape=(4, 8))
        self.assertLess(report["after"]["recon_mse"], report["before"]["recon_mse"])
        self.assertIn("fid", report)

    def test_summarize(self):
        cfg = Config()
        p = DistillPipeline(cfg)
        report = p.run(epochs=5, shape=(4, 8))
        text = p.summarize(report)
        self.assertIn("skill", text)


if __name__ == "__main__":
    unittest.main()
