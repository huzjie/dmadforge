import unittest

from dmadforge.config import Config
from dmadforge.model.engine import DmadEngine
from dmadforge.bench.summary import run_all, summarize


class TestBench(unittest.TestCase):
    def test_run_all_keys(self):
        cfg = Config()
        engine = DmadEngine(cfg)
        results = run_all(engine, shape=(4, 8), seed="t")
        for k in ("speed", "quality", "distribution", "audio"):
            self.assertIn(k, results)

    def test_summarize_text(self):
        cfg = Config()
        engine = DmadEngine(cfg)
        engine.train(5)
        text = summarize(run_all(engine, shape=(4, 8), seed="t"))
        self.assertIn("speed", text)
        self.assertIn("distribution", text)


if __name__ == "__main__":
    unittest.main()
