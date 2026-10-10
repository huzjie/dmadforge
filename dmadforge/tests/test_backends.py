import unittest

from dmadforge.config import Config
from dmadforge.backends import list_backends, get_backend


class TestBackends(unittest.TestCase):
    def test_all_backends_registered(self):
        names = list_backends()
        for expected in ("mock", "cpu", "openai", "vllm", "transformers"):
            self.assertIn(expected, names)

    def test_backend_train_and_generate(self):
        cfg = Config()
        for name in list_backends():
            b = get_backend(name, cfg)
            out = b.generate("hello", n_frames=2, seed="t")
            self.assertEqual(len(out), 2)
            self.assertGreaterEqual(len(out[0]), 1)


if __name__ == "__main__":
    unittest.main()
