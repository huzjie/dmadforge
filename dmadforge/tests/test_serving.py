import unittest

from dmadforge.config import Config
from dmadforge.model.engine import DmadEngine
from dmadforge.serving.http import create_app


class TestServing(unittest.TestCase):
    def test_create_app(self):
        cfg = Config()
        engine = DmadEngine(cfg)
        handler = create_app(engine)
        self.assertTrue(handler is not None)


if __name__ == "__main__":
    unittest.main()
