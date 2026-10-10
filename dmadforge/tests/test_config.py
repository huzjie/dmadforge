import unittest
from pathlib import Path

from dmadforge.config import Config, load_config


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Config()
        self.assertEqual(c.model.teacher_steps, 50)
        self.assertEqual(c.model.student_steps, 4)
        self.assertEqual(c.model.lora_rank, 128)

    def test_load_yaml(self):
        p = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        c = load_config(p)
        self.assertEqual(c.model.dim, 64)
        self.assertEqual(c.backend.name, "mock")

    def test_roundtrip(self):
        c = Config()
        c2 = Config.from_dict(c.to_dict())
        self.assertEqual(c.model.lora_rank, c2.model.lora_rank)


if __name__ == "__main__":
    unittest.main()
