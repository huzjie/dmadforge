import unittest

from dmadforge.data.prompts import PromptDataset, DEFAULT_PROMPTS


class TestData(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(PromptDataset()), len(DEFAULT_PROMPTS))

    def test_getitem_wraps(self):
        d = PromptDataset()
        self.assertEqual(d[0], d[len(d)])

    def test_seed_for_deterministic(self):
        d = PromptDataset()
        self.assertEqual(d.seed_for(3), d.seed_for(3))

    def test_batches(self):
        d = PromptDataset()
        batches = list(d.iter_batches(batch_size=4))
        self.assertGreaterEqual(len(batches), 2)


if __name__ == "__main__":
    unittest.main()
