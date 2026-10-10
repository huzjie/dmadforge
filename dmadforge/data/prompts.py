"""A small curated prompt dataset for the mock distillation loop."""
from ..utils.stable import stable_seed

DEFAULT_PROMPTS = [
    "a cat walking in the rain",
    "a rocket launching at dusk",
    "waves crashing on a rocky shore",
    "a dancer spinning in a spotlight",
    "a train crossing a snowy bridge",
    "fireflies over a summer meadow",
    "a chef plating a gourmet dish",
    "a falcon diving through clouds",
]


class PromptDataset:
    def __init__(self, prompts=None, seed="data"):
        self.prompts = list(prompts or DEFAULT_PROMPTS)
        self._seed = seed

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        return self.prompts[i % len(self.prompts)]

    def seed_for(self, i):
        return "%s:%d" % (self._seed, i)

    def iter_batches(self, batch_size=4):
        for i in range(0, len(self.prompts), batch_size):
            yield self.prompts[i:i + batch_size]
