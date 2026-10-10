"""Benchmark subcommand."""
from ..model.engine import DmadEngine
from ..bench.summary import run_all, summarize


def run_bench(cfg):
    engine = DmadEngine(cfg)
    engine.train()
    results = run_all(engine)
    print(summarize(results))
    return results
