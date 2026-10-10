"""Demo: run all benchmarks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmadforge.config import load_config
from dmadforge.model.engine import DmadEngine
from dmadforge.bench.summary import run_all, summarize


def main():
    cfg = load_config()
    engine = DmadEngine(cfg)
    engine.train()
    print(summarize(run_all(engine)))


if __name__ == "__main__":
    main()
