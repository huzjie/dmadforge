"""Demo: full distillation pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmadforge.config import load_config
from dmadforge.pipeline.distill_pipeline import DistillPipeline


def main():
    cfg = load_config()
    p = DistillPipeline(cfg)
    report = p.run(epochs=20, shape=(8, 16))
    print(p.summarize(report))


if __name__ == "__main__":
    main()
