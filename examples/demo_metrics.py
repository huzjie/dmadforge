"""Demo: compute generation-quality metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmadforge.config import load_config
from dmadforge.model.engine import DmadEngine
from dmadforge.metrics.fid import fid_proxy
from dmadforge.metrics.psnr import psnr


def main():
    cfg = load_config()
    engine = DmadEngine(cfg)
    gold = engine.trainer.generate_teacher((8, 16), seed="m")
    stu = engine.trainer.generate_student((8, 16), seed="m")
    print("FID : %.5f" % fid_proxy(gold, stu))
    print("PSNR: %s" % psnr(gold, stu))
    engine.train(20)
    gold = engine.trainer.generate_teacher((8, 16), seed="m")
    stu = engine.trainer.generate_student((8, 16), seed="m")
    print("after train -> FID : %.5f" % fid_proxy(gold, stu))


if __name__ == "__main__":
    main()
