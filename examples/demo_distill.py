"""Demo: run distillation and print before/after metrics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmadforge.config import load_config
from dmadforge.model.engine import DmadEngine


def main():
    cfg = load_config()
    engine = DmadEngine(cfg)
    report = engine.distill_report(epochs=20)
    b, a = report["before"], report["after"]
    print("skill      : %.3f -> %.3f" % (b["skill"], a["skill"]))
    print("mmd2       : %.5f -> %.5f" % (b["mmd2"], a["mmd2"]))
    print("recon_mse  : %.5f -> %.5f" % (b["recon_mse"], a["recon_mse"]))
    print("disc_acc   : %.3f -> %.3f" % (b["disc_acc"], a["disc_acc"]))


if __name__ == "__main__":
    main()
