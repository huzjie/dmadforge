"""Train subcommand."""
from ..model.engine import DmadEngine


def run_train(cfg, epochs=None):
    engine = DmadEngine(cfg)
    report = engine.distill_report(epochs)
    print("[train] skill %.3f -> %.3f | recon_mse %.5f -> %.5f | mmd2 %.5f -> %.5f | disc_acc %.3f -> %.3f"
          % (report["before"]["skill"], report["after"]["skill"],
             report["before"]["recon_mse"], report["after"]["recon_mse"],
             report["before"]["mmd2"], report["after"]["mmd2"],
             report["before"]["disc_acc"], report["after"]["disc_acc"]))
    return report
