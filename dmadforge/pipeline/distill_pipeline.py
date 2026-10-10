"""Orchestrate a full DMAD distillation run: data -> distill -> eval -> report."""
from ..model.engine import DmadEngine
from ..bench.summary import run_all, summarize
from ..data.prompts import PromptDataset
from ..metrics.fid import fid_proxy


class DistillPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.engine = DmadEngine(cfg)
        self.dataset = PromptDataset()

    def run(self, epochs=None, shape=(8, 16)):
        before = self.engine.evaluate(shape, seed="pipeline")
        self.engine.train(epochs)
        after = self.engine.evaluate(shape, seed="pipeline")
        gold = self.engine.trainer.generate_teacher(shape, seed="fid")
        stu = self.engine.trainer.generate_student(shape, seed="fid")
        report = {
            "before": before,
            "after": after,
            "fid": fid_proxy(gold, stu),
            "history": list(self.engine.trainer.history),
        }
        return report

    def summarize(self, report):
        lines = []
        lines.append("pipeline: skill %.3f -> %.3f, recon_mse %.5f -> %.5f, disc_acc %.3f -> %.3f, fid=%.5f"
                     % (report["before"]["skill"], report["after"]["skill"],
                        report["before"]["recon_mse"], report["after"]["recon_mse"],
                        report["before"]["disc_acc"], report["after"]["disc_acc"],
                        report["fid"]))
        return "\n".join(lines)
