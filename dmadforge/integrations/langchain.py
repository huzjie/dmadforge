"""LangChain tool wrapper (optional)."""
try:
    from langchain_core.tools import BaseTool
except Exception:  # pragma: no cover
    BaseTool = object


class DmadGenerateTool(BaseTool if BaseTool is not object else object):
    name = "dmadforge_generate"
    description = "Generate a few-step video latent from a text prompt using DMAD distillation."

    def __init__(self, engine):
        if BaseTool is object:
            raise ImportError("langchain is not installed")
        super().__init__()
        self.engine = engine

    def _run(self, prompt: str) -> str:
        r = self.engine.generate(prompt)
        return "generated %d video frames (skill=%.3f, av_consistency=%.4f)" % (
            len(r["video_latent"]), r["skill"], r["av_consistency"])
