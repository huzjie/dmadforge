"""Demo: generate video/audio latents from a prompt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmadforge.config import load_config
from dmadforge.model.engine import DmadEngine


def main():
    cfg = load_config()
    engine = DmadEngine(cfg)
    r = engine.generate("a cat walking in the rain", n_frames=4, seed="demo")
    print("prompt      :", r["prompt"])
    print("video_frames:", len(r["video_latent"]))
    print("steps       : %d -> %d (%.1fx)" % (r["teacher_steps"], r["student_steps"], r["step_ratio"]))
    print("av_consist  : %.4f" % r["av_consistency"])


if __name__ == "__main__":
    main()
