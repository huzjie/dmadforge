"""Run all benchmarks and produce a summary table."""
from .speed import run_speed
from .quality import run_quality
from .distribution import run_distribution
from .audio import run_audio


def run_all(engine, shape=(8, 16), seed="bench", n_frames=4):
    out = {}
    out["speed"] = run_speed(engine, n_frames=n_frames)
    out["quality"] = run_quality(engine, shape=shape, seed=seed)
    out["distribution"] = run_distribution(engine, shape=shape, seed=seed)
    out["audio"] = run_audio(engine, shape=shape, seed=seed)
    return out


def summarize(results):
    lines = []
    lines.append("=" * 62)
    lines.append("dmadforge benchmark summary")
    lines.append("=" * 62)
    s = results.get("speed", {})
    lines.append("speed        : %d -> %d steps (%.1fx fewer steps, %.1fx wall-clock speedup)"
                 % (s.get("teacher_steps", 0), s.get("student_steps", 0),
                    s.get("step_ratio", 0), s.get("speedup_x", 0)))
    q = results.get("quality", {})
    lines.append("quality      : recon_mse=%.5f  av_consistency=%.4f  skill=%.3f"
                 % (q.get("recon_mse", 0), q.get("av_consistency", 0), q.get("skill", 0)))
    d = results.get("distribution", {})
    lines.append("distribution : mmd2=%.5f  wasserstein1=%.5f  skill=%.3f"
                 % (d.get("mmd2", 0), d.get("wasserstein1", 0), d.get("skill", 0)))
    a = results.get("audio", {})
    lines.append("audio        : audio_recon_mse=%.5f  av_consistency=%.4f"
                 % (a.get("audio_recon_mse", 0), a.get("av_consistency", 0)))
    lines.append("=" * 62)
    return "\n".join(lines)
