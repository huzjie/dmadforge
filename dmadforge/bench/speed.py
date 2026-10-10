"""Speed benchmark: step reduction and inference latency proxy."""
import time


def run_speed(engine, n_frames=4, n_runs=5):
    m = engine.cfg.model
    t0 = time.perf_counter()
    for _ in range(n_runs):
        engine.teacher.sample((n_frames, m.latent_dim), seed="speed_t")
    t1 = time.perf_counter()
    for _ in range(n_runs):
        engine.student.sample((n_frames, m.latent_dim), seed="speed_s")
    t2 = time.perf_counter()
    return {
        "teacher_steps": m.teacher_steps,
        "student_steps": m.student_steps,
        "step_ratio": m.teacher_steps / m.student_steps,
        "teacher_time_s": t1 - t0,
        "student_time_s": t2 - t1,
        "speedup_x": (t1 - t0) / max(1e-9, t2 - t1),
    }
