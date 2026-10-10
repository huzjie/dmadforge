"""Diagnostics: instantiate every component and print health."""
from ..config import Config
from ..model.engine import DmadEngine
from ..backends import list_backends, get_backend


def run_doctor(cfg):
    m = cfg.model
    print("[doctor] config: dim=%d latent=%d audio=%d frames=%d teacher_steps=%d student_steps=%d rank=%d"
          % (m.dim, m.latent_dim, m.audio_dim, m.n_frames, m.teacher_steps, m.student_steps, m.lora_rank))
    engine = DmadEngine(cfg)
    print("[doctor] engine ok: step_ratio=%.1fx" % engine.student.step_ratio())
    gen = engine.generate("a test prompt", seed="doctor")
    print("[doctor] generate ok: video_latent=%d frames, av_consistency=%.4f"
          % (len(gen["video_latent"]), gen["av_consistency"]))
    print("[doctor] backends available: %s" % ", ".join(list_backends()))
    for name in list_backends():
        b = get_backend(name, cfg)
        print("[doctor]   backend %-14s -> %s" % (name, type(b).__name__))
    report = engine.distill_report(epochs=5)
    print("[doctor] distill demo: skill %.3f -> %.3f | recon_mse %.5f -> %.5f | disc_acc %.3f -> %.3f"
          % (report["before"]["skill"], report["after"]["skill"],
             report["before"]["recon_mse"], report["after"]["recon_mse"],
             report["before"]["disc_acc"], report["after"]["disc_acc"]))
    return engine
