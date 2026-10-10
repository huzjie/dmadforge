"""Quality benchmark: reconstruction fidelity and av-consistency vs skill."""
from ..utils.stable import stable_vector


def run_quality(engine, shape=(8, 16), seed="quality"):
    from ..distillation.losses import reconstruction_loss
    from ..audiovideo.fusion import JointLatentFusion
    gold = engine.trainer.generate_teacher(shape, seed=seed)
    stu = engine.trainer.generate_student(shape, seed=seed)
    video_z, audio_z = engine.trainer.generate_av(shape, seed=seed)
    fusion = JointLatentFusion(latent_dim=engine.cfg.model.latent_dim)
    return {
        "recon_mse": reconstruction_loss(stu, gold),
        "av_consistency": fusion.consistency(video_z, audio_z),
        "skill": engine.trainer.skill,
    }
