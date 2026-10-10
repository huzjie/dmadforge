"""Audio benchmark: audio latent reconstruction and sync consistency."""
from ..distillation.losses import reconstruction_loss
from ..audiovideo.fusion import JointLatentFusion


def run_audio(engine, shape=(8, 16), seed="audio"):
    gold_a = engine.trainer.generate_teacher(shape, seed=seed + "a")
    stu_a = engine.trainer.generate_student(shape, seed=seed + "a")
    video_z, audio_z = engine.trainer.generate_av(shape, seed=seed)
    fusion = JointLatentFusion(latent_dim=engine.cfg.model.latent_dim)
    return {
        "audio_recon_mse": reconstruction_loss(stu_a, gold_a),
        "av_consistency": fusion.consistency(video_z, audio_z),
    }
