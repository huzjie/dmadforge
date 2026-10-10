"""High-level inference engine: assemble teacher/student/discriminator and generate."""
from ..distillation.teacher import TeacherDiffusion
from ..distillation.student import StudentGenerator
from ..distillation.discriminator import Discriminator
from ..distillation.trainer import DMADTrainer
from ..audiovideo.fusion import JointLatentFusion
from ..utils.stable import stable_vector


class DmadEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        m = cfg.model
        self.teacher = TeacherDiffusion(latent_dim=m.latent_dim, dim=m.dim, steps=m.teacher_steps, seed="engine.teacher")
        self.student = StudentGenerator(
            latent_dim=m.latent_dim, dim=m.dim, teacher_steps=m.teacher_steps,
            student_steps=m.student_steps, lora_rank=m.lora_rank, seed="engine.student")
        self.discriminator = Discriminator(dim=m.latent_dim, seed="engine.disc")
        self.fusion = JointLatentFusion(latent_dim=m.latent_dim)
        self.trainer = DMADTrainer(cfg, discriminator=self.discriminator)

    def generate(self, prompt, n_frames=None, seed=""):
        """Generate video + audio latents for a text prompt (deterministic mock)."""
        m = self.cfg.model
        n_frames = n_frames or m.n_frames
        shape = (n_frames, m.latent_dim)
        video_z, audio_z = self.trainer.generate_av(shape, seed=seed)
        return {
            "prompt": prompt,
            "video_latent": video_z,
            "audio_latent": audio_z,
            "student_steps": m.student_steps,
            "teacher_steps": m.teacher_steps,
            "step_ratio": self.student.step_ratio(),
            "av_consistency": self.fusion.consistency(video_z, audio_z),
            "skill": self.trainer.skill,
        }

    def train(self, epochs=None):
        return self.trainer.train(epochs)

    def evaluate(self, shape=(8, 16), seed="eval"):
        return self.trainer.evaluate(shape, seed)

    def distill_report(self, epochs=None):
        """Run distillation and return before/after metrics.

        NOTE: before/after MUST share the same seed so the only thing that
        changes is `skill` -- otherwise the noise baseline differs and MMD
        looks non-monotonic.
        """
        before = self.evaluate(seed="report")
        self.train(epochs)
        after = self.evaluate(seed="report")
        return {"before": before, "after": after, "history": list(self.trainer.history)}
