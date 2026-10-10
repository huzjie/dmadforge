import unittest

from dmadforge.config import Config
from dmadforge.distillation.losses import mmd2, wasserstein1, adversarial_loss, reconstruction_loss
from dmadforge.distillation.trainer import DMADTrainer
from dmadforge.distillation.discriminator import Discriminator
from dmadforge.distillation.teacher import TeacherDiffusion
from dmadforge.distillation.student import StudentGenerator


class TestLosses(unittest.TestCase):
    def test_mmd2_zero_identical(self):
        x = [[1.0, 2.0], [3.0, 4.0]]
        self.assertAlmostEqual(mmd2(x, x, 1.0), 0.0, places=5)

    def test_mmd2_positive_different(self):
        x = [[1.0, 2.0], [3.0, 4.0]]
        y = [[-1.0, -2.0], [-3.0, -4.0]]
        self.assertGreater(mmd2(x, y, 1.0), 0.0)

    def test_wasserstein_zero_identical(self):
        x = [[1.0], [2.0]]
        self.assertAlmostEqual(wasserstein1(x, x), 0.0, places=5)

    def test_reconstruction_zero(self):
        x = [[1.0, 2.0]]
        self.assertAlmostEqual(reconstruction_loss(x, x), 0.0, places=5)


class TestTrainer(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        self.trainer = DMADTrainer(cfg, discriminator=Discriminator(dim=8))

    def test_skill_monotonic(self):
        cfg = Config()
        cfg.train.lr = 0.05
        t = DMADTrainer(cfg, discriminator=Discriminator(dim=8))
        t.train(10)
        self.assertGreater(t.skill, cfg.train.skill_start)
        self.assertLessEqual(t.skill, 1.0)

    def test_student_converges_to_teacher(self):
        shape = (8, 8)
        before = self.trainer.evaluate(shape, seed="c")
        self.trainer.train(20)
        after = self.trainer.evaluate(shape, seed="c")
        # recon_mse is guaranteed monotonic: (1-skill)^2 * const -> 0
        self.assertLess(after["recon_mse"], before["recon_mse"])
        self.assertGreater(after["skill"], before["skill"])

    def test_disc_accuracy_falls(self):
        shape = (8, 8)
        gold = self.trainer.generate_teacher(shape, seed="d")
        stu0 = self.trainer.generate_student(shape, seed="d")
        acc0 = self.trainer.discriminator.accuracy(gold, stu0)
        self.trainer.train(20)
        stu1 = self.trainer.generate_student(shape, seed="d")
        acc1 = self.trainer.discriminator.accuracy(gold, stu1)
        self.assertLessEqual(acc1, acc0 + 1e-9)


class TestTeacherStudent(unittest.TestCase):
    def test_teacher_student_shapes(self):
        teacher = TeacherDiffusion(latent_dim=8, dim=8, steps=50)
        student = StudentGenerator(latent_dim=8, dim=8, teacher_steps=50, student_steps=4, lora_rank=8)
        t = teacher.sample((2, 8), seed="s")
        s = student.sample((2, 8), seed="s")
        self.assertEqual(len(t), 2)
        self.assertEqual(len(s), 2)

    def test_step_ratio(self):
        student = StudentGenerator(latent_dim=8, dim=8, teacher_steps=50, student_steps=4)
        self.assertAlmostEqual(student.step_ratio(), 12.5)


if __name__ == "__main__":
    unittest.main()
