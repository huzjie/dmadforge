"""Distribution-matching adversarial distillation components."""
from .teacher import TeacherDiffusion
from .student import StudentGenerator
from .discriminator import Discriminator
from .losses import mmd2, wasserstein1, adversarial_loss, reconstruction_loss
from .trainer import DMADTrainer

__all__ = [
    "TeacherDiffusion", "StudentGenerator", "Discriminator",
    "mmd2", "wasserstein1", "adversarial_loss", "reconstruction_loss",
    "DMADTrainer",
]
