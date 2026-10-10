"""Distribution benchmark: MMD and Wasserstein distance to the teacher."""
from ..distillation.losses import mmd2, wasserstein1


def run_distribution(engine, shape=(8, 16), seed="dist", sigma=1.0):
    gold = engine.trainer.generate_teacher(shape, seed=seed)
    stu = engine.trainer.generate_student(shape, seed=seed)
    return {
        "mmd2": mmd2(stu, gold, sigma),
        "wasserstein1": wasserstein1(stu, gold),
        "skill": engine.trainer.skill,
    }
