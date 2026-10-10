"""Benchmark suite for few-step distillation quality and speed."""
from .speed import run_speed
from .quality import run_quality
from .distribution import run_distribution
from .audio import run_audio
from .summary import run_all, summarize

__all__ = ["run_speed", "run_quality", "run_distribution", "run_audio", "run_all", "summarize"]
