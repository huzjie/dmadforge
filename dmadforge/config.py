"""Configuration loading with a YAML fallback.

When PyYAML is unavailable (common in isolated runtimes) we fall back to our
bundled minimal `utils.yamlish` parser so a `.yaml` config still works.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    dim: int = 64
    latent_dim: int = 4
    audio_dim: int = 4
    frame_size: int = 16
    n_frames: int = 4
    teacher_steps: int = 50
    student_steps: int = 4
    lora_rank: int = 128


@dataclass
class BackendConfig:
    name: str = "mock"
    temperature: float = 1.0
    max_tokens: int = 256


@dataclass
class DistillationConfig:
    adv_weight: float = 1.0
    mmd_weight: float = 0.5
    recon_weight: float = 0.5
    mmd_sigma: float = 1.0
    gp_weight: float = 10.0


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 0.05
    skill_start: float = 0.5
    batch_size: int = 8


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cfg = cls()
        if not d:
            return cfg
        m = d.get("model", {}) or {}
        cfg.model = ModelConfig(
            dim=int(m.get("dim", 64)),
            latent_dim=int(m.get("latent_dim", 4)),
            audio_dim=int(m.get("audio_dim", 4)),
            frame_size=int(m.get("frame_size", 16)),
            n_frames=int(m.get("n_frames", 4)),
            teacher_steps=int(m.get("teacher_steps", 50)),
            student_steps=int(m.get("student_steps", 4)),
            lora_rank=int(m.get("lora_rank", 128)),
        )
        b = d.get("backend", {}) or {}
        cfg.backend = BackendConfig(
            name=str(b.get("name", "mock")),
            temperature=float(b.get("temperature", 1.0)),
            max_tokens=int(b.get("max_tokens", 256)),
        )
        dl = d.get("distillation", {}) or {}
        cfg.distillation = DistillationConfig(
            adv_weight=float(dl.get("adv_weight", 1.0)),
            mmd_weight=float(dl.get("mmd_weight", 0.5)),
            recon_weight=float(dl.get("recon_weight", 0.5)),
            mmd_sigma=float(dl.get("mmd_sigma", 1.0)),
            gp_weight=float(dl.get("gp_weight", 10.0)),
        )
        t = d.get("train", {}) or {}
        cfg.train = TrainConfig(
            epochs=int(t.get("epochs", 20)),
            lr=float(t.get("lr", 0.05)),
            skill_start=float(t.get("skill_start", 0.5)),
            batch_size=int(t.get("batch_size", 8)),
        )
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.__dict__,
            "backend": self.backend.__dict__,
            "distillation": self.distillation.__dict__,
            "train": self.train.__dict__,
        }


def _load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text)
        return data or {}
    except Exception:
        from .utils.yamlish import parse
        return parse(text)


def load_config(path=None) -> Config:
    if path is None:
        path = Path(__file__).resolve().parent / "configs" / "default.yaml"
    path = Path(path)
    if not path.exists():
        return Config()
    if path.suffix in (".yaml", ".yml"):
        return Config.from_dict(_load_yaml(path))
    if path.suffix == ".json":
        import json
        return Config.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return Config()
