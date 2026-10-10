"""FastAPI adapter (optional)."""
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None
    BaseModel = object


def build_app(engine):
    if FastAPI is None:
        raise ImportError("fastapi is not installed")
    app = FastAPI(title="dmadforge")

    @app.get("/health")
    def health():
        return {"status": "ok", "skill": engine.trainer.skill}

    @app.post("/v1/video/generations")
    def generate(prompt: str = "", n_frames: int = 4, seed: str = "fastapi"):
        return engine.generate(prompt, n_frames=n_frames, seed=seed)

    return app
