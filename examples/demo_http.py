"""Demo: start the HTTP server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmadforge.config import load_config
from dmadforge.model.engine import DmadEngine
from dmadforge.serving.http import run_server


def main():
    cfg = load_config()
    engine = DmadEngine(cfg)
    run_server(engine, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
