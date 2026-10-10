"""Serve subcommand."""
from ..model.engine import DmadEngine
from ..serving.http import run_server


def run_serve(cfg, host="127.0.0.1", port=8765):
    engine = DmadEngine(cfg)
    run_server(engine, host=host, port=port)
