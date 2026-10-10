"""CLI entrypoint: python -m dmadforge <command> [--config PATH]."""
import argparse

from ..config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dmadforge", description="DMAD few-step video distillation framework")
    ap.add_argument("--config", default=None, help="path to config yaml/json")
    sub = ap.add_subparsers(dest="command")

    sub.add_parser("doctor", help="run diagnostics")
    p_train = sub.add_parser("train", help="run distillation training")
    p_train.add_argument("--epochs", type=int, default=None)
    sub.add_parser("bench", help="run benchmarks")
    p_serve = sub.add_parser("serve", help="start HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "doctor":
        from .doctor import run_doctor
        run_doctor(cfg)
    elif args.command == "train":
        from .train import run_train
        run_train(cfg, args.epochs)
    elif args.command == "bench":
        from .bench import run_bench
        run_bench(cfg)
    elif args.command == "serve":
        from .serve import run_serve
        run_serve(cfg, args.host, args.port)
    else:
        ap.print_help()
