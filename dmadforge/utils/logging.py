"""Tiny logger that mirrors stdlib logging without extra deps."""
import sys
import time

LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


class Logger:
    def __init__(self, name="dmadforge", level="INFO"):
        self.name = name
        self.level = LEVELS.get(str(level).upper(), 20)

    def _emit(self, level, msg):
        if LEVELS[level] >= self.level:
            ts = time.strftime("%H:%M:%S")
            print("[%s] %s %s: %s" % (ts, level, self.name, msg), flush=True)

    def debug(self, msg):
        self._emit("DEBUG", msg)

    def info(self, msg):
        self._emit("INFO", msg)

    def warn(self, msg):
        self._emit("WARN", msg)

    def error(self, msg):
        self._emit("ERROR", msg)


def get_logger(name="dmadforge", level="INFO"):
    return Logger(name, level)
