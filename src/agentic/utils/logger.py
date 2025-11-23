import os
import sys

_ENABLED = os.getenv("AGENTIC_DEBUG", "false").lower() in ("1", "true", "yes")


def debug(msg: str):
    if _ENABLED:
        print(msg, file=sys.stderr)


def info(msg: str):
    if _ENABLED:
        print(msg, file=sys.stderr)


def warn(msg: str):
    if _ENABLED:
        print(msg, file=sys.stderr)
