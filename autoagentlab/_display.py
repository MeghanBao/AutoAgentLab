"""ANSI display helpers shared across loop.py and population.py."""

from __future__ import annotations

BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
RESET   = "\033[0m"


def bar(value: float, width: int = 30) -> str:
    """ASCII progress bar. *value* is clamped to [0, 1]."""
    filled = int(max(0.0, min(1.0, value)) * width)
    return f"{'█' * filled}{'░' * (width - filled)}"
