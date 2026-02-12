"""Centralized logging setup.

This project historically relied on print() for status/errors.
We now standardize on Python's `logging`.

Notes
-----
- The optional Qt "Log Console" dock redirects stdout/stderr, so a normal
  StreamHandler is sufficient for messages to appear there as well.
- This module avoids importing any Qt types to keep it usable in CLI tools.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int | str = "INFO", *, fmt: str = _DEFAULT_FORMAT) -> None:
    """Configure root logging.

    Safe to call multiple times.

    Parameters
    ----------
    level:
        Logging level (e.g. "DEBUG" or logging.DEBUG).
    fmt:
        Formatter format string.
    """
    if isinstance(level, str):
        level = level.upper()

    # If root already has handlers, don't double-configure.
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root.addHandler(handler)
    root.setLevel(level)

    # Optional: make noisy third-party modules less verbose by default.
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.WARNING))


def env_log_level(default: str = "INFO") -> str:
    """Return LOG_LEVEL env var if set, else default."""
    return os.environ.get("LOG_LEVEL", default)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name if name else __name__)
