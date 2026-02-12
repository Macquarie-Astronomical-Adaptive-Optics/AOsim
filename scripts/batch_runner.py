"""Backward-compatible import path for the long-run batch runner.

Implementation now lives in `scripts.batch.runner`.
"""

from scripts.batch import AOBatchRunner

__all__ = ["AOBatchRunner"]
