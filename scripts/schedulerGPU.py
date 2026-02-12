"""Backward-compatible import path for the GPU scheduler.

Historically the project used `scripts.schedulerGPU.SimWorker`.
The implementation now lives in `scripts.scheduler.worker`.
"""

from scripts.scheduler import SimWorker, time_n, remove_tt, strip_tt_from_slopes

__all__ = ["SimWorker", "time_n", "remove_tt", "strip_tt_from_slopes"]
