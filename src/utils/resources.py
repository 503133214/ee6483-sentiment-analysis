"""Tiny helpers for the resource columns of results/runs.csv (report section g)."""
from __future__ import annotations

import platform
import resource
import time


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if platform.system() == "Darwin" else rss / 1e3  # bytes on macOS, KB on Linux


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def machine_tag() -> str:
    return f"{platform.system()} {platform.machine()} {platform.processor() or ''}".strip()
