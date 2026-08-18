"""PF-01 delegates to official ``benchmarks/manifest.toml`` case ``arith_halo``.

This directory is not a second benchmark stack. Timed MultiFab arithmetic
and periodic halo fill live in the official ``pops_benchmark`` harness.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.official_benchmark import (
    OFFICIAL_MANIFEST,
    OfficialBenchmarkUnavailable,
    run_official_benchmark,
)

OFFICIAL_CASE = "arith_halo"
CASE_ID = "PF-01"


class NativeUnavailable(RuntimeError):
    """Raised when the official benchmark harness cannot run."""


def official_authority() -> dict:
    """Return the official bench this PF id maps to."""
    return {
        "verification_id": CASE_ID,
        "manifest": str(OFFICIAL_MANIFEST),
        "case_id": OFFICIAL_CASE,
    }


def run_native(n_cells=None, t_end=None, **_kwargs) -> dict:
    """Run official ``arith_halo``. Cell counts are harness defaults."""
    del n_cells, t_end
    try:
        result = run_official_benchmark(OFFICIAL_CASE)
    except OfficialBenchmarkUnavailable as exc:
        raise NativeUnavailable(str(exc)) from exc
    result.update(official_authority())
    return result
