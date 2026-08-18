"""PF-01 delegates to official ``benchmarks/manifest.toml`` case ``arith_halo``.

This directory is not a second benchmark stack. Timed MultiFab arithmetic
and periodic halo fill live in the official ``pops_benchmark`` harness.
"""
from __future__ import annotations

from pathlib import Path

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.official_benchmark import (
    OFFICIAL_MANIFEST,
    OfficialBenchmarkUnavailable,
    run_official_benchmark,
)

_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")
OFFICIAL_CASE = "arith_halo"
CASE_ID = "PF-01"


class NativeUnavailable(RuntimeError):
    """Raised when the official benchmark harness cannot run."""


def official_authority() -> dict:
    """Return the official bench this PF id maps to in benchmarks/manifest.toml."""
    authority = _v15.official_authority(CASE_ID)
    assert authority["manifest"] == str(OFFICIAL_MANIFEST)
    return authority


def run_native(n_cells=None, t_end=None, request=None, **_kwargs) -> dict:
    """Run official ``arith_halo``. Cell counts are harness defaults."""
    del n_cells, t_end
    try:
        return _v15.run_mapped_or_refuse(CASE_ID, request)
    except OfficialBenchmarkUnavailable as exc:
        raise NativeUnavailable(str(exc)) from exc
