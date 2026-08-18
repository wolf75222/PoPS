"""PF-02 delegates to official ``benchmarks/manifest.toml`` case ``scalar_mg``.

This directory is not a second benchmark stack. Timed GeometricMG lives
in the official ``pops_benchmark`` harness.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.official_benchmark import (
    OFFICIAL_MANIFEST,
    OfficialBenchmarkUnavailable,
    run_official_benchmark,
)

_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")
OFFICIAL_CASE = "scalar_mg"
CASE_ID = "PF-02"


class NativeUnavailable(RuntimeError):
    """Raised when the official benchmark harness cannot run."""


def official_authority() -> dict:
    """Return the official bench this PF id maps to in benchmarks/manifest.toml."""
    authority = _v15.official_authority(CASE_ID)
    assert authority["manifest"] == str(OFFICIAL_MANIFEST)
    return authority


def run_native(n_cells=None, t_end=None, request=None, **_kwargs) -> dict:
    """Run official ``scalar_mg``. Cell counts are harness defaults."""
    del n_cells, t_end
    try:
        return _v15.run_mapped_or_refuse(CASE_ID, request)
    except OfficialBenchmarkUnavailable as exc:
        raise NativeUnavailable(str(exc)) from exc
