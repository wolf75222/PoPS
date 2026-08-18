"""PF-05 two-level composite residual stand-in plus optional AM-10 timing.

Reuses AM-10 helpers. Covered parents are excluded from the leaf residual.
Refined fraction is an observation, not an acceptance threshold.
Optional ``run_native`` times AM-10 ``run_native`` when present and returns
``{elapsed_s, residual_or_error, cells_per_second}``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_AM10_RUN = _CASE_DIR.parents[1] / "amr" / "composite_poisson" / "run.py"

_exact = load_sibling_module(_CASE_DIR / "exact.py")
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")
_am10 = load_sibling_module(_AM10_RUN)


class NativeUnavailable(RuntimeError):
    """Raised when the timed AM-10 native path cannot run."""


def refined_fraction(sample) -> float:
    """Return the fine-patch volume fraction of the unit interval."""
    n_fine = int(sample["n_fine"])
    fine_volume = float(np.sum(np.asarray(sample["volumes"], dtype=np.float64)[-n_fine:]))
    return fine_volume


def two_level_residual(**kwargs):
    """AM-10 two-level residual plus the refined-fraction observation."""
    sample = dict(_am10.two_level_residual(**kwargs))
    sample["refined_fraction"] = refined_fraction(sample)
    return sample


def leaf_residual_errors(sample=None, **kwargs):
    """Task 13 leaf-only residual norms. Covered parents are excluded."""
    if sample is None:
        sample = two_level_residual(**kwargs)
    return _am10.leaf_residual_errors(sample)



def official_authority() -> dict:
    """PF-05 is absent from benchmarks/manifest.toml."""
    return _v15.official_authority("PF-05")


def run_native(*args, request=None, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import OfficialBenchmarkUnavailable

    try:
        return _v15.run_mapped_or_refuse("PF-05", request)
    except OfficialBenchmarkUnavailable as exc:
        raise NativeUnavailable(str(exc)) from exc

