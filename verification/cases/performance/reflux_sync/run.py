"""PF-08 in-memory reflux / AMR sync stand-in.

Closed with reflux; open without (negative control). Residuals come from
Task 16 ``conservation_residual``. Elapsed wall time is an observation.
Optional ``run_native`` times public AM-09 when that path exists.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.conservation import conservation_residual

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_AM09_RUN = Path(__file__).resolve().parents[2] / "amr" / "reflux" / "run.py"


class NativeUnavailable(RuntimeError):
    """Raised when the optional AM-09 reflux timer cannot run."""


def residual(*, reflux: bool = True):
    """Return the discrete conservation residual of the manufactured statement."""
    terms = _exact.closed_balance_terms() if reflux else _exact.open_balance_terms()
    return conservation_residual(
        terms["storage_change"],
        terms["outward_boundary_flux"],
        terms["sources"],
        reflux=terms["reflux"],
        projection=terms["projection"],
    )


def timed_sync(*, reflux: bool = True) -> dict:
    """Evaluate the residual and record elapsed wall time as an observation."""
    started = time.perf_counter()
    value = residual(reflux=reflux)
    elapsed_s = time.perf_counter() - started
    return {
        "residual": value,
        "elapsed_s": float(elapsed_s),
        "reflux": bool(reflux),
    }


def public_reflux_native():
    """Return AM-09 ``run.py`` if it exposes ``run_native``, else ``None``."""
    if not _AM09_RUN.is_file():
        return None
    sibling = load_sibling_module(_AM09_RUN)
    if not callable(getattr(sibling, "run_native", None)):
        return None
    return sibling


def refuse_public_reflux() -> str:
    """Return why a native reflux timer cannot wrap AM-09."""
    if not _AM09_RUN.is_file():
        return "public AM-09 run.py is not available"
    return "public AM-09 run_native is not available"



def run_native(*args, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import refuse_unofficial_pf

    raise NativeUnavailable(refuse_unofficial_pf('PF-08'))

