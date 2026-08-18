"""PF-07 in-memory tag + cluster of the TR-02 pulse.

Samples the TR-02 exact Gaussian, tags cells above the documented amplitude,
and clusters contiguous tagged runs into patches of min width 4. Optional
``run_native`` times public AM-02 / AM-03 / AM-05 when those paths exist.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_CASES = Path(__file__).resolve().parents[2]
_TR02_DIR = _CASES / "transport" / "gaussian_pulse"
_exact = load_sibling_module(_CASE_DIR / "exact.py")
_tr02 = load_sibling_module(_TR02_DIR / "exact.py")

NATIVE_CASES = ("am02", "am03", "am05")
DEFAULT_NATIVE_CASE = "am02"
_NATIVE_RUNS = {
    "am02": _CASES / "amr" / "prescribed_patch" / "run.py",
    "am03": _CASES / "amr" / "tagging" / "run.py",
    "am05": _CASES / "amr" / "regrid_frequency" / "run.py",
}


class NativeUnavailable(RuntimeError):
    """Raised when the optional AM-02 / AM-03 / AM-05 timer cannot run."""


def sample_field(n_cells=None, t=0.0):
    """Return (centers, TR-02 exact Gaussian) on a uniform periodic mesh."""
    count = _exact.N_CELLS if n_cells is None else int(n_cells)
    centers = _exact.uniform_centers(count)
    field = _tr02.exact_gaussian(centers, t)
    return centers, field


def cluster_tagged_pulse(n_cells=None, t=0.0, *, threshold=None, min_width=None):
    """Tag the TR-02 pulse and cluster contiguous runs into min-width patches."""
    centers, field = sample_field(n_cells, t)
    tags = _exact.raw_tag_mask(
        field,
        _exact.TAG_THRESHOLD if threshold is None else threshold,
    )
    width = _exact.MIN_PATCH_WIDTH if min_width is None else int(min_width)
    patches = _exact.cluster_runs(tags, width)
    covered = _exact.coverage_mask(patches, tags.size)
    return {
        "centers": centers,
        "field": field,
        "tags": tags,
        "patches": patches,
        "covered": covered,
        "raw_tag_count": int(np.count_nonzero(tags)),
        "patch_count": len(patches),
    }


def default_native_case() -> str:
    """Return the default public regrid sibling (AM-02 prescribed patch)."""
    return DEFAULT_NATIVE_CASE


def public_regrid_native(case: str = DEFAULT_NATIVE_CASE):
    """Return the sibling module if it exposes ``run_native``, else ``None``."""
    name = str(case)
    path = _NATIVE_RUNS.get(name)
    if path is None or not path.is_file():
        return None
    sibling = load_sibling_module(path)
    if not callable(getattr(sibling, "run_native", None)):
        return None
    return sibling


def refuse_public_regrid(case: str = DEFAULT_NATIVE_CASE) -> str:
    """Return why a native regrid timer cannot wrap the named sibling."""
    name = str(case)
    if name not in _NATIVE_RUNS:
        return f"public regrid sibling {name} is not AM-02 / AM-03 / AM-05"
    path = _NATIVE_RUNS[name]
    if not path.is_file():
        return f"public {name} run.py is not available"
    return f"public {name} run_native is not available"



def run_native(*args, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import refuse_unofficial_pf

    raise NativeUnavailable(refuse_unofficial_pf('PF-07'))

