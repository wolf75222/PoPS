"""Plan Serial/OpenMP × Dim1/Dim2 native-order jobs.

In-memory manufactured series only. No compile, bind, or pops.run.
Live ROMEO jobs live under verification/machines/.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

SPACES = ("Serial", "OpenMP")
DIMENSIONS = ("Dim1", "Dim2")


def plan_jobs():
    """Emit Serial/OpenMP × Dim1/Dim2 labels."""
    return [f"{space}-{dim}" for space in SPACES for dim in DIMENSIONS]


def manufactured_series(resolutions=None):
    """Return the manufactured L2 ∝ h² series used by the in-memory gate."""
    counts = _exact.RESOLUTIONS if resolutions is None else resolutions
    return {
        "resolutions": list(counts),
        "spacings": _exact.spacings(counts),
        "l2": _exact.manufactured_l2(counts),
    }
