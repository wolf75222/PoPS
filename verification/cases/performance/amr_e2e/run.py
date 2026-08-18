"""PF-11 in-memory dynamic AMR e2e.

Walk 2 warmup steps then 50 measured fake steps. Rebuild every 8 global
steps. Warmup rebuilds are recorded but not counted. Leaf-cell throughput
is cells / fake time of the measured window. Optional ``run_native`` times
a short AM-01 or AM-02 public path.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_CASES = Path(__file__).resolve().parents[2]
NATIVE_CASES = ("am01", "am02")
DEFAULT_NATIVE_CASE = "am01"
_NATIVE_RUNS = {
    "am01": _CASES / "amr" / "static_cf_wave" / "run.py",
    "am02": _CASES / "amr" / "prescribed_patch" / "run.py",
}


class NativeUnavailable(RuntimeError):
    """Raised when the optional AM-01 / AM-02 e2e timer cannot run."""


def advance_fake_amr(
    *,
    n_warmup=_exact.N_WARMUP,
    n_steps=_exact.N_STEPS,
    regrid_every=_exact.REGRID_EVERY,
):
    """Advance warmup + measured fake steps and return rebuild/throughput stats."""
    prefix = int(n_warmup)
    measured = int(n_steps)
    interval = int(regrid_every)
    rebuild_steps: list[int] = []
    warmup_rebuild_steps: list[int] = []
    for step in range(prefix + measured):
        if not _exact.should_rebuild(step, interval):
            continue
        if _exact.is_warmup(step, prefix):
            warmup_rebuild_steps.append(int(step))
        else:
            rebuild_steps.append(int(step))
    leaf_cells = _exact.leaf_cell_count()
    fake_time = _exact.fake_duration(measured)
    cell_updates = float(leaf_cells) * float(measured)
    return {
        "n_warmup": prefix,
        "n_steps": measured,
        "regrid_every": interval,
        "rebuilds": len(rebuild_steps),
        "rebuild_steps": rebuild_steps,
        "warmup_rebuilds": len(warmup_rebuild_steps),
        "warmup_rebuild_steps": warmup_rebuild_steps,
        "rebuilds_including_warmup": len(rebuild_steps) + len(warmup_rebuild_steps),
        "leaf_cells": leaf_cells,
        "fake_time": fake_time,
        "throughput": _exact.leaf_cell_throughput(cell_updates, fake_time),
    }


def count_rebuilds(
    *,
    n_warmup=_exact.N_WARMUP,
    n_steps=_exact.N_STEPS,
    regrid_every=_exact.REGRID_EVERY,
) -> int:
    """Return measured rebuilds after dropping the warmup prefix."""
    return advance_fake_amr(
        n_warmup=n_warmup, n_steps=n_steps, regrid_every=regrid_every
    )["rebuilds"]


def leaf_cell_throughput(
    *,
    n_warmup=_exact.N_WARMUP,
    n_steps=_exact.N_STEPS,
    regrid_every=_exact.REGRID_EVERY,
) -> float:
    """Return leaf-cell throughput (cells / fake time) of the measured window."""
    return advance_fake_amr(
        n_warmup=n_warmup, n_steps=n_steps, regrid_every=regrid_every
    )["throughput"]


def default_native_case() -> str:
    """Return the default short native sibling (AM-01 static CF wave)."""
    return DEFAULT_NATIVE_CASE


def public_amr_e2e_native(case: str = DEFAULT_NATIVE_CASE):
    """Return the sibling module if it exposes ``run_native``, else ``None``."""
    name = str(case)
    path = _NATIVE_RUNS.get(name)
    if path is None or not path.is_file():
        return None
    sibling = load_sibling_module(path)
    if not callable(getattr(sibling, "run_native", None)):
        return None
    return sibling


def refuse_public_amr_e2e(case: str = DEFAULT_NATIVE_CASE) -> str:
    """Return why a short native AMR e2e timer cannot wrap the named sibling."""
    name = str(case)
    if name not in _NATIVE_RUNS:
        return f"public AMR e2e sibling {name} is not AM-01 / AM-02"
    path = _NATIVE_RUNS[name]
    if not path.is_file():
        return f"public {name} run.py is not available"
    return f"public {name} run_native is not available"



def run_native(*args, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import refuse_unofficial_pf

    raise NativeUnavailable(refuse_unofficial_pf('PF-11'))

