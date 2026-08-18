"""PF-04 in-memory Rusanov Euler step plus optional EU-01 native timing.

One local Lax-Friedrichs / Rusanov update of a uniform free stream.
The python face loop is timed only as an observation.

``run_native`` times EU-01 ``linear_waves.run_native`` and returns elapsed
plus cells/s. GPU spaces are refused. ``pops.run`` stays inside EU-01.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")
_EU01_RUN = (
    Path(__file__).resolve().parents[2] / "euler" / "linear_waves" / "run.py"
)

N_REPEATS = 32
DEFAULT_NATIVE_T_END = 0.05
GPU_SPACES = ("KokkosCuda", "KokkosHIP")
CUDA_UNAVAILABLE = "no public CUDA space"


class NativeUnavailable(RuntimeError):
    """Optional native EU-01 timing cannot run in this environment."""


def initial_primitives(n_cells=_exact.N_CELLS):
    """Uniform primitive IC at t=0. Each field has shape (n,)."""
    centers = _exact.cell_centers(n_cells)
    return _exact.exact_primitives(centers, 0.0)


def initial_conserved(n_cells=_exact.N_CELLS):
    """Conserved IC from the uniform free stream."""
    return _exact.primitives_to_conserved(initial_primitives(n_cells))


def one_rusanov_step(n_cells=_exact.N_CELLS, cfl=_exact.CFL) -> dict:
    """Apply one periodic Rusanov step to the uniform free stream."""
    initial = initial_primitives(n_cells)
    conserved = _exact.primitives_to_conserved(initial)
    dx = _exact.cell_width(n_cells)
    dt = _exact.cfl_dt(dx, conserved, cfl)
    updated = _exact.rusanov_step(conserved, dt, dx)
    primitives = _exact.conserved_to_primitives(updated)
    return {
        "rho": primitives["rho"],
        "u": primitives["u"],
        "p": primitives["p"],
        "initial": initial,
        "dt": float(dt),
        "dx": float(dx),
        "n_cells": int(n_cells),
        "volumes": _exact.cell_volumes(n_cells),
    }


def time_python_loop(
    n_cells=_exact.N_CELLS, n_repeats: int = N_REPEATS, cfl=_exact.CFL
) -> dict:
    """Time repeated Rusanov python loops. Observation only; not a gate."""
    conserved = initial_conserved(n_cells)
    dx = _exact.cell_width(n_cells)
    dt = _exact.cfl_dt(dx, conserved, cfl)
    _exact.rusanov_step(conserved, dt, dx)
    started = time.perf_counter()
    for _ in range(int(n_repeats)):
        _exact.rusanov_step(conserved, dt, dx)
    elapsed = time.perf_counter() - started
    cells = float(n_cells) * float(n_repeats)
    rate = cells / elapsed if elapsed > 0.0 else None
    return {
        "elapsed_s": float(elapsed),
        "n_repeats": int(n_repeats),
        "n_cells": int(n_cells),
        "cells_per_second": None if rate is None else float(rate),
    }


def _eu01_run():
    """Load EU-01 ``run.py`` via ``load_sibling_module``."""
    return load_sibling_module(_EU01_RUN)


def _reraise_native_unavailable(exc: BaseException) -> None:
    if exc.__class__.__name__ == "NativeUnavailable":
        raise NativeUnavailable(str(exc)) from exc
    raise exc


def _cells_per_second(n_cells: int, elapsed_s: float):
    if elapsed_s <= 0.0:
        return None
    return float(n_cells) / float(elapsed_s)



def official_authority() -> dict:
    """PF-04 is absent from benchmarks/manifest.toml."""
    return _v15.official_authority("PF-04")


def run_native(*args, request=None, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import OfficialBenchmarkUnavailable

    try:
        return _v15.run_mapped_or_refuse("PF-04", request)
    except OfficialBenchmarkUnavailable as exc:
        raise NativeUnavailable(str(exc)) from exc

