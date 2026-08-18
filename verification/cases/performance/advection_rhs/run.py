"""PF-03 in-memory periodic halo fill and upwind FV RHS, plus optional TR-01 timer.

Periodic halo of width 1, then interior first-order upwind of q_t + a q_x.
``run_native`` times TR-01 ``run_native``. ``pops.run`` stays in TR-01.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_TR01_RUN = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "run.py"
)


class NativeUnavailable(RuntimeError):
    """Raised when the optional TR-01 native timer cannot run."""


def _tr01_run():
    """Load TR-01 ``run.py`` via ``load_sibling_module``."""
    return load_sibling_module(_TR01_RUN)


def _reraise_native_unavailable(exc: BaseException) -> None:
    if exc.__class__.__name__ == "NativeUnavailable":
        raise NativeUnavailable(str(exc)) from exc
    raise exc


def expected_n_steps(n_cells, t_end, *, cfl, speed) -> int:
    """Return ceil(t_end / dt) for 1-d advection with dt = cfl * dx / |a|."""
    count = int(n_cells)
    horizon = float(t_end)
    if count < 1:
        raise ValueError(f"n_cells must be >= 1, got {n_cells!r}")
    if horizon < 0.0:
        raise ValueError(f"t_end must be >= 0, got {t_end!r}")
    if horizon == 0.0:
        return 0
    dt = float(cfl) * (1.0 / count) / abs(float(speed))
    return int(math.ceil(horizon / dt))



def run_native(*args, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import refuse_unofficial_pf

    raise NativeUnavailable(refuse_unofficial_pf('PF-03'))



def periodic_halo_fill(n_cells=_exact.N_CELLS, halo_width=_exact.HALO_WIDTH):
    """Pad the TR-01 sine at cell centres and fill a periodic halo."""
    centers = _exact.cell_centers(n_cells)
    interior = _exact.exact_sine(centers, 0.0)
    padded = _exact.pad_interior(interior, halo_width)
    return _exact.fill_periodic_halo(padded, halo_width)


def interior_rhs(n_cells=_exact.N_CELLS) -> dict:
    """Return the interior upwind RHS after a separate periodic halo fill."""
    count = int(n_cells)
    centers = _exact.cell_centers(count)
    volumes = _exact.cell_volumes(count)
    filled = periodic_halo_fill(count)
    rhs = _exact.upwind_rhs(filled, float(volumes[0]))
    return {
        "rhs": rhs,
        "centers": centers,
        "volumes": volumes,
        "analytic": _exact.exact_rhs(centers, 0.0),
    }
