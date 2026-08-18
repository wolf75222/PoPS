"""PF-11 dynamic AMR e2e oracle: rebuild cadence and leaf-cell throughput.

50 measured fake steps after 2 warmup steps; regrid every 8. Warmup
rebuilds are excluded. Throughput is cells / fake time.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

N_WARMUP = 2
N_STEPS = 50
REGRID_EVERY = 8
N_COARSE = 16
INTERFACE = 0.5
REFINEMENT_RATIO = 2
FAKE_STEP_TIME = 1.0e-3


def expected_rebuilds(
    n_steps: int = N_STEPS, regrid_every: int = REGRID_EVERY
) -> int:
    """Return the documented measured rebuild count: n_steps / regrid_every."""
    steps = int(n_steps)
    interval = int(regrid_every)
    if steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    if interval < 1:
        raise ValueError(f"regrid interval must be >= 1, got {regrid_every!r}")
    return steps // interval


def is_warmup(step, n_warmup: int = N_WARMUP) -> bool:
    """True on the documented warmup prefix: step < n_warmup."""
    prefix = int(n_warmup)
    if prefix < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup!r}")
    return int(step) < prefix


def should_rebuild(step, regrid_every: int = REGRID_EVERY) -> bool:
    """True on the documented cadence: rebuild when step is a multiple of k."""
    interval = int(regrid_every)
    if interval < 1:
        raise ValueError(f"regrid interval must be >= 1, got {regrid_every!r}")
    return int(step) % interval == 0


def counts_as_rebuild(
    step, *, n_warmup: int = N_WARMUP, regrid_every: int = REGRID_EVERY
) -> bool:
    """True when a rebuild falls on a measured (post-warmup) step."""
    return should_rebuild(step, regrid_every) and not is_warmup(step, n_warmup)


def leaf_cell_count(
    n_coarse: int = N_COARSE,
    *,
    interface: float = INTERFACE,
    refinement_ratio: int = REFINEMENT_RATIO,
) -> int:
    """Return uncovered coarse cells plus fine cells on the refined half."""
    coarse = int(n_coarse)
    ratio = int(refinement_ratio)
    if coarse < 1 or ratio < 1:
        raise ValueError("n_coarse and refinement_ratio must be >= 1")
    n_covered = coarse // 2 if float(interface) == 0.5 else 0
    n_fine = n_covered * ratio
    return (coarse - n_covered) + n_fine


def fake_duration(n_steps: int = N_STEPS, step_time: float = FAKE_STEP_TIME) -> float:
    """Return the documented fake wall time of the measured steps."""
    steps = int(n_steps)
    dt = float(step_time)
    if steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    if dt <= 0.0:
        raise ValueError(f"step_time must be > 0, got {step_time!r}")
    return float(steps) * dt


def leaf_cell_throughput(n_leaf_cells, fake_time) -> float:
    """Return leaf-cell throughput: cells / fake time."""
    cells = float(n_leaf_cells)
    duration = float(fake_time)
    if cells <= 0.0:
        raise ValueError(f"n_leaf_cells must be > 0, got {n_leaf_cells!r}")
    if duration <= 0.0:
        raise ValueError(f"fake_time must be > 0, got {fake_time!r}")
    return cells / duration
