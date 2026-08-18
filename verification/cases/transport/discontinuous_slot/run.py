"""TR-07 in-memory TV / overshoot / undershoot. No compile, bind, or pops.run.

Helpers compare a manufactured smeared slot to the exact discontinuous pair.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

OVERSHOOT_VALUE = 1.1
UNDERSHOOT_VALUE = -0.1


def total_variation(field) -> float:
    """Return the periodic 1-d total variation sum |q_{i+1}-q_i|."""
    values = np.asarray(field, dtype=np.float64).ravel()
    if values.size < 2:
        return 0.0
    jumps = np.abs(np.diff(values))
    return float(np.sum(jumps) + abs(values[0] - values[-1]))


def overshoot(field, reference) -> float:
    """Return max(0, max(field) - max(reference))."""
    sampled = np.asarray(field, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    return float(max(0.0, np.max(sampled) - np.max(exact)))


def undershoot(field, reference) -> float:
    """Return max(0, min(reference) - min(field))."""
    sampled = np.asarray(field, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    return float(max(0.0, np.min(exact) - np.min(sampled)))


def smear_slot(exact):
    """Return a three-point periodic moving average of the exact slot."""
    values = np.asarray(exact, dtype=np.float64)
    return (np.roll(values, 1) + values + np.roll(values, -1)) / 3.0


def manufactured_smeared_pair(
    n_cells: int = _exact.DEFAULT_N_CELLS,
    t=0.0,
    *,
    overshoot_value: float = OVERSHOOT_VALUE,
    undershoot_value: float = UNDERSHOOT_VALUE,
):
    """Return centers, smeared field, exact slot, and volumes."""
    centers, volumes = _exact.cell_centers(n_cells)
    reference = _exact.exact_slot(centers, t)
    field = smear_slot(reference)
    inside = np.flatnonzero(reference >= 0.5)
    outside = np.flatnonzero(reference < 0.5)
    if inside.size:
        field[int(inside[inside.size // 2])] = float(overshoot_value)
    if outside.size:
        field[int(outside[0])] = float(undershoot_value)
    return centers, field, reference, volumes
