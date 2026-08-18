"""AM-11 Euler–Poisson AMR leaf-only charge.

Two-level 1-d hierarchy: coarse left leaf plus a covered parent on the
right, with two fine children covering that parent. Net charge is the
volume-weighted sum of q n on leaf cells only. Adding the covered parent
double-counts the fine-patch charge.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

Q = 1.0
N0 = 1.0
DELTA = 0.25
K = 2.0 * np.pi
X_LO = 0.0
X_HI = 1.0
INTERFACE = 0.5
PARENT_INDEX = 1
FINE_INDICES = (2, 3)
LEAF_INDICES = (0, 2, 3)


def density(x, *, n0: float = N0, delta: float = DELTA, k: float = K) -> np.ndarray:
    """Number density n = n0 + δ cos(k x)."""
    samples = np.atleast_1d(np.asarray(x, dtype=np.float64))
    return float(n0) + float(delta) * np.cos(float(k) * samples)


def charge_density(x, *, q: float = Q, **kwargs) -> np.ndarray:
    """Euler–Poisson charge density ρ_q = q n."""
    return float(q) * density(x, **kwargs)


def two_level_hierarchy() -> dict:
    """Return a two-level 1-d charge fixture with a conservatively restricted parent."""
    lo = np.array([X_LO, INTERFACE, INTERFACE, 0.75], dtype=np.float64)
    hi = np.array([INTERFACE, X_HI, 0.75, X_HI], dtype=np.float64)
    volumes = hi - lo
    centers = 0.5 * (lo + hi)
    leaf_mask = np.array([True, False, True, True])
    rho = charge_density(centers)
    fine = np.array(FINE_INDICES)
    rho[PARENT_INDEX] = float(np.sum(rho[fine] * volumes[fine]) / volumes[PARENT_INDEX])
    return {
        "x": centers,
        "volumes": volumes,
        "charge_density": rho,
        "leaf_mask": leaf_mask,
        "parent_index": PARENT_INDEX,
        "fine_indices": fine,
        "leaf_indices": np.array(LEAF_INDICES),
    }


def _broadcast_charge(rho, volumes, leaf_mask=None):
    density = np.asarray(rho, dtype=np.float64)
    cell_volumes = np.asarray(volumes, dtype=np.float64)
    if leaf_mask is None:
        try:
            density, cell_volumes = np.broadcast_arrays(density, cell_volumes)
        except ValueError as exc:
            raise ValueError("shape mismatch") from exc
        return density, cell_volumes, None
    mask = np.asarray(leaf_mask)
    if mask.dtype != np.bool_:
        raise ValueError("leaf_mask must be boolean")
    try:
        density, cell_volumes, mask = np.broadcast_arrays(density, cell_volumes, mask)
    except ValueError as exc:
        raise ValueError("shape mismatch") from exc
    return density, cell_volumes, mask


def leaf_net_charge(rho, volumes, leaf_mask) -> float:
    """Volume-weighted net charge on leaf cells only."""
    density, cell_volumes, mask = _broadcast_charge(rho, volumes, leaf_mask)
    if not np.any(mask):
        raise ValueError("empty leaf set")
    return float(np.sum(density[mask] * cell_volumes[mask]))


def naive_net_charge(rho, volumes) -> float:
    """Volume-weighted net charge on every cell, including covered parents."""
    density, cell_volumes, _ = _broadcast_charge(rho, volumes)
    return float(np.sum(density * cell_volumes))


def covered_parent_charge(rho, volumes, leaf_mask) -> float:
    """Volume-weighted charge on covered/parent cells only."""
    density, cell_volumes, mask = _broadcast_charge(rho, volumes, leaf_mask)
    if np.all(mask):
        return 0.0
    return float(np.sum(density[~mask] * cell_volumes[~mask]))


def restricted_parent_charge(hierarchy) -> float:
    """Conservative average-down: sum of fine-child charges covering the parent."""
    density = np.asarray(hierarchy["charge_density"], dtype=np.float64)
    volumes = np.asarray(hierarchy["volumes"], dtype=np.float64)
    fine = np.asarray(hierarchy["fine_indices"])
    return float(np.sum(density[fine] * volumes[fine]))
