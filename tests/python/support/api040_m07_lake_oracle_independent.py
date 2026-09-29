"""Independent finite-volume oracle for the bounded M07 lake-at-rest variant.

No PoPS import or generated flux. The two momentum face contributions follow
Audusse hydrostatic reconstruction; a shared face flux cannot replace them.
"""

from __future__ import annotations

from math import erf, pi, sqrt

import numpy as np


RESOLUTIONS = (40, 80, 160)
FINAL_TIME = 1.0
GRAVITY = 1.0
SURFACE = 1.0
EQUILIBRIUM_TOLERANCE = 1.0e-12


def bottom_cell_mean(x_left: float, x_right: float) -> float:
    """Exact average of 0.2 exp(-50 x²), including ghost cells."""
    if not x_right > x_left:
        raise ValueError("cell bounds must increase")
    return (0.2 * sqrt(pi) / (2.0 * sqrt(50.0))
            * (erf(sqrt(50.0) * x_right) - erf(sqrt(50.0) * x_left))
            / (x_right - x_left))


def initial_with_equilibrium_ghosts(cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (U,z) in the N+2 cell order left ghost, interior, right ghost."""
    if type(cells) is not int or cells < 2:
        raise ValueError("at least two integer cells required")
    dx = 2.0 / cells
    x_left = -1.0 + dx * np.arange(-1, cells + 1)
    bottom = np.array([bottom_cell_mean(x, x + dx) for x in x_left])
    state = np.vstack((SURFACE - bottom, np.zeros_like(bottom)))
    return state, bottom


def physical_flux(state: np.ndarray) -> np.ndarray:
    depth, momentum = state
    if np.any(depth <= 0.0) or not np.isfinite(state).all():
        raise ValueError("wet finite state required")
    return np.array((momentum, momentum * momentum / depth + 0.5 * GRAVITY * depth * depth))


def hydrostatic_face(left: np.ndarray, right: np.ndarray,
                     z_left: float, z_right: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared HLL flux and distinct left/right momentum contributions."""
    if not np.isfinite([z_left, z_right]).all():
        raise ValueError("finite bottom required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("finite fluid states required")
    depth_left, depth_right = float(left[0]), float(right[0])
    if depth_left <= 0.0 or depth_right <= 0.0:
        raise ValueError("this wet-domain oracle does not define a dry state")
    crest = max(z_left, z_right)
    velocity_left, velocity_right = float(left[1])/depth_left, float(right[1])/depth_right
    wet_left = max(0.0, depth_left + z_left - crest)
    wet_right = max(0.0, depth_right + z_right - crest)
    reconstructed_left = np.array([wet_left, wet_left * velocity_left])
    reconstructed_right = np.array([wet_right, wet_right * velocity_right])
    if min(reconstructed_left[0], reconstructed_right[0]) <= 0.0:
        raise ValueError("the bounded M07 witness remains strictly wet")
    flux_left = physical_flux(reconstructed_left)
    flux_right = physical_flux(reconstructed_right)
    u_left = reconstructed_left[1] / reconstructed_left[0]
    u_right = reconstructed_right[1] / reconstructed_right[0]
    lower = min(u_left - sqrt(GRAVITY * reconstructed_left[0]),
                u_right - sqrt(GRAVITY * reconstructed_right[0]))
    upper = max(u_left + sqrt(GRAVITY * reconstructed_left[0]),
                u_right + sqrt(GRAVITY * reconstructed_right[0]))
    if lower >= 0.0:
        common = flux_left
    elif upper <= 0.0:
        common = flux_right
    else:
        common = ((upper * flux_left - lower * flux_right
                   + lower * upper * (reconstructed_right - reconstructed_left))
                  / (upper - lower))
    # Separate hydrostatic pressure corrections belong to the two adjacent cells.
    from_left = common + np.array([0.0, 0.5 * GRAVITY *
                                    (depth_left**2 - reconstructed_left[0]**2)])
    from_right = common + np.array([0.0, 0.5 * GRAVITY *
                                     (depth_right**2 - reconstructed_right[0]**2)])
    return common, from_left, from_right


def rhs(state: np.ndarray, bottom: np.ndarray, *, corrected: bool = True) -> np.ndarray:
    if state.shape != (2, len(bottom)) or len(bottom) < 4:
        raise ValueError("state and bottom require the same ghosted cells")
    faces = [hydrostatic_face(state[:, i], state[:, i + 1], bottom[i], bottom[i + 1])
             for i in range(len(bottom) - 1)]
    right = np.stack([face[1 if corrected else 0] for face in faces[1:]], axis=1)
    left = np.stack([face[2 if corrected else 0] for face in faces[:-1]], axis=1)
    dx = 2.0 / (len(bottom) - 2)
    return -(right - left) / dx


def trajectory(cells: int) -> tuple[np.ndarray, float]:
    state, bottom = initial_with_equilibrium_ghosts(cells)
    initial = state[:, 1:-1].copy()
    dt = 0.1 * (2.0 / cells)
    steps = 5 * cells  # T=1 and dt=0.2/N.
    if abs(steps * dt - FINAL_TIME) > 1.0e-14:
        raise AssertionError("predeclared FE calendar does not land on T=1")
    for _ in range(steps):
        state[:, 1:-1] += dt * rhs(state, bottom)
        if np.any(state[0, 1:-1] <= 0.0) or not np.isfinite(state).all():
            raise ValueError("native comparison must not repair inadmissible depths")
    return state[:, 1:-1], float(np.max(np.abs(state[:, 1:-1] - initial)))
