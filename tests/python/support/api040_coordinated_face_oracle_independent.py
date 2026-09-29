"""Pure oracle for an asymmetric, genuinely nonconservative coordinated face.

The three-state system is unrelated to shallow water or stationary topography:
F=(c, p/2, -r), B[p,c]=beta*r, B[r,p]=gamma*c. B is not a flux Jacobian
for beta != 0. The full face result has shared flux and separately signed
left/right cell sources; the split is deliberately 30/70 rather than 50/50.
"""

from __future__ import annotations

import numpy as np


COMPONENTS = ("c", "p", "r")
PERMUTATIONS = (COMPONENTS, ("r", "c", "p"))
SPLIT_LEFT = 0.3


def initial_cell_means(cells: int) -> np.ndarray:
    if type(cells) is not int or cells < 4:
        raise ValueError("integer periodic grid with at least four cells required")
    x = (np.arange(cells) + 0.5) / cells
    return np.vstack((0.2 + 0.05*np.sinc(1/cells)*np.cos(2*np.pi*x),
                      -0.1 + 0.04*np.sinc(1/cells)*np.sin(2*np.pi*x),
                      0.3 + 0.03*np.sinc(2/cells)*np.cos(4*np.pi*x)))


def physical_flux(state: np.ndarray) -> np.ndarray:
    if np.shape(state)[0] != 3 or not np.isfinite(state).all():
        raise ValueError("three finite components required")
    return np.array((state[0], 0.5*state[1], -state[2]))


def path_integral(left: np.ndarray, right: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """Exact straight-path integral of B(U) dU, ordered c,p,r."""
    if not np.isfinite([beta, gamma]).all():
        raise ValueError("finite nonconservative coefficients required")
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != (3,) or right.shape != (3,) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("finite three-state face endpoints required")
    jump = right-left
    return np.array((0.0, beta*0.5*(left[2]+right[2])*jump[0],
                     gamma*0.5*(left[0]+right[0])*jump[1]))


def face(left: np.ndarray, right: np.ndarray, *, beta: float, gamma: float,
         order: tuple[str, str, str] = COMPONENTS,
         margin: float = 0.) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(order) != 3 or set(order) != set(COMPONENTS):
        raise ValueError("face storage order must permute c,p,r")
    if not np.isfinite(margin) or margin < 0.:
        raise ValueError("finite nonnegative numerical margin required")
    canonical = [order.index(name) for name in COMPONENTS]
    left_c = np.asarray(left)[canonical]
    right_c = np.asarray(right)[canonical]
    integral = path_integral(left_c, right_c, beta, gamma)
    # ||DF||_inf + max_path ||B||_inf bounds the composed DF+B, and leaves
    # dissipation for the intentionally unequal side split (not a full theorem
    # of FE stability for arbitrary states).
    speed = 1.0 + margin + max(abs(beta)*max(abs(left_c[2]), abs(right_c[2])),
                      abs(gamma)*max(abs(left_c[0]), abs(right_c[0])))
    common = 0.5*(physical_flux(left_c)+physical_flux(right_c)) - 0.5*speed*(right_c-left_c)
    left_source = -SPLIT_LEFT*integral
    right_source = -(1.0-SPLIT_LEFT)*integral
    back = [COMPONENTS.index(name) for name in order]
    return common[back], left_source[back], right_source[back], speed


def rhs(state: np.ndarray, *, beta: float, gamma: float,
        order: tuple[str, str, str] = COMPONENTS, margin: float = 0.) -> np.ndarray:
    if np.shape(state)[0] != 3 or not np.isfinite(state).all():
        raise ValueError("finite three-component periodic state required")
    cells = state.shape[1]
    interfaces = [face(state[:, i], state[:, (i+1) % cells], beta=beta, gamma=gamma,
                       order=order, margin=margin)
                  for i in range(cells)]
    common = np.stack([entry[0] for entry in interfaces], axis=1)
    left = np.stack([entry[1] for entry in interfaces], axis=1)
    right = np.stack([entry[2] for entry in interfaces], axis=1)
    return cells*(-common+np.roll(common, 1, axis=1)
                  +left+np.roll(right, 1, axis=1))


def forward_euler(cells: int, *, beta: float, gamma: float,
                  order: tuple[str, str, str] = COMPONENTS, steps: int = 8,
                  margin: float = 0.) -> np.ndarray:
    canonical = initial_cell_means(cells)
    state = canonical[[COMPONENTS.index(name) for name in order]].copy()
    dt = 0.05/cells
    for _ in range(steps):
        state += dt*rhs(state, beta=beta, gamma=gamma, order=order, margin=margin)
        if not np.isfinite(state).all():
            raise ValueError("nonfinite FE state")
    return state
