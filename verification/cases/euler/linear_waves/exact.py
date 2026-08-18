"""1-d gamma-law Euler linear eigenmodes in primitive variables W=(rho, u, p).

Background: rho=1, u=0, p=1/gamma so the sound speed is
c = sqrt(gamma p / rho) = 1. gamma=1.4.

Right eigenvectors use the Athena/Athena++ primitive scaling
dW = (d rho, d u, d p):

- left acoustic  (lambda = u-c): (1, -c/rho, c^2)
- entropy/contact (lambda = u):  (1, 0, 0)  — density only
- right acoustic (lambda = u+c): (1, +c/rho, c^2)

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import math

import numpy as np

GAMMA = 1.4
MODES = ("left", "entropy", "right")


def background() -> dict:
    """Uniform rest state. p=1/gamma so c=1 at gamma=1.4."""
    return {"rho": 1.0, "u": 0.0, "p": 1.0 / GAMMA}


def acoustic_speed(W) -> float:
    """c = sqrt(gamma p / rho) for a gamma-law ideal gas."""
    rho = float(W["rho"])
    pressure = float(W["p"])
    if rho <= 0.0 or pressure <= 0.0:
        raise ValueError("non-positive thermodynamic state")
    return float(math.sqrt(GAMMA * pressure / rho))


def right_eigenvectors(W) -> dict:
    """Primitive right eigenvectors. Each value has shape (3,)."""
    rho = float(W["rho"])
    speed = acoustic_speed(W)
    return {
        "left": np.array([1.0, -speed / rho, speed * speed], dtype=np.float64),
        "entropy": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "right": np.array([1.0, speed / rho, speed * speed], dtype=np.float64),
    }


def eigenvalue(mode: str, W) -> float:
    """Wave speed of one eigenmode: u-c, u, or u+c."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    velocity = float(W["u"])
    speed = acoustic_speed(W)
    if mode == "left":
        return velocity - speed
    if mode == "entropy":
        return velocity
    return velocity + speed


def exact_mode(x, t, *, mode, eps=1e-6, k=2.0 * math.pi) -> np.ndarray:
    """Linear eigenmode in primitives. Shape (3, n)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    state = background()
    vector = right_eigenvectors(state)[mode]
    lam = eigenvalue(mode, state)
    samples = np.atleast_1d(np.asarray(x, dtype=np.float64))
    phase = float(k) * samples - lam * abs(float(k)) * float(t)
    wave = float(eps) * np.sin(phase)
    bar = np.array([state["rho"], state["u"], state["p"]], dtype=np.float64)
    return bar[:, None] + vector[:, None] * wave
