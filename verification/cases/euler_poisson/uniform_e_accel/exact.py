"""1-d uniform-E acceleration oracle.

Prescribed uniform field E0. Spatial dynamics are absent: density is uniform
and unchanged, velocity is spatially uniform.

    u(t) = u0 + (q/m) E0 t
    n(x, t) = n0
    ke(t) = (1/2) n0 m u(t)^2

Opposite charges have opposite accelerations. Dimensionless defaults:
q=1, m=1, E0=1, u0=0, n0=1. Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

Q = 1.0
MASS = 1.0
E0 = 1.0
U0 = 0.0
N0 = 1.0
P0 = 1.0
GAMMA = 1.4


def _as_samples(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def acceleration(*, q=Q, mass=MASS, e0=E0) -> float:
    """du/dt = (q/m) E0 for a prescribed uniform field."""
    mass_value = float(mass)
    if mass_value == 0.0:
        raise ValueError("mass must be nonzero")
    return float(q) / mass_value * float(e0)


def velocity(t, *, q=Q, mass=MASS, e0=E0, u0=U0) -> float:
    """u(t) = u0 + (q/m) E0 t."""
    return float(u0) + acceleration(q=q, mass=mass, e0=e0) * float(t)


def density(x, t, *, n0=N0) -> np.ndarray:
    """Uniform n(x, t) = n0. Independent of time."""
    del t
    samples = _as_samples(x)
    return np.full(samples.shape, float(n0), dtype=np.float64)


def kinetic_energy_density(t, *, q=Q, mass=MASS, e0=E0, u0=U0, n0=N0) -> float:
    """ke(t) = (1/2) n0 m u(t)^2 for a uniform species."""
    speed = velocity(t, q=q, mass=mass, e0=e0, u0=u0)
    return 0.5 * float(n0) * float(mass) * speed * speed


def primitives(x, t, *, q=Q, mass=MASS, e0=E0, u0=U0, n0=N0, p0=P0) -> np.ndarray:
    """Primitive W=(n, u, p). Shape (3, n)."""
    number = density(x, t, n0=n0)
    speed = velocity(t, q=q, mass=mass, e0=e0, u0=u0)
    pressure = np.full(number.shape, float(p0), dtype=np.float64)
    return np.stack((number, np.full(number.shape, speed, dtype=np.float64), pressure))
