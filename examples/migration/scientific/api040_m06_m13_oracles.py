"""Independent homogeneous M06/M13 oracles; no PoPS imports or solver calls."""
from __future__ import annotations

import math
import numpy as np


def enthalpy_thermometer(h, *, capacity=2., latent=3., melting=1.):
    """Return T and liquid fraction for H=c*T+L*phase at T=Tm."""
    if not all(math.isfinite(v) and v > 0 for v in (capacity, latent, melting)):
        raise ValueError("capacity, latent and melting temperature must be positive finite")
    values = np.asarray(h, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("enthalpy must be finite")
    lower = capacity * melting
    upper = lower + latent
    temperature = np.where(values < lower, values / capacity,
                           np.where(values <= upper, melting,
                                    (values - latent) / capacity))
    fraction = np.clip((values - lower) / latent, 0., 1.)
    return temperature, fraction


def enthalpy_exact(h0, input_rate, time, **law):
    if not math.isfinite(input_rate) or not math.isfinite(time) or time < 0:
        raise ValueError("invalid input or time")
    enthalpy = np.asarray(h0, dtype=float) + input_rate * time
    return enthalpy, *enthalpy_thermometer(enthalpy, **law)


def chain_exact(initial, k_ab, k_bc, time):
    """Closed-form exp(t*S*K) for A→B→C, including the repeated eigenvalue."""
    if not all(math.isfinite(v) and v >= 0 for v in (k_ab, k_bc, time)):
        raise ValueError("reaction rates and time must be finite and nonnegative")
    state = np.asarray(initial, dtype=float)
    if state.shape[0] != 3 or not np.isfinite(state).all() or (state < 0).any():
        raise ValueError("initial species must be three finite nonnegative fields")
    a0, b0, c0 = state
    a = a0 * np.exp(-k_ab * time)
    if math.isclose(k_ab, k_bc, rel_tol=0., abs_tol=1.e-12):
        b = (b0 + k_ab * a0 * time) * np.exp(-k_bc * time)
    else:
        b = b0 * np.exp(-k_bc * time) + k_ab * a0 * (
            np.exp(-k_ab * time) - np.exp(-k_bc * time)) / (k_bc - k_ab)
    c = a0 + b0 + c0 - a - b
    return np.stack((a, b, c))
