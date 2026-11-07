"""Independent finite-volume/Fourier and continuum freestreaming equations.

NumPy/stdlib only. This computes references, never advances a PoPS runtime.
"""
import math

import numpy as np


def cell_centers(nx, nv):
    return (np.arange(nx) + .5) / nx, -1 + 2 * (np.arange(nv) + .5) / nv


def initial(nx, nv):
    x, v = cell_centers(nx, nv)
    return ((1 + .1 * np.sinc(1 / nx) * np.cos(2 * math.pi * x[:, None]))
            * (1 + .25 * v[None, :]))[None, ...]


def exact_cell_average(nx, nv, time):
    """Closed integral of (1+.25v) cos(2π(x-vt)) over each v cell.

    Differentiating sinc gives the first centered velocity moment. x averaging
    contributes sinc(π dx). This is the original continuum PDE, distinct from
    the CellMidpoint velocity realization in discrete_fourier().
    """
    x, v = cell_centers(nx, nv)
    half = 1 / nv
    b = 2 * math.pi * time
    z = b * half
    if z == 0:
        return initial(nx, nv)
    sinc = math.sin(z) / z
    # mean s*sin(bs), s∈[-half,half]
    first = half * (math.sin(z) - z * math.cos(z)) / (z * z)
    phase = 2 * math.pi * x[:, None] - b * v[None, :]
    wave = (1 + .25 * v[None, :]) * sinc * np.cos(phase) + .25 * first * np.sin(phase)
    return ((1 + .25 * v[None, :]) + .1 * np.sinc(1 / nx) * wave)[None, ...]


def discrete_fourier(nx, nv, dt, steps, *, sign=1, temporal_order=2):
    x, v = cell_centers(nx, nv)
    velocity = sign * v
    theta = 2 * math.pi / nx
    eigenvalue = -np.abs(velocity) * nx * (1 - np.exp(-1j * np.sign(velocity) * theta))
    factor = 1 + dt * eigenvalue
    if temporal_order == 2:
        factor += .5 * (dt * eigenvalue) ** 2
    elif temporal_order != 1:
        raise ValueError("reference requires explicit ForwardEuler or SSPRK2")
    wave = np.real(np.exp(2j * math.pi * x[:, None]) * factor[None, :] ** steps)
    return ((1 + .25 * v[None, :]) * (1 + .1 * np.sinc(1 / nx) * wave))[None, ...]


def moments(values, nx, nv):
    values = np.asarray(values)
    if values.shape != (1, nx, nv) or not np.isfinite(values).all():
        raise ValueError("freestreaming state rank/shape/finitude differs")
    _, velocities = cell_centers(nx, nv)
    # Explicit CellMidpoint measure dx*dv; no normalization by number of cells.
    measure = 2 / (nx * nv)
    return tuple(math.fsum(float(values[0, i, j]) * float(velocities[j]) ** k * measure
                           for i in range(nx) for j in range(nv)) for k in (0, 1, 2))


def receive(values, seed, nx, nv, dt, steps):
    values, seed = np.asarray(values), np.asarray(seed)
    expected = discrete_fourier(nx, nv, dt, steps)
    if values.shape != expected.shape or not np.isfinite(values).all():
        raise ValueError("freestreaming physical state differs")
    discrete_error = float(np.max(np.abs(values - expected)))
    if discrete_error > 3e-12:
        raise ValueError("original signed FV/SSPRK2 transport equation differs")
    time = steps * dt
    l1 = float(np.mean(np.abs(values - exact_cell_average(nx, nv, time))))
    if l1 > .65 * time / nx or np.max(np.abs(values - seed)) < .005:
        raise ValueError("continuum displacement error or nontrivial evolution differs")
    before, after = moments(seed, nx, nv), moments(values, nx, nv)
    if max(abs(a - b) for a, b in zip(before, after, strict=True)) > 3e-12:
        raise ValueError("phase-space mass or velocity moment is not conserved")
    return dict(discrete_error=discrete_error, continuum_l1=l1, moments=after)
