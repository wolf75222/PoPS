"""Independent constant-coefficient Fourier oracle, with no PoPS dependency.

Canonical physical component order is (temperature, heat_flux_x, heat_flux_y).
Both the continuum exponential and the specified FV/IMEX discretization are
computed from small complex matrices, independently of native flux assembly.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

MODE = (1, 2)
MEAN = np.array((1., .03, -.02))
AMPLITUDE = np.array((.12 + 0j, .015j, -.01 + .005j))


def coefficients(tau, kappa):
    tau, kappa = float(tau), float(kappa)
    if not np.isfinite(tau) or not np.isfinite(kappa) or tau <= 0 or kappa <= 0:
        raise ValueError("Maxwell-Cattaneo requires finite tau>0 and kappa>0")
    a = kappa / tau
    x = np.array(((0., 1., 0.), (a, 0., 0.), (0., 0., 0.)))
    y = np.array(((0., 0., 1.), (0., 0., 0.), (a, 0., 0.)))
    relaxation = np.diag((0., -1. / tau, -1. / tau))
    return x, y, relaxation


def symbols(tau, kappa, *, cells=None, mode=MODE):
    x, y, relaxation = coefficients(tau, kappa)
    wave = 2 * np.pi * np.asarray(mode, dtype=float)
    if cells is None:
        principal = -1j * (wave[0] * x + wave[1] * y)
    else:
        n = int(cells)
        if n < 2 or n != cells:
            raise ValueError("cells must be an integer >=2")
        speed = np.sqrt(kappa / tau)
        principal = -1j * n * (np.sin(wave[0] / n) * x + np.sin(wave[1] / n) * y)
        principal -= speed * n * np.sum(1 - np.cos(wave / n)) * np.eye(3)
    return principal, relaxation


def field(mean, amplitude, cells, *, mode=MODE):
    n = int(cells)
    centers = (np.arange(n) + .5) / n
    x, y = np.meshgrid(centers, centers)
    average = np.prod(np.sinc(np.asarray(mode, dtype=float) / n))
    phase = np.exp(2j * np.pi * (mode[0] * x + mode[1] * y))
    return np.ascontiguousarray(np.asarray(mean)[:, None, None]
                                + np.real(np.asarray(amplitude)[:, None, None] * phase) * average)


def exact(cells, time, tau, kappa, *, discrete_space=False, mode=MODE):
    principal, relaxation = symbols(tau, kappa, cells=cells if discrete_space else None, mode=mode)
    amplitude = expm(time * (principal + relaxation)) @ AMPLITUDE
    mean = expm(time * relaxation) @ MEAN
    return field(mean, amplitude, cells, mode=mode)


def imex_euler(cells, dt, steps, tau, kappa, *, mode=MODE):
    principal, relaxation = symbols(tau, kappa, cells=cells, mode=mode)
    backward = np.linalg.inv(np.eye(3) - dt * relaxation)
    update = backward @ (np.eye(3) + dt * principal)
    amplitude = np.linalg.matrix_power(update, steps) @ AMPLITUDE
    mean = np.linalg.matrix_power(backward, steps) @ MEAN
    return field(mean, amplitude, cells, mode=mode)
