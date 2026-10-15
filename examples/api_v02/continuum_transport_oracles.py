"""Independent NumPy receivers for the declared periodic discrete equations.

These functions assess native results; they do not construct or advance a PoPS
state. No authored symbolic expression or generated operator is imported here.
"""
from __future__ import annotations

import numpy as np


def anisotropic_ssprk2(initial, *, velocity, tensor, lengths, dt, steps):
    """Fourier symbol of upwind advection and centered full-tensor diffusion."""
    values = np.asarray(initial, dtype=float)
    ny, nx = values.shape
    hx, hy = lengths[0] / nx, lengths[1] / ny
    tx = 2 * np.pi * np.fft.fftfreq(nx)[None, :]
    ty = 2 * np.pi * np.fft.fftfreq(ny)[:, None]
    ax, ay = velocity
    symbol = (
        -abs(ax) * (1 - np.cos(tx)) / hx - 1j * ax * np.sin(tx) / hx
        - abs(ay) * (1 - np.cos(ty)) / hy - 1j * ay * np.sin(ty) / hy
        - 4 * tensor[0][0] * np.sin(tx / 2)**2 / hx**2
        - 4 * tensor[1][1] * np.sin(ty / 2)**2 / hy**2
        - (tensor[0][1] + tensor[1][0]) * np.sin(tx) * np.sin(ty) / (hx * hy)
    )
    amplification = 1 + dt * symbol + .5 * (dt * symbol)**2
    return np.fft.ifft2(np.fft.fft2(values) * amplification**steps).real


def periodic_poisson(values, lengths):
    """Mean-zero inverse of the positive five-point Laplacian."""
    ny, nx = values.shape
    hx, hy = lengths[0] / nx, lengths[1] / ny
    eigenvalues = (
        4 * np.sin(np.pi * np.fft.fftfreq(nx)[None, :])**2 / hx**2
        + 4 * np.sin(np.pi * np.fft.fftfreq(ny)[:, None])**2 / hy**2
    )
    transformed = np.fft.fft2(values)
    potential = np.zeros_like(transformed)
    np.divide(transformed, eigenvalues, out=potential, where=eigenvalues != 0)
    return np.fft.ifft2(potential).real


def guiding_center_rhs(values, lengths, *, velocity_sign=1):
    """Cell-centered rotated potential gradient, two-trace Rusanov face law."""
    ny, nx = values.shape
    hx, hy = lengths[0] / nx, lengths[1] / ny
    phi = periodic_poisson(values, lengths)
    gx = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2 * hx)
    gy = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2 * hy)
    result = np.zeros_like(values)
    for axis, spacing, speed in ((1, hx, -velocity_sign * gy), (0, hy, velocity_sign * gx)):
        right = np.roll(values, -1, axis)
        right_speed = np.roll(speed, -1, axis)
        face = .5 * (speed * values + right_speed * right) - .5 * np.maximum(
            abs(speed), abs(right_speed)) * (right - values)
        result -= (face - np.roll(face, 1, axis)) / spacing
    return result, phi


def guiding_center_ssprk2(initial, *, lengths, dt, steps, velocity_sign=1):
    values = np.array(initial, dtype=float, copy=True)
    last_stage = None
    for _ in range(steps):
        first, _ = guiding_center_rhs(values, lengths, velocity_sign=velocity_sign)
        predictor = values + dt * first
        second, phi = guiding_center_rhs(predictor, lengths, velocity_sign=velocity_sign)
        last_stage = (predictor.copy(), phi)
        values = .5 * values + .5 * (predictor + dt * second)
    return values, last_stage
