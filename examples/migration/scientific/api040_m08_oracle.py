"""Independent periodic cell-centred Poisson oracle for the M08 variant.

Array order is (y, x).  This module imports no PoPS solver or field code.
"""
from __future__ import annotations

import numpy as np


DOMAIN_LENGTH = 2.0 * np.pi


def cell_means(n: int, *, charge_amplitude: float = .1,
               tracer_amplitude: float = .2) -> tuple[np.ndarray, np.ndarray]:
    if type(n) is not int or n < 4:
        raise ValueError("M08 needs a Cartesian grid with at least four cells per axis")
    center = (np.arange(n) + .5) * DOMAIN_LENGTH / n
    factor = np.sinc(1.0 / n)
    charge = charge_amplitude * factor**2 * np.cos(center)[:, None] * np.cos(center)[None, :]
    tracer = 1.0 + tracer_amplitude * factor * np.cos(center)[None, :]
    return np.ascontiguousarray(charge), np.ascontiguousarray(np.broadcast_to(tracer, (n, n)))


def poisson_discrete(charge: np.ndarray) -> np.ndarray:
    """Invert the five-point periodic -Laplacian by its Fourier eigenvalues."""
    q = np.asarray(charge, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.isfinite(q).all():
        raise ValueError("charge must be a finite square array")
    n = q.shape[0]
    if abs(float(np.mean(q))) > 1.e-13:
        raise ValueError("periodic Poisson charge must have zero mean")
    h = DOMAIN_LENGTH / n
    k = np.arange(n)
    eigen = 4.0 / h**2 * (np.sin(np.pi * k[:, None] / n)**2
                           + np.sin(np.pi * k[None, :] / n)**2)
    eigen[0, 0] = 1.0
    hat = np.fft.fft2(q)
    hat[0, 0] = 0.0
    potential = np.fft.ifft2(hat / eigen).real
    return np.ascontiguousarray(potential)


def negative_laplacian(potential: np.ndarray) -> np.ndarray:
    h = DOMAIN_LENGTH / potential.shape[0]
    return (4 * potential - np.roll(potential, 1, 0) - np.roll(potential, -1, 0)
            - np.roll(potential, 1, 1) - np.roll(potential, -1, 1)) / h**2


def guiding_velocity(potential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = DOMAIN_LENGTH / potential.shape[0]
    ux = (np.roll(potential, -1, 0) - np.roll(potential, 1, 0)) / (2 * h)
    uy = -(np.roll(potential, -1, 1) - np.roll(potential, 1, 1)) / (2 * h)
    return ux, uy


def centered_divergence(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    h = DOMAIN_LENGTH / ux.shape[0]
    return ((np.roll(ux, -1, 1) - np.roll(ux, 1, 1)
             + np.roll(uy, -1, 0) - np.roll(uy, 1, 0)) / (2 * h))


def _face_flux(quantity: np.ndarray, velocity: np.ndarray, axis: int) -> np.ndarray:
    right = np.roll(quantity, -1, axis)
    velocity_right = np.roll(velocity, -1, axis)
    return (.5 * (velocity * quantity + velocity_right * right)
            - .5 * np.maximum(np.abs(velocity), np.abs(velocity_right))
            * (right - quantity))


def _rhs(charge: np.ndarray, tracer: np.ndarray,
         *, fixed_potential: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    potential = poisson_discrete(charge) if fixed_potential is None else fixed_potential
    ux, uy = guiding_velocity(potential)
    h = DOMAIN_LENGTH / charge.shape[0]
    rates = []
    for state in (charge, tracer):
        fx = _face_flux(state, ux, 1)
        fy = _face_flux(state, uy, 0)
        rates.append(-((fx - np.roll(fx, 1, 1)) + (fy - np.roll(fy, 1, 0))) / h)
    return rates[0], rates[1]


def ssprk2_reference(charge: np.ndarray, tracer: np.ndarray, *, dt: float,
                     steps: int, stale_second_stage: bool = False
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Separate first-order FV/Rusanov and SSPRK2 implementation.

    ``stale_second_stage`` is a deliberately incorrect comparison trajectory.
    """
    q, c = np.array(charge, copy=True), np.array(tracer, copy=True)
    for _ in range(steps):
        original_potential = poisson_discrete(q)
        rq0, rc0 = _rhs(q, c, fixed_potential=original_potential)
        q1, c1 = q + dt * rq0, c + dt * rc0
        rq1, rc1 = _rhs(q1, c1, fixed_potential=(
            original_potential if stale_second_stage else None))
        q, c = .5 * (q + q1 + dt * rq1), .5 * (c + c1 + dt * rc1)
    return q, c


def field_metrics(charge: np.ndarray, potential: np.ndarray,
                  gradient: np.ndarray) -> dict[str, float]:
    q = np.asarray(charge)
    phi = np.asarray(potential)
    grad = np.asarray(gradient)
    if q.shape != phi.shape or grad.shape != (2, *q.shape):
        raise ValueError("field archive has incompatible component shapes")
    oracle = poisson_discrete(q)
    ux, uy = guiding_velocity(phi)
    # Native gradient is ordered (x,y); velocity is its quarter-turn.
    native_ux, native_uy = grad[1], -grad[0]
    return {
        "poisson_residual_max": float(np.max(np.abs(negative_laplacian(phi) - q))),
        "fft_potential_max_error": float(np.max(np.abs(phi - oracle))),
        "gradient_max_error": float(max(np.max(np.abs(native_ux - ux)),
                                           np.max(np.abs(native_uy - uy)))),
        "velocity_divergence_max": float(np.max(np.abs(centered_divergence(native_ux, native_uy)))),
    }
