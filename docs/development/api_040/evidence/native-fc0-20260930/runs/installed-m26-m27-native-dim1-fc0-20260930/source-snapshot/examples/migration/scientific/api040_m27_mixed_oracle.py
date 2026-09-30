"""Independent periodic Fourier and original-equation oracle for mixed linear M27.

No PoPS imports. The two unknowns remain c and chemical potential mu; the
Fourier 2x2 solve is only an oracle for the saved native cell means.
"""
from __future__ import annotations

import math

import numpy as np


EPSILON = .08
STEP = .01
STEPS = 10


def initial_means(cells: int) -> np.ndarray:
    center = (np.arange(cells, dtype=float) + .5) / cells
    return (.4 + .1 * np.sinc(1 / cells) * np.cos(2 * math.pi * center)
            + .05 * np.sinc(2 / cells) * np.sin(4 * math.pi * center))


def integrated_initial_means(cells: int) -> np.ndarray:
    edges = np.arange(cells + 1, dtype=float) / cells
    first = .1 * cells * (np.sin(2 * math.pi * edges[1:])
                         - np.sin(2 * math.pi * edges[:-1])) / (2 * math.pi)
    second = .05 * cells * (np.cos(4 * math.pi * edges[:-1])
                           - np.cos(4 * math.pi * edges[1:])) / (4 * math.pi)
    return .4 + first + second


def laplacian(values: np.ndarray) -> np.ndarray:
    cells = len(values)
    return cells**2 * (np.roll(values, -1) - 2 * values + np.roll(values, 1))


def quadratic_energy(c: np.ndarray, epsilon: float = EPSILON) -> float:
    cells = len(c)
    jumps = np.roll(c, -1) - c
    return .5 * (float(np.dot(c, c)) / cells
                 + epsilon**2 * cells * float(np.dot(jumps, jumps)))


def fourier_mixed_step(c: np.ndarray, dt: float = STEP,
                       epsilon: float = EPSILON) -> tuple[np.ndarray, np.ndarray]:
    """Solve the two original BE relations mode by mode, without eliminating mu."""
    cells = len(c)
    transformed = np.fft.fft(c)
    wave = np.arange(cells)
    eigenvalue = -4 * cells**2 * np.sin(math.pi * wave / cells)**2
    solved_c = np.empty(cells, dtype=complex)
    solved_mu = np.empty(cells, dtype=complex)
    for index, symbol in enumerate(eigenvalue):
        matrix = np.array(((1., -dt * symbol),
                           (-1. + epsilon**2 * symbol, 1.)))
        solved_c[index], solved_mu[index] = np.linalg.solve(
            matrix, np.array((transformed[index], 0j)))
    return np.fft.ifft(solved_c).real, np.fft.ifft(solved_mu).real


def trajectory(cells: int, steps: int = STEPS) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initial = initial_means(cells)
    c = initial.copy()
    before = c.copy()
    mu = np.zeros_like(c)
    for _ in range(steps):
        before = c.copy()
        c, mu = fourier_mixed_step(c)
    return initial, before, c, mu


def original_residuals(before: np.ndarray, after: np.ndarray,
                       mu: np.ndarray, dt: float = STEP,
                       epsilon: float = EPSILON) -> tuple[float, float]:
    first = after - before - dt * laplacian(mu)
    second = mu - after + epsilon**2 * laplacian(after)
    return float(np.max(np.abs(first))), float(np.max(np.abs(second)))
