"""Independent cell-mean and energy oracle for the closed M05 shear subcase.

This module has no PoPS import. It evaluates periodic finite-volume differences
directly and keeps continuous, semidiscrete, and time-discrete claims separate.
"""
from __future__ import annotations

import math

import numpy as np


def initial_means(cells: int) -> np.ndarray:
    """Exact averages of sin(2*pi*x) over cells in [0,1)."""
    center = (np.arange(cells, dtype=float) + .5) / cells
    return np.sinc(1. / cells) * np.sin(2 * math.pi * center)


def integrated_initial_means(cells: int) -> np.ndarray:
    """Second expression from the antiderivative, independent of sinc."""
    edges = np.arange(cells + 1, dtype=float) / cells
    return cells * (np.cos(2 * math.pi * edges[:-1]) -
                    np.cos(2 * math.pi * edges[1:])) / (2 * math.pi)


def periodic_faces(state: np.ndarray, viscosity: float) -> np.ndarray:
    """Flux at i+1/2 from explicit adjacent cell values."""
    cells = len(state)
    return viscosity * cells * (np.roll(state, -1) - state)


def periodic_rate(state: np.ndarray, viscosity: float) -> np.ndarray:
    faces = periodic_faces(state, viscosity)
    return len(state) * (faces - np.roll(faces, 1))


def energy(state: np.ndarray) -> float:
    return .5 * float(np.dot(state, state)) / len(state)


def semidiscrete_work(state: np.ndarray, viscosity: float) -> tuple[float, float]:
    """Rate inner product and negative face-jump quadratic form."""
    cells = len(state)
    work = float(np.dot(state, periodic_rate(state, viscosity))) / cells
    jumps = np.roll(state, -1) - state
    dissipation = -viscosity * cells * float(np.dot(jumps, jumps))
    return work, dissipation


def fourier_amplitudes(cells: int, viscosity: float, duration: float,
                       steps: int) -> tuple[float, float, float, float]:
    """Continuous, spatially discrete, and Forward Euler amplitudes."""
    h = 1. / cells
    wave = 2 * math.pi
    spatial_eigenvalue = 4 * math.sin(math.pi / cells)**2 / h**2
    continuous = math.exp(-viscosity * wave**2 * duration)
    semidiscrete = math.exp(-viscosity * spatial_eigenvalue * duration)
    step = duration / steps
    forward_euler = (1 - viscosity * spatial_eigenvalue * step)**steps
    return continuous, semidiscrete, forward_euler, spatial_eigenvalue


def forward_euler_states(cells: int, viscosity: float, duration: float,
                         steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Oracle for the penultimate and final *cell means*."""
    initial = initial_means(cells)
    step = duration / steps
    _, _, _, eigenvalue = fourier_amplitudes(cells, viscosity, duration, steps)
    factor = 1 - step * viscosity * eigenvalue
    return initial * factor**(steps - 1), initial * factor**steps


def forward_euler_energy_defect(state: np.ndarray, viscosity: float,
                                step: float) -> tuple[float, float]:
    """Actual FE energy increment and its exact temporal correction."""
    rate = periodic_rate(state, viscosity)
    increment = energy(state + step * rate) - energy(state)
    work, dissipation = semidiscrete_work(state, viscosity)
    correction = .5 * step**2 * float(np.dot(rate, rate)) / len(state)
    return increment, step * dissipation + correction
