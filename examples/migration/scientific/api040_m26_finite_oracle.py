"""Independent Fourier oracle for the closed N=12 periodic M26 finite witness."""
from __future__ import annotations

import math

import numpy as np


NODES = 12
WEIGHT = 1. / NODES
COORDINATES = (np.arange(NODES, dtype=float) + .5) / NODES
PERMUTATION = (6, 0, 9, 3, 11, 2, 8, 5, 1, 10, 4, 7)


def density_pair(*, variation: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x = COORDINATES
    if variation:
        first = .9 + .15 * np.cos(2 * math.pi * x) - .03 * np.sin(6 * math.pi * x)
        second = 1.2 - .11 * np.sin(2 * math.pi * x) + .04 * np.cos(4 * math.pi * x)
    else:
        first = 1. + .2 * np.cos(2 * math.pi * x) + .1 * np.sin(4 * math.pi * x)
        second = .85 + .12 * np.sin(2 * math.pi * x) + .07 * np.cos(6 * math.pi * x)
    assert np.all(first > 0) and np.all(second > 0)
    return first, second


def fourier_convolution(density: np.ndarray) -> np.ndarray:
    """W*rho from only the first discrete Fourier coefficient, not a kernel matrix."""
    if density.shape != (NODES,):
        raise ValueError("the M26 finite oracle requires twelve ordered nodes")
    angle = 2 * math.pi * COORDINATES
    cosine = WEIGHT * float(np.dot(density, np.cos(angle)))
    sine = WEIGHT * float(np.dot(density, np.sin(angle)))
    return cosine * np.cos(angle) + sine * np.sin(angle)


def metrics(first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray | float]:
    potential_first = fourier_convolution(first)
    potential_second = fourier_convolution(second)
    direction = second - first
    first_energy = .5 * WEIGHT * float(np.dot(first, potential_first))
    second_energy = .5 * WEIGHT * float(np.dot(second, potential_second))
    return {
        "potential_first": potential_first,
        "potential_second": potential_second,
        "energy_first": first_energy,
        "energy_second": second_energy,
        "directional_derivative": WEIGHT * float(np.dot(direction, potential_first)),
        "pair_first_adjoint_second": WEIGHT * float(np.dot(first, potential_second)),
        "pair_action_first_second": WEIGHT * float(np.dot(potential_first, second)),
        "mass_first": WEIGHT * float(np.sum(first)),
        "mass_second": WEIGHT * float(np.sum(second)),
    }


def exact_quadratic_increment(first: np.ndarray, second: np.ndarray) -> float:
    direction = second - first
    return metrics(first, second)["directional_derivative"] \
        + .5 * WEIGHT * float(np.dot(direction, fourier_convolution(direction)))
