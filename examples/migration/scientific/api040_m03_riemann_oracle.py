"""Independent exact Riemann oracle for ideal and stiffened-gas Euler.

Only NumPy is needed. The finite-volume implementation never imports this file;
it is used for prescribed cell averages and post-run comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Gas:
    gamma: float
    p_inf: float = 0.0

    def __post_init__(self):
        if not (np.isfinite(self.gamma) and self.gamma > 1 and
                np.isfinite(self.p_inf) and self.p_inf >= 0):
            raise ValueError("gamma must exceed one and p_inf must be nonnegative")


@dataclass(frozen=True)
class Primitive:
    rho: float
    u: float
    p: float

    def __post_init__(self):
        if not (np.isfinite(self.rho) and self.rho > 0 and
                np.isfinite(self.u) and np.isfinite(self.p) and self.p > 0):
            raise ValueError("Riemann states require finite rho > 0 and p > 0")


def conservative(state: Primitive, gas: Gas) -> np.ndarray:
    """2D Euler state, with zero transverse momentum, in physical units."""
    energy = (state.p + gas.gamma * gas.p_inf) / (gas.gamma - 1)
    energy += .5 * state.rho * state.u * state.u
    return np.array((state.rho, state.rho * state.u, 0., energy))


def _sound(state: Primitive, gas: Gas) -> float:
    return float(np.sqrt(gas.gamma * (state.p + gas.p_inf) / state.rho))


def _wave_function(p: float, state: Primitive, gas: Gas) -> float:
    shifted = p + gas.p_inf
    base = state.p + gas.p_inf
    if p <= state.p:
        exponent = (gas.gamma - 1) / (2 * gas.gamma)
        return 2 * _sound(state, gas) / (gas.gamma - 1) * ((shifted / base)**exponent - 1)
    a = 2 / ((gas.gamma + 1) * state.rho)
    b = (gas.gamma - 1) / (gas.gamma + 1) * base
    return (p - state.p) * np.sqrt(a / (shifted + b))


@lru_cache(maxsize=32)
def star_state(left: Primitive, right: Primitive, gas: Gas) -> tuple[float, float]:
    """Solve Toro's pressure residual with p -> p+p_inf, by monotone bisection."""
    def residual(p):
        return (_wave_function(p, left, gas) + _wave_function(p, right, gas)
                + right.u - left.u)

    lo = 0.
    if residual(lo) >= 0:
        raise ValueError("chosen positive-pressure Riemann family has no p_star > 0")
    hi = max(left.p, right.p, 1.)
    while residual(hi) <= 0:
        hi *= 2
        if hi > 1.e12:
            raise ValueError("could not bracket exact Riemann pressure")
    for _ in range(100):
        mid = (lo + hi) / 2
        if residual(mid) < 0:
            lo = mid
        else:
            hi = mid
    p = (lo + hi) / 2
    u = .5 * (left.u + right.u +
              _wave_function(p, right, gas) - _wave_function(p, left, gas))
    return float(p), float(u)


def _star_density(state: Primitive, p: float, gas: Gas) -> float:
    ratio = (p + gas.p_inf) / (state.p + gas.p_inf)
    if p <= state.p:
        return state.rho * ratio**(1 / gas.gamma)
    beta = (gas.gamma - 1) / (gas.gamma + 1)
    return state.rho * (ratio + beta) / (beta * ratio + 1)


def wave_locations(left: Primitive, right: Primitive, gas: Gas) -> tuple[float, ...]:
    """Sorted self-similar wave speeds, including both fan edges and contact."""
    p, u = star_state(left, right, gas)
    locations = [u]
    for side, state in ((-1, left), (1, right)):
        c = _sound(state, gas)
        if p <= state.p:
            c_star = c * ((p + gas.p_inf) / (state.p + gas.p_inf))**(
                (gas.gamma - 1) / (2 * gas.gamma))
            locations.extend((state.u + side * c, u + side * c_star))
        else:
            ratio = (p + gas.p_inf) / (state.p + gas.p_inf)
            shock = c * np.sqrt((gas.gamma + 1) / (2 * gas.gamma) * ratio
                                + (gas.gamma - 1) / (2 * gas.gamma))
            locations.append(state.u + side * shock)
    return tuple(sorted(locations))


def sample(x: float, time: float, left: Primitive, right: Primitive,
           gas: Gas) -> Primitive:
    if time < 0:
        raise ValueError("time must be nonnegative")
    if time == 0:
        return left if x < 0 else right
    xi = x / time
    p, u = star_state(left, right, gas)
    if xi < u:
        state, side = left, -1
    else:
        state, side = right, 1
    c = _sound(state, gas)
    if p > state.p:
        ratio = (p + gas.p_inf) / (state.p + gas.p_inf)
        shock = c * np.sqrt((gas.gamma + 1) / (2 * gas.gamma) * ratio
                            + (gas.gamma - 1) / (2 * gas.gamma))
        if side * (xi - state.u) > shock:
            return state
        return Primitive(_star_density(state, p, gas), u, p)
    head = state.u + side * c
    c_star = c * ((p + gas.p_inf) / (state.p + gas.p_inf))**(
        (gas.gamma - 1) / (2 * gas.gamma))
    tail = u + side * c_star
    if side * (xi - head) > 0:
        return state
    if side * (xi - tail) < 0:
        return Primitive(_star_density(state, p, gas), u, p)
    fan_u = 2 / (gas.gamma + 1) * (-side * c + (gas.gamma - 1) / 2 * state.u + xi)
    fan_c = 2 / (gas.gamma + 1) * (c - side * (gas.gamma - 1) / 2 * (state.u - xi))
    rho = state.rho * (fan_c / c)**(2 / (gas.gamma - 1))
    pressure = (state.p + gas.p_inf) * (fan_c / c)**(
        2 * gas.gamma / (gas.gamma - 1)) - gas.p_inf
    return Primitive(float(rho), float(fan_u), float(pressure))


def cell_averages(n: int, time: float, left: Primitive, right: Primitive,
                  gas: Gas, *, quadrature_order: int = 32) -> np.ndarray:
    """Exact self-similar solution integrated over each [-1/2,1/2] cell.

    Gauss integration is applied to smooth subintervals split at every fan
    edge, shock and contact; a cell-centered point sample is never substituted.
    """
    if n <= 0 or quadrature_order < 2:
        raise ValueError("positive n and quadrature_order >= 2 required")
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    edges = np.linspace(-.5, .5, n + 1)
    breaks = [time * speed for speed in wave_locations(left, right, gas)] if time else [0.]
    result = np.empty((4, n), dtype=float)
    for i, (cell_lo, cell_hi) in enumerate(zip(edges[:-1], edges[1:])):
        cuts = [cell_lo] + [v for v in breaks if cell_lo < v < cell_hi] + [cell_hi]
        integral = np.zeros(4)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            half = (hi - lo) / 2
            center = (hi + lo) / 2
            for node, weight in zip(nodes, weights):
                integral += half * weight * conservative(
                    sample(center + half * node, time, left, right, gas), gas)
        result[:, i] = integral / (cell_hi - cell_lo)
    return result
