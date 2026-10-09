"""Analytic entropy solution of u_t + (u^2/2)_x = 0.

Independent of PoPS and its numerical face body. Integrals are closed form;
neither quadrature nor cell-center sampling replaces conservative averages.
"""
from __future__ import annotations

import numpy as np


def entropy_cell_averages(n: int, time: float, left: float, right: float,
                          *, lower: float = -1., upper: float = 1.,
                          diaphragm: float = 0.) -> np.ndarray:
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if not all(np.isfinite(v) for v in (time, left, right, lower, upper, diaphragm)):
        raise ValueError("finite Riemann data, time and domain required")
    if time < 0 or upper <= lower:
        raise ValueError("nonnegative time and increasing domain required")
    edges = np.linspace(lower, upper, n + 1) - diaphragm
    lo, hi = edges[:-1], edges[1:]
    width = hi - lo
    if time == 0 or left >= right:
        position = .5 * (left + right) * time
        # These min/max operations measure interval intersections; they do not
        # clip a numerical state or repair a failed admissibility condition.
        left_length = np.maximum(0., np.minimum(hi, position) - lo)
        left_length = np.minimum(left_length, width)
        return (left * left_length + right * (width - left_length)) / width
    fan_left, fan_right = time * left, time * right
    left_length = np.maximum(0., np.minimum(hi, fan_left) - lo)
    right_length = np.maximum(0., hi - np.maximum(lo, fan_right))
    fan_lo = np.maximum(lo, fan_left)
    fan_hi = np.minimum(hi, fan_right)
    fan_length = np.maximum(0., fan_hi - fan_lo)
    # Integral of x/time over the fan; the factored difference of squares
    # avoids subtracting two large nearly equal antiderivative values.
    fan_integral = .5 * fan_length * (fan_lo + fan_hi) / time
    return (left * left_length + fan_integral + right * right_length) / width


def godunov_flux(left, right):
    """Convex-flux extremum characterization, independent of authored branches.

The entropy flux is min f over [left,right] for a rarefaction, and max f over
[right,left] for a shock. NumPy scalars and arrays are accepted.
"""
    left, right = np.asarray(left), np.asarray(right)
    fl, fr = .5 * left**2, .5 * right**2
    interval_contains_zero = (left <= 0) & (right >= 0)
    minimum = np.where(interval_contains_zero, 0., np.minimum(fl, fr))
    return np.where(left <= right, minimum, np.maximum(fl, fr))


def weak_shock_speed(left: float, right: float) -> float:
    if not (np.isfinite(left) and np.isfinite(right) and left > right):
        raise ValueError("a nontrivial entropy shock requires finite left > right")
    return .5 * (left + right)
