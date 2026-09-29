"""Independent Fan--Li15 oracle for the M17 reception, never used by PoPS.

Adapted from tests/python/support/fan_li15_oracle.py (2026-09-29).
That oracle derives Hermite coefficients from a bivariate generating function
and the regularization from Fan--Li equation (4.16) by multi-index sums.
Gaussian fifth moments use integration by parts, not the production closure
recurrence. The full periodic semidiscrete problem is integrated by SciPy
DOP853 with a 24-point Gauss path; 48 points independently check the path.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial, sqrt

import numpy as np


INDICES = tuple((p, q) for q in range(5) for p in range(5-q))
TOP = tuple(k for k, alpha in enumerate(INDICES) if sum(alpha) == 4)
CONSERVATIVE = tuple(k for k in range(15) if k not in TOP)


def gaussian_raw(velocity, temperature, degree=5):
    u, v = velocity
    a, b, c = temperature

    @lru_cache(None)
    def moment(p, q):
        if p < 0 or q < 0:
            return 0.
        if p == q == 0:
            return 1.
        if p:
            return u*moment(p-1, q) + (p-1)*a*moment(p-2, q) + q*b*moment(p-1, q-1)
        return v*moment(0, q-1) + (q-1)*c*moment(0, q-2)

    return {(p, q): moment(p, q) for q in range(degree+1)
            for p in range(degree+1-q)}


def gaussian_mixture(components):
    values = [0.] * 15
    for weight, velocity, temperature in components:
        gaussian = gaussian_raw(velocity, temperature, 4)
        values = [old + weight*gaussian[alpha]
                  for old, alpha in zip(values, INDICES, strict=True)]
    return tuple(values)


def _product(a, b):
    result = {}
    for (p, q), x in a.items():
        for (r, s), y in b.items():
            alpha = p+r, q+s
            if sum(alpha) <= 4:
                result[alpha] = result.get(alpha, 0.) + x*y
    return result


def generating_hermite(raw):
    rho = raw[0]
    u, v = raw[1]/rho, raw[5]/rho
    a, b, c = raw[2]/rho-u*u, raw[6]/rho-u*v, raw[9]/rho-v*v
    exponent = {(1, 0): -u, (0, 1): -v, (2, 0): -a/2,
                (1, 1): -b, (0, 2): -c/2}
    power, exponential = {(0, 0): 1.}, {(0, 0): 1.}
    for degree in range(1, 5):
        power = _product(power, exponent)
        for alpha, value in power.items():
            exponential[alpha] = exponential.get(alpha, 0.) + value/factorial(degree)
    moments = {alpha: value/(factorial(alpha[0])*factorial(alpha[1]))
               for alpha, value in zip(INDICES, raw, strict=True)}
    return _product(exponential, moments), (u, v), (a, b, c)


def flux(raw, direction):
    """Gaussian integration-by-parts of the truncated Hermite distribution."""
    h, velocity, temperature = generating_hermite(raw)
    gaussian = gaussian_raw(velocity, temperature)

    def moment(p, q):
        return sum(value*(factorial(p)//factorial(p-i))
                   *(factorial(q)//factorial(q-j))*gaussian[p-i, q-j]
                   for (i, j), value in h.items() if i <= p and j <= q)

    gx, gy = direction
    return np.asarray([gx*moment(p+1, q)+gy*moment(p, q+1)
                       for p, q in INDICES], dtype=float)


def primary_product(raw, increment, direction):
    """Fan--Li multi-index B_g(U)dU, independent of the production five-row code."""
    h, velocity, temperature = generating_hermite(raw)
    rho = raw[0]
    u, v = velocity
    a, b, c = temperature
    dr = increment[0]
    du = ((increment[1]-u*dr)/rho, (increment[5]-v*dr)/rho)
    da = (increment[2]-2*u*increment[1]+(u*u-a)*dr)/rho
    db = (increment[6]-v*increment[1]-u*increment[5]+(u*v-b)*dr)/rho
    dc = (increment[9]-2*v*increment[5]+(v*v-c)*dr)/rho
    dtheta = ((da, db), (db, dc))
    result = np.zeros(15)
    for slot in TOP:
        alpha = INDICES[slot]
        for axis in range(2):
            for i in range(2):
                beta = tuple(alpha[k]+(k == axis)-(k == i) for k in range(2))
                value = h.get(beta, 0.)*du[i]
                for j in range(2):
                    gamma = tuple(beta[k]-(k == j) for k in range(2))
                    value += h.get(gamma, 0.)*dtheta[i][j]/2
                result[slot] -= (factorial(alpha[0])*factorial(alpha[1])
                                 *direction[axis]*(alpha[axis]+1)*value)
    return result


@lru_cache(None)
def _gauss_rule(points):
    nodes, weights = np.polynomial.legendre.leggauss(points)
    return tuple(zip((nodes+1)/2, weights/2, strict=True))


def independent_path(left, right, direction, *, points=24):
    """Independent Gauss integral of B(U(s))*(R-L) on a raw straight path."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.shape[0] != 15:
        raise ValueError("path states must have shape (15, interfaces)")
    jump = right-left
    integral = np.zeros_like(left)
    for k in range(left.shape[1]):
        for node, weight in _gauss_rule(points):
            integral[:, k] += weight*primary_product(
                left[:, k]+node*jump[:, k], jump[:, k], direction)
    return integral


def _speed(raw):
    # Complete DF+B fixed-direction raw-second-moment majorant.
    return sqrt(6+sqrt(10))*np.sqrt(raw[2]/raw[0])


def semidiscrete_rhs(_time, flattened):
    u = np.asarray(flattened).reshape(15, -1)
    n = u.shape[1]
    right = np.roll(u, -1, axis=1)
    flux_left = np.stack([flux(u[:, k], (1., 0.)) for k in range(n)], axis=1)
    flux_right = np.roll(flux_left, -1, axis=1)
    speed = np.maximum(_speed(u), _speed(right))
    face = .5*(flux_left+flux_right-speed[None, :]*(right-u))
    integral = independent_path(u, right, (1., 0.), points=24)
    rhs = -n*(face-np.roll(face, 1, axis=1))
    rhs -= .5*n*(integral+np.roll(integral, 1, axis=1))
    return rhs.ravel()


def solve_reference(initial, t_end):
    from scipy.integrate import solve_ivp
    initial = np.asarray(initial, dtype=float)
    if initial.ndim != 2 or initial.shape[0] != 15:
        raise ValueError("the full oracle requires shape (15, N)")
    result = solve_ivp(semidiscrete_rhs, (0., float(t_end)), initial.ravel(),
                       method="DOP853", rtol=2.e-12, atol=2.e-14)
    if not result.success or not np.all(np.isfinite(result.y[:, -1])):
        raise RuntimeError("independent Fan--Li DOP853 reference failed")
    return result.y[:, -1].reshape(initial.shape)
