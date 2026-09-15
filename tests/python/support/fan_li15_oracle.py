"""Independent Fan--Li formula oracle, never imported by production code.

Hermite coefficients come from a bivariate formal generating function, flux
moments from Gaussian integration by parts, and the path from numerical
quadrature. No production constitutive builder or density-integral recurrence
is used here. Exact rational inputs remain rational outside quadrature.
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial


INDICES = tuple((p, q) for q in range(5) for p in range(5 - q))
TOP = tuple(k for k, alpha in enumerate(INDICES) if sum(alpha) == 4)


def gaussian_raw(velocity, temperature, degree=5):
    """Stein recursion for normalized noncentral Gaussian moments."""
    u, v = velocity
    a, b, c = temperature

    @lru_cache(None)
    def moment(p, q):
        if p < 0 or q < 0:
            return 0
        if p == q == 0:
            return 1
        if p:
            return (
                u * moment(p - 1, q) + (p - 1) * a * moment(p - 2, q) + q * b * moment(p - 1, q - 1)
            )
        return v * moment(0, q - 1) + (q - 1) * c * moment(0, q - 2)

    return {(p, q): moment(p, q) for q in range(degree + 1) for p in range(degree + 1 - q)}


def gaussian_mixture(components):
    """Each component is (unnormalized weight, velocity, full covariance)."""
    values = [0] * 15
    for weight, velocity, temperature in components:
        gaussian = gaussian_raw(velocity, temperature, 4)
        values = [
            old + weight * gaussian[alpha] for old, alpha in zip(values, INDICES, strict=True)
        ]
    return tuple(values)


def _product(a, b):
    result = {}
    for (p, q), x in a.items():
        for (r, s), y in b.items():
            alpha = p + r, q + s
            if sum(alpha) <= 4:
                result[alpha] = result.get(alpha, 0) + x * y
    return result


def generating_hermite(raw):
    """Coefficients of exp(-u.z-z.Theta.z/2) sum M_alpha z^alpha/alpha!."""
    rho = raw[0]
    u, v = raw[1] / rho, raw[5] / rho
    a, b, c = raw[2] / rho - u * u, raw[6] / rho - u * v, raw[9] / rho - v * v
    exponent = {(1, 0): -u, (0, 1): -v, (2, 0): -a / 2, (1, 1): -b, (0, 2): -c / 2}
    power, exponential = {(0, 0): 1}, {(0, 0): 1}
    for degree in range(1, 5):
        power = _product(power, exponent)
        for alpha, value in power.items():
            exponential[alpha] = exponential.get(alpha, 0) + value / factorial(degree)
    moments = {
        alpha: value / (factorial(alpha[0]) * factorial(alpha[1]))
        for alpha, value in zip(INDICES, raw, strict=True)
    }
    return _product(exponential, moments), (u, v), (a, b, c)


def derivative_gaussian_flux(raw, direction):
    """Integrate the Hermite distribution using Gaussian integration by parts."""
    h, velocity, temperature = generating_hermite(raw)
    gaussian = gaussian_raw(velocity, temperature)

    def moment(p, q):
        return sum(
            value
            * (factorial(p) // factorial(p - i))
            * (factorial(q) // factorial(q - j))
            * gaussian[p - i, q - j]
            for (i, j), value in h.items()
            if i <= p and j <= q
        )

    gx, gy = direction
    return tuple(gx * moment(p + 1, q) + gy * moment(p, q + 1) for p, q in INDICES)


def primary_product(raw, increment, direction):
    """Primary (4.16) multi-index double sum, with both mixed tensor entries."""
    h, velocity, temperature = generating_hermite(raw)
    rho = raw[0]
    u, v = velocity
    a, b, c = temperature
    dr = increment[0]
    du = ((increment[1] - u * dr) / rho, (increment[5] - v * dr) / rho)
    da = (increment[2] - 2 * u * increment[1] + (u * u - a) * dr) / rho
    db = (increment[6] - v * increment[1] - u * increment[5] + (u * v - b) * dr) / rho
    dc = (increment[9] - 2 * v * increment[5] + (v * v - c) * dr) / rho
    dtheta = ((da, db), (db, dc))
    result = [0] * 15
    for slot in TOP:
        alpha = INDICES[slot]
        for d in range(2):
            for i in range(2):
                beta = tuple(alpha[k] + (k == d) - (k == i) for k in range(2))
                value = h.get(beta, 0) * du[i]
                for j in range(2):
                    gamma = tuple(beta[k] - (k == j) for k in range(2))
                    value += h.get(gamma, 0) * dtheta[i][j] / 2
                result[slot] -= (
                    factorial(alpha[0])
                    * factorial(alpha[1])
                    * direction[d]
                    * (alpha[d] + 1)
                    * value
                )
    return tuple(result)


def quadrature_path(raw_left, raw_right, direction, dps=65):
    """mpmath quadrature of the primary one-form along the exact raw path.

    A logarithmic density parameter resolves large density ratios without
    subtracting nearly equal raw density derivatives. It is a reparameterization
    of the same raw segment, not a different path or an analytic I_k recurrence.
    """
    import mpmath as mp

    with mp.workdps(dps):
        left, right = tuple(map(mp.mpf, raw_left)), tuple(map(mp.mpf, raw_right))
        normal_left = tuple(x / left[0] for x in left)
        normal_right = tuple(x / right[0] for x in right)
        delta = tuple(b - a for a, b in zip(normal_left, normal_right, strict=True))
        g = tuple(map(mp.mpf, direction))
        logarithm = mp.log(left[0] / right[0])
        factor = left[0] if logarithm == 0 else left[0] * logarithm / mp.expm1(logarithm)
        samples = {}

        def integrand(t):
            if t not in samples:
                w = t if logarithm == 0 else mp.expm1(logarithm * t) / mp.expm1(logarithm)
                normal = tuple(a + w * d for a, d in zip(normal_left, delta, strict=True))
                samples[t] = primary_product(normal, delta, g)
            return samples[t]

        # The steepest exponential varies by at most exp(8) on each interval.
        panels = max(1, int(mp.ceil(abs(logarithm) / 8)))
        knots = [mp.mpf(i) / panels for i in range(panels + 1)]
        result = [mp.mpf(0)] * 15
        for slot in TOP:
            result[slot] = factor * mp.quad(lambda t, slot=slot: integrand(t)[slot], knots)
        return tuple(result)
