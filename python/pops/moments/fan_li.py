"""Full-temperature Fan--Li D=2, M=4 constitutive expressions.

Reference: Fan and Li, arXiv:1401.4639v1, equations (4.16), (5.9),
Theorem 4.13. These are flux *and nonconservative product* expressions;
using the flux alone does not implement the Fan--Li system. This module
does not implement LocalClosure or change HYQMOM15.

All fifteen values are unnormalized raw moments in the existing q-outer
ordering. Algebra accepts numbers or PoPS expressions. Runtime consumers
must enforce finite values, positive density and SPD covariance; no floor
or full-moment-realizability assertion is supplied by this algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial, sqrt
from typing import Any

from pops._ir.expr import Expr
from pops._ir.ops import sqrt as symbolic_sqrt

FAN_LI15_INDICES = tuple((p, q) for q in range(5) for p in range(5 - q))
FAN_LI15_REGULARIZED_COMPONENTS = (4, 8, 11, 13, 14)
_TOP = tuple(FAN_LI15_INDICES[k] for k in FAN_LI15_REGULARIZED_COMPONENTS)
_HERMITE_EDGE = sqrt(5 + sqrt(10))
_RAW_SECOND_BOUND = sqrt(6 + sqrt(10))


def _sqrt(value: Any) -> Any:
    return symbolic_sqrt(value) if isinstance(value, Expr) else sqrt(value)


def _pair(value: Any, name: str) -> tuple:
    result = tuple(value)
    if len(result) != 2:
        raise ValueError(f"{name} requires two Cartesian components")
    return result


def _raw(value: Any) -> tuple:
    result = tuple(value)
    if len(result) != 15:
        raise ValueError("FanLi15 requires the complete fifteen raw moments")
    return result


def _translate(central: dict, u: Any, v: Any, p: int, q: int) -> Any:
    return sum(
        comb(p, i) * comb(q, j) * u ** (p - i) * v ** (q - j) * central.get((i, j), 0)
        for i in range(p + 1)
        for j in range(q + 1)
    )


@dataclass(frozen=True)
class FanLi15Expressions:
    """Constitutive expressions; attach both transport parts to the balance.

    ``x``/``y`` are the conservative parts only. The directional product
    methods return B_g dU with the sign on the LEFT of U_t+div(F)+B grad(U)=S.
    Every direction argument is one common physical/mapped covector g.
    """

    raw: tuple
    rho: Any
    velocity: tuple
    temperature: tuple
    central: dict
    hermite: dict
    x: tuple
    y: tuple

    def directional_flux(self, direction: Any) -> tuple:
        gx, gy = _pair(direction, "direction")
        return tuple(gx * fx + gy * fy for fx, fy in zip(self.x, self.y, strict=True))

    def directional_nonconservative_action(self, direction: Any, increment: Any) -> tuple:
        """Exact B_g(U) dU, including raw factorials and the regularization minus sign."""
        gx, gy = _pair(direction, "direction")
        delta = _raw(increment)
        rho = self.rho
        u, v = self.velocity
        a, b, c = self.temperature
        dr = delta[0]
        du, dv = (delta[1] - u * dr) / rho, (delta[5] - v * dr) / rho
        da = (delta[2] - 2 * u * delta[1] + (u * u - a) * dr) / rho
        db = (delta[6] - v * delta[1] - u * delta[5] + (u * v - b) * dr) / rho
        dc = (delta[9] - 2 * v * delta[5] + (v * v - c) * dr) / rho
        h = self.hermite
        result = [0] * 15
        for slot, (p, q) in zip(FAN_LI15_REGULARIZED_COMPONENTS, _TOP, strict=True):
            rx = (p + 1) * (
                h.get((p, q), 0) * du
                + h.get((p + 1, q - 1), 0) * dv
                + h.get((p - 1, q), 0) * da / 2
                + h.get((p, q - 1), 0) * db
                + h.get((p + 1, q - 2), 0) * dc / 2
            )
            ry = (q + 1) * (
                h.get((p - 1, q + 1), 0) * du
                + h.get((p, q), 0) * dv
                + h.get((p - 2, q + 1), 0) * da / 2
                + h.get((p - 1, q), 0) * db
                + h.get((p, q - 1), 0) * dc / 2
            )
            result[slot] = -factorial(p) * factorial(q) * (gx * rx + gy * ry)
        return tuple(result)

    def directional_nonconservative_matrix(self, direction: Any) -> tuple:
        """The 15x15 coefficient matrix B_g, not the flux-only Jacobian DF_g.

        The complete characteristic matrix is DF_g+B_g. Only six kinematic
        columns can be nonzero; the ten conserved rows are literal zeros.
        """
        result = [[0] * 15 for _ in range(15)]
        for column in (0, 1, 2, 5, 6, 9):
            unit = [0] * 15
            unit[column] = 1
            action = self.directional_nonconservative_action(direction, unit)
            for row in FAN_LI15_REGULARIZED_COMPONENTS:
                result[row][column] = action[row]
        return tuple(tuple(row) for row in result)

    def directional_spectral_radius(self, direction: Any) -> Any:
        """Exact radius of the complete A_g on rho>0, Theta SPD."""
        gx, gy = _pair(direction, "direction")
        u, v = self.velocity
        a, b, c = self.temperature
        return abs(gx * u + gy * v) + _HERMITE_EDGE * _sqrt(
            gx * gx * a + 2 * gx * gy * b + gy * gy * c
        )

    def fixed_direction_path_speed_majorant(self, direction: Any) -> Any:
        """One endpoint majorant; max of both endpoints bounds the raw path.

        C*sqrt(E[(g.v)^2]), C=sqrt(6+sqrt(10)). The SAME g must be used at
        both endpoints. Separate trace auxiliary covectors do not satisfy
        this contract. This is not an invariant-domain or realizability proof.
        """
        gx, gy = _pair(direction, "direction")
        return _RAW_SECOND_BOUND * _sqrt(
            (gx * gx * self.raw[2] + 2 * gx * gy * self.raw[6] + gy * gy * self.raw[9]) / self.rho
        )


def fan_li15_expressions(variables: Any) -> FanLi15Expressions:
    """Build the exact generalized-Hermite flux and regularization algebra.

    This pure constitutive builder neither registers a fake conservative
    model nor evaluates a Riemann solver. Pass both its flux and B product
    to a qualified path-conservative balance implementation.
    """
    raw = _raw(variables)
    m = dict(zip(FAN_LI15_INDICES, raw, strict=True))
    rho = raw[0]
    u, v = raw[1] / rho, raw[5] / rho
    a, b, c = raw[2] / rho - u * u, raw[6] / rho - u * v, raw[9] / rho - v * v
    central = {(0, 0): rho, (1, 0): 0, (0, 1): 0, (2, 0): rho * a, (1, 1): rho * b, (0, 2): rho * c}
    for p, q in FAN_LI15_INDICES:
        if p + q >= 3:
            central[p, q] = sum(
                comb(p, i) * comb(q, j) * (-u) ** (p - i) * (-v) ** (q - j) * m[i, j]
                for i in range(p + 1)
                for j in range(q + 1)
            )
    hermite = {(0, 0): rho}
    for p, q in FAN_LI15_INDICES:
        if p + q in (1, 2):
            hermite[p, q] = 0
        elif p + q == 3:
            hermite[p, q] = central[p, q] / (factorial(p) * factorial(q))
    gaussian4 = {
        (4, 0): 3 * a * a,
        (3, 1): 3 * a * b,
        (2, 2): a * c + 2 * b * b,
        (1, 3): 3 * b * c,
        (0, 4): 3 * c * c,
    }
    for p, q in _TOP:
        hermite[p, q] = (central[p, q] - rho * gaussian4[p, q]) / (factorial(p) * factorial(q))
    c30, c21, c12, c03 = (central[pq] for pq in ((3, 0), (2, 1), (1, 2), (0, 3)))
    central.update(
        {
            (5, 0): 10 * a * c30,
            (4, 1): 6 * a * c21 + 4 * b * c30,
            (3, 2): 3 * a * c12 + 6 * b * c21 + c * c30,
            (2, 3): a * c03 + 6 * b * c12 + 3 * c * c21,
            (1, 4): 4 * b * c03 + 6 * c * c12,
            (0, 5): 10 * c * c03,
        }
    )
    closed = dict(m)
    for q in range(6):
        p = 5 - q
        closed[p, q] = _translate(central, u, v, p, q)
    return FanLi15Expressions(
        raw,
        rho,
        (u, v),
        (a, b, c),
        central,
        hermite,
        tuple(closed[p + 1, q] for p, q in FAN_LI15_INDICES),
        tuple(closed[p, q + 1] for p, q in FAN_LI15_INDICES),
    )


def fan_li15_from_hermite(rho: Any, velocity: Any, temperature: Any, coefficients: Any) -> tuple:
    """Inverse transform; temperature=(theta_xx,theta_xy,theta_yy), f3/f4 only.

    Coefficients are unnormalized generalized-Hermite coefficients, not
    central moments or standardized moments. Missing f3/f4 mean zero.
    """
    u, v = _pair(velocity, "velocity")
    theta = tuple(temperature)
    if len(theta) != 3:
        raise ValueError("temperature requires theta_xx, theta_xy, theta_yy")
    a, b, c = theta
    h = dict(coefficients)
    valid = {pq for pq in FAN_LI15_INDICES if sum(pq) in (3, 4)}
    if set(h) - valid:
        raise ValueError("only degree-three/four Hermite coefficients may be supplied")
    central = {(0, 0): rho, (1, 0): 0, (0, 1): 0, (2, 0): rho * a, (1, 1): rho * b, (0, 2): rho * c}
    gaussian4 = {
        (4, 0): 3 * a * a,
        (3, 1): 3 * a * b,
        (2, 2): a * c + 2 * b * b,
        (1, 3): 3 * b * c,
        (0, 4): 3 * c * c,
    }
    for p, q in valid:
        central[p, q] = factorial(p) * factorial(q) * h.get((p, q), 0)
        if p + q == 4:
            central[p, q] += rho * gaussian4[p, q]
    return tuple(_translate(central, u, v, p, q) for p, q in FAN_LI15_INDICES)


__all__ = [
    "FAN_LI15_INDICES",
    "FAN_LI15_REGULARIZED_COMPONENTS",
    "FanLi15Expressions",
    "fan_li15_expressions",
    "fan_li15_from_hermite",
]
