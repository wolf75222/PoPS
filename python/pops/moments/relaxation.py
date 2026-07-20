"""Native pointwise relaxation policies for moment hierarchies.

The formulas in this module are assembled in Python while the physical model is authored.  They
remain ordinary PoPS expressions and are emitted into the generated C++ Program kernel; no Python
callback participates in a time step.
"""
from __future__ import annotations

import math
import sys
from math import comb
from typing import Any

from pops._ir.expr import Expr
from pops._ir.ops import abs_ as _abs
from pops._ir.ops import eig_real_status as _eig_real_status
from pops._ir.ops import eig_lmin as _eig_lmin
from pops._ir.ops import maximum as _maximum
from pops._ir.ops import minimum as _minimum
from pops._ir.ops import sqrt as _sqrt
from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet


_ORDER = 4
_INDICES = tuple((p, q) for q in range(_ORDER + 1) for p in range(_ORDER + 1 - q))
_COMPONENTS = tuple("M%d%d" % index for index in _INDICES)


def _select(mask: Any, when_true: Any, when_false: Any) -> Any:
    """Arithmetic pointwise selection whose two arms are already eager-safe."""

    return when_false + mask * (when_true - when_false)


def _either(left: Any, right: Any) -> Any:
    return 1.0 - (1.0 - left) * (1.0 - right)


def _pow(value: Any, exponent: int) -> Any:
    if exponent == 0:
        return 1.0
    result = value
    for _ in range(exponent - 1):
        result = result * value
    return result


def _central_moments(moments: dict[tuple[int, int], Any]) -> tuple[Any, Any, Any, dict]:
    rho = moments[(0, 0)]
    velocity_x = moments[(1, 0)] / rho
    velocity_y = moments[(0, 1)] / rho
    normalized = {
        index: (1.0 if index == (0, 0) else value / rho)
        for index, value in moments.items()
    }
    central: dict[tuple[int, int], Any] = {
        (0, 0): 1.0,
        (1, 0): 0.0,
        (0, 1): 0.0,
    }
    for degree in range(2, _ORDER + 1):
        for q in range(degree + 1):
            p = degree - q
            value: Any = 0.0
            for i in range(p + 1):
                for j in range(q + 1):
                    coefficient = float(
                        comb(p, i) * comb(q, j) * (-1) ** (p - i + q - j)
                    )
                    value = value + (
                        coefficient
                        * _pow(velocity_x, p - i)
                        * _pow(velocity_y, q - j)
                        * normalized[(i, j)]
                    )
            central[(p, q)] = value
    return rho, velocity_x, velocity_y, central


def _standardized(central: dict[tuple[int, int], Any]) -> tuple[Any, Any, dict]:
    scale_x = _sqrt(central[(2, 0)])
    scale_y = _sqrt(central[(0, 2)])
    result: dict[tuple[int, int], Any] = {
        (2, 0): 1.0,
        (0, 2): 1.0,
    }
    for index, value in central.items():
        p, q = index
        if p + q >= 2 and index not in result:
            result[index] = value / (_pow(scale_x, p) * _pow(scale_y, q))
    return scale_x, scale_y, result


def _p2p2(values: dict[tuple[int, int], Any]) -> tuple[tuple[Any, ...], ...]:
    """Exact symbolic transcription of ``p2p2_2D.m`` with one eager-safe denominator."""

    a03 = values[(0, 3)]
    a04 = values[(0, 4)]
    a11 = values[(1, 1)]
    a12 = values[(1, 2)]
    a13 = values[(1, 3)]
    a21 = values[(2, 1)]
    a22 = values[(2, 2)]
    a30 = values[(3, 0)]
    a31 = values[(3, 1)]
    a40 = values[(4, 0)]

    t2 = a03 * a12
    t3 = a03 * a21
    t4 = a12 * a21
    t5 = a12 * a30
    t6 = a21 * a30
    t7 = a11 * a11
    t8 = t7 * a11
    t9 = a12 * a12
    t10 = a21 * a21
    t12 = a03 * a11 * a30
    t11 = a11 * t3
    t13 = a11 * t4
    t14 = a11 * t5
    t19 = a11 * t9
    t20 = a13 * t7
    t21 = a11 * t10
    t22 = a22 * t7
    t23 = a31 * t7
    denominator = t7 - 1.0
    inverse = 1.0 / denominator
    t33 = a22 + t12 + t13 - t3 - t5 + t7 - 1.0 - t22
    t34 = a11 + t2 + t4 - a13 + t20 - t8 - t11 - t19
    t35 = a11 + t4 + t6 - a31 + t23 - t8 - t14 - t21
    p11 = inverse * (
        -a40 + t10 - t7 - 2.0 * a11 * t6 + a40 * t7 + a30 * a30 + 1.0
    )
    p12 = inverse * t35
    p13 = -inverse * t33
    p22 = inverse * (-a22 + t7 + t9 + t10 - 2.0 * t13 - t7 * t7 + t22)
    p23 = inverse * t34
    p33 = inverse * (
        -a04 + t9 - t7 + a04 * t7 - 2.0 * a11 * t2 + a03 * a03 + 1.0
    )
    return ((p11, p12, p13), (p12, p22, p23), (p13, p23, p33))


def _transverse_jacobian(
    s03: Any,
    s04: Any,
    s11: Any,
    s12: Any,
    s13: Any,
    s21: Any,
    s22: Any,
) -> tuple[tuple[Any, ...], ...]:
    """Last 3x3 block of the HyQMOM15 flux Jacobian in standardized coordinates."""

    derivative_s23_s03 = -3.0 * s03 * s21 + 0.5 * (3.0 * s22 - 1.0)
    derivative_s14_s03 = (
        -0.25 * (8.0 * s04 - 27.0 * s03 * s03 - 4.0) * s11
        - 7.5 * s03 * s12
        + 2.0 * s13
    )
    derivative_s14_s04 = -2.0 * s03 * s11 + 2.5 * s12
    return (
        (0.0, 1.0, 0.0),
        (derivative_s23_s03, 0.0, s21),
        (derivative_s14_s03, 2.0 * s03, derivative_s14_s04),
    )


def _raw_moments(
    rho: Any,
    velocity_x: Any,
    velocity_y: Any,
    central: dict[tuple[int, int], Any],
) -> tuple[Any, ...]:
    raw = []
    for p, q in _INDICES:
        value: Any = 0.0
        for i in range(p + 1):
            for j in range(q + 1):
                centered = central.get((i, j), 0.0)
                if isinstance(centered, (int, float)) and centered == 0.0:
                    continue
                value = value + (
                    float(comb(p, i) * comb(q, j))
                    * _pow(velocity_x, p - i)
                    * _pow(velocity_y, q - j)
                    * centered
                )
        raw.append(rho * value)
    return tuple(raw)


class HyQMOM15Relaxation(Descriptor):
    """One explicit application of MATLAB ``relaxation15`` as a native local transform."""

    category = "moment_transform"

    def __init__(
        self,
        *,
        eigenvalue_cutoff: Any = 1.0e-12,
        mach: Any = 20.0,
        small: Any = 1.0e-6,
        spectral_tolerance: Any = sys.float_info.min,
    ) -> None:
        values = {
            "eigenvalue_cutoff": eigenvalue_cutoff,
            "mach": mach,
            "small": small,
            "spectral_tolerance": spectral_tolerance,
        }
        converted = {}
        for name, value in values.items():
            if isinstance(value, bool):
                raise TypeError("HyQMOM15Relaxation %s must be numeric" % name)
            converted[name] = float(value)
            if not math.isfinite(converted[name]):
                raise ValueError("HyQMOM15Relaxation %s must be finite" % name)
        if converted["eigenvalue_cutoff"] < 0.0:
            raise ValueError("HyQMOM15Relaxation eigenvalue_cutoff must be >= 0")
        if converted["mach"] < 0.0:
            raise ValueError("HyQMOM15Relaxation mach must be >= 0")
        if converted["small"] <= 0.0:
            raise ValueError("HyQMOM15Relaxation small must be > 0")
        if converted["spectral_tolerance"] <= 0.0:
            raise ValueError("HyQMOM15Relaxation spectral_tolerance must be > 0")
        self.eigenvalue_cutoff = converted["eigenvalue_cutoff"]
        self.mach = converted["mach"]
        self.small = converted["small"]
        self.spectral_tolerance = converted["spectral_tolerance"]

    def options(self) -> dict[str, Any]:
        return {
            "eigenvalue_cutoff": self.eigenvalue_cutoff,
            "mach": self.mach,
            "small": self.small,
            "spectral_tolerance": self.spectral_tolerance,
        }

    def capabilities(self) -> Any:
        return CapabilitySet({
            "execution": "native_pointwise",
            "state": "hyqmom15",
        })

    def expressions(self, variables: Any) -> tuple[Any, ...]:
        supplied = tuple(variables)
        if len(supplied) != len(_INDICES):
            raise ValueError(
                "HyQMOM15Relaxation requires 15 canonical moment variables; got %d"
                % len(supplied)
            )
        if any(not isinstance(value, Expr) for value in supplied):
            raise TypeError("HyQMOM15Relaxation variables must be symbolic PoPS expressions")

        moments = dict(zip(_INDICES, supplied, strict=True))
        rho, velocity_x, velocity_y, central = _central_moments(moments)
        scale_x, scale_y, s = _standardized(central)

        # Large third standardized moments retain their Hankel margin while being capped.
        third_limit = 4.0 + self.mach / 2.0
        cap_x = _abs(s[(3, 0)]) > third_limit
        hankel_x = s[(4, 0)] - s[(3, 0)] * s[(3, 0)] - 1.0
        limited_s30 = _select(
            s[(3, 0)] >= 0.0, third_limit, -third_limit,
        )
        s[(3, 0)] = _select(cap_x, limited_s30, s[(3, 0)])
        s[(4, 0)] = _select(
            cap_x, hankel_x + s[(3, 0)] * s[(3, 0)] + 1.0, s[(4, 0)],
        )
        cap_y = _abs(s[(0, 3)]) > third_limit
        hankel_y = s[(0, 4)] - s[(0, 3)] * s[(0, 3)] - 1.0
        limited_s03 = _select(
            s[(0, 3)] >= 0.0, third_limit, -third_limit,
        )
        s[(0, 3)] = _select(cap_y, limited_s03, s[(0, 3)])
        s[(0, 4)] = _select(
            cap_y, hankel_y + s[(0, 3)] * s[(0, 3)] + 1.0, s[(0, 4)],
        )

        # Boundary states receive the exact MATLAB minimum Hankel margin.
        boundary = _either(
            s[(4, 0)] - s[(3, 0)] * s[(3, 0)] - 1.0 < self.small,
            s[(0, 4)] - s[(0, 3)] * s[(0, 3)] - 1.0 < self.small,
        )
        s[(4, 0)] = _select(
            boundary, s[(3, 0)] * s[(3, 0)] + 1.0 + self.small, s[(4, 0)],
        )
        s[(0, 4)] = _select(
            boundary, s[(0, 3)] * s[(0, 3)] + 1.0 + self.small, s[(0, 4)],
        )
        s[(1, 1)] = _select(
            boundary,
            _select(s[(1, 1)] > 0.0, 1.0, _select(s[(1, 1)] < 0.0, -1.0, 0.0)),
            s[(1, 1)],
        )

        # Near-perfect correlations are mapped to the same crossing-jet fixed points as MATLAB.
        correlation_threshold = 1.0 - self.small
        upper = s[(1, 1)] >= correlation_threshold
        lower = (1.0 - upper) * (s[(1, 1)] <= -correlation_threshold)
        correlation_repair = _either(upper, lower)
        common_s3 = _sqrt(_abs(s[(0, 3)] * s[(3, 0)])) * _select(
            s[(3, 0)] > 0.0, 1.0, _select(s[(3, 0)] < 0.0, -1.0, 0.0),
        )
        common_s4 = _sqrt(_abs(s[(0, 4)] * s[(4, 0)]))
        common_s4 = _select(
            common_s4 - common_s3 * common_s3 - 1.0 <= 0.0,
            common_s3 * common_s3 + 1.0 + self.small,
            common_s4,
        )
        s[(1, 1)] = _select(
            upper, 1.0 - self.small,
            _select(lower, -1.0 + self.small, s[(1, 1)]),
        )
        s[(3, 0)] = _select(correlation_repair, common_s3, s[(3, 0)])
        s[(2, 1)] = _select(
            upper, common_s3, _select(lower, -common_s3, s[(2, 1)]),
        )
        s[(1, 2)] = _select(correlation_repair, common_s3, s[(1, 2)])
        s[(0, 3)] = _select(
            upper, common_s3, _select(lower, -common_s3, s[(0, 3)]),
        )
        s[(4, 0)] = _select(correlation_repair, common_s4, s[(4, 0)])
        s[(3, 1)] = _select(
            upper, common_s4, _select(lower, -common_s4, s[(3, 1)]),
        )
        s[(2, 2)] = _select(correlation_repair, common_s4, s[(2, 2)])
        s[(1, 3)] = _select(
            upper, common_s4, _select(lower, -common_s4, s[(1, 3)]),
        )
        s[(0, 4)] = _select(correlation_repair, common_s4, s[(0, 4)])

        real_x = _eig_real_status(
            _transverse_jacobian(
                s[(0, 3)], s[(0, 4)], s[(1, 1)], s[(1, 2)],
                s[(1, 3)], s[(2, 1)], s[(2, 2)],
            ),
            im_tol=self.spectral_tolerance,
        )
        real_y = _eig_real_status(
            _transverse_jacobian(
                s[(3, 0)], s[(4, 0)], s[(1, 1)], s[(2, 1)],
                s[(3, 1)], s[(1, 2)], s[(2, 2)],
            ),
            im_tol=self.spectral_tolerance,
        )
        complex_flux = 1.0 - real_x * real_y
        s[(2, 1)] = _select(complex_flux, 0.0, s[(2, 1)])
        s[(1, 2)] = _select(complex_flux, 0.0, s[(1, 2)])
        s[(2, 2)] = _select(complex_flux, _maximum(s[(2, 2)], 1.0 / 3.0), s[(2, 2)])

        preprocessed = dict(s)
        collision = self._collision(preprocessed)
        needs_collision = (
            (1.0 - correlation_repair)
            * (_eig_lmin(_p2p2(preprocessed)) <= self.eigenvalue_cutoff)
        )
        s = dict(preprocessed)
        s.update({
            index: _select(needs_collision, value, preprocessed[index])
            for index, value in collision.items()
        })

        repaired_central = dict(central)
        repaired_central[(1, 1)] = s[(1, 1)] * scale_x * scale_y
        for p, q in _INDICES:
            if p + q < 3:
                continue
            repaired_central[(p, q)] = s[(p, q)] * _pow(scale_x, p) * _pow(scale_y, q)
        return _raw_moments(rho, velocity_x, velocity_y, repaired_central)

    def _collision(self, source: dict[tuple[int, int], Any]) -> dict[tuple[int, int], Any]:
        """Eager-safe transcription of ``collision15_anisotropic.m``."""

        small = self.small
        s03, s04 = source[(0, 3)], source[(0, 4)]
        s11, s12, s13 = source[(1, 1)], source[(1, 2)], source[(1, 3)]
        s21, s22 = source[(2, 1)], source[(2, 2)]
        s30, s31, s40 = source[(3, 0)], source[(3, 1)], source[(4, 0)]

        delta1 = _maximum(1.0 - s11 * s11, sys.float_info.epsilon)
        crossing_s3 = _sqrt(_abs(s30 * s03)) * _select(
            s30 > 0.0, 1.0, _select(s30 < 0.0, -1.0, 0.0),
        )
        crossing_s4 = _sqrt(s40 * s04)
        crossing_s4 = _select(
            crossing_s4 - crossing_s3 * crossing_s3 - 1.0 < small,
            crossing_s3 * crossing_s3 + 1.0 + small,
            crossing_s4,
        )
        delta = _either(
            s40 - s30 * s30 - 1.0 < small,
            s04 - s03 * s03 - 1.0 < small,
        )
        delta_s11 = _select(s11 > 0.0, 1.0, _select(s11 < 0.0, -1.0, 0.0))
        delta_output = {
            (0, 3): crossing_s3 * delta_s11,
            (0, 4): crossing_s4,
            (1, 1): delta_s11,
            (1, 2): crossing_s3,
            (1, 3): crossing_s4 * delta_s11,
            (2, 1): crossing_s3 * delta_s11,
            (2, 2): crossing_s4,
            (3, 0): crossing_s3,
            (3, 1): crossing_s4 * delta_s11,
            (4, 0): crossing_s4,
        }

        target = {
            (1, 1): s11,
            (3, 0): 0.75 * s12 + 0.25 * s30,
            (2, 1): 0.25 * s03 + 0.75 * s21,
            (3, 1): 0.5 * (s13 + s31),
        }
        target[(1, 2)] = target[(3, 0)]
        target[(0, 3)] = target[(2, 1)]
        target[(4, 0)] = (
            0.125 * s04 + 0.75 * s22 + 0.125 * s40 + 1.5 * (1.0 - s11 * s11)
        )
        target[(2, 2)] = (
            0.125 * s04 + 0.75 * s22 + 0.125 * s40 - 0.5 * (1.0 - s11 * s11)
        )
        target[(1, 3)] = target[(3, 1)]
        target[(0, 4)] = target[(4, 0)]

        first_third_bound = _sqrt(
            delta1 * (target[(4, 0)] - target[(3, 0)] * target[(3, 0)] - 1.0)
        )
        repair_s21 = _abs(target[(2, 1)] - target[(1, 1)] * target[(3, 0)]) >= first_third_bound
        target[(2, 1)] = _select(
            repair_s21, target[(1, 1)] * target[(3, 0)], target[(2, 1)],
        )
        second_third_bound = _sqrt(
            delta1 * (target[(0, 4)] - target[(0, 3)] * target[(0, 3)] - 1.0)
        )
        repair_s12 = (1.0 - repair_s21) * (
            _abs(target[(1, 2)] - target[(1, 1)] * target[(0, 3)]) >= second_third_bound
        )
        target[(1, 2)] = _select(
            repair_s12, target[(1, 1)] * target[(0, 3)], target[(1, 2)],
        )

        target_matrix = _p2p2(target)
        target_invalid = _either(
            target_matrix[0][0] * target_matrix[1][1]
            - target_matrix[0][1] * target_matrix[1][0] < 0.0,
            target_matrix[0][0] < 0.0,
        )
        crossing_repair_s11 = _select(
            target[(1, 1)] > 0.0, 1.0,
            _select(target[(1, 1)] < 0.0, -1.0, 0.0),
        )
        crossing_repair_s3 = _sqrt(_abs(s03 * s30)) * _select(
            s30 > 0.0, 1.0, _select(s30 < 0.0, -1.0, 0.0),
        )
        crossing_repair_s4 = _sqrt(s40 * s04)
        crossing_repair_s4 = _select(
            crossing_repair_s4 - crossing_repair_s3 * crossing_repair_s3 - 1.0 < small,
            crossing_repair_s3 * crossing_repair_s3 + 1.0 + small,
            crossing_repair_s4,
        )
        crossing_repair = {
            (0, 3): crossing_repair_s3 * crossing_repair_s11,
            (0, 4): crossing_repair_s4,
            (1, 1): crossing_repair_s11 * (1.0 - small),
            (1, 2): crossing_repair_s3,
            (1, 3): crossing_repair_s4 * crossing_repair_s11,
            (2, 1): crossing_repair_s3 * crossing_repair_s11,
            (2, 2): crossing_repair_s4,
            (3, 0): crossing_repair_s3,
            (3, 1): crossing_repair_s4 * crossing_repair_s11,
            (4, 0): crossing_repair_s4,
        }
        target = {
            index: _select(target_invalid, crossing_repair[index], target[index])
            for index in target
        }

        denominator_plus = 3.0 * target[(1, 1)] / 16.0 + 3.0 / 16.0
        denominator_minus = 3.0 * target[(1, 1)] / 16.0 - 3.0 / 16.0
        a03, a04 = target[(0, 3)], target[(0, 4)]
        a11, a12, a13 = target[(1, 1)], target[(1, 2)], target[(1, 3)]
        a21, a22 = target[(2, 1)], target[(2, 2)]
        a30, a31, a40 = target[(3, 0)], target[(3, 1)], target[(4, 0)]
        s22_1 = (
            3.0 * a11 / 8.0 - a04 / 32.0 - a13 / 8.0 - a31 / 8.0 - a40 / 32.0
            + 3.0 * a03 * a12 / 32.0 - a04 * a11 / 32.0
            + 3.0 * a03 * a21 / 32.0 - a11 * a13 / 8.0
            + a03 * a30 / 32.0 + 9.0 * a12 * a21 / 32.0
            - a11 * a31 / 8.0 + 3.0 * a12 * a30 / 32.0
            - a11 * a40 / 32.0 + 3.0 * a21 * a30 / 32.0
            + a03 * a03 / 64.0 + 3.0 * a11 * a11 / 8.0 + a11 * a11 * a11 / 8.0
            + 9.0 * a12 * a12 / 64.0 + 9.0 * a21 * a21 / 64.0
            + a30 * a30 / 64.0 + 1.0 / 8.0
        ) / denominator_plus
        s22_2 = -(
            a13 / 8.0 - 3.0 * a11 / 8.0 - a04 / 32.0 + a31 / 8.0 - a40 / 32.0
            - 3.0 * a03 * a12 / 32.0 + a04 * a11 / 32.0
            + 3.0 * a03 * a21 / 32.0 - a11 * a13 / 8.0
            - a03 * a30 / 32.0 - 9.0 * a12 * a21 / 32.0
            - a11 * a31 / 8.0 + 3.0 * a12 * a30 / 32.0
            + a11 * a40 / 32.0 - 3.0 * a21 * a30 / 32.0
            + a03 * a03 / 64.0 + 3.0 * a11 * a11 / 8.0 - a11 * a11 * a11 / 8.0
            + 9.0 * a12 * a12 / 64.0 + 9.0 * a21 * a21 / 64.0
            + a30 * a30 / 64.0 + 1.0 / 8.0
        ) / denominator_minus
        target[(2, 2)] = _maximum(_maximum(s22_1, s22_2), target[(2, 2)]) * 1.001
        target[(2, 2)] = _minimum(target[(2, 2)], target[(4, 0)])
        target[(2, 2)] = _maximum(target[(2, 2)], 1.0 / 3.0)

        return {
            index: _select(delta, delta_output[index], target[index])
            for index in target
        }

    def declare(self, model: Any, state: Any, *, name: str = "relaxation15") -> Any:
        """Declare the map and return the typed handle used by ``Program.transform``."""

        components = tuple(getattr(state, "components", ()))
        if components != _COMPONENTS:
            raise ValueError(
                "HyQMOM15Relaxation requires canonical components %r; got %r"
                % (_COMPONENTS, components)
            )
        variables = tuple(state)
        moments = dict(zip(_INDICES, variables, strict=True))
        rho, _, _, central = _central_moments(moments)
        valid_if = (rho > 0.0) * (central[(2, 0)] > 0.0) * (central[(0, 2)] > 0.0)
        return model.local_transform(
            name, self.expressions(variables), on=state, valid_if=valid_if)

    def __repr__(self) -> str:
        return (
            "HyQMOM15Relaxation(eigenvalue_cutoff=%g, mach=%g, small=%g, "
            "spectral_tolerance=%g)"
            % (
                self.eigenvalue_cutoff,
                self.mach,
                self.small,
                self.spectral_tolerance,
            )
        )


__all__ = ["HyQMOM15Relaxation"]
