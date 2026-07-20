"""NumPy reference for one HyQMOM15 relaxation application.

This is a host-side oracle for tests and authoring inspection.  Production time stepping uses the
symbolic transform emitted into C++/Kokkos by :mod:`pops.moments.relaxation`.
"""
from __future__ import annotations

from math import comb
from typing import Any

import numpy as np


_ORDER = 4
_INDICES = tuple((p, q) for q in range(_ORDER + 1) for p in range(_ORDER + 1 - q))


def _central(raw: dict[tuple[int, int], float]) -> tuple[float, float, float, dict]:
    rho = raw[(0, 0)]
    ux = raw[(1, 0)] / rho
    uy = raw[(0, 1)] / rho
    normalized = {index: value / rho for index, value in raw.items()}
    central = {(0, 0): 1.0, (1, 0): 0.0, (0, 1): 0.0}
    for degree in range(2, _ORDER + 1):
        for q in range(degree + 1):
            p = degree - q
            central[(p, q)] = sum(
                comb(p, i) * comb(q, j) * (-1) ** (p - i + q - j)
                * ux ** (p - i) * uy ** (q - j) * normalized[(i, j)]
                for i in range(p + 1) for j in range(q + 1)
            )
    return rho, ux, uy, central


def _standardized(central: dict[tuple[int, int], float]) -> tuple[float, float, dict]:
    sx = np.sqrt(central[(2, 0)])
    sy = np.sqrt(central[(0, 2)])
    result = {(2, 0): 1.0, (0, 2): 1.0}
    for (p, q), value in central.items():
        if p + q >= 2 and (p, q) not in result:
            result[(p, q)] = value / (sx**p * sy**q)
    return float(sx), float(sy), result


def _p2p2(s: dict[tuple[int, int], float]) -> np.ndarray:
    a03, a04 = s[(0, 3)], s[(0, 4)]
    a11, a12, a13 = s[(1, 1)], s[(1, 2)], s[(1, 3)]
    a21, a22 = s[(2, 1)], s[(2, 2)]
    a30, a31, a40 = s[(3, 0)], s[(3, 1)], s[(4, 0)]
    t2, t3, t4 = a03 * a12, a03 * a21, a12 * a21
    t5, t6 = a12 * a30, a21 * a30
    t7, t8 = a11**2, a11**3
    t9, t10 = a12**2, a21**2
    t12 = a03 * a11 * a30
    t11, t13, t14 = a11 * t3, a11 * t4, a11 * t5
    t19, t20 = a11 * t9, a13 * t7
    t21, t22, t23 = a11 * t10, a22 * t7, a31 * t7
    inverse = 1.0 / (t7 - 1.0)
    t33 = a22 + t12 + t13 - t3 - t5 + t7 - 1.0 - t22
    t34 = a11 + t2 + t4 - a13 + t20 - t8 - t11 - t19
    t35 = a11 + t4 + t6 - a31 + t23 - t8 - t14 - t21
    p11 = inverse * (-a40 + t10 - t7 - 2 * a11 * t6 + a40 * t7 + a30**2 + 1)
    p12 = inverse * t35
    p13 = -inverse * t33
    p22 = inverse * (-a22 + t7 + t9 + t10 - 2 * t13 - t7**2 + t22)
    p23 = inverse * t34
    p33 = inverse * (-a04 + t9 - t7 + a04 * t7 - 2 * a11 * t2 + a03**2 + 1)
    return np.array([[p11, p12, p13], [p12, p22, p23], [p13, p23, p33]])


def _transverse(s03: float, s04: float, s11: float, s12: float, s13: float,
                s21: float, s22: float) -> np.ndarray:
    return np.array([
        [0.0, 1.0, 0.0],
        [-3.0 * s03 * s21 + 0.5 * (3.0 * s22 - 1.0), 0.0, s21],
        [
            -0.25 * (8.0 * s04 - 27.0 * s03**2 - 4.0) * s11
            - 7.5 * s03 * s12 + 2.0 * s13,
            2.0 * s03,
            -2.0 * s03 * s11 + 2.5 * s12,
        ],
    ])


def _collision(source: dict[tuple[int, int], float], small: float) -> dict:
    s03, s04 = source[(0, 3)], source[(0, 4)]
    s11, s12, s13 = source[(1, 1)], source[(1, 2)], source[(1, 3)]
    s21, s22 = source[(2, 1)], source[(2, 2)]
    s30, s31, s40 = source[(3, 0)], source[(3, 1)], source[(4, 0)]
    delta1 = max(1.0 - s11**2, np.finfo(float).eps)
    crossing_s3 = np.sign(s30) * np.sqrt(abs(s30 * s03))
    crossing_s4 = np.sqrt(s40 * s04)
    if crossing_s4 - crossing_s3**2 - 1.0 < small:
        crossing_s4 = crossing_s3**2 + 1.0 + small
    if s40 - s30**2 - 1.0 < small or s04 - s03**2 - 1.0 < small:
        crossing_s11 = np.sign(s11)
        return {
            (0, 3): crossing_s3 * crossing_s11, (0, 4): crossing_s4,
            (1, 1): crossing_s11, (1, 2): crossing_s3,
            (1, 3): crossing_s4 * crossing_s11,
            (2, 1): crossing_s3 * crossing_s11, (2, 2): crossing_s4,
            (3, 0): crossing_s3, (3, 1): crossing_s4 * crossing_s11,
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
    target[(4, 0)] = 0.125 * s04 + 0.75 * s22 + 0.125 * s40 + 1.5 * (1 - s11**2)
    target[(2, 2)] = 0.125 * s04 + 0.75 * s22 + 0.125 * s40 - 0.5 * (1 - s11**2)
    target[(1, 3)] = target[(3, 1)]
    target[(0, 4)] = target[(4, 0)]
    if abs(target[(2, 1)] - s11 * target[(3, 0)]) >= np.sqrt(
        delta1 * (target[(4, 0)] - target[(3, 0)]**2 - 1)
    ):
        target[(2, 1)] = s11 * target[(3, 0)]
    elif abs(target[(1, 2)] - s11 * target[(0, 3)]) >= np.sqrt(
        delta1 * (target[(0, 4)] - target[(0, 3)]**2 - 1)
    ):
        target[(1, 2)] = s11 * target[(0, 3)]

    matrix = _p2p2(target)
    if np.linalg.det(matrix[:2, :2]) < 0.0 or matrix[0, 0] < 0.0:
        crossing_repair_s11 = np.sign(target[(1, 1)])
        crossing_repair_s3 = np.sign(s30) * np.sqrt(abs(s03 * s30))
        crossing_repair_s4 = np.sqrt(s40 * s04)
        if crossing_repair_s4 - crossing_repair_s3**2 - 1.0 < small:
            crossing_repair_s4 = crossing_repair_s3**2 + 1.0 + small
        target = {
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

    a03, a04 = target[(0, 3)], target[(0, 4)]
    a11, a12, a13 = target[(1, 1)], target[(1, 2)], target[(1, 3)]
    a21, a22 = target[(2, 1)], target[(2, 2)]
    a30, a31, a40 = target[(3, 0)], target[(3, 1)], target[(4, 0)]
    numerator1 = (
        3*a11/8-a04/32-a13/8-a31/8-a40/32+3*a03*a12/32-a04*a11/32
        +3*a03*a21/32-a11*a13/8+a03*a30/32+9*a12*a21/32-a11*a31/8
        +3*a12*a30/32-a11*a40/32+3*a21*a30/32+a03**2/64+3*a11**2/8
        +a11**3/8+9*a12**2/64+9*a21**2/64+a30**2/64+1/8
    )
    numerator2 = (
        a13/8-3*a11/8-a04/32+a31/8-a40/32-3*a03*a12/32+a04*a11/32
        +3*a03*a21/32-a11*a13/8-a03*a30/32-9*a12*a21/32-a11*a31/8
        +3*a12*a30/32+a11*a40/32-3*a21*a30/32+a03**2/64+3*a11**2/8
        -a11**3/8+9*a12**2/64+9*a21**2/64+a30**2/64+1/8
    )
    lower_s22 = max(
        numerator1 / (3*a11/16 + 3/16),
        -numerator2 / (3*a11/16 - 3/16),
    )
    target[(2, 2)] = max(lower_s22, a22) * 1.001
    target[(2, 2)] = min(target[(2, 2)], a40)
    target[(2, 2)] = max(target[(2, 2)], 1.0 / 3.0)
    return target


def _raw(rho: float, ux: float, uy: float, central: dict) -> np.ndarray:
    result = []
    for p, q in _INDICES:
        result.append(rho * sum(
            comb(p, i) * comb(q, j) * ux ** (p-i) * uy ** (q-j)
            * central.get((i, j), 0.0)
            for i in range(p+1) for j in range(q+1)
        ))
    return np.asarray(result)


def _transform_cell(values: np.ndarray, cutoff: float, mach: float, small: float,
                    spectral_tolerance: float) -> np.ndarray:
    raw = dict(zip(_INDICES, values, strict=True))
    rho, ux, uy, central = _central(raw)
    sx, sy, s = _standardized(central)
    limit = 4.0 + mach / 2.0
    if abs(s[(3, 0)]) > limit:
        margin = s[(4, 0)] - s[(3, 0)]**2 - 1.0
        s[(3, 0)] = np.sign(s[(3, 0)]) * limit
        s[(4, 0)] = margin + s[(3, 0)]**2 + 1.0
    if abs(s[(0, 3)]) > limit:
        margin = s[(0, 4)] - s[(0, 3)]**2 - 1.0
        s[(0, 3)] = np.sign(s[(0, 3)]) * limit
        s[(0, 4)] = margin + s[(0, 3)]**2 + 1.0
    if s[(4, 0)] - s[(3, 0)]**2 - 1.0 < small \
            or s[(0, 4)] - s[(0, 3)]**2 - 1.0 < small:
        s[(4, 0)] = s[(3, 0)]**2 + 1.0 + small
        s[(0, 4)] = s[(0, 3)]**2 + 1.0 + small
        s[(1, 1)] = np.sign(s[(1, 1)])
    correlation_repair = False
    correlation_threshold = 1.0 - small
    if s[(1, 1)] >= correlation_threshold:
        correlation_repair = True
        sign = 1.0
    elif s[(1, 1)] <= -correlation_threshold:
        correlation_repair = True
        sign = -1.0
    if correlation_repair:
        common_s3 = np.sign(s[(3, 0)]) * np.sqrt(abs(s[(0, 3)] * s[(3, 0)]))
        common_s4 = np.sqrt(abs(s[(0, 4)] * s[(4, 0)]))
        if common_s4 - common_s3**2 - 1.0 <= 0.0:
            common_s4 = common_s3**2 + 1.0 + small
        s[(1, 1)] = sign * (1.0 - small)
        s[(3, 0)], s[(1, 2)] = common_s3, common_s3
        s[(2, 1)], s[(0, 3)] = sign * common_s3, sign * common_s3
        s[(4, 0)], s[(2, 2)], s[(0, 4)] = common_s4, common_s4, common_s4
        s[(3, 1)], s[(1, 3)] = sign * common_s4, sign * common_s4
    blocks = (
        _transverse(s[(0, 3)], s[(0, 4)], s[(1, 1)], s[(1, 2)], s[(1, 3)],
                    s[(2, 1)], s[(2, 2)]),
        _transverse(s[(3, 0)], s[(4, 0)], s[(1, 1)], s[(2, 1)], s[(3, 1)],
                    s[(1, 2)], s[(2, 2)]),
    )
    for block in blocks:
        eigenvalues = np.linalg.eigvals(block)
        scale = max(float(np.max(np.abs(np.real(eigenvalues)))), 1.0)
        if float(np.max(np.abs(np.imag(eigenvalues)))) > spectral_tolerance * scale:
            s[(2, 1)] = s[(1, 2)] = 0.0
            s[(2, 2)] = max(s[(2, 2)], 1.0 / 3.0)
    minimum_eigenvalue = float(np.min(np.real(np.linalg.eigvals(_p2p2(s)))))
    if not correlation_repair and minimum_eigenvalue <= cutoff:
        s.update(_collision(s, small))
    repaired = dict(central)
    repaired[(1, 1)] = s[(1, 1)] * sx * sy
    for p, q in _INDICES:
        if p + q >= 3:
            repaired[(p, q)] = s[(p, q)] * sx**p * sy**q
    return _raw(rho, ux, uy, repaired)


def _apply_hyqmom15_relaxation_array(values: Any, *, cutoff: float, mach: float,
                                     small: float, spectral_tolerance: float) -> np.ndarray:
    state = np.asarray(values, dtype=np.float64)
    if state.ndim < 1 or state.shape[0] != len(_INDICES):
        raise ValueError("HyQMOM15 state must have 15 components on axis zero")
    if not np.isfinite(state).all():
        raise FloatingPointError("HyQMOM15 relaxation received a non-finite state")
    flat = state.reshape((len(_INDICES), -1))
    result = np.empty_like(flat)
    cell_count = flat.size // len(_INDICES)
    for cell in range(cell_count):
        _, _, _, central = _central(dict(zip(_INDICES, flat[:, cell], strict=True)))
        if flat[0, cell] <= 0.0 or central[(2, 0)] <= 0.0 or central[(0, 2)] <= 0.0:
            raise ValueError("HyQMOM15 relaxation input has non-positive density or variance")
        result[:, cell] = _transform_cell(
            flat[:, cell], cutoff, mach, small, spectral_tolerance,
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("HyQMOM15 relaxation produced a non-finite state")
    return result.reshape(state.shape)
