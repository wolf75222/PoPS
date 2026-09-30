"""Frozen-source/math counterprobe only; no AMR/native solver execution."""
from fractions import Fraction
import math
from pathlib import Path
import re
import subprocess

FROZEN = "9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6"
HEADER = "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp"
ROOT = Path(__file__).resolve().parents[2]


def dot(left, right):
    # An explicit binary64 accumulation; Python 3.12 sum(float) compensates.
    value = 0.
    for a, b in zip(left, right, strict=True):
        value += a * b
    return value


def norm(value):
    return math.sqrt(dot(value, value))


def two_columns(matrix, rhs):
    """Independent 2D Arnoldi/MGS + Givens algebra, not a field/runtime prototype."""
    beta = norm(rhs)
    basis = [[value / beta for value in rhs]]
    h = [[0., 0.], [0., 0.], [0., 0.]]
    cosine, sine, rotated = [0., 0.], [0., 0.], [beta, 0., 0.]
    for column in range(2):
        work = [dot(row, basis[column]) for row in matrix]
        for row in range(column + 1):
            h[row][column] = dot(work, basis[row])
            work = [value - h[row][column] * direction
                    for value, direction in zip(work, basis[row], strict=True)]
        h[column + 1][column] = norm(work)
        basis.append([value / h[column + 1][column] for value in work]
                     if h[column + 1][column] > 0 else [0., 0.])
        for rotation in range(column):
            upper, lower = h[rotation][column], h[rotation + 1][column]
            h[rotation][column] = cosine[rotation] * upper + sine[rotation] * lower
            h[rotation + 1][column] = -sine[rotation] * upper + cosine[rotation] * lower
        magnitude = math.hypot(h[column][column], h[column + 1][column])
        cosine[column], sine[column] = h[column][column] / magnitude, h[column + 1][column] / magnitude
        h[column][column], h[column + 1][column] = magnitude, 0.
        rotated[column + 1] = -sine[column] * rotated[column]
        rotated[column] *= cosine[column]
    coefficients = [0., rotated[1] / h[1][1]]
    coefficients[0] = (rotated[0] - h[0][1] * coefficients[1]) / h[0][0]
    solution = [coefficients[0] * a + coefficients[1] * b
                for a, b in zip(basis[0], basis[1], strict=True)]
    return solution, abs(rotated[2]), 1e-5 * beta


def test_frozen_amr_accepts_projected_convergence_before_actual_linear_recheck():
    source = subprocess.check_output(["git", "show", FROZEN + ":" + HEADER], cwd=ROOT, text=True)
    assert re.search(r"if \(cycle_converged\) \{\s*result.converged = true;\s*return result;\s*\}", source)
    recheck = source.index("apply_jvp(iterate, correction_, image_, nonlinear_iteration)")
    assert recheck > source.index("if (cycle_converged)")
    assert "options_.linear_tolerance * report.residual_norm" in source


def test_finite_spd_countermodel_needs_true_residual_guard_without_relaxing_stop():
    matrix = [[64000000000000.36, -47999999999999.52],
              [-47999999999999.52, 36000000000000.64]]
    rhs = [.3, .7]
    exact = [[Fraction(value) for value in row] for row in matrix]
    assert matrix[0][1] == matrix[1][0] and exact[0][0] > 0
    assert exact[0][0] * exact[1][1] - exact[0][1] ** 2 > 0  # exact binary64 SPD
    solution, projected, stop = two_columns(matrix, rhs)
    residual = [Fraction(value) - sum(coefficient * Fraction(x)
                for coefficient, x in zip(row, solution, strict=True))
                for value, row in zip(rhs, exact, strict=True)]
    true_norm = math.sqrt(float(sum(value * value for value in residual)))
    assert all(math.isfinite(value) for value in solution)
    assert projected <= stop
    assert true_norm > 20 * stop
