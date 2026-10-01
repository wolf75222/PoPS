"""Autonomous preparation oracles, not a reception of future production code.

Fraction arrays and the isolated AMR interface are mathematical witnesses only.
"""

from fractions import Fraction as Q
from itertools import product

import pytest


def base_matrix(width):
    rows = [tuple(Q((-1) ** (i + j) * (i + 2 * j + 1)) for j in range(width)) for i in range(width)]
    if width > 1:
        rows[-1] = tuple(
            2 * value for value in rows[0]
        )  # Exact singularity; generally nonsymmetric.
    else:
        rows[0] = (Q(-3),)
    return tuple(rows)


def coefficient(value, capture, base, weights):
    factor = capture + sum(
        weight * component**2 for weight, component in zip(weights, value, strict=True)
    )
    return tuple(tuple(factor * element for element in row) for row in base)


def derivative_coefficient(value, direction, base, weights):
    factor = sum(2 * weight * a * b for weight, a, b in zip(weights, value, direction, strict=True))
    return tuple(tuple(factor * element for element in row) for row in base)


def principal(values, matrices, dx):
    n, width = len(values), len(values[0])
    faces = tuple(
        tuple(
            sum(
                (matrices[i][row][column] + matrices[(i + 1) % n][row][column])
                / 2
                * (values[(i + 1) % n][column] - values[i][column])
                / dx
                for column in range(width)
            )
            for row in range(width)
        )
        for i in range(n)
    )
    return tuple(
        tuple(-(faces[i][row] - faces[i - 1][row]) / dx for row in range(width)) for i in range(n)
    )


def residual(values, captures, base, weights, dx, *, coefficient_seed=None):
    source = values if coefficient_seed is None else coefficient_seed
    matrices = tuple(
        coefficient(value, capture, base, weights)
        for value, capture in zip(source, captures, strict=True)
    )
    image = principal(values, matrices, dx)
    return tuple(
        tuple(value + value**3 + image[i][row] for row, value in enumerate(cell))
        for i, cell in enumerate(values)
    )


def shift(q, direction, step):
    return tuple(
        tuple(a + step * b for a, b in zip(cell, vector, strict=True))
        for cell, vector in zip(q, direction, strict=True)
    )


def difference(left, right, scale=Q(1)):
    return tuple(
        tuple((a - b) * scale for a, b in zip(cell, other, strict=True))
        for cell, other in zip(left, right, strict=True)
    )


def matrix_witness(n, width):
    q = tuple(
        tuple(Q(2 + (3 * cell + 2 * component) % 9, 7) for component in range(width))
        for cell in range(n)
    )
    direction = tuple(
        tuple(
            Q((-1) ** (cell + component) * (component + cell + 1), 11) for component in range(width)
        )
        for cell in range(n)
    )
    captures = tuple(Q((-1) ** cell * (cell + 1), 13) for cell in range(n))
    weights = tuple(Q(component + 1, 5) for component in range(width))
    return q, direction, captures, base_matrix(width), weights, Q(17, 3 * n)


@pytest.mark.parametrize("n,width", [(4, 1), (7, 3), (9, 5)])
def test_signed_singular_matrix_full_jvp_includes_all_candidate_dependencies(n, width):
    q, direction, captures, base, weights, dx = matrix_witness(n, width)
    epsilon = Q(1, 1013)
    d = tuple(
        coefficient(cell, capture, base, weights) for cell, capture in zip(q, captures, strict=True)
    )
    delta_d = tuple(
        derivative_coefficient(cell, vector, base, weights)
        for cell, vector in zip(q, direction, strict=True)
    )
    d2 = tuple(coefficient(vector, Q(0), base, weights) for vector in direction)
    first, omitted, cubic = (
        principal(direction, d, dx),
        principal(q, delta_d, dx),
        principal(direction, d2, dx),
    )
    derivative = tuple(
        tuple(
            (1 + 3 * value**2) * vector[row] + first[i][row] + omitted[i][row]
            for row, value in enumerate(cell)
        )
        for i, (cell, vector) in enumerate(zip(q, direction, strict=True))
    )
    fd = difference(
        residual(shift(q, direction, epsilon), captures, base, weights, dx),
        residual(shift(q, direction, -epsilon), captures, base, weights, dx),
        Q(1) / (2 * epsilon),
    )
    expected = tuple(
        tuple(
            derivative[i][row] + epsilon**2 * (direction[i][row] ** 3 + cubic[i][row])
            for row in range(width)
        )
        for i in range(n)
    )
    assert fd == expected
    assert any(value for cell in omitted for value in cell)
    frozen_fd = difference(
        residual(shift(q, direction, epsilon), captures, base, weights, dx, coefficient_seed=q),
        residual(shift(q, direction, -epsilon), captures, base, weights, dx, coefficient_seed=q),
        Q(1) / (2 * epsilon),
    )
    assert frozen_fd != fd
    for row in range(width):
        assert sum(cell[row] for cell in principal(q, d, dx)) == 0
    if width > 1:
        assert all(matrix[-1] == tuple(2 * value for value in matrix[0]) for matrix in d)
        assert any(matrix[0][1] != matrix[1][0] for matrix in d)


@pytest.mark.parametrize("n,width", [(4, 1), (7, 3), (9, 5)])
def test_component_permutation_and_owned_capture_snapshot_are_separate(n, width):
    q, direction, captures, base, weights, dx = matrix_witness(n, width)
    order = tuple(reversed(range(width)))

    def permute(cells):
        return tuple(tuple(cell[index] for index in order) for cell in cells)

    matrix = tuple(tuple(base[i][j] for j in order) for i in order)
    assert residual(permute(q), captures, matrix, tuple(weights[j] for j in order), dx) == permute(
        residual(q, captures, base, weights, dx)
    )
    caller_capture = list(captures)
    owned_capture = tuple(caller_capture)
    before = residual(q, owned_capture, base, weights, dx)
    caller_capture[1] += Q(23, 19)
    assert residual(q, owned_capture, base, weights, dx) == before
    assert residual(q, tuple(caller_capture), base, weights, dx) != before
    seed = shift(q, direction, Q(1, 5))
    false_load = residual(q, owned_capture, base, weights, dx, coefficient_seed=seed)
    assert residual(q, owned_capture, base, weights, dx) != false_load


def average(values):
    return tuple(sum(cell[i] for cell in values) / len(values) for i in range(len(values[0])))


def average_matrices(matrices):
    width = len(matrices[0])
    return tuple(
        tuple(sum(matrix[i][j] for matrix in matrices) / len(matrices) for j in range(width))
        for i in range(width)
    )


def interface_flux(
    left, children, capture_left, capture_children, base, weights, *, wrong_order=False
):
    # Restrict candidate q; assemble D on valid children; restrict D; use one shared face.
    right = average(children)
    dl = coefficient(left, capture_left, base, weights)
    dr = (
        coefficient(right, sum(capture_children) / len(capture_children), base, weights)
        if wrong_order
        else average_matrices(
            tuple(
                coefficient(cell, cap, base, weights)
                for cell, cap in zip(children, capture_children, strict=True)
            )
        )
    )
    return tuple(
        sum((dl[i][j] + dr[i][j]) / 2 * (right[j] - left[j]) for j in range(len(left)))
        for i in range(len(left))
    )


@pytest.mark.parametrize(
    "ratio,origin,width", [((2,), (-3,), 1), ((2, 3), (-2, 5), 3), ((3, 2), (7, -4), 5)]
)
def test_amr_restriction_assembly_restriction_order_and_full_directional_derivative(
    ratio, origin, width
):
    # Explicit child coordinates avoid conflating a spatial embedding with tuple order.
    coordinates = tuple(
        product(*(range(start, start + count) for start, count in zip(origin, ratio, strict=True)))
    )
    n = len(coordinates)
    q, h, captures, base, weights, _ = matrix_witness(n + 1, width)
    left, children, hl, hc = q[0], q[1:], h[0], h[1:]
    cl, cc = captures[0], captures[1:]
    correct = interface_flux(left, children, cl, cc, base, weights)
    assert correct != interface_flux(left, children, cl, cc, base, weights, wrong_order=True)
    # Different child/patch iteration order preserves this exact linear restriction.
    assert correct == interface_flux(
        left, tuple(reversed(children)), cl, tuple(reversed(cc)), base, weights
    )
    restricted_h, restricted_q = average(hc), average(children)
    dl, dr = (
        coefficient(left, cl, base, weights),
        average_matrices(
            tuple(
                coefficient(cell, cap, base, weights)
                for cell, cap in zip(children, cc, strict=True)
            )
        ),
    )
    ddl = derivative_coefficient(left, hl, base, weights)
    ddr = average_matrices(
        tuple(
            derivative_coefficient(cell, vector, base, weights)
            for cell, vector in zip(children, hc, strict=True)
        )
    )
    full = tuple(
        sum(
            (dl[i][j] + dr[i][j]) / 2 * (restricted_h[j] - hl[j])
            + (ddl[i][j] + ddr[i][j]) / 2 * (restricted_q[j] - left[j])
            for j in range(width)
        )
        for i in range(width)
    )
    incomplete = tuple(
        sum((dl[i][j] + dr[i][j]) / 2 * (restricted_h[j] - hl[j]) for j in range(width))
        for i in range(width)
    )
    assert full != incomplete
    epsilon = Q(1, 1019)
    plus = interface_flux(
        shift((left,), (hl,), epsilon)[0], shift(children, hc, epsilon), cl, cc, base, weights
    )
    minus = interface_flux(
        shift((left,), (hl,), -epsilon)[0], shift(children, hc, -epsilon), cl, cc, base, weights
    )
    d2l = coefficient(hl, Q(0), base, weights)
    d2r = average_matrices(tuple(coefficient(vector, Q(0), base, weights) for vector in hc))
    remainder = tuple(
        epsilon**2
        * sum((d2l[i][j] + d2r[i][j]) / 2 * (restricted_h[j] - hl[j]) for j in range(width))
        for i in range(width)
    )
    assert tuple((a - b) / (2 * epsilon) for a, b in zip(plus, minus, strict=True)) == tuple(
        a + b for a, b in zip(full, remainder, strict=True)
    )
