"""Exact generic countermodels; no PoPS import for the mathematical witnesses.

The last three cases receive the current public refusal, not a native solver.
No synthetic state is classified as a saved M22/Marshak observation.
"""

from fractions import Fraction as Q
from functools import partial
from unittest.mock import patch

import pytest


def principal(values, coefficients, spacing):
    """Periodic arithmetic-face -div(D grad(q)), with independent exact arithmetic."""
    n = len(values)
    face = tuple(
        (coefficients[i] + coefficients[(i + 1) % n])
        / 2
        * (values[(i + 1) % n] - values[i])
        / spacing
        for i in range(n)
    )
    return tuple(-(face[i] - face[i - 1]) / spacing for i in range(n))


def original(values, spacing):
    diffusion = principal(values, tuple(1 + value**2 for value in values), spacing)
    return tuple(value + value**3 + image for value, image in zip(values, diffusion, strict=True))


def frozen(values, seed, spacing):
    diffusion = principal(values, tuple(1 + value**2 for value in seed), spacing)
    return tuple(value + value**3 + image for value, image in zip(values, diffusion, strict=True))


def subtract(left, right):
    return tuple(a - b for a, b in zip(left, right, strict=True))


def witness(n):
    q = tuple(Q(2 + (3 * i) % 7, 5) for i in range(n))
    direction = tuple(Q((-1) ** i * (i + 1), 7) for i in range(n))
    seed = tuple(Q(1 + (2 * i) % 5, 3) for i in range(n))
    return q, direction, seed, Q(11, 5 * n)


@pytest.mark.parametrize("n", [5, 7, 11])
def test_seed_frozen_zero_is_not_an_original_equation_zero(n):
    q, _, seed, dx = witness(n)
    # f is manufactured from the substituted equation, not from the original.
    load = frozen(q, seed, dx)
    assert subtract(frozen(q, seed, dx), load) == (Q(0),) * n
    defect = subtract(original(q, dx), load)
    assert any(value != 0 for value in defect)
    assert sum(defect) == 0  # A mass/conservation test alone cannot detect this.
    assert frozen(q, seed, dx) != frozen(q, tuple(reversed(seed)), dx)


@pytest.mark.parametrize("n", [5, 7, 11])
def test_full_original_jvp_has_the_omitted_candidate_coefficient_term(n):
    q, direction, _, dx = witness(n)
    epsilon = Q(1, 1009)
    plus = tuple(a + epsilon * b for a, b in zip(q, direction, strict=True))
    minus = tuple(a - epsilon * b for a, b in zip(q, direction, strict=True))
    full_fd = tuple(
        value / (2 * epsilon) for value in subtract(original(plus, dx), original(minus, dx))
    )
    fixed_image = principal(direction, tuple(1 + value**2 for value in q), dx)
    omitted = principal(q, tuple(2 * a * b for a, b in zip(q, direction, strict=True)), dx)
    cubic_image = principal(direction, tuple(value**2 for value in direction), dx)
    fixed_jvp = tuple(
        (1 + 3 * a**2) * b + image for a, b, image in zip(q, direction, fixed_image, strict=True)
    )
    full_jvp = tuple(a + b for a, b in zip(fixed_jvp, omitted, strict=True))
    # This polynomial witness has an exact central-FD remainder, no tolerance fit.
    remainder = tuple(epsilon**2 * (a**3 + b) for a, b in zip(direction, cubic_image, strict=True))
    assert full_fd == tuple(a + b for a, b in zip(full_jvp, remainder, strict=True))
    assert any(value != 0 for value in omitted)
    frozen_fd = tuple(
        value / (2 * epsilon) for value in subtract(frozen(plus, q, dx), frozen(minus, q, dx))
    )
    assert frozen_fd == tuple(
        a + epsilon**2 * b**3 for a, b in zip(fixed_jvp, direction, strict=True)
    )
    assert frozen_fd != full_fd


@pytest.mark.parametrize("n", [5, 7, 11])
def test_generic_countermodel_is_conservative_and_rotation_covariant(n):
    q, _, _, dx = witness(n)
    image = principal(q, tuple(1 + value**2 for value in q), dx)
    assert sum(image) == 0
    for offset in (1, n // 2):

        def rotate(values, offset=offset):
            return values[offset:] + values[:offset]

        assert original(rotate(q), dx) == rotate(original(q, dx))


def test_h05_exchange_does_not_establish_spatial_or_temperature_equations():
    e0, reservoir0, dt, rate = Q(2), Q(1, 2), Q(2, 5), Q(4, 5)
    total = e0 + reservoir0
    difference = (e0 - reservoir0) / (1 + 2 * dt * rate)
    e1, reservoir1 = (total + difference) / 2, (total - difference) / 2
    assert (e1, reservoir1) == (Q(70, 41), Q(65, 82))
    assert e1 - e0 + dt * rate * (e1 - reservoir1) == 0
    assert reservoir1 - reservoir0 - dt * rate * (e1 - reservoir1) == 0
    assert e1 + reservoir1 == total
    # Equality of conserved totals does not equate linear exchange with T^4.
    assert e1 - reservoir1 != e1 - reservoir1**4


def test_scalar_m1_closure_can_change_while_radiation_energy_is_constant():
    # Algebraic closure identity only: f=8/13 gives sqrt(4-3 f^2)=22/13.
    f, root = Q(8, 13), Q(22, 13)
    assert root**2 == 4 - 3 * f**2
    chi = (3 + 4 * f**2) / (5 + 2 * root)
    assert chi == Q(7, 13)
    assert chi != Q(1, 3)
    energy = Q(13)
    assert energy * chi == 7
    assert energy * Q(1, 3) != 7
    # This does not prescribe the missing physical tensor/flux convention.


@pytest.mark.parametrize(
    "width,order,uniform", [(1, (0,), True), (2, (1, 0), False), (4, (2, 0, 3, 1), True)]
)
def test_public_original_field_refuses_candidate_diffusion_before_solve(width, order, uniform):
    import test_sol61_amr_public_original as physical
    from pops._ir.elliptic import DivCoeffGrad
    from pops.fields import CellCenteredNonlinearCoupled, FieldProblemError
    from pops.math import ValueExpr

    def candidate_diffusion(field, coefficient):
        return DivCoeffGrad(field, coefficient * (1 + ValueExpr(field) ** 2))

    with (
        patch.object(physical, "DivCoeffGrad", candidate_diffusion),
        patch.object(
            physical,
            "CellCenteredNonlinearCoupled",
            partial(CellCenteredNonlinearCoupled, face_policy="Arithmetic@1"),
        ),
    ):
        with pytest.raises(
            FieldProblemError,
            match="unknown-dependent D has no full spatial nonlinear JVP realization",
        ):
            physical.authored(width, order, seed=True, uniform=uniform)
