"""Exact coefficient contracts for the final ready-made time-scheme library."""
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

import pops.lib.time as libtime
from pops import time as adctime
from pops.physics._facade import Model
from pops.time._methods.tableau import AdditiveRungeKuttaTableau, RungeKuttaTableau
from typed_program_support import state_refs


def _authoring(name):
    model = Model(name + "_model")
    model.conservative_vars("u", "v")
    rate = model.rate("R", flux=False, sources=())
    linear = model.local_linear_map("L", [[-1, 0], [0, -1]])
    block, state = state_refs(adctime.Program("refs"), "block", model=model)
    return block[state], rate, linear


def _node(program, op):
    return next(node for node in program._serialize()["nodes"] if node["op"] == op)


def _one_stage_tableau(theta):
    explicit = RungeKuttaTableau(A=[[]], b=[1], c=[0], name="exact-explicit")
    return AdditiveRungeKuttaTableau(
        explicit,
        implicit_A=[[theta]],
        implicit_b=[1],
        name="exact-imex",
    )


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (Fraction(1, 3), {"kind": "rational", "numerator": "1", "denominator": "3"}),
        (Decimal("0.125"), {"kind": "decimal", "value": "0.125"}),
        (0.25, {"kind": "binary64", "value": 0.25.hex()}),
    ],
)
def test_imex_preserves_the_authored_diagonal_coefficient_domain(theta, expected):
    state, rate, linear = _authoring("imex_exact")
    program = libtime.IMEX(
        state,
        explicit_operator=rate,
        implicit_operator=linear,
        tableau=_one_stage_tableau(theta),
    )
    assert program.validate() is True
    assert _node(program, "solve_local_linear")["attrs"]["a_coeff"] == [[1, expected]]


def test_exact_coefficients_reject_bool_nan_and_numeric_domain_mixing():
    from pops.time.values import _Coeff

    with pytest.raises(TypeError, match="bool is not a real scalar literal"):
        _Coeff({0: True})
    with pytest.raises(ValueError, match="Decimal scalar literal must be finite"):
        _Coeff({0: Decimal("NaN")})
    with pytest.raises(TypeError, match="cannot mix Decimal and Fraction"):
        _ = _Coeff({0: Decimal("0.5")}) * Fraction(3, 5)


def test_decimal_affine_products_ignore_the_ambient_context():
    from pops.time.values import _Coeff

    left = Decimal("0.123456789012345678901234567890123456789")
    right = Decimal("0.200000000000000000000000000000000000001")
    with localcontext() as context:
        context.prec = 5
        affine_sum = (_Coeff({0: left}) + right).as_dict()[0]
        affine_product = (_Coeff({0: left}) * right).as_dict()[0]

    assert affine_sum == Decimal("0.323456789012345678901234567890123456790")
    assert affine_product == Decimal(
        "0.024691357802469135780246913578024691357923456789012345678901234567890123456789")


def test_repeating_decimal_division_is_never_silently_rounded():
    from pops.time.values import _Coeff

    with localcontext() as context:
        context.prec = 3
        with pytest.raises(TypeError, match="must terminate"):
            _ = _Coeff({0: Decimal(1)}) / Decimal(3)


def test_multistep_factories_serialize_integer_and_rational_weights():
    state, rate, _ = _authoring("ab3_exact")
    ab3 = libtime.AdamsBashforth(state, rate=rate, order=3)
    coeffs = _node(ab3, "linear_combine")["attrs"]["coeffs"]
    assert coeffs == [
        [[0, {"kind": "integer", "value": "1"}]],
        [[1, {"kind": "rational", "numerator": "23", "denominator": "12"}]],
        [[1, {"kind": "rational", "numerator": "-4", "denominator": "3"}]],
        [[1, {"kind": "rational", "numerator": "5", "denominator": "12"}]],
    ]

    state, _, linear = _authoring("bdf2_exact")
    bdf2 = libtime.BDF(state, implicit=linear, order=2)
    assert _node(bdf2, "solve_local_linear")["attrs"]["a_coeff"] == [[
        1, {"kind": "rational", "numerator": "2", "denominator": "3"}
    ]]


def test_strang_and_lie_pass_exact_builtin_step_fractions():
    seen = []

    def first(program, state, fraction, *, at):
        seen.append(("first", fraction))
        return program.value("first-flow", 1 * state, at=at)

    def second(program, state, fraction, *, at):
        seen.append(("second", fraction))
        return program.value("second-flow", 1 * state, at=at)

    state, _, _ = _authoring("split")
    assert libtime.Strang(state, first=first, second=second).validate() is True
    assert seen == [
        ("first", Fraction(1, 2)),
        ("second", 1),
        ("first", Fraction(1, 2)),
    ]
    assert type(seen[1][1]) is int

    seen.clear()
    assert libtime.Lie(state, first=first, second=second).validate() is True
    assert seen == [("first", 1), ("second", 1)]
    assert all(type(fraction) is int for _, fraction in seen)
