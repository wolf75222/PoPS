"""Exact coefficient contracts for the ready-made time-scheme library."""
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

import pops.lib.time as libtime
from pops import time as adctime
from pops.physics.facade import Model
from typed_program_support import state_refs


def _bound_program(name):
    model = Model(name + "_model")
    u, v = model.conservative_vars("u", "v")
    model.flux(x=[u, v], y=[u, v])
    linear = model.local_linear_map("L", [[-1, 0], [0, -1]])
    return adctime.Program(name).bind_operators(model), linear


def _node(program, op):
    return next(node for node in program._serialize()["nodes"] if node["op"] == op)


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (
            Fraction(1, 3),
            {"kind": "rational", "numerator": "1", "denominator": "3"},
        ),
        (Decimal("0.125"), {"kind": "decimal", "value": "0.125"}),
        (0.25, {"kind": "binary64", "value": 0.25.hex()}),
    ],
)
def test_imex_preserves_the_authored_theta_domain(theta, expected):
    program, linear = _bound_program("imex_exact")
    libtime.imex_local(
        program,
        *state_refs(program, "block"),
        linear_source=linear,
        sources=(),
        flux=False,
        theta=theta,
    )
    assert program.validate() is True
    solve = _node(program, "solve_local_linear")
    assert solve["attrs"]["a_coeff"] == [[1, expected]]


def test_imex_rejects_values_that_cannot_be_exact_coefficients():
    bad_bool, bool_linear = _bound_program("bad_bool")
    with pytest.raises(TypeError, match="finite real coefficient"):
        libtime.imex_local(
            bad_bool,
            *state_refs(bad_bool, "block"),
            linear_source=bool_linear,
            sources=(),
            flux=False,
            theta=True,
        )
    bad_nan, nan_linear = _bound_program("bad_nan")
    with pytest.raises(ValueError, match="finite real coefficient"):
        libtime.imex_local(
            bad_nan,
            *state_refs(bad_nan, "block"),
            linear_source=nan_linear,
            sources=(),
            flux=False,
            theta=Decimal("NaN"),
        )


def test_condensed_schur_composes_rational_coefficients_exactly():
    program, linear = _bound_program("schur_exact")
    libtime.CondensedSchur(
        program,
        *state_refs(program, "block"),
        theta=Fraction(1, 2),
        alpha=Fraction(3, 5),
        linear_operator=linear,
    )
    assert program.validate() is True
    coeffs = _node(program, "condensed_coeffs")["attrs"]
    rhs = _node(program, "condensed_rhs")["attrs"]
    assert coeffs["c"] == [[
        2, {"kind": "rational", "numerator": "3", "denominator": "20"}
    ]]
    assert coeffs["th_dt"] == [[
        1, {"kind": "rational", "numerator": "1", "denominator": "2"}
    ]]
    assert rhs["g"] == [[
        1, {"kind": "rational", "numerator": "3", "denominator": "10"}
    ]]
    extrap = next(
        node for node in program._serialize()["nodes"]
        if node["op"] == "linear_combine" and node["name"] == "block.schur_extrap"
    )
    assert extrap["attrs"]["coeffs"] == [
        [[0, {"kind": "rational", "numerator": "-1", "denominator": "1"}]],
        [[0, {"kind": "rational", "numerator": "2", "denominator": "1"}]],
    ]


def test_condensed_schur_composes_decimal_coefficients_without_binary64():
    program, linear = _bound_program("schur_decimal")
    libtime.CondensedSchur(
        program,
        *state_refs(program, "block"),
        theta=Decimal("0.5"),
        alpha=Decimal("0.6"),
        linear_operator=linear,
    )
    coeffs = _node(program, "condensed_coeffs")["attrs"]
    rhs = _node(program, "condensed_rhs")["attrs"]
    assert coeffs["c"] == [[2, {"kind": "decimal", "value": "0.150"}]]
    assert coeffs["th_dt"] == [[1, {"kind": "decimal", "value": "0.5"}]]
    assert rhs["g"] == [[1, {"kind": "decimal", "value": "0.30"}]]


def test_decimal_affine_and_lib_time_products_ignore_the_ambient_context():
    from pops.time.values import _Coeff

    left = Decimal("0.123456789012345678901234567890123456789")
    right = Decimal("0.200000000000000000000000000000000000001")
    with localcontext() as context:
        context.prec = 5
        affine_sum = (_Coeff({0: left}) + right).as_dict()[0]
        affine_product = (_Coeff({0: left}) * right).as_dict()[0]
        program, linear = _bound_program("schur_decimal_context")
        libtime.CondensedSchur(
            program, *state_refs(program, "block"),
            theta=Decimal("0.1250000000000000000000000000000000000000"),
            alpha=right, linear_operator=linear)

    assert affine_sum == Decimal("0.323456789012345678901234567890123456790")
    exact_product = Decimal(
        "0.024691357802469135780246913578024691357923456789012345678901234567890123456789")
    assert affine_product == exact_product
    rhs = _node(program, "condensed_rhs")["attrs"]
    assert rhs["g"] == [[1, {
        "kind": "decimal",
        "value": (
            "0.0250000000000000000000000000000000000001250000000000000000000000000000000000000"
        ),
    }]]


def test_repeating_decimal_division_is_never_silently_rounded():
    from pops.time.values import _Coeff

    program, linear = _bound_program("schur_repeating")
    with localcontext() as context:
        context.prec = 3
        with pytest.raises(TypeError, match="must terminate"):
            _ = _Coeff({0: Decimal(1)}) / Decimal(3)
        with pytest.raises(TypeError, match="non-terminating Decimal reciprocal"):
            libtime.CondensedSchur(
                program,
                *state_refs(program, "block"),
                theta=Decimal("0.3"),
                alpha=Decimal(1),
                linear_operator=linear,
            )


def test_condensed_schur_refuses_implicit_numeric_domain_mixing():
    program, linear = _bound_program("mixed_domains")
    with pytest.raises(TypeError, match="cannot mix Decimal and Fraction"):
        libtime.CondensedSchur(
            program,
            *state_refs(program, "block"),
            theta=Decimal("0.5"),
            alpha=Fraction(3, 5),
            linear_operator=linear,
        )


def test_multistep_presets_serialize_integer_and_rational_weights():
    ab3 = adctime.Program("ab3_exact")
    libtime.adams_bashforth(
        ab3, *state_refs(ab3, "block"), order=3)
    coeffs = _node(ab3, "linear_combine")["attrs"]["coeffs"]
    assert coeffs == [
        [[0, {"kind": "integer", "value": "1"}]],
        [[1, {"kind": "rational", "numerator": "23", "denominator": "12"}]],
        [[1, {"kind": "rational", "numerator": "-4", "denominator": "3"}]],
        [[1, {"kind": "rational", "numerator": "5", "denominator": "12"}]],
    ]

    bdf2, linear = _bound_program("bdf2_exact")
    libtime.bdf(
        bdf2, *state_refs(bdf2, "block"), order=2,
        linear_source=linear)
    solve = _node(bdf2, "solve_local_linear")
    assert solve["attrs"]["a_coeff"] == [[
        1, {"kind": "rational", "numerator": "2", "denominator": "3"}
    ]]


def test_strang_and_lie_pass_exact_builtin_step_fractions():
    seen = []

    def flow(program, state, fraction, *, at):
        seen.append(("flow", fraction))
        return program.value(None, 1 * state, at=at)

    def source(program, state, fraction, *, at):
        seen.append(("source", fraction))
        return program.value(None, 1 * state, at=at)

    strang = adctime.Program("strang")
    libtime.strang(
        strang, *state_refs(strang, "block"), half_flow=flow,
        source=source, commit=False)
    assert seen == [
        ("flow", Fraction(1, 2)),
        ("source", 1),
        ("flow", Fraction(1, 2)),
    ]
    assert type(seen[1][1]) is int

    seen.clear()
    lie = adctime.Program("lie")
    libtime.lie(
        lie, *state_refs(lie, "block"), half_flow=flow,
        source=source, commit=False)
    assert seen == [("flow", 1), ("source", 1)]
    assert all(type(fraction) is int for _, fraction in seen)
