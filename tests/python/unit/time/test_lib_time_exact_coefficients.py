"""Exact coefficient contracts for the final ready-made time-scheme library."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

import pops.lib.time as libtime
from pops import time as adctime
from pops.physics._facade import Model
from pops.solvers.nonlinear import LocalNewton
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
        (0.25, {"kind": "binary64", "value": (0.25).hex()}),
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


def test_imex_lowers_a_nonlinear_local_source_through_typed_newton():
    model = Model("nonlinear_imex_model")
    (u,) = model.conservative_vars("u")
    explicit = model.rate("transport", flux=False, sources=())
    implicit = model.source_term("reaction", [-u * u])
    block, state = state_refs(adctime.Program("nonlinear_refs"), "block", model=model)

    program = libtime.IMEX(
        block[state],
        explicit_operator=explicit,
        implicit_operator=implicit,
        implicit_solver=LocalNewton(
            tolerance=1.0e-11,
            max_iterations=17,
            finite_difference_step=1.0e-6,
        ),
    )

    assert program.validate() is True
    solve = _node(program, "solve_local_nonlinear")
    assert solve["attrs"]["problem_kind"] == "local_residual"
    assert solve["attrs"]["max_iter"] == 17
    assert solve["attrs"]["tol"] == {"kind": "binary64", "value": (1.0e-11).hex()}
    assert solve["attrs"]["fd_eps"] == {"kind": "binary64", "value": (1.0e-6).hex()}
    residual_sources = [node for node in solve["attrs"]["residual_block"] if node["op"] == "source"]
    assert len(residual_sources) == 1
    assert residual_sources[0]["attrs"]["source"] == "reaction"
    assert _node(program, "solve_outcome")["attrs"]["action"]["kind"] == "fail_run"
    assert _node(program, "source")["attrs"]["source"] == "reaction"


def test_ars222_tableau_retains_stable_binary64_coefficients_and_stage_topology():
    tableau = libtime.IMEX_ARS222_TABLEAU

    assert tableau.name == "imex-ars222"
    assert tableau.stages == 3
    assert tableau.explicit.A == (
        (),
        (float.fromhex("0x1.2bec333018868p-2"),),
        (
            float.fromhex("-0x1.6a09e667f3bcap-1"),
            float.fromhex("0x1.b504f333f9de5p+0"),
        ),
    )
    assert tableau.implicit_A == (
        (0,),
        (0, float.fromhex("0x1.2bec333018868p-2")),
        (
            0,
            float.fromhex("0x1.6a09e667f3bccp-1"),
            float.fromhex("0x1.2bec333018868p-2"),
        ),
    )

    model = Model("nonlinear_ars222_model")
    (u,) = model.conservative_vars("u")
    explicit = model.rate("transport", flux=False, sources=())
    implicit = model.source_term("reaction", [-u * u])
    block, state = state_refs(adctime.Program("ars222_refs"), "block", model=model)
    program = libtime.IMEX(
        block[state],
        explicit_operator=explicit,
        implicit_operator=implicit,
        tableau=tableau,
        implicit_solver=LocalNewton(),
    )

    assert program.validate() is True
    operations = [node["op"] for node in program._serialize()["nodes"]]
    assert operations.count("solve_local_nonlinear") == 2
    assert operations.count("solve_outcome") == 2
    assert operations.count("source") == 3


def test_imex_nonlinear_source_requires_an_explicit_solver():
    model = Model("nonlinear_imex_missing_solver")
    (u,) = model.conservative_vars("u")
    explicit = model.rate("transport", flux=False, sources=())
    implicit = model.source_term("reaction", [-u * u])
    block, state = state_refs(adctime.Program("missing_solver_refs"), "block", model=model)

    with pytest.raises(ValueError, match="requires an explicit implicit_solver"):
        libtime.IMEX(
            block[state],
            explicit_operator=explicit,
            implicit_operator=implicit,
        )


def test_imex_rejects_a_nonlinear_source_that_requires_fields():
    from pops.model import Module, Rate
    from pops.problem import Case

    model = Module("nonlinear_imex_field_coupled")
    state = model.state_space("U", ("u",))
    fields = model.field_space("electric", ("electric",))
    state_handle = model.state_handle(state)
    explicit = model.operator(
        "transport",
        signature=(state,) >> Rate(state),
        kind="local_rate",
        expr={"test": "transport"},
    )
    implicit = model.operator(
        "reaction",
        signature=(state, fields) >> Rate(state),
        kind="local_source",
        expr={"test": "reaction"},
    )
    block = Case("field_source_case").block(
        "block", model=model, states=(state_handle,)
    )

    with pytest.raises(ValueError, match="must consume exactly one State"):
        libtime.IMEX(
            block[state_handle],
            explicit_operator=explicit,
            implicit_operator=implicit,
            implicit_solver=LocalNewton(),
        )


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
        "0.024691357802469135780246913578024691357923456789012345678901234567890123456789"
    )


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
    assert _node(bdf2, "solve_local_linear")["attrs"]["a_coeff"] == [
        [1, {"kind": "rational", "numerator": "2", "denominator": "3"}]
    ]


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
