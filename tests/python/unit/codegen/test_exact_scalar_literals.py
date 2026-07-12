"""ADC-652: authoring literals stay exact until an explicit target lowering."""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from pops.ir import Const, ScalarLiteral, scalar_data
from pops.model import StateSpace
from pops.time import Program
from typed_program_support import typed_state


_SCALAR_SPACE = StateSpace("U", ("u",))


def _state(program, *, temporal=False):
    return typed_state(
        program, "transport", state_name="U" if temporal else None, space=_SCALAR_SPACE)


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        (7, "integer", {"kind": "integer", "value": "7"}),
        (Fraction(1, 3), "rational",
         {"kind": "rational", "numerator": "1", "denominator": "3"}),
        (Decimal("0.1000000000000000000000000001"), "decimal",
         {"kind": "decimal", "value": "0.1000000000000000000000000001"}),
        (0.1, "binary64", {"kind": "binary64", "value": 0.1.hex()}),
    ],
)
def test_scalar_literal_preserves_the_authoring_representation(value, kind, expected):
    literal = ScalarLiteral.from_value(value)

    assert literal.kind == kind
    assert literal.to_data() == expected
    assert scalar_data(value) == expected


def test_const_keeps_rational_and_decimal_values_without_float_coercion():
    rational = Const(Fraction(2, 7))
    decimal = Const(Decimal("1.0000000000000000000000000001"))

    assert rational.value == Fraction(2, 7)
    assert isinstance(rational.value, Fraction)
    assert decimal.value == Decimal("1.0000000000000000000000000001")
    assert isinstance(decimal.value, Decimal)
    assert rational.to_cpp() == "(pops::Real(2) / pops::Real(7))"
    assert decimal.to_cpp() == "pops::Real(1.0000000000000000000000000001)"


def test_binary_float_payload_is_recorded_explicitly():
    literal = ScalarLiteral.from_value(0.1)

    assert literal.payload == "0x1.999999999999ap-4"
    assert literal.to_python() == 0.1


def test_unit_and_target_annotations_survive_until_lowering():
    literal = ScalarLiteral.from_value(Fraction(5, 8), unit="m/s", target="Real32")

    assert literal.to_data() == {
        "kind": "rational",
        "numerator": "5",
        "denominator": "8",
        "unit": "m/s",
        "target": "Real32",
    }
    with pytest.raises(TypeError, match="explicit unit-system conversion"):
        literal.to_cpp()
    converted = ScalarLiteral.from_value(literal.to_python(), target=literal.target)
    assert converted.to_cpp() == "(Real32(5) / Real32(8))"


def test_algebraic_constant_has_explicit_symbolic_and_target_spellings():
    literal = ScalarLiteral.algebraic("sqrt(2)", cpp="std::sqrt(pops::Real(2))")

    assert literal.to_data()["value"] == "sqrt(2)"
    assert literal.to_cpp() == "std::sqrt(pops::Real(2))"
    with pytest.raises(TypeError, match="target lowering"):
        literal.to_python()


@pytest.mark.parametrize("value", [True, False, float("inf"), float("nan")])
def test_ambiguous_or_non_finite_literals_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        ScalarLiteral.from_value(value)


def test_exact_affine_coefficient_has_one_typed_codec_for_hash_and_cpp():
    program = Program("exact_coeff")
    state = _state(program, temporal=True)
    result = program.linear_combine(
        "next", Fraction(1, 3) * state.n, at=state.next.point)
    program.commit(state.next, result)

    coeff = program._serialize()["nodes"][-1]["attrs"]["coeffs"][0]
    assert coeff == [[0, {
        "kind": "rational", "numerator": "1", "denominator": "3"}]]
    assert "(pops::Real(1) / pops::Real(3))" in program.emit_cpp_program()


def test_unit_target_and_algebraic_program_constants_reach_target_lowering_losslessly():
    annotated = ScalarLiteral.from_value(
        Fraction(5, 8), unit="m/s", target="pops::Real")
    algebraic = ScalarLiteral.algebraic(
        "sqrt(2)", cpp="std::sqrt(pops::Real(2))")
    program = Program("annotated_scalars")
    state = _state(program, temporal=True)
    reduced = program.norm2(state.n)
    program.record_scalar("annotated", reduced + annotated)
    program.record_scalar("algebraic", reduced + algebraic)
    program.commit(
        state.next,
        program.linear_combine("next", state.n, at=state.next.point),
    )

    serialized = program._serialize()
    scalar_nodes = [node for node in serialized["nodes"] if node["op"] == "scalar_op"]
    literals = [node["attrs"]["operands"][1][1] for node in scalar_nodes]
    assert literals[0]["unit"] == "m/s" and literals[0]["target"] == "pops::Real"
    assert literals[1]["kind"] == "algebraic" and literals[1]["value"] == "sqrt(2)"
    with pytest.raises(TypeError, match="explicit unit-system conversion"):
        program.emit_cpp_program()


def test_affine_coefficients_refuse_to_drop_annotations_or_evaluate_algebraic_literals():
    program = Program("coefficient_gate")
    state = _state(program)
    annotated = ScalarLiteral.from_value(Fraction(1, 2), unit="s")
    algebraic = ScalarLiteral.algebraic("sqrt(2)", cpp="std::sqrt(pops::Real(2))")

    with pytest.raises(TypeError, match="unit or target annotation"):
        _ = annotated * state
    with pytest.raises(TypeError, match="algebraic scalar"):
        _ = algebraic * state


def test_custom_literal_payload_is_transitively_immutable_and_detached():
    payload = {"coefficients": [1, 2]}

    class CustomLiteral:
        def __pops_scalar_literal__(self):
            return {"kind": "custom", "payload": payload, "cpp": "pops::Real(3)"}

    literal = ScalarLiteral.from_value(CustomLiteral())
    payload["coefficients"].append(3)

    assert literal.to_data()["value"] == {"coefficients": [1, 2]}
    with pytest.raises(TypeError):
        literal.payload["coefficients"] = (9,)
    assert literal.to_cpp() == "pops::Real(3)"

    direct_payload = {"nested": [1]}
    direct = ScalarLiteral("custom", direct_payload, cpp="pops::Real(1)")
    direct_payload["nested"].append(2)
    assert direct.to_data()["value"] == {"nested": [1]}


def test_direct_known_literal_payloads_are_strictly_validated():
    with pytest.raises(TypeError, match="decimal string"):
        ScalarLiteral("decimal", True)
    with pytest.raises(TypeError, match=r"float\.hex string"):
        ScalarLiteral("binary64", 1.0)
    with pytest.raises(TypeError, match="strict JSON"):
        ScalarLiteral("custom", {"x": Decimal("0.1")}, cpp="pops::Real(1)")
    with pytest.raises(TypeError, match="string keys"):
        ScalarLiteral("custom", {1: "a", "1": "b"}, cpp="pops::Real(1)")
    assert ScalarLiteral("decimal", " 1_0 ").to_cpp() == "pops::Real(10)"
    assert ScalarLiteral.from_value(Decimal("-0")).to_cpp() == "pops::Real(-0.0)"
    assert ScalarLiteral("rational", (2, -4)).payload == (-1, 2)
    assert ScalarLiteral("binary64", "1").payload == 1.0.hex()
    with pytest.raises(ValueError, match="algebraic"):
        ScalarLiteral("algebraic", "", cpp="pops::Real(1)")


def test_wide_integer_and_rational_literals_lower_only_at_the_real_target_boundary():
    wide = 1 << 63
    integer_cpp = ScalarLiteral.from_value(wide).to_cpp()
    rational_cpp = ScalarLiteral.from_value(Fraction(wide, 3)).to_cpp()

    assert str(wide) not in integer_cpp
    assert str(wide) not in rational_cpp
    assert integer_cpp.startswith("pops::Real(")
    assert rational_cpp.startswith("pops::Real(")


def test_target_typed_literals_never_round_through_pops_real_first():
    rational = ScalarLiteral.from_value(Fraction(1, 3), target="Real128")
    decimal = ScalarLiteral.from_value(Decimal("0.123456789012345678901"), target="Real128")
    wide = ScalarLiteral.from_value(1 << 80, target="Real128")

    for token in (rational.to_cpp(), decimal.to_cpp(), wide.to_cpp()):
        assert "pops::Real" not in token
        assert "Real128" in token
    assert rational.to_cpp() == "(Real128(1) / Real128(3))"
    assert str(1 << 80) not in wide.to_cpp()
    with pytest.raises(ValueError, match=r"qualified C\+\+ scalar type"):
        ScalarLiteral.from_value(1, target="bad); injected(")


def test_binary64_rational_boundary_rounds_the_exact_fraction_only_once():
    value = Fraction(125968702744266439, 2969781818958378174)

    assert value.numerator.bit_length() > 53
    assert ScalarLiteral.from_value(value).to_cpp() == "pops::Real(%s)" % repr(float(value))


@pytest.mark.parametrize("value", [10 ** 1000, Fraction(10 ** 1000, 3), Decimal("1e1000")])
def test_literals_outside_the_real_target_range_fail_with_a_clear_lowering_error(value):
    with pytest.raises(OverflowError, match="finite pops::Real target"):
        ScalarLiteral.from_value(value).to_cpp()


def test_solver_controls_keep_exact_literals_until_codegen():
    from pops.solvers.krylov import Richardson

    program = Program("exact_solver_controls")
    time_state = _state(program, temporal=True)
    state = time_state.n
    operator = program.matrix_free_operator("A")
    program.set_apply(operator, lambda P, out, in_: in_)
    rhs = program.linear_combine("rhs", state, at=time_state.next.point)
    result = program.solve_linear(
        operator=operator,
        rhs=rhs,
        method=Richardson(max_iter=4, omega=Fraction(2, 3)),
        tol=Decimal("1e-12"),
        max_iter=4,
    )
    program.commit(time_state.next, result)

    solve = next(node for node in program._values if node.op == "solve_linear")
    assert solve.attrs["tol"].to_data() == {"kind": "decimal", "value": "1E-12"}
    assert solve.attrs["omega"].to_data() == {
        "kind": "rational", "numerator": "2", "denominator": "3"}
    source = program.emit_cpp_program()
    assert "pops::Real(1E-12)" in source
    assert "(pops::Real(2) / pops::Real(3))" in source


def test_solver_iteration_budget_rejects_bool():
    program = Program("bool_budget")
    state = _state(program)
    operator = program.matrix_free_operator("A")
    program.set_apply(operator, lambda P, out, in_: in_)

    with pytest.raises(ValueError, match="max_iter"):
        program.solve_linear(operator=operator, rhs=state, max_iter=True)


def test_board_operator_scales_never_erase_annotations_or_mix_number_domains():
    from pops.ir.expr import Partial

    annotated = ScalarLiteral.from_value(Fraction(1, 3), unit="m")
    with pytest.raises(TypeError, match="unit or target annotation"):
        Partial("phi", 0, annotated)

    partial = Partial("phi", 0, Fraction(1, 3))
    assert partial.scale == Fraction(1, 3)
    with pytest.raises(TypeError, match="explicit target conversion"):
        _ = partial * 0.5


def test_symbolic_diff_preserves_fraction_and_decimal_exponents_exactly():
    from pops.ir.expr import Var
    from pops.ir.lowering import diff

    x = Var("x", "cons")
    fraction_cpp = diff(x ** Fraction(2, 3), x).to_cpp()
    decimal_cpp = diff(x ** Decimal("0.666666666666666666"), x).to_cpp()

    assert "pops::Real(-1) / pops::Real(3)" in fraction_cpp
    assert "-0.333333" not in fraction_cpp
    assert "pops::Real(-0.333333333333333334)" in decimal_cpp


def test_symbolic_constant_folding_refuses_implicit_number_domain_mixing():
    from pops.ir.expr import Const
    from pops.ir.lowering import _s_add

    with pytest.raises(TypeError, match="explicit target conversion"):
        _s_add(Const(Fraction(1, 3)), Const(0.5))


def test_symbolic_annotation_algebra_preserves_or_refuses_semantics():
    from pops.ir.expr import Mul, Var
    from pops.ir.lowering import _s_add, _s_div, _s_mul, _s_sub

    metres_a = Const(ScalarLiteral.from_value(Fraction(1, 3), unit="m", target="Real32"))
    metres_b = Const(ScalarLiteral.from_value(Fraction(2, 3), unit="m", target="Real32"))
    total = _s_add(metres_a, metres_b)
    delta = _s_sub(metres_b, metres_a)
    assert total.literal.to_data()["unit"] == "m"
    assert total.literal.to_data()["target"] == "Real32"
    assert total.value == 1
    assert delta.value == Fraction(1, 3)

    seconds = Const(ScalarLiteral.from_value(1, unit="s", target="Real32"))
    real64 = Const(ScalarLiteral.from_value(1, unit="m", target="Real64"))
    with pytest.raises(TypeError, match="identical units"):
        _s_add(metres_a, seconds)
    with pytest.raises(TypeError, match="compatible scalar targets"):
        _s_sub(metres_a, real64)
    with pytest.raises(TypeError, match="explicit unit-system operation"):
        _s_mul(metres_a, metres_b)
    with pytest.raises(TypeError, match="explicit unit-system operation"):
        _s_div(metres_a, seconds)

    target_zero = Const(ScalarLiteral.from_value(0, target="Real32"))
    target_one = Const(ScalarLiteral.from_value(1, target="Real32"))
    folded_zero = _s_mul(target_zero, Const(7))
    assert folded_zero.value == 0 and folded_zero.literal.target == "Real32"
    unsimplified_one = _s_mul(target_one, Var("x", "cons"))
    assert isinstance(unsimplified_one, Mul)


def test_diff_and_decimal_folding_are_context_independent_and_annotation_safe():
    from decimal import localcontext
    from pops.ir.expr import Div, Var
    from pops.ir.lowering import _s_add, _s_div, _s_mul, diff

    left = Decimal("1.123456789012345678901234567890123456789")
    right = Decimal("2.000000000000000000000000000000000000001")
    with localcontext() as context:
        context.prec = 5
        summed = _s_add(Const(left), Const(right))
        product = _s_mul(Const(left), Const(right))
        repeating = _s_div(Const(Decimal(1)), Const(Decimal(3)))
    assert summed.value == Decimal("3.123456789012345678901234567890123456790")
    assert product.value == Decimal(
        "2.246913578024691357802469135780246913579123456789012345678901234567890123456789")
    assert isinstance(repeating, Div), "a repeating Decimal quotient must remain symbolic"

    x = Var("x", "cons")
    coefficient = Const(ScalarLiteral.from_value(left, target="Real32"))
    derivative = diff(coefficient * x, x)
    assert derivative.value == left
    assert derivative.literal.target == "Real32"


def test_direct_expression_construction_rejects_annotation_mismatches():
    left = Const(ScalarLiteral.from_value(1, unit="m", target="Real32"))
    wrong_unit = Const(ScalarLiteral.from_value(1, unit="s", target="Real32"))
    wrong_target = Const(ScalarLiteral.from_value(1, unit="m", target="Real64"))

    with pytest.raises(TypeError, match="identical units"):
        _ = left + wrong_unit
    with pytest.raises(TypeError, match="compatible scalar targets"):
        _ = left - wrong_target
    with pytest.raises(TypeError, match="explicit unit-system operation"):
        _ = left * left
