"""ADC-652: ProgramValue is symbolic SSA data, never a Python value."""
from __future__ import annotations

from typed_program_support import state_refs, typed_state

from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from pops.math import Equation, SymbolicTruthValueError
from pops.numerics.terms import DefaultSource, Flux
from pops.provenance import ProvenanceRecord, SourceSpan
from pops.time import Program, ProgramValue, TimePoint


def _direct_provenance(program: Program) -> ProvenanceRecord:
    span = SourceSpan(__file__, 0)
    return ProvenanceRecord(
        primary=span, owner=program.owner_path,
        authoring_api="tests.ProgramValue", origins=(span,))


def test_program_state_is_an_immutable_nonhashable_program_value():
    program = Program("data_model")
    value = typed_state(program, "plasma")

    assert isinstance(value, ProgramValue)
    with pytest.raises(TypeError, match="unhashable"):
        hash(value)
    with pytest.raises(AttributeError, match="immutable"):
        value.name = "renamed"
    with pytest.raises(TypeError):
        value.attrs["new"] = "metadata"
    assert not hasattr(value, "_set_internal")
    assert not hasattr(value, "_update_attrs")


def test_program_metadata_is_deeply_immutable():
    program = Program("deep_immutable")
    state = typed_state(program, "plasma")
    rate = program.rhs(state=state, terms=[Flux(), DefaultSource()])

    assert rate.attrs["sources"] == ("default",)
    with pytest.raises(AttributeError):
        rate.attrs["sources"].append("other")


def test_program_metadata_rejects_mutable_or_opaque_leaves():
    class Box:
        def __init__(self):
            self.value = 1

    program = Program("opaque_metadata")
    block, _ = state_refs(program, "transport")
    with pytest.raises(TypeError, match="not an immutable IR value"):
        ProgramValue(
            program, 0, "state", "state", (), {"box": Box()},
            "U", block, point=TimePoint(program.clock),
            provenance=_direct_provenance(program),
        )


def test_program_value_rejects_untyped_mutable_space_and_field_context():
    program = Program("strict_metadata")
    block, _ = state_refs(program, "transport")
    for keyword in ("space", "field_context"):
        with pytest.raises(TypeError):
            ProgramValue(
                program, 0, "state", "state", (), {}, "U", block,
                point=TimePoint(program.clock),
                provenance=_direct_provenance(program),
                **{keyword: []},
            )


def test_program_value_has_no_python_truth_value_and_reports_typed_provenance():
    program = Program("truth")
    value = typed_state(program, "plasma")

    with pytest.raises(SymbolicTruthValueError) as raised:
        bool(value)

    assert raised.value.code == "symbolic_truth_value"
    provenance = value.provenance
    assert isinstance(provenance, ProvenanceRecord)
    assert isinstance(provenance.primary, SourceSpan)
    assert provenance.authoring_api == "pops.time.Program.state"
    assert provenance.transformation == "direct"
    assert Path(raised.value.location.file) == Path(provenance.primary.file)
    assert raised.value.location.line == provenance.primary.line


def test_program_value_equality_builds_an_equation_instead_of_identity_truth():
    program = Program("equation")
    left = typed_state(program, "left")
    right = typed_state(program, "right")

    equation = left == right

    assert isinstance(equation, Equation)
    assert equation.lhs is left
    assert equation.rhs is right
    with pytest.raises(SymbolicTruthValueError):
        bool(equation)


def test_scalar_program_value_equality_builds_a_runtime_bool_predicate():
    program = Program("scalar_equality")
    scalar = program.norm2(typed_state(program, "plasma"))

    predicate = scalar == Fraction(1, 3)

    assert isinstance(predicate, ProgramValue)
    assert predicate.vtype == "bool"
    assert predicate.op == "compare"
    assert predicate.attrs["cmp"] == "=="
    assert predicate.attrs["rhs"].to_data() == {
        "kind": "rational", "numerator": "1", "denominator": "3"}
    with pytest.raises(SymbolicTruthValueError):
        bool(predicate)


def test_program_value_from_a_different_program_still_never_falls_back_to_boolean_identity():
    left_program = Program("left")
    right_program = Program("right")
    left = typed_state(left_program, "plasma")
    right = typed_state(right_program, "plasma")

    equation = left == right

    assert isinstance(equation, Equation)
    assert equation.lhs is left
    assert equation.rhs is right
    with pytest.raises(SymbolicTruthValueError):
        bool(equation)


def test_forged_same_program_value_cannot_be_laundered_by_ssa_id():
    program = Program("forgery")
    real = typed_state(program, "plasma")
    forged = ProgramValue(
        program, real.id, real.vtype, real.op, real.inputs, real.attrs,
        real.name, real.block, space=real.space, region=real.region,
        point=real.point,
        provenance=real.provenance)

    assert program._canonical_value(forged) is forged
    with pytest.raises(ValueError, match="not authored"):
        program.value("forged", forged)


@pytest.mark.parametrize(
    ("field", "value"),
    [("vid", True), ("vid", -1), ("vtype", ""), ("op", ""), ("block", ""),
     ("region", True), ("region", -1)],
)
def test_direct_program_value_construction_validates_identity_fields(field, value):
    program = Program("constructor_validation")
    block, _ = state_refs(program, "transport")
    kwargs = {
        "prog": program, "vid": 0, "vtype": "state",
        "op": "state", "inputs": (), "attrs": {}, "name": "U", "block": block,
        "region": 0, "point": TimePoint(program.clock),
        "provenance": _direct_provenance(program),
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        ProgramValue(**kwargs)


def test_affine_coefficients_keep_exact_rationals():
    program = Program("rational_coeff")
    state = typed_state(program, "plasma")

    affine = Fraction(1, 3) * state + Fraction(2, 3) * state
    [(merged_state, coefficient)] = affine._merge()

    assert merged_state is state
    assert coefficient.as_dict() == {0: Fraction(1, 1)}
    one_third = (Fraction(1, 3) * state)._merge()[0][1].as_dict()[0]
    assert one_third == Fraction(1, 3)
    assert isinstance(one_third, Fraction)


@pytest.mark.parametrize("other", [0.5, Decimal("0.5")])
def test_affine_coefficients_refuse_implicit_cross_domain_coercion(other):
    program = Program("mixed_coeff_domain")
    state = typed_state(program, "plasma")

    with pytest.raises(TypeError, match="explicit target conversion"):
        program.value("mixed", Fraction(1, 3) * state + other * state)


def test_affine_coefficients_refuse_a_wide_integer_binary64_coercion():
    program = Program("wide_integer_domain")
    state = typed_state(program, "plasma")

    with pytest.raises(TypeError, match="explicit target conversion"):
        program.value("wide", (10 ** 100) * state + 0.5 * state)


def test_debug_name_is_immutable_and_part_of_ir_identity():
    def build(label):
        program = Program("named_identity")
        state = typed_state(program, "transport", state_name="U")
        value = program.value(label, state.n, at=state.next.point)
        program.commit(state.next, value)
        return program

    alpha = build("alpha")
    beta = build("beta")

    assert alpha._ir_hash() != beta._ir_hash()
    assert any(node["name"] == "alpha" for node in alpha.ir_nodes())
    assert any(node["name"] == "beta" for node in beta.ir_nodes())
    with pytest.raises(ValueError, match="non-empty string"):
        Program("")


def test_ssa_aliases_are_canonical_in_every_public_inspection_view():
    program = Program("canonical_inspection")
    state = typed_state(program, "transport")
    first = program.value("first", state)
    program.value("consumer", first)

    program.value("renamed", first)

    assert "renamed" in program.dump_operator_ir()
    assert "renamed" in program.dump_cpp_plan()
    nodes = program.ir_nodes()
    assert [node["name"] for node in nodes] == ["renamed"]
    assert nodes[0]["inputs"] == []
