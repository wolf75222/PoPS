"""Native field expressions bind exact owners/components and refuse altered source data."""
from __future__ import annotations

from fractions import Fraction

import pytest

from pops._ir.expr import Sqrt, Var
from pops._ir.quantity import QuantityRef
from pops.fields._program_expression import (
    decode_field_literal, encode_field_expression, field_expression_cpp,
    field_expression_dependencies,
)
from pops.codegen.program_emit_field_problem import emit_field_problem_value
from pops.fields._identity import field_identity
from pops.identity.scalar import scalar_data
from pops.model import Handle, Module, OwnerPath
from pops.problem import Case
from pops.time import Program
from pops.time.references import canonical_handle


def _state(program, case, name, components):
    module = Module("model-" + name)
    space = module.state_space("U", components)
    state = module.state_handle(space)
    block = case.block(name, module, states=(state,))
    return program.state(block[state]).n


def _expression():
    program, case = Program("field-expression"), Case("assembly")
    states = (_state(program, case, "first", ("u",)),
              _state(program, case, "second", ("v", "w")))
    first = QuantityRef(canonical_handle(states[0].state_ref), "u", space=states[0].space)
    second = QuantityRef(canonical_handle(states[1].state_ref), "w", space=states[1].space)
    expression = Sqrt(first * first + 1) + Fraction(2, 3) * second
    return states, expression


def test_two_owner_expression_preserves_unequal_component_indices_and_exact_literal():
    states, expression = _expression()
    encoded = encode_field_expression(expression, states)
    cpp, reads = field_expression_cpp(encoded, states, views=("left", "right"))
    assert "left(index, 0)" in cpp and "right(index, 1)" in cpp
    assert "std::sqrt" in cpp
    assert {row.qualified_id for row in reads} == {canonical_handle(row.state_ref).qualified_id for row in states}
    assert field_expression_dependencies((encoded,), states) == tuple(
        canonical_handle(state.state_ref).canonical_identity() for state in states)
    assert decode_field_literal(scalar_data(Fraction(2, 3))).to_python() == Fraction(2, 3)


def test_field_expression_refuses_foreign_owner_and_ambiguous_binding():
    states, expression = _expression()
    with pytest.raises(ValueError, match="qualified State binding"):
        encode_field_expression(expression, (states[0],))
    with pytest.raises(ValueError, match="ambiguous"):
        encode_field_expression(expression, (*states, states[0]))
    with pytest.raises(NotImplementedError, match="Var"):
        encode_field_expression(Var("u", "state"), states)


@pytest.mark.parametrize("mutate", [
    lambda node: ("input", 3, 0, node[3]),
    lambda node: ("input", node[1], 19, node[3]),
    lambda node: ("input", node[1], node[2], {**node[3], "local_id": "other"}),
])
def test_encoded_field_expression_refuses_changed_index_width_or_authority(mutate):
    states, _expression_value = _expression()
    expression = QuantityRef(canonical_handle(states[0].state_ref), "u", space=states[0].space)
    node = mutate(encode_field_expression(expression, states))
    with pytest.raises(ValueError):
        field_expression_cpp(node, states, views=("left", "right"))


def test_field_literal_cannot_carry_raw_cpp_or_noncanonical_metadata():
    with pytest.raises(TypeError, match="without raw C"):
        decode_field_literal({"kind": "algebraic", "value": "opaque", "cpp": "unsafe()"})
    with pytest.raises(ValueError, match="canonical"):
        decode_field_literal({"kind": "integer", "value": "2", "cpp": "unsafe()"})


def test_native_load_emits_collective_preflight_and_domain_refusal_before_use():
    states, expression = _expression()
    encoded = encode_field_expression(expression, states)
    attrs = {"ncomp": 1, "expressions": (encoded,),
             "field_problem_identity": field_identity("field-problem", {"test": True}).token,
             "field_dependencies": field_expression_dependencies((encoded,), states)}
    program = states[0].prog
    value = program._new("scalar_field", "field_problem_load", states, attrs,
                         "field_load", None, inherit_state_ref=False)
    var = {state.id: "state%d" % i for i, state in enumerate(states)}
    lines, prelude = [], []
    emit_field_problem_value(value, var, lines, prelude, target="system")
    cpp = "\n".join(lines)
    assert cpp.index("all_reduce_max(field_layout_invalid") < cpp.index("for_each_cell")
    assert "local_rank()" in cpp and "distribution()" in cpp
    assert "input1(index, 1)" in cpp
    assert "std::isfinite" in cpp and "StepAttemptRejected" in cpp
    assert "ctx.prepared_execution_lane()" in cpp
    assert len(prelude) == 2
    assert var[value.id] == "(*program_field_%d)" % value.id
    altered = program._replace_value(value, attrs={**value.attrs, "field_dependencies": ()})
    with pytest.raises(ValueError, match="actual input reads"):
        emit_field_problem_value(altered, var, [], [], target="system")


def test_component_emission_requires_its_consumed_exact_unknown_and_native_problem():
    from pops.time._graph.base import CanonicalData

    unknown = Handle("potential", kind="field", owner=OwnerPath.case("field-owner"))
    identity = field_identity("field-problem", {"observation": True}).token
    program = Program("field_observation_authority")
    load = program._new("scalar_field", "field_problem_load", (),
                        {"field_problem_identity": identity}, "load", None)
    solve = program._new("scalar_field", "solve_linear", (load,), {"solve_request": {
        "physical_problem": CanonicalData({"unknown_components": (
            unknown.canonical_identity(),)}).to_data()}}, "solve", None)
    outcome = program._new("solve_outcome", "solve_outcome", (solve,), {}, "outcome", None)
    packed = program._new("scalar_field", "solve_outcome_component", (outcome,),
                          {"ncomp": 1}, "packed", None)
    observed = program._new("scalar_field", "field_component", (packed,), {
        "ncomp": 1, "component": 0, "field_problem_identity": identity,
        "field_unknown": unknown.canonical_identity()}, "observation", None)
    var, lines = {packed.id: "packed"}, []
    emit_field_problem_value(observed, var, lines, [], target="system")
    assert any("input0(index, 0)" in line for line in lines)
    foreign = Handle("other", kind="field", owner=unknown.owner_path)
    observed = program._replace_value(observed,
        attrs={**observed.attrs, "field_unknown": foreign.canonical_identity()})
    with pytest.raises(ValueError, match="declared solved unknown"):
        emit_field_problem_value(observed, var, [], [], target="system")
