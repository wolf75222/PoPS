"""Independent Field scientific identity survives structure without a State owner."""
from types import SimpleNamespace

import pytest

from pops.codegen.program_emit_versioned import schema_cpp
from pops.fields._observation_contract import validate_field_observation
from pops.time import FailRun
from pops.time._program.detach import detach_compiled_program
from pops.time.value_signatures import same_signature, value_signature
from tests.python.unit.fields.test_program_field_problem import field_case


def source(*, joint=False):
    _case, field, problem, program, values, point = field_case(joint=joint)
    solved = program.solve(field, values=values, at=point).consume(action=FailRun())
    observation = field.observe(solved)
    return program, observation, field, problem


def test_scientific_provider_survives_product_projection_and_detach():
    program, observation, field, problem = source()
    value = observation[field[problem.unknowns[0]]]
    validate_field_observation(value)
    projected = program.product(stored=value)["stored"]
    assert same_signature(value_signature(value), value_signature(projected))
    assert value.block is value.state_ref is None
    code = schema_cpp(value_signature(projected), {})
    assert "ContinuationKind::Field" in code and "field_problem" in code
    detached = detach_compiled_program(program)
    restored = next(node for node in detached._values if node.id == projected.id)
    assert same_signature(value_signature(projected), value_signature(restored))
    assert detached._ir_hash() == program._ir_hash()


def test_two_unknowns_share_problem_but_not_provider():
    _program, observation, field, problem = source(joint=True)
    first, second = [observation[field[unknown]] for unknown in problem.unknowns]
    assert first.attrs["field_authority"]["problem"] == second.attrs["field_authority"]["problem"]
    assert first.attrs["field_authority"]["unknown"] != second.attrs["field_authority"]["unknown"]
    assert not same_signature(value_signature(first), value_signature(second))


@pytest.mark.parametrize("key,change", [("problem", "other"), ("components", 2),
    ("projection", "other"), ("unknown", {"name": "same"})])
def test_observation_rejects_forged_provider_despite_same_storage_shape(key, change):
    _program, observation, field, problem = source()
    value = observation[field[problem.unknowns[0]]]
    forged = dict(value.attrs["field_authority"])
    forged[key] = change
    value = SimpleNamespace(op=value.op, vtype=value.vtype, inputs=value.inputs,
                            attrs={**value.attrs, "field_authority": forged}, space=value.space, prog=value.prog)
    with pytest.raises(ValueError, match="provider authority"):
        validate_field_observation(value)


def test_gradient_has_distinct_exact_provider_and_survives_structural_projection():
    from pops.fields._observation_contract import validate_field_gradient
    program, observation, field, problem = source()
    potential = observation[field[problem.unknowns[0]]]
    gradient = observation.gradient(field[problem.unknowns[0]], dimension=2)
    assert validate_field_gradient(gradient)[0] == 2
    assert not same_signature(value_signature(potential), value_signature(gradient))
    assert gradient.attrs["field_authority"]["problem"] == potential.attrs["field_authority"]["problem"]
    projected = program.product(stored=gradient)["stored"]
    assert same_signature(value_signature(gradient), value_signature(projected))
    assert "gradient" in schema_cpp(value_signature(projected), {})
    forged = SimpleNamespace(op=gradient.op, vtype=gradient.vtype, inputs=gradient.inputs,
        attrs={**gradient.attrs, "field_authority": potential.attrs["field_authority"]},
        space=gradient.space, point=gradient.point, region=gradient.region)
    with pytest.raises(ValueError, match="provider authority"):
        validate_field_gradient(forged)
