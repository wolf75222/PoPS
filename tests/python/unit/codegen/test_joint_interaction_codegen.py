"""Joint balance lowering preserves one context and every mathematical occurrence."""
import pytest

import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.math import ddt
from typed_program_support import typed_state


def _program(*, repeated=True):
    physical = pops.Model("joint_exchange")
    left = physical.species("left", state=("p", "E"))
    right = physical.species("right", state=("p", "E", "m"))
    force = right[0] / right[2] - left[0]
    power = (left[0] + right[0] / right[2]) * force / 2
    joint = physical.interaction("exchange", outputs={
        left: (force, power), right: (-force, -power, 0)})
    left_rate = physical.rate("left_balance", equation=ddt(left) == (
        joint[left] + joint[left] if repeated else joint[left]))
    right_rate = physical.rate("right_balance", equation=ddt(right) == joint[right])
    module = physical.module
    program = pops.Program("joint_step")._bind_operators(module)
    a = typed_state(program, "recipient_a", space=left.space, model=module, state=left)
    b = typed_state(program, "recipient_b", space=right.space, model=module, state=right)
    ra = left_rate(a, b)
    rb = right_rate(b, a)
    return physical, program, joint, left_rate, right_rate, a, b, ra, rb


def test_distinct_balance_projections_share_one_owned_joint_node():
    physical, program, joint, left_rate, right_rate, a, b, ra, rb = _program()
    nodes = [value for value in program._values if value.op == "coupled_rate"]
    assert len(nodes) == 1
    assert nodes[0].attrs["joint_application"] is joint
    assert nodes[0].attrs["output_bindings"] == {"left": a.block, "right": b.block}
    assert len(left_rate.occurrences) == 2 and len(right_rate.occurrences) == 1
    assert ra.block == a.block and rb.block == b.block
    a_next = typed_state(program, "recipient_a", state_name="left", space=a.space,
                         model=physical.module, state=physical._states["left"]).next
    b_next = typed_state(program, "recipient_b", state_name="right", space=b.space,
                         model=physical.module, state=physical._states["right"]).next
    program.commit(a_next, program.value("a_next", a + program.dt * ra, at=a_next.point))
    program.commit(b_next, program.value("b_next", b + program.dt * rb, at=b_next.point))
    source = emit_cpp_program(program)
    assert source.count("native_status_view(index, 0) = pops::Real(0);") == 1
    assert source.count("consume_pointwise_evaluation_status") == 1
    assert "pops_input_0_component_0" in source
    assert "pops_input_1_component_2" in source
    from pops.time._program.detach import detach_compiled_program
    detached = detach_compiled_program(program)
    assert detached._ir_hash() == program._ir_hash()
    detached_joint, = [value.attrs["joint_application"] for value in detached._values
                      if value.op == "coupled_rate"]
    assert all(handle.is_resolved for handle in detached_joint.declaration_references())


def test_new_ssa_inputs_create_another_joint_evaluation():
    _physical, program, _joint, left_rate, _right_rate, a, b, _ra, _rb = _program()
    fresh_a = program.value("fresh_a", 1 * a)
    left_rate(fresh_a, b)
    assert len([value for value in program._values if value.op == "coupled_rate"]) == 2


def test_foreign_quantity_cannot_enter_interaction_by_matching_spelling():
    physical = pops.Model("owner")
    a = physical.species("left", state=("u",))
    b = physical.species("right", state=("v", "e"))
    foreign = pops.Model("foreign").state("left", components=("u",))
    with pytest.raises(ValueError, match="exact quantity"):
        physical.interaction("invalid", outputs={a: (foreign[0],), b: (b[0], b[1])})


def test_joint_numerical_method_checks_exact_target_and_sampling():
    from pops.numerics import JointEvaluation
    physical, _time, _joint, left_rate, _right_rate, a, b, _ra, _rb = _program()
    method = JointEvaluation(physical._states["left"])
    assert method.validate_rate_contract(physical.rate_contract(left_rate))
    assert method.validate_balance_view(left_rate.view)
    with pytest.raises(ValueError, match="exact joint-source"):
        JointEvaluation(physical._states["right"]).validate_balance_view(left_rate.view)
    with pytest.raises(ValueError, match="face laws"):
        JointEvaluation(physical._states["left"], sampling="face")


def test_independent_source_and_joint_projection_share_only_the_joint_call():
    physical = pops.Model("joint_and_source")
    a = physical.species("a", state=("p", "E"))
    b = physical.species("b", state=("p", "E", "m"))
    interaction = physical.interaction("exchange", outputs={a: (b[0], 0), b: (-b[0], 0, 0)})
    source = physical.source("forcing", on=a, value=(1, 0))
    rate = physical.rate("sum", equation=ddt(a) == interaction[a] + 2 * source)
    module = physical.module
    program = pops.Program("mixed")._bind_operators(module)
    av = typed_state(program, "a", space=a.space, model=module, state=a)
    bv = typed_state(program, "b", space=b.space, model=module, state=b)
    result = rate(av, bv)
    assert result.block == av.block
    assert len([value for value in program._values if value.op == "coupled_rate"]) == 1
    assert len([value for value in program._values if value.op == "source"]) == 1
