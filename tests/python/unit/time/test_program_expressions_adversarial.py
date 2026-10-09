"""Independent adversarial cases for exact scientific expression captures.

These cases exercise the public authoring boundary. They deliberately use
homonymous declarations and a seed distinct from the equation's frozen data.
"""

import pytest

from pops.frames import Cartesian2D
from pops.physics import Model
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, LocalResidual, Program
from typed_program_support import state_refs


def _state(program, *, block="fluid", model_name="same", state_name="U", model=None, state=None):
    if model is None:
        model = Model(model_name, frame=Cartesian2D())
        state = model.state(state_name, components=("rho", "m"))
    owner, _ = state_refs(program, block, model=model, state=state)
    return program.state(owner[state])


def test_homonymous_states_on_distinct_blocks_cannot_form_one_pointwise_body():
    program = Program("homonymous")
    model = Model("same", frame=Cartesian2D())
    state = model.state("U", components=("rho", "m"))
    left = _state(program, block="left", model=model, state=state)
    right = _state(program, block="right", model=model, state=state)
    expression = (left.n["rho"] + right.n["rho"], left.n["m"])
    before = program._ir_hash()
    with pytest.raises(ValueError, match="same block support"):
        program.value("mixed", expression)
    assert program._ir_hash() == before


def test_foreign_program_with_matching_names_cannot_be_captured():
    first = Program("identical_name")
    second = Program("identical_name")
    left = _state(first)
    right = _state(second)
    expression = (left.n["rho"] + right.n["rho"], left.n["m"])
    before = first._ir_hash()
    with pytest.raises(ValueError, match="different Program"):
        first.value("mixed", expression)
    assert first._ir_hash() == before


def test_method_coefficient_can_multiply_a_selected_component():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program

    model = Model("coefficient_model", frame=Cartesian2D())
    state = model.state("U", components=("rho", "m"))
    program = Program("coefficient_body")
    block, _ = state_refs(program, "fluid", model=model, state=state)
    q = program.state(block[state])
    expression = (q.n["rho"] + program.dt * q.n["m"], q.n["m"] * program.dt)
    value = program.value("weighted", expression, at=q.next.point)
    assert value.inputs == (q.n,)
    assert value.op == "pointwise_expression"
    assert value.attrs["expression_nodes"][value.attrs["expressions"][1]][0] == "mul"
    program.commit(q.next, value)
    lowered, _ = lower_and_validate(model, facade=model)
    generated = emit_cpp_program(program, model=lowered)
    assert "pointwise_active_mask" in generated


def test_equation_capture_does_not_alias_a_different_newton_seed():
    program = Program("frozen_equation_data")
    q = _state(program)
    old = q.n
    seed_a = program.value("seed_a", 2 * old, at=q.next.point)
    seed_b = program.value("seed_b", 3 * old, at=q.next.point)

    def residual(_, unknown, frozen):
        return (unknown[0] - frozen[0] - 1, unknown[1] - frozen[1])

    for seed in (seed_a, seed_b):
        result = program.solve(
            LocalResidual(residual, seed, captures={"frozen": old}),
            solver=LocalNewton(),
        ).consume(action=FailRun())
        assert result.vtype == "state"

    solves = [value for value in program._values if value.op == "solve_local_nonlinear"]
    assert len(solves) == 2
    assert solves[0].inputs[0] is seed_a
    assert solves[1].inputs[0] is seed_b
    assert solves[0].inputs[1] is old
    assert solves[1].inputs[1] is old
    assert solves[0].attrs["capture_names"] == ("frozen",)
    assert solves[1].attrs["capture_names"] == ("frozen",)
