"""One scientific body on physical declarations and native temporal values."""
import pytest

from pops.frames import Cartesian2D
from pops.physics import Model
from pops.time import Program
from typed_program_support import state_refs


def fixture(*, with_source=False):
    model = Model("expression_body", frame=Cartesian2D())
    state = model.state("U", components=("rho", "m"))
    if with_source:
        model.source("square", on=state, value=(state[0] ** 2, state[1] ** 2))
    program = Program("expression_program")
    block, _ = state_refs(program, "fluid", model=model, state=state)
    temporal = program.state(block[state])
    return model, state, program, temporal


def body(q):
    return (q["rho"] + q["m"] * q["m"] / q["rho"], q["m"] / q["rho"])


def test_same_body_materializes_components_at_a_stage():
    model, state, program, q = fixture()
    assert len(body(state)) == 2
    value = program.value("nonlinear", body(q.n), at=q.next.point)
    assert value.op == "pointwise_expression"
    assert value.inputs == (q.n,)
    assert value.space == q.n.space
    program.commit(q.next, value)
    assert program.validate()


def test_whole_field_product_and_quotient_are_pointwise():
    _, _, program, q = fixture()
    value = program.value("square", q.n * q.n / q.n, at=q.next.point)
    assert value.op == "pointwise_expression"
    assert len(value.attrs["expressions"]) == 2


def test_foreign_program_component_refuses_before_publication():
    _, _, program, q = fixture()
    _, _, foreign, other = fixture()
    _ = q.n, other.n
    before = program._ir_hash()
    with pytest.raises(ValueError, match="different Program"):
        program.value("foreign", (q.n[0] + other.n[0], q.n[1]))
    assert program._ir_hash() == before


def test_component_selection_rejects_invalid_keys():
    _, _, _, q = fixture()
    for index in (-1, 2):
        with pytest.raises(IndexError):
            q.n[index]
    with pytest.raises((KeyError, ValueError)):
        q.n["absent"]
    with pytest.raises(TypeError):
        q.n[True]


def emit(model, program):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    lowered, _ = lower_and_validate(model, facade=model)
    return emit_cpp_program(program, model=lowered)


def test_pointwise_expression_emits_native_component_reads():
    model, _, program, q = fixture()
    value = program.value("nonlinear", body(q.n), at=q.next.point)
    program.commit(q.next, value)
    source = emit(model, program)
    assert "pops::for_each_cell" in source
    assert "u0A(index, 1)" in source
    assert "const pops::Real cse" in source
    assert "expression_has_mask_" in source
    assert "ctx.scratch_state" in source


def test_body_reuses_a_materialized_stage_and_method_coefficients():
    model, _, program, q = fixture()
    from pops.time import StagePoint, TimePoint
    stage = q.stage("intermediate", point=StagePoint("intermediate", {"main": TimePoint(q.clock, 1)}))
    program.value(stage, body(q.n))
    value = program.value("weighted", (stage[0] * program.dt,
                                      program.dt * stage[1] + (program.dt - stage[0])),
                          at=q.next.point)
    assert value.inputs == (stage._as_value(),)
    program.commit(q.next, value)
    assert "dt" in emit(model, program)


def test_failed_stage_expression_does_not_publish_partial_value():
    from pops.time import StagePoint, TimePoint
    model, state, program, q = fixture()
    other_block, _ = state_refs(program, "other", model=model, state=state)
    other = program.state(other_block[state])
    stage = q.stage("invalid", point=StagePoint("invalid", {"main": TimePoint(q.clock, 1)}))
    expression = body(other.n)
    before = program._ir_hash()
    with pytest.raises(ValueError, match="block"):
        program.value(stage, expression)
    assert program._ir_hash() == before


def test_local_nonlinear_expression_captures_equation_data_separately():
    from pops.solvers.nonlinear import LocalNewton
    from pops.time import LocalResidual, FailRun
    model, _, program, q = fixture()
    old = q.n
    seed = program.value("seed", 2 * old, at=q.next.point)

    def residual(p, unknown, old):
        return (unknown[0] + unknown[0] ** 2 - old[0] - old[0] ** 2 - 1,
                unknown[1] - old[1])

    result = program.solve(LocalResidual(residual, seed, captures={"old": old}),
                           solver=LocalNewton()).consume(action=FailRun())
    program.commit(q.next, result)
    token = next(value for value in program._values if value.op == "solve_local_nonlinear")
    assert token.inputs[0] is seed
    assert token.inputs[1] is old
    assert token.attrs["capture_names"] == ("old",)
    source = emit(model, program)
    assert "Cval0[0]" in source
    assert "std::pow(expression_leaf_" in source
    assert "pops::solve_prepared_local_nonlinear" in source
    assert program.to_graph() is not None


def test_source_in_local_residual_uses_its_actual_frozen_argument():
    from pops.solvers.nonlinear import LocalNewton
    from pops.time import LocalResidual, FailRun
    model, state, program, q = fixture(with_source=True)
    source = model.module.operator_handle("square")
    old = q.n
    seed = program.value("seed", 2 * old, at=q.next.point)

    def residual(p, unknown, old):
        return p.value("r", unknown - p.source(source, old), at=unknown.point)

    result = program.solve(LocalResidual(residual, seed, captures={"old": old}),
                           solver=LocalNewton()).consume(action=FailRun())
    program.commit(q.next, result)
    emitted = emit(model, program)
    assert " = Cval0[0];" in emitted
    assert "residual_value_" in emitted


def test_hidden_nonfinite_intermediate_is_checked_before_publication():
    from pops.math import minimum
    model, _, program, q = fixture()
    zero = q.n[0] - q.n[0]
    value = program.value("masked_nan", (minimum(zero / zero, q.n[0]), q.n[1]),
                          at=q.next.point)
    program.commit(q.next, value)
    source = emit(model, program)
    assert "Kokkos::fmin" in source
    assert "!Kokkos::isfinite(cse" in source
    assert "pointwise_status_max" in source
    assert "non-finite scientific expression input or intermediate" in source


def test_shared_expression_dag_is_not_expanded_into_a_tree():
    model, _, program, q = fixture()
    expression = q.n[0]
    for _ in range(32):
        expression = expression + expression
    value = program.value("shared", (expression, q.n[1]), at=q.next.point)
    assert len(value.attrs["expression_nodes"]) == 34
    assert len(str(value.attrs["expression_nodes"])) < 2000
    program.commit(q.next, value)
    source = emit(model, program)
    assert source.count("const pops::Real cse") == 32
    assert len(source) < 30000


def test_local_residual_checks_intermediates_hidden_by_minimum():
    from pops.math import minimum
    from pops.solvers.nonlinear import LocalNewton
    from pops.time import FailRun, LocalResidual
    model, _, program, q = fixture()

    def residual(p, unknown):
        zero = unknown[0] - unknown[0]
        return (minimum(zero / zero, unknown[0]), unknown[1])

    candidate = program.solve(LocalResidual(residual, q.n), solver=LocalNewton()).consume(action=FailRun())
    program.commit(q.next, program.value("endpoint", candidate, at=q.next.point))
    source = emit(model, program)
    assert "!Kokkos::isfinite(cse" in source
    assert "rout[0] = std::numeric_limits<pops::Real>::quiet_NaN();" in source
