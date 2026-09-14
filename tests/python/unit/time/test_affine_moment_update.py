"""All-moment Cayley authoring keeps solved means and qualifies AMR dependencies."""
from __future__ import annotations

import re

import pytest

import pops
from pops.codegen._resolution import CapabilityResolutionError, _resolve_amr_program
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import ddt, div
from pops.moments import moment_names
from pops.params import RuntimeParam
from pops.runtime.amr_program_support import AMRProgramSupportContext
from pops.time._program.affine_moments import validate_affine_moment_prefix


def _program(*, transported=False, recorded=False, runtime_rotation=False, history_use=None,
             transport_order=0, affine=True):
    frame = Rectangle("moment_square", (0, 0), (1, 1)).frame(Cartesian2D())
    model = pops.Model("affine_moments", frame=frame)
    state = model.state("U", components=tuple(moment_names(4)))
    flux = model.flux("transport", state=state, frame=frame,
                      components={axis: tuple(state) for axis in frame.axes},
                      waves={axis: (1.0,) * 15 for axis in frame.axes})
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    matrix = [[0.0] * 15 for _ in range(15)]
    omega = model.value(model.param(RuntimeParam("omega", default=1e12))) if runtime_rotation else 1e12
    matrix[1][5], matrix[5][1] = omega, -omega
    rotation = model.operator("rotation", returns=model.local_linear_operator(
        "rotation", on=state, matrix=tuple(tuple(row) for row in matrix)))
    case = pops.Case("affine_moment_case")
    block = case.block("moments", model=model)
    program = pops.Program("source_first")
    q = program.state(block[state])
    old = q.n
    if transported:
        if recorded:
            old = program.range(q.n, 1, lambda builder, value: builder.value(
                "transported", value + builder.dt * rate(value), at=value.point))
        else:
            old = program.value("transported", q.n + program.dt * rate(q.n), at=q.n.point)
    mean = old
    if history_use is not None:
        from pops.linalg import LinearProblem
        from pops.solvers import CompositeTensorFAC, Hierarchy
        from pops.time import FailRun

        history = program.history("moments.phi", lag=1, ncomp=1, block=block)
        coefficients = program.condensed_coeffs(
            state=old, linear_operator=rotation, subset=(1, 5), c=program.dt * program.dt,
            th_dt=program.dt / 2)
        rhs = program.condensed_rhs(
            program.scalar_field("charge_rhs"), history if history_use == "rhs" else None,
            old, linear_operator=rotation, subset=(1, 5),
            th_dt=program.dt / 2, g=program.dt / 2,
            charge_component=None if history_use == "rhs" else 0)
        operator = program.matrix_free_operator("elliptic", scope=Hierarchy())
        program.set_apply(operator, lambda builder, _out, value:
            -builder.apply_laplacian_coeff(builder.scalar_field("action"), value, coefficients))
        potential = program.solve(LinearProblem(
            operator, rhs, initial_guess=history, scope=Hierarchy(), nullspace=None),
            solver=CompositeTensorFAC()).consume(action=FailRun())
        working = program.value("working_mean", 1 * old, at=old.point)
        mean = program.condensed_reconstruct(
            state=working, phi=history if history_use == "reconstruction" else potential,
            linear_operator=rotation, subset=(1, 5), th_dt=program.dt / 2)
    # This source-only witness supplies unchanged first moments. Runtime tests
    # supply the independently computed coupled endpoint for nonzero electric force.
    updated = program.affine_moment_update(
        old, mean, linear_operator=rotation, theta_dt=program.dt / 2, name="source") if affine else mean
    candidate = updated
    if transport_order:
        k0 = rate(updated)
        if transport_order == 2:
            predictor = program.value("predictor", updated + program.dt * k0,
                                      at=program.stage("predictor", c=1))
            candidate = updated + (program.dt / 2) * (k0 + rate(predictor))
        else:
            candidate = updated + program.dt * k0
    program.commit(q.next, program.value("accepted", candidate, at=q.next.point))
    return model, program, updated


def test_common_affine_moment_update_is_typed_and_emits_native_kernel():
    model, program, update = _program()
    assert update.op == "affine_moment_update"
    assert update.attrs["order"] == 4
    assert update.inputs[0] is update.inputs[1]
    emit_model, _ = lower_and_validate(model, facade=model)
    source = emit_cpp_program(program, model=emit_model)
    assert "affine_velocity_push_forward<4>" in source
    assert "pops/numerics/moments/affine_velocity.hpp" in source
    assert "jyx == -jxy" in source
    assert "ctx.pointwise_status_max(" in source
    assert "ctx.apply_projection" not in source


@pytest.mark.parametrize("recorded", (False, True), ids=("direct", "recorded_body"))
def test_refined_amr_accepts_source_prefix_and_refuses_transported_moments(recorded):
    _, prefix, _ = _program()
    context = AMRProgramSupportContext(
        hierarchy_level_count=2, frozen_hierarchy=True,
        shared_block_interfaces=False, field_routes_validated=True)
    assert _resolve_amr_program("amr", prefix, context=context)["status"] == "proven"
    validate_affine_moment_prefix(prefix)
    model, transported, _ = _program(transported=True, recorded=recorded)
    with pytest.raises(CapabilityResolutionError, match="source-first prefix"):
        _resolve_amr_program("amr", transported, context=context)
    emit_model, _ = lower_and_validate(model, facade=model)
    with pytest.raises(ValueError, match="source-first prefix"):
        emit_cpp_program(transported, model=emit_model, target="amr_system")


def test_affine_moment_rotation_runtime_parameter_has_metadata_and_per_block_read():
    model, program, _ = _program(runtime_rotation=True)
    emit_model, _ = lower_and_validate(model, facade=model)
    source = emit_cpp_program(program, model=emit_model)
    assert "ctx.program_params(0)" in source
    assert "const pops::Real jxy = params.get(0)" in source
    assert 'case 0: return "omega"' in source


def test_amr_affine_result_has_an_authenticated_level_block_scratch_owner():
    model, program, update = _program()
    emit_model, _ = lower_and_validate(model, facade=model)
    source = emit_cpp_program(program, model=emit_model, target="amr_system")
    assert f"transform_state_resource_{update.id} = &ctx.scratch_state({update.id}, 0, ctx.state(0))" in source
    assert "ctx.scratch_state_like(" not in source
    assert "ctx.pointwise_level_status_max(" in source


@pytest.mark.parametrize("transport_order", (1, 2), ids=("FE", "SSPRK2"))
@pytest.mark.parametrize("source_kind", ("affine", "condensed", "none"))
def test_amr_source_descendants_capture_every_exact_transport_input(source_kind, transport_order):
    model, program, _ = _program(
        transport_order=transport_order, affine=source_kind == "affine",
        history_use="guess" if source_kind == "condensed" else None)
    emit_model, _ = lower_and_validate(model, facade=model)
    source = emit_cpp_program(program, model=emit_model, target="amr_system")
    count = transport_order if source_kind != "none" else 0
    captured = set(re.findall(r"auto (rhs_input_trace_\d+) = ctx.capture_rhs_input_trace\(", source))
    assert len(captured) == count
    if count:
        # Every generated transport invocation carries its own unforgeable native
        # token, including SSPRK2's source-descended predictor at the later stage.
        assert set(re.findall(r"&(rhs_input_trace_\d+)", source)) == captured
    if source_kind != "condensed":
        uniform = emit_cpp_program(program, model=emit_model, target="system")
        assert "ctx.capture_rhs_input_trace(" not in uniform


def test_scalar_history_is_qualified_only_as_an_elliptic_initial_guess():
    _, program, _ = _program(history_use="guess")
    validate_affine_moment_prefix(program)
    context = AMRProgramSupportContext(
        hierarchy_level_count=2, frozen_hierarchy=True,
        shared_block_interfaces=False, field_routes_validated=True)
    assert _resolve_amr_program("amr", program, context=context)["status"] == "proven"
    for use in ("rhs", "reconstruction"):
        _, contaminated, _ = _program(history_use=use)
        with pytest.raises(ValueError, match="source-first prefix"):
            validate_affine_moment_prefix(contaminated)


def test_affine_moment_update_rejects_incomplete_basis_and_foreign_mean():
    _, program, update = _program()
    with pytest.raises(ValueError, match="complete canonical"):
        program.affine_moment_update(
            update.inputs[0], update.inputs[1], linear_operator=None,
            theta_dt=program.dt / 2, order=3)
    _, foreign, foreign_update = _program()
    with pytest.raises(ValueError, match="Program|program"):
        program.affine_moment_update(
            update.inputs[0], foreign_update.inputs[1], linear_operator=None,
            theta_dt=program.dt / 2)
