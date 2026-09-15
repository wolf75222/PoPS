"""An exact centered phase is explicit, identity-bearing, and keeps source guards."""
from __future__ import annotations

import pytest

import pops
from pops.codegen._resolution import CapabilityResolutionError, _resolve_amr_program
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_emit_affine_moments import emit_affine_moment_kernel
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.identity.semantic import program_semantic_data, semantic_identity_of
from pops.math import ddt, div
from pops.moments import moment_names
from pops.runtime.amr_program_support import AMRProgramSupportContext


_OMITTED = object()


def _program(policy=_OMITTED, *, order=4, transported=False):
    frame = Rectangle("rotation_square", (0, 0), (1, 1)).frame(Cartesian2D())
    model = pops.Model("affine_rotation_policy", frame=frame)
    state = model.state("U", components=tuple(moment_names(order)))
    count = len(state)
    flux = model.flux("transport", state=state, frame=frame,
                      components={axis: tuple(state) for axis in frame.axes},
                      waves={axis: (1.0,) * count for axis in frame.axes})
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    matrix = [[0.0] * count for _ in range(count)]
    matrix[1][order + 1], matrix[order + 1][1] = -6.28318531e12, 6.28318531e12
    operator = model.operator("rotation", returns=model.local_linear_operator(
        "rotation", on=state, matrix=tuple(tuple(row) for row in matrix)))
    block = pops.Case("affine_rotation_case").block("moments", model=model)
    program = pops.Program("affine_rotation_policy")
    q = program.state(block[state])
    old = program.value("transported", q.n + program.dt * rate(q.n), at=q.n.point) \
        if transported else q.n
    kwargs = {} if policy is _OMITTED else {"rotation": policy}
    update = program.affine_moment_update(
        old, old, linear_operator=operator, theta_dt=program.dt / 2,
        order=order, name="source", **kwargs)
    program.commit(q.next, program.value("accepted", update, at=q.next.point))
    return model, program, update, operator


def _source(model, program, target="system"):
    emit_model, _ = lower_and_validate(model, facade=model)
    return emit_cpp_program(program, model=emit_model, target=target)


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_explicit_default_has_identical_ir_identity_and_generated_bytes(target):
    left_model, left, left_update, _ = _program()
    right_model, right, right_update, _ = _program("cayley")
    assert "rotation" not in left_update.attrs and "rotation" not in right_update.attrs
    assert left._serialize(include_provenance=False) == right._serialize(include_provenance=False)
    assert left._ir_hash() == right._ir_hash()
    assert semantic_identity_of(program=left) == semantic_identity_of(program=right)
    assert _source(left_model, left, target) == _source(right_model, right, target)
    assert "affine_velocity_push_forward<4>" in _source(left_model, left, target)


@pytest.mark.parametrize("order", (1, 2, 3, 4))
def test_exponential_policy_changes_scientific_identity_and_selects_native_map(order):
    _, default, _, _ = _program(order=order)
    model, program, update, _ = _program("exponential", order=order)
    assert update.attrs["rotation"] == "exponential"
    assert program_semantic_data(program) != program_semantic_data(default)
    assert semantic_identity_of(program=program) != semantic_identity_of(program=default)
    assert program._ir_hash() != default._ir_hash()
    source = _source(model, program)
    assert (f"affine_velocity_push_forward<{order}, "
            "pops::moments::AffineVelocityRotation::exponential>") in source
    assert "jyx == -jxy" in source and "jxx == pops::Real(0)" in source
    assert "jyy == pops::Real(0)" in source and "std::isfinite(jxy)" in source
    assert "ctx.pointwise_status_max(" in source
    assert "ctx.apply_projection" not in source


@pytest.mark.parametrize("policy", (None, True, 1, "", "EXPONENTIAL", "exact", [], {}))
def test_invalid_policy_is_refused_without_authoring_residue(policy):
    _, program, update, operator = _program()
    half_dt = program.dt / 2
    before = (program._ir_hash(), program._next_id, tuple(map(id, program._values)))
    with pytest.raises(ValueError, match="rotation must be 'cayley' or 'exponential'"):
        program.affine_moment_update(
            update.inputs[0], update.inputs[1], linear_operator=operator,
            theta_dt=half_dt, rotation=policy)
    assert (program._ir_hash(), program._next_id, tuple(map(id, program._values))) == before


@pytest.mark.parametrize("policy", (None, True, "EXACT", []))
def test_codegen_refuses_invalid_rotation_attr_instead_of_defaulting(policy):
    model, _, update, _ = _program()
    emit_model, _ = lower_and_validate(model, facade=model)
    attrs = dict(update.attrs, rotation=policy)
    with pytest.raises(ValueError, match="rotation must be 'cayley' or 'exponential'"):
        emit_affine_moment_kernel(
            emit_model, attrs, "old", "mean", "out", "status", "active", 0,
            provider_plans=None, consumer_qid="invalid_rotation")


def test_exponential_amr_policy_retains_source_prefix_and_status_contract():
    context = AMRProgramSupportContext(
        hierarchy_level_count=2, frozen_hierarchy=True,
        shared_block_interfaces=False, field_routes_validated=True)
    model, prefix, _, _ = _program("exponential")
    assert _resolve_amr_program("amr", prefix, context=context)["status"] == "proven"
    source = _source(model, prefix, target="amr_system")
    assert "ctx.pointwise_level_status_max(" in source
    assert "AffineVelocityRotation::exponential" in source
    transported_model, transported, _, _ = _program("exponential", transported=True)
    with pytest.raises(CapabilityResolutionError, match="source-first prefix"):
        _resolve_amr_program("amr", transported, context=context)
    with pytest.raises(ValueError, match="source-first prefix"):
        _source(transported_model, transported, target="amr_system")
