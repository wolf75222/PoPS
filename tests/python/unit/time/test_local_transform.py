"""Generic explicit local-transform authoring and Program code generation."""
from __future__ import annotations

import numpy as np
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.frames import Cartesian2D
from pops.physics import Model
from pops.time import Program

from typed_program_support import state_refs


def _transform_program():
    model = Model("local_transform_model", frame=Cartesian2D())
    state = model.state("U", components=("q",))
    cached = model.module
    transform = model.local_transform(
        "bounded_shift",
        (state[0] + 1.0,),
        valid_if=state[0] > 0.0,
    )
    assert model.module is not cached

    program = Program("local_transform_program")
    block, _ = state_refs(program, "fluid", model=model, state=state)
    temporal = program.state(block[state])
    candidate = program.value("candidate", temporal.n, at=temporal.next.point)
    transformed = program.transform(
        candidate, transform=transform, name="transformed_candidate")
    program.commit(temporal.next, transformed)
    return model, state, transform, program


def test_local_transform_is_typed_explicit_and_fresh() -> None:
    model, state, transform, program = _transform_program()
    assert transform.kind == "local_transform"
    values = [value for value in program._values if value.op == "local_transform"]
    assert len(values) == 1
    assert values[0].inputs[0].id != values[0].id
    assert values[0].attrs["transform"] == "bounded_shift"
    assert not any(value.op == "project" for value in program._values)

    assert np.array_equal(
        model.local_transform_value("bounded_shift", np.array([[[2.0, 3.0]]])),
        np.array([[[3.0, 4.0]]]),
    )
    lowered = model.module.to_dsl()
    assert np.array_equal(
        lowered.local_transform_value("bounded_shift", np.array([[[2.0, 3.0]]])),
        np.array([[[3.0, 4.0]]]),
    )
    with pytest.raises(ValueError, match="outside its domain"):
        model.local_transform_value("bounded_shift", np.array([[[-1.0]]]))
    with pytest.raises(FloatingPointError, match="non-finite state"):
        model.local_transform_value("bounded_shift", np.array([[[np.nan]]]))


def test_local_transform_program_emits_one_collective_fail_closed_kernel() -> None:
    model, _, _, program = _transform_program()
    source = emit_cpp_program(program, model=model)
    assert source.count("transform_failed_") >= 1
    assert "ctx.scratch_state_like(" in source
    install_prelude, step_body = source.split("ctx.install([=](double dt)", 1)
    assert "transform_state_resource_" in install_prelude
    assert "transform_status_resource_" in install_prelude
    assert "ctx.scratch_state_like(" not in step_body
    assert "ctx.alloc_scalar_field(1, 0)" not in step_body
    assert "require_cartesian_generated_operator(0, \"local_transform\")" not in source
    assert "ctx.pointwise_active_mask(0," in source
    assert "transform_has_active_mask_" in source
    assert "outA(i, j, 0) = u" in source
    assert "ctx.pointwise_status_max(0," in source
    assert "StepAttemptRejected" in source
    assert "Kokkos::isfinite" in source
    assert "ctx.apply_projection" not in source

    amr_source = emit_cpp_program(program, model=model, target="amr_system")
    assert "if (ctx.nlev() > 1)" in amr_source
    assert "post-synchronization Program phase" in amr_source
    refresh = amr_source.index("auto _refresh_level_programs")
    refreshed_guard = amr_source.index(
        "_require_local_transform_level_contract();", refresh)
    resource_mutation = amr_source.index("_level_programs->clear();", refresh)
    assert refresh < refreshed_guard < resource_mutation
    assert "ctx.pointwise_active_mask(0," in amr_source
    assert "ctx.pointwise_status_max(0," in amr_source
    assert "inherit_state_metadata" not in amr_source


def test_local_transform_name_collisions_are_rejected() -> None:
    model = Model("local_transform_collision", frame=Cartesian2D())
    state = model.state("U", components=("q",))
    model.local_transform("repair", (state[0],))
    with pytest.raises(ValueError, match="local_transform"):
        model.source("repair", on=state, value=(state[0],))


def test_local_transform_formula_and_domain_are_part_of_module_identity() -> None:
    first = Model("transform_identity", frame=Cartesian2D())
    first_state = first.state("U", components=("q",))
    first.local_transform("repair", (first_state[0] + 1.0,), valid_if=first_state[0] > 0.0)

    changed_formula = Model("transform_identity", frame=Cartesian2D())
    formula_state = changed_formula.state("U", components=("q",))
    changed_formula.local_transform(
        "repair", (formula_state[0] + 2.0,), valid_if=formula_state[0] > 0.0)

    changed_domain = Model("transform_identity", frame=Cartesian2D())
    domain_state = changed_domain.state("U", components=("q",))
    changed_domain.local_transform(
        "repair", (domain_state[0] + 1.0,), valid_if=domain_state[0] > 1.0)

    identities = {
        first.module.module_hash(),
        changed_formula.module.module_hash(),
        changed_domain.module.module_hash(),
    }
    assert len(identities) == 3
