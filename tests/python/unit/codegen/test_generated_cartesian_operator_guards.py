"""Emitted-C++ contract for Uniform-only Cartesian generated-operator preflights.

These tests author real Programs and inspect ``emit_cpp_program`` output. They do not
read ``program_emit_ops.py`` / ``program_emit_solve.py`` as text.

Top-level ``laplacian`` / ``gradient`` / ``divergence`` may themselves be unqualified;
Uniform emit must still name the unique authenticated dataflow owner. Matrix-free
apply graphs stay under the outer ``matrix_free_stencil`` preflight.
"""

from __future__ import annotations

import json
import re

import pytest

from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.frames import Cartesian2D
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.physics import Model
from pops.solvers.krylov import BiCGStab
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, LocalResidual, Program

from typed_program_support import state_refs, typed_state


CARTESIAN_GUARD = "ctx.require_cartesian_generated_operator"


def _guard_stmt(owner: int, label: str) -> str:
    return "%s(%d, %s);" % (CARTESIAN_GUARD, owner, json.dumps(label))


def _owner_index(program: Program, block) -> int:
    index = program._block_indices()[block]
    assert index > 0
    return index


def _step_body(source: str) -> str:
    marker = "state->step = [ctx_owner = state->ctx_owner](double dt)"
    assert marker in source
    return source.split(marker, 1)[1]


def _apply_fn_regions(source: str) -> list[str]:
    regions: list[str] = []
    needle = "pops::ApplyFn<"
    cursor = 0
    while True:
        start = source.find(needle, cursor)
        if start < 0:
            return regions
        brace = source.find("{", start)
        assert brace >= 0
        depth = 0
        end = None
        for offset, char in enumerate(source[brace:], brace):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = offset + 1
                    break
        assert end is not None
        regions.append(source[start:end])
        cursor = end


def _assert_guard_before(source: str, owner: int, label: str, evaluation: str) -> None:
    guard = _guard_stmt(owner, label)
    assert source.count(guard) == 1
    at = source.index(guard)
    assert source.index(evaluation, at) > at


def _pin_leading_block(program: Program, name: str = "first", **state_kwargs):
    block, state = state_refs(program, name, **state_kwargs)
    temporal = program.state(block[state])
    program.commit(
        temporal.next,
        program.value("%s_next" % name, 1 * temporal.n, at=temporal.next.point),
    )
    return block, state


def _canonical_emit_model(model):
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert source_module is model.module
    pack = getattr(emit_model, "_auxiliary_provider_pack", None)
    assert type(pack).__name__ == "ProviderPack"
    return emit_model


def _where_program():
    program = Program("cartesian_where_guards")
    _pin_leading_block(program)
    state = typed_state(program, "second")
    half = program.value("half", 0.5 * state)
    mask = program.cell_ge(state, 0.0, name="mask")
    selected = program.where(mask, state, half, name="selected")
    endpoint = typed_state(program, "second", state_name="U").next
    program.commit(endpoint, program.value("selected_next", 1 * selected, at=endpoint.point))
    return program, mask, selected


def _rotation_model():
    from pops.lib.models import author_electrostatic_lorentz
    from pops.math import sqrt
    from pops.params import ConstParam
    from pops.physics._facade import Model as FacadeModel

    model = FacadeModel("cartesian_guard_rotation")
    rho, mx, my = model.conservative_vars("rho", "mx", "my")
    cs2 = model.value(model.param(ConstParam("cs2", 0.5)))
    velocity_x = model.primitive("u", mx / rho)
    velocity_y = model.primitive("v", my / rho)
    pressure = model.primitive("p", cs2 * rho)
    model.primitive_vars(rho=rho, u=velocity_x, v=velocity_y)
    model.conservative_from([rho, rho * velocity_x, rho * velocity_y])
    model.flux(
        x=[mx, mx * velocity_x + pressure, my * velocity_x],
        y=[my, mx * velocity_y, my * velocity_y + pressure],
    )
    sound = sqrt(cs2)
    model.eigenvalues(
        x=[velocity_x - sound, velocity_x, velocity_x + sound],
        y=[velocity_y - sound, velocity_y, velocity_y + sound],
    )
    model.elliptic_rhs(rho)
    model.aux("grad_x")
    model.aux("grad_y")
    for component in ("magnetic_x", "magnetic_y", "magnetic_z"):
        model.aux(component)
    author_electrostatic_lorentz(
        model,
        magnetic_components=("magnetic_x", "magnetic_y", "magnetic_z"),
        dimension=2,
    )
    return model


def _linear_handle(model):
    from pops.model import OperatorHandle

    registry = model.operator_registry()
    operator = registry.operators_of_kind("local_linear_operator")[0]
    return OperatorHandle(
        operator.name,
        kind=operator.kind,
        owner=registry.owner_path,
        signature=operator.signature,
    )


def _condensed_rhs_program():
    emit_model = _canonical_emit_model(_rotation_model())
    program = Program("cartesian_condensed_rhs_guard")._bind_operators(emit_model)
    _pin_leading_block(program, "dummy", model=emit_model)
    block, state = state_refs(program, "blk", model=emit_model)
    temporal = program.state(block[state])
    rhs = program.condensed_rhs(
        program.scalar_field("condensed_rhs_storage"),
        program.history("blk.tensor_phi", lag=1, ncomp=1, block=block),
        temporal.n,
        linear_operator=_linear_handle(emit_model),
        subset=(1, 2),
        th_dt=1.0,
        g=1.0,
    )
    reconstructed = program.condensed_reconstruct(
        state=temporal.n,
        phi=rhs,
        linear_operator=_linear_handle(emit_model),
        subset=(1, 2),
        th_dt=1.0,
        c_rho=0,
    )
    program.commit(
        temporal.next,
        program.value("next", reconstructed, at=temporal.next.point),
    )
    return program, emit_model, rhs


def _transform_program():
    model = Model("cartesian_guard_transform", frame=Cartesian2D())
    state = model.state("U", components=("q",))
    transform = model.local_transform(
        "bounded_shift",
        (state[0] + 1.0,),
        valid_if=state[0] > 0.0,
    )
    emit_model = _canonical_emit_model(model)
    program = Program("cartesian_guard_transform")
    _pin_leading_block(program, model=model, state=state)
    block, _ = state_refs(program, "fluid", model=model, state=state)
    temporal = program.state(block[state])
    candidate = program.value("candidate", temporal.n, at=temporal.next.point)
    transformed = program.transform(
        candidate, transform=transform, name="transformed_candidate"
    )
    program.commit(temporal.next, transformed)
    return program, emit_model


def _passive_model(name: str):
    from pops.physics._facade import Model as FacadeModel

    model = FacadeModel(name)
    (rho,) = model.conservative_vars("rho")
    model.primitive_vars(rho)
    model.conservative_from([rho])
    model.flux(x=[0.0 * rho], y=[0.0 * rho])
    model.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    return model


def _nonlinear_program():
    emit_model = _canonical_emit_model(_passive_model("cartesian_guard_nonlinear"))
    program = Program("cartesian_guard_nonlinear")._bind_operators(emit_model)
    _pin_leading_block(program, model=emit_model)
    block, state = state_refs(program, "second", model=emit_model)
    temporal = program.state(block[state])

    def residual(builder, iterate, guess):
        return builder.value("residual", iterate - guess)

    guess = program.value("guess", temporal.n, at=temporal.next.point)
    solved = program.solve(
        LocalResidual(residual, guess),
        name="solved",
        solver=LocalNewton(tolerance=1e-12, max_iterations=20),
    ).consume(action=FailRun())
    program.commit(temporal.next, solved)
    return program, emit_model


def _top_level_stencil_program():
    program = Program("cartesian_top_level_stencils")
    _pin_leading_block(program)
    block, state = state_refs(program, "second")
    temporal = program.state(block[state])
    phi = program.history("second.phi", lag=1, ncomp=1, block=block)
    lap = program.laplacian(program.scalar_field("lap"), phi)
    gradient = program.gradient(program.scalar_field("grad", ncomp=2), phi)
    divergence = program.divergence(program.scalar_field("div"), gradient)
    # Top-level stencil values are executable only when consumed by a side effect.  Keep all three
    # generated kernels live so the sealed resource plan and the emitted body traverse the same
    # authenticated graph after dead declarations are filtered.
    program.fill_boundary(lap)
    program.fill_boundary(divergence)
    program.commit(
        temporal.next,
        program.value("next", 1 * temporal.n, at=temporal.next.point),
    )
    return program, phi


def _helmholtz_program(*, kind: str = "laplacian"):
    program = Program("cartesian_matrix_free_stencil_%s" % kind)
    _pin_leading_block(program)
    state = typed_state(program, "second")
    operator = program.matrix_free_operator("helmholtz")

    def apply_laplacian(builder, _out, value):
        laplacian = builder.scalar_field("lap")
        builder.laplacian(laplacian, value)
        return value - 0.1 * laplacian

    def apply_divgrad(builder, _out, value):
        gradient = builder.scalar_field("grad", ncomp=2)
        builder.gradient(gradient, value)
        divergence = builder.scalar_field("div")
        builder.divergence(divergence, gradient)
        return value - 0.1 * divergence

    program.set_apply(
        operator, apply_laplacian if kind == "laplacian" else apply_divgrad
    )
    solved = program.solve(
        LinearProblem(
            operator,
            state,
            properties=LinearOperatorProperties.general(),
            nullspace=None,
        ),
        solver=BiCGStab(max_iter=8, rel_tol=1e-8),
    ).consume(action=FailRun())
    endpoint = typed_state(program, "second", state_name="U").next
    program.commit(endpoint, program.value("phi_next", solved, at=endpoint.point))
    solve = next(value for value in program._values if value.op == "solve_linear")
    return program, solve


def _hierarchy_direct_program():
    from pops.solvers import CompositeTensorFAC, Hierarchy

    emit_model = _canonical_emit_model(_rotation_model())
    program = Program("cartesian_hierarchy_direct")._bind_operators(emit_model)
    _pin_leading_block(program, "dummy", model=emit_model)
    block, state = state_refs(program, "blk", model=emit_model)
    temporal = program.state(block[state])
    current = temporal.n
    linear = _linear_handle(emit_model)
    coefficients = program.condensed_coeffs(
        "tensor_coefficients",
        state=current,
        linear_operator=linear,
        subset=(1, 2),
        c=1.0,
        th_dt=1.0,
        c_rho=0,
    )
    previous = program.history("blk.tensor_phi", lag=1, ncomp=1, block=block)
    rhs = program.condensed_rhs(
        program.scalar_field("tensor_rhs"),
        previous,
        current,
        linear_operator=linear,
        subset=(1, 2),
        th_dt=1.0,
        g=1.0,
    )
    operator = program.matrix_free_operator("tensor_operator", scope=Hierarchy())

    def apply(builder, _out, value):
        laplacian = builder.scalar_field("tensor_laplacian")
        return -1 * builder.apply_laplacian_coeff(laplacian, value, coefficients)

    program.set_apply(operator, apply)
    phi = program.solve(
        LinearProblem(
            operator,
            rhs,
            initial_guess=previous,
            scope=Hierarchy(),
            nullspace=None,
        ),
        solver=CompositeTensorFAC(),
        name="phi",
    ).consume(action=FailRun())
    program.store_history("blk.tensor_phi", phi)
    program.commit(
        temporal.next,
        program.value("next", 1 * current, at=temporal.next.point),
    )
    return program, emit_model


def test_uniform_cell_compare_and_where_guard_exact_nonzero_owner() -> None:
    program, mask, selected = _where_program()
    owner = _owner_index(program, mask.block)
    assert selected.block is mask.block
    source = emit_cpp_program(program)
    step = _step_body(source)

    assert step.count(CARTESIAN_GUARD + "(") == 2
    _assert_guard_before(step, owner, "cell_compare", "ctx.scalar_scratch(")
    _assert_guard_before(step, owner, "where", "ctx.scratch_state(")
    assert _guard_stmt(0, "cell_compare") not in source
    assert _guard_stmt(0, "where") not in source


def test_amr_cell_compare_and_where_omit_uniform_cartesian_guard() -> None:
    program, _, _ = _where_program()
    source = emit_cpp_program(program, target="amr_system")
    assert CARTESIAN_GUARD not in source
    assert "ctx.scalar_scratch(" in source
    assert "ctx.scratch_state(" in source


def test_uniform_condensed_rhs_guard_exact_nonzero_owner() -> None:
    program, emit_model, rhs = _condensed_rhs_program()
    owner = _owner_index(program, rhs.block)
    source = emit_cpp_program(program, model=emit_model)
    step = _step_body(source)

    assert step.count(CARTESIAN_GUARD + "(") == 1
    _assert_guard_before(step, owner, "condensed_rhs", "ctx.laplacian(")
    assert _guard_stmt(0, "condensed_rhs") not in source
    begin_step = source.index("state->step = [ctx_owner = state->ctx_owner](double dt)")
    assert source.count("ctx.prepare_scalar_scratch(") >= 3
    assert source.count("ctx.scalar_scratch(") >= 3
    assert source.index("ctx.prepare_scalar_scratch(") < begin_step
    assert "ctx.scalar_scratch(" in step
    assert "ctx.alloc_scalar_field(" not in source
    assert "ctx.scratch_state_like(" not in source
    assert re.search(
        r"ctx\.prepare_scalar_scratch\(\d+, 0, %d, 1, 1\);" % owner,
        source,
    )
    assert re.search(
        r"ctx\.scalar_scratch\(\d+, 0, ctx\.state\(%d\), 1, 1\)" % owner,
        step,
    )
    preparations = [
        line.strip()
        for line in source.splitlines()
        if "ctx.prepare_" in line
    ]
    assert len(preparations) == len(set(preparations))


def test_amr_condensed_rhs_omits_uniform_cartesian_guard() -> None:
    program, emit_model, _ = _condensed_rhs_program()
    source = emit_cpp_program(program, model=emit_model, target="amr_system")
    assert CARTESIAN_GUARD not in source
    assert "ctx.laplacian(" in source


def test_supported_local_transform_has_no_cartesian_guard() -> None:
    program, emit_model = _transform_program()
    source = emit_cpp_program(program, model=emit_model)
    assert CARTESIAN_GUARD not in source
    assert "ctx.pointwise_active_mask(" in source
    assert "ctx.scratch_state_like(" not in source
    assert "ctx.alloc_scalar_field(" not in source
    assert "ctx.prepare_state_scratch(" in source
    assert "ctx.scratch_state(" in _step_body(source)
    assert source.index("ctx.prepare_state_scratch(") < source.index("ctx.begin_step(")
    amr = emit_cpp_program(program, model=emit_model, target="amr_system")
    assert CARTESIAN_GUARD not in amr
    assert "ctx.scratch_state_like(" not in amr
    assert "ctx.alloc_scalar_field(" not in amr


def test_supported_solve_local_nonlinear_has_no_cartesian_guard() -> None:
    program, emit_model = _nonlinear_program()
    source = emit_cpp_program(program, model=emit_model)
    assert CARTESIAN_GUARD not in source
    assert "ctx.pointwise_active_mask(" in source
    amr = emit_cpp_program(program, model=emit_model, target="amr_system")
    assert CARTESIAN_GUARD not in amr


def test_uniform_matrix_free_stencil_guard_exact_nonzero_owner() -> None:
    for kind in ("laplacian", "divgrad"):
        program, solve = _helmholtz_program(kind=kind)
        owner = _owner_index(program, solve.block)
        source = emit_cpp_program(program)
        step = _step_body(source)

        assert step.count(CARTESIAN_GUARD + "(") == 1
        _assert_guard_before(step, owner, "matrix_free_stencil", "ctx.solve_prepared_linear(")
        assert _guard_stmt(0, "matrix_free_stencil") not in source
        assert _guard_stmt(owner, "laplacian") not in source
        assert _guard_stmt(owner, "gradient") not in source
        assert _guard_stmt(owner, "divergence") not in source


def test_matrix_free_stencil_guard_is_outside_apply_fn() -> None:
    program, solve = _helmholtz_program()
    owner = _owner_index(program, solve.block)
    source = emit_cpp_program(program)
    regions = _apply_fn_regions(source)
    assert regions
    for region in regions:
        assert CARTESIAN_GUARD not in region
        assert "ctx.laplacian(" in region
    assert _guard_stmt(owner, "matrix_free_stencil") in _step_body(source)


def test_amr_matrix_free_stencil_omits_uniform_cartesian_guard() -> None:
    program, _ = _helmholtz_program()
    source = emit_cpp_program(program, target="amr_system")
    assert CARTESIAN_GUARD not in source
    assert json.dumps("matrix_free_stencil") not in source
    assert "ctx.solve_prepared_linear(" in source
    assert "ctx.laplacian(" in source


def test_amr_direct_hierarchy_provider_omits_matrix_free_stencil_guard() -> None:
    program, emit_model = _hierarchy_direct_program()
    source = emit_cpp_program(program, model=emit_model, target="amr_system")
    assert CARTESIAN_GUARD not in source
    assert json.dumps("matrix_free_stencil") not in source
    assert "ctx.solve_hierarchy_tensor(" in source


def test_uniform_top_level_stencils_guard_exact_nonzero_owner() -> None:
    program, phi = _top_level_stencil_program()
    owner = _owner_index(program, phi.block)
    source = emit_cpp_program(program)
    step = _step_body(source)

    assert step.count(CARTESIAN_GUARD + "(") == 3
    _assert_guard_before(step, owner, "laplacian", "ctx.laplacian(")
    _assert_guard_before(step, owner, "gradient", "ctx.gradient(")
    _assert_guard_before(step, owner, "divergence", "ctx.divergence(")
    assert _guard_stmt(0, "laplacian") not in source
    assert _guard_stmt(0, "gradient") not in source
    assert _guard_stmt(0, "divergence") not in source


def test_amr_top_level_stencils_omit_uniform_cartesian_guard() -> None:
    program, _ = _top_level_stencil_program()
    source = emit_cpp_program(program, target="amr_system")
    assert CARTESIAN_GUARD not in source
    assert "ctx.laplacian(" in source
    assert "ctx.gradient(" in source
    assert "ctx.divergence(" in source


def test_top_level_stencil_refuses_missing_owner() -> None:
    program = Program("cartesian_top_level_missing_owner")
    _pin_leading_block(program)
    _pin_leading_block(program, "second")
    scratch = program.scalar_field("buf")
    program.fill_boundary(program.laplacian(scratch, scratch))
    with pytest.raises(ValueError, match="no unique authenticated owner block"):
        emit_cpp_program(program)


def test_top_level_stencil_refuses_conflicting_owners() -> None:
    program = Program("cartesian_top_level_conflicting_owners")
    first, _ = _pin_leading_block(program)
    second, _ = _pin_leading_block(program, "second")
    program.fill_boundary(program.laplacian(
        program.history("first.phi", lag=1, ncomp=1, block=first),
        program.history("second.phi", lag=1, ncomp=1, block=second),
    ))
    with pytest.raises(ValueError, match="conflicting owner blocks"):
        emit_cpp_program(program)
