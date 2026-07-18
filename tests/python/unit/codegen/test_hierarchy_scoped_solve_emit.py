"""Direct scalar tensor-FAC scheduling on the AMR target."""

from __future__ import annotations
from pops.codegen.program_codegen import emit_cpp_program

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pops._ir.literals import scalar_cpp, scalar_data
from pops.linalg import LinearProblem
from pops.params import ConstParam
from pops.solvers import CompositeTensorFAC, Hierarchy
from pops.time import FailRun, Program

from typed_program_support import state_refs


def _coupled_model(name):
    from pops.math import sqrt
    from pops.lib.models import author_electrostatic_lorentz
    from pops.physics._facade import Model

    model = Model(name)
    rho, mx, my = model.conservative_vars("rho", "mx", "my")
    cs2 = model.value(model.param(ConstParam("cs2", 0.5)))
    u = model.primitive("u", mx / rho)
    v = model.primitive("v", my / rho)
    pressure = model.primitive("p", cs2 * rho)
    model.primitive_vars(rho=rho, u=u, v=v, p=pressure)
    model.conservative_from([rho, rho * u, rho * v])
    model.flux(
        x=[mx, mx * u + pressure, my * u],
        y=[my, mx * v, my * v + pressure],
    )
    sound_speed = sqrt(cs2)
    model.eigenvalues(
        x=[u - sound_speed, u, u + sound_speed],
        y=[v - sound_speed, v, v + sound_speed],
    )
    model.elliptic_rhs(rho)
    model.aux("grad_x")
    model.aux("grad_y")
    model.aux("B_z")
    author_electrostatic_lorentz(model)
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


def _build(solver, *, nullspace=None, gauge=None, properties=None, _return_model=False):
    model = _coupled_model("hierarchy_tensor_model")
    program = Program("hierarchy_tensor_step")._bind_operators(model)

    # Keep the tensor owner at Program block index 1. This makes the emitted/native block binding
    # test catch any regression to the former sys_block(0) shortcut.
    dummy_block, state = state_refs(program, "dummy", model=model)
    dummy_temporal = program.state(dummy_block[state])
    dummy_next = program.value(
        "dummy_next", 1 * dummy_temporal.n, at=dummy_temporal.next.point
    )
    program.commit(dummy_temporal.next, dummy_next)

    block, state = state_refs(program, "blk", model=model)
    temporal = program.state(block[state])
    current = temporal.n
    linear = _linear_handle(model)

    coefficients = program.condensed_coeffs(
        "tensor_coefficients",
        state=current,
        linear_operator=linear,
        subset=(1, 2),
        c=1.0,
        th_dt=1.0,
        c_rho=0,
    )
    phi_previous = program.history("blk.tensor_phi", lag=1, ncomp=1, block=block)
    rhs_storage = program.scalar_field("tensor_rhs")
    rhs = program.condensed_rhs(
        rhs_storage,
        phi_previous,
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
    problem_options = {
        "initial_guess": phi_previous,
        "scope": Hierarchy(),
        "nullspace": nullspace,
        "gauge": gauge,
    }
    if properties is not None:
        problem_options["properties"] = properties
    phi = program.solve(
        LinearProblem(operator, rhs, **problem_options),
        solver=solver,
        name="phi",
    ).consume(action=FailRun())
    program.store_history("blk.tensor_phi", phi)
    reconstructed = program.condensed_reconstruct(
        "reconstructed",
        state=current,
        phi=phi,
        linear_operator=linear,
        subset=(1, 2),
        th_dt=1.0,
        c_rho=0,
    )
    next_state = program.value("next", 1 * reconstructed, at=temporal.next.point)
    program.commit(temporal.next, next_state)
    source = emit_cpp_program(program, model=model, target="amr_system")
    if _return_model:
        return program, source, model
    return program, source


def test_refined_hierarchy_uses_one_direct_solve_and_flat_path_executes_apply():
    solver = CompositeTensorFAC(
        max_iter=23,
        rel_tol=3.0e-8,
        abs_tol=4.0e-13,
        fine_sweeps=7,
        coarse_rel_tol=2.0e-7,
        coarse_abs_tol=5.0e-14,
        coarse_cycles=9,
        verbose=True,
    )
    program, source = _build(solver)

    amr = source.split('extern "C" void pops_install_program_amr', 1)[1]
    configure = amr.index("ctx.configure_hierarchy_tensor_solver(")
    flat_phase = amr.index("ctx.solve_prepared_linear(")
    direct_phase = amr.index("ctx.solve_hierarchy_tensor(")
    branch = amr.index("if (ctx.uses_prepared_krylov_fallback())")
    hierarchy_advance = amr.index("auto _advance_hierarchy")
    refresh = amr.index("_refresh_level_programs();", hierarchy_advance)
    gather_call = amr.index(".gather(hierarchy_dt)", branch)
    direct_call = amr.index("_level_programs->front().solve(hierarchy_dt)", gather_call)
    publish_call = amr.index(".publish(hierarchy_dt)", direct_call)
    synchronized = amr.index("ctx.advance_synchronized_hierarchy", publish_call)
    assert configure < flat_phase < direct_phase < branch
    assert hierarchy_advance < refresh < branch < gather_call < direct_call < publish_call < synchronized

    assert amr.count("ctx.solve_prepared_linear(") == 1
    assert amr.count("ctx.solve_hierarchy_tensor(") == 1
    assert "pops::PureFieldAlgebra::copy_allocated(*frozen_A" in source
    assert "pops::PureFieldAlgebra::copy(*frozen_A" not in source
    assert 'ctx.history_zero_start("blk.tensor_phi", 1, 1, 1)' in amr[:direct_phase]
    assert "ctx.stage_linear_initial_guess(" in amr[:direct_phase]
    assert "ctx.stage_linear_initial_guess();" not in amr[:direct_phase]
    assert "assembly_source(" in amr[direct_phase:branch]

    configuration_line = next(
        line for line in amr[:branch].splitlines()
        if "ctx.configure_hierarchy_tensor_solver(" in line
    )
    assert 'ctx.configure_hierarchy_tensor_solver(1, 1, "pops.hierarchy.composite-tensor-fac"' in configuration_line
    assert '"pops.hierarchy.composite-tensor-fac.options@1"' in configuration_line
    assert '{"fac.fine_sweeps", std::int64_t{7}}' in configuration_line
    assert '{"fac.coarse_rel_tol", static_cast<double>(%s)}' % scalar_cpp(2.0e-7) in configuration_line
    assert '{"fac.coarse_abs_tol", static_cast<double>(%s)}' % scalar_cpp(5.0e-14) in configuration_line
    assert '{"fac.coarse_cycles", std::int64_t{9}}' in configuration_line
    assert '{"fac.verbose", true}' in configuration_line
    solve_line = next(
        line for line in amr[direct_phase:].splitlines()
        if "ctx.solve_hierarchy_tensor(" in line
    )
    assert "ctx.solve_hierarchy_tensor(1, 1," in solve_line
    assert "static_cast<pops::Real>(%s)" % scalar_cpp(3.0e-8) in solve_line
    assert "static_cast<pops::Real>(%s)" % scalar_cpp(4.0e-13) in solve_line
    assert solve_line.rstrip().endswith(", 23);")

    solve = next(value for value in program._values if value.op == "solve_linear")
    assert solve.attrs["hierarchy_block_index"] == 1
    assert solve.attrs["ncomp"] == 1
    assert solve.attrs["method_provider"]["provider_id"] == "pops.krylov.bicgstab"
    assert solve.attrs["method_options"] == {}
    assert solve.attrs["preconditioner_provider"]["provider_id"] == (
        "pops.preconditioner.identity"
    )
    assert solve.attrs["preconditioner_options"] == {}
    assert "restart" not in solve.attrs
    assert scalar_data(solve.attrs["abs_tol"]) == scalar_data(4.0e-13)
    frozen_identity = solve.attrs["hierarchy_solver_identity"]
    expected_identity = solver.canonical_identity()
    assert frozen_identity == solver.identity.token
    assert solve.attrs["solver_identity"] == frozen_identity
    assert solve.attrs["hierarchy_solver_provider"]["provider_id"] == (
        expected_identity["provider"]["provider_id"]
    )
    assert dict(solve.attrs["hierarchy_solver_options"]) == expected_identity["options"]
    assert "hierarchy_solver" not in solve.attrs


def test_program_reauthenticates_composite_fac_native_iteration_capacity():
    prepared = CompositeTensorFAC().prepare_program_solve()
    options = prepared.options
    options["max_iter"] = 1 << 31
    prepared = replace(
        prepared,
        _options_json=json.dumps(options, sort_keys=True, separators=(",", ":")),
    )

    class ForgedDescriptor:
        def prepare_program_solve(self):
            return prepared

    with pytest.raises(ValueError, match="max_iter"):
        _build(ForgedDescriptor())


def test_omitted_fac_controls_emit_native_default_sentinels_only():
    _, source = _build(CompositeTensorFAC(max_iter=13, rel_tol=4.0e-8))
    configuration_line = next(
        line for line in source.splitlines()
        if "ctx.configure_hierarchy_tensor_solver(" in line
    )
    assert 'ctx.configure_hierarchy_tensor_solver(1, 1, "pops.hierarchy.composite-tensor-fac"' in configuration_line
    assert '"pops.hierarchy.composite-tensor-fac.options@1", {}});' in configuration_line
    solve_line = next(
        line for line in source.splitlines()
        if "ctx.solve_hierarchy_tensor(" in line
    )
    assert "ctx.solve_hierarchy_tensor(1, 1," in solve_line
    assert scalar_cpp(4.0e-8) in solve_line
    assert solve_line.rstrip().endswith(", 13);")


def test_refined_solution_publishes_atomically_before_reflux_then_average_down():
    """Lock the complete refined-stage ordering without a wall-clock or legacy oracle."""
    _, source = _build(CompositeTensorFAC(max_iter=13, rel_tol=4.0e-8))
    amr = source.split('extern "C" void pops_install_program_amr', 1)[1]
    branch = amr.index("if (ctx.uses_prepared_krylov_fallback())")
    gather = amr.index(".gather(hierarchy_dt)", branch)
    solve = amr.index("_level_programs->front().solve(hierarchy_dt)", gather)
    publish = amr.index(".publish(hierarchy_dt)", solve)
    synchronize = amr.index("ctx.advance_synchronized_hierarchy", publish)
    assert gather < solve < publish < synchronize
    assert amr.count("ctx.solve_hierarchy_tensor(") == 1

    root = Path(__file__).resolve().parents[4]
    provider = (root / "include" / "pops" / "runtime" / "amr"
                / "amr_tensor_elliptic.hpp").read_text(encoding="utf-8")
    solved = provider.index("if (!report.solved_value_available())")
    publication = provider.index("copy0(levels_[", solved)
    assert solved < publication, "a failed hierarchy solve must not publish a partial iterate"

    context = (root / "include" / "pops" / "runtime" / "program"
               / "amr_program_context.hpp").read_text(encoding="utf-8")
    coupling = context.index("void couple_levels() const")
    reflux = context.index("route_reflux_program", coupling)
    average_down = context.index("average_down_level", reflux)
    assert reflux < average_down, "accepted synchronization is reflux then average-down"
