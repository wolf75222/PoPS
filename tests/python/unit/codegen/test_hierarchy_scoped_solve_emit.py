"""Direct scalar tensor-FAC scheduling on the AMR target."""

from __future__ import annotations
from pops.codegen.program_codegen import emit_cpp_program

from pathlib import Path

from pops._ir.literals import scalar_cpp
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


def _build(solver):
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
    phi = program.solve(
        LinearProblem(operator, rhs, initial_guess=phi_previous, scope=Hierarchy()),
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
    return program, emit_cpp_program(program, model=model, target="amr_system")


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

    configure = source.index("ctx.configure_composite_tensor_fac(")
    branch = source.index("if (!ctx.has_refined_hierarchy())")
    gather = source.index("Gather every level before the unique hierarchy-scoped solve")
    direct = source.index("ctx.solve_composite_tensor_fac(", gather)
    publish = source.index("The composite solution is complete", direct)
    synchronized = source.index("ctx.advance_synchronized_hierarchy", publish)
    assert configure < branch < gather < direct < publish < synchronized

    flat = source[branch:gather]
    refined = source[gather:synchronized]
    assert flat.count("ctx.solve_linear_matfree(") == 1
    assert "ctx.solve_composite_tensor_fac(" not in flat
    assert refined.count("ctx.solve_composite_tensor_fac(") == 1
    assert "ctx.solve_linear_matfree(" not in refined
    assert 'ctx.history_zero_start("blk.tensor_phi", 1, 1, 1)' in refined[:direct]
    assert "ctx.stage_linear_initial_guess(" in refined[:direct]
    assert "ctx.stage_linear_initial_guess();" not in refined[:direct]
    assert "assembly_source(" in source[publish:synchronized]

    expected_configuration = (
        "ctx.configure_composite_tensor_fac(1, 1, 7, static_cast<pops::Real>(%s), "
        "static_cast<pops::Real>(%s), 9, 1);" % (scalar_cpp(2.0e-7), scalar_cpp(5.0e-14))
    )
    assert expected_configuration in source[:branch]
    solve_line = next(
        line for line in source[direct:].splitlines()
        if "ctx.solve_composite_tensor_fac(" in line
    )
    assert "ctx.solve_composite_tensor_fac(1, 1," in solve_line
    assert "static_cast<pops::Real>(%s)" % scalar_cpp(3.0e-8) in solve_line
    assert "static_cast<pops::Real>(%s)" % scalar_cpp(4.0e-13) in solve_line
    assert solve_line.rstrip().endswith(", 23);")

    solve = next(value for value in program._values if value.op == "solve_linear")
    assert solve.attrs["hierarchy_block_index"] == 1
    assert solve.attrs["ncomp"] == 1
    assert solve.attrs["method"] == "bicgstab"
    assert solve.attrs["preconditioner"] == "identity"
    assert solve.attrs["restart"] is None
    frozen_identity = solve.attrs["hierarchy_solver_identity"]
    expected_identity = solver.canonical_identity()
    assert frozen_identity["solver_id"] == expected_identity["solver_id"]
    assert tuple(frozen_identity["capabilities"]) == tuple(expected_identity["capabilities"])
    assert dict(frozen_identity["options"]) == expected_identity["options"]
    assert "hierarchy_provider" not in solve.attrs


def test_omitted_fac_controls_emit_native_default_sentinels_only():
    _, source = _build(CompositeTensorFAC(max_iter=13, rel_tol=4.0e-8))
    expected = (
        "ctx.configure_composite_tensor_fac(1, 1, 0, static_cast<pops::Real>(%s), "
        "static_cast<pops::Real>(%s), 0, -1);" % (scalar_cpp(0), scalar_cpp(0))
    )
    assert expected in source
    solve_line = next(
        line for line in source.splitlines()
        if "ctx.solve_composite_tensor_fac(" in line
    )
    assert "ctx.solve_composite_tensor_fac(1, 1," in solve_line
    assert scalar_cpp(4.0e-8) in solve_line
    assert solve_line.rstrip().endswith(", 13);")


def test_refined_solution_publishes_atomically_before_reflux_then_average_down():
    """Lock the complete refined-stage ordering without a wall-clock or legacy oracle."""
    _, source = _build(CompositeTensorFAC(max_iter=13, rel_tol=4.0e-8))
    gather = source.index("Gather every level before the unique hierarchy-scoped solve")
    solve = source.index("ctx.solve_composite_tensor_fac(", gather)
    publish = source.index("The composite solution is complete", solve)
    synchronize = source.index("ctx.advance_synchronized_hierarchy", publish)
    assert gather < solve < publish < synchronize
    assert source[gather:publish].count("ctx.solve_composite_tensor_fac(") == 1

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
