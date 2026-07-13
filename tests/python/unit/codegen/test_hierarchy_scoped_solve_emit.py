"""Generic hierarchy-scoped tensor solve scheduling on the AMR target.

This is intentionally authored from the final Program primitives: no preset selects the provider,
scope, initial guess, or condensed reconstruction on the caller's behalf.
"""

from __future__ import annotations

from pops.ir.literals import scalar_cpp
from pops.linalg import LinearProblem
from pops.params import ConstParam
from pops.solvers import BiCGStab, CompositeTensorFAC, Hierarchy
from pops.time import FailRun, Program

from typed_program_support import state_refs


def _coupled_model(name):
    from pops.ir.ops import sqrt
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


def _emit(provider):
    model = _coupled_model("hierarchy_tensor_model")
    program = Program("hierarchy_tensor_step")._bind_operators(model)
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
    operator = program.matrix_free_operator("tensor_operator", scope=Hierarchy(), provider=provider)

    def apply(builder, _out, value):
        laplacian = builder.scalar_field("tensor_laplacian")
        builder.apply_laplacian_coeff(laplacian, value, coefficients)
        return -1 * laplacian

    program.set_apply(operator, apply)
    phi = program.solve(
        LinearProblem(operator, rhs, initial_guess=phi_previous, scope=Hierarchy()),
        solver=BiCGStab(max_iter=23, rel_tol=3.0e-8),
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
    return program.emit_cpp_program(model=model, target="amr_system")


def test_refined_hierarchy_orders_gather_one_solve_then_publish_and_reconstruct():
    provider = CompositeTensorFAC(
        fine_sweeps=7, coarse_rel_tol=2.0e-7, coarse_cycles=9, verbose=True
    )
    source = _emit(provider)

    gather = source.index("Gather every level before the unique hierarchy-scoped solve")
    configure = source.index("ctx.configure_composite_tensor_fac(", gather)
    solve = source.index("ctx.solve_linear_matfree(", configure)
    publish = source.index("The composite solution is complete", solve)
    synchronized = source.index("ctx.advance_synchronized_hierarchy", publish)
    assert gather < configure < solve < publish < synchronized
    assert source[gather:synchronized].count("ctx.solve_linear_matfree(") == 1

    gathered = source[gather:solve]
    assert 'ctx.history_zero_start("blk.tensor_phi", 1, 1, 0)' in gathered
    assert "ctx.stage_linear_initial_guess(" in gathered
    assert "ctx.stage_linear_initial_guess();" not in gathered
    assert "assembly_source(" in source[publish:synchronized]

    expected_configuration = (
        "ctx.configure_composite_tensor_fac(7, static_cast<pops::Real>(%s), 9, 1);"
        % scalar_cpp(2.0e-7)
    )
    assert expected_configuration in source[configure:solve]
    solve_line = next(
        line for line in source[solve:].splitlines() if "ctx.solve_linear_matfree(" in line
    )
    assert "static_cast<pops::Real>(%s)" % scalar_cpp(3.0e-8) in solve_line
    assert ", 23," in solve_line


def test_omitted_provider_controls_emit_native_default_sentinels_only():
    source = _emit(CompositeTensorFAC())
    expected = (
        "ctx.configure_composite_tensor_fac(0, static_cast<pops::Real>(%s), 0, -1);" % scalar_cpp(0)
    )
    assert expected in source
