"""pops.time Phase 4 IR ops (epic ADC-399 / ADC-403): named sources + local linear operators.

Pins the builder surface for split sources and cell-local implicit solves -- `P.source`,
`P.linear_source`, `P.apply`, typed `LocalLinear` problems, and the operator algebra `P.I - a*L`. These
build typed IR (validated structurally); the codegen that LOWERS them is a later PR, so
`emit_cpp_program` still refuses a Program that uses them with a clear NotImplementedError (never a
mis-lowering). Pure Python (no compile / no _pops runtime); skips if pops is unavailable.
"""
from types import SimpleNamespace

from typed_program_support import solve_field, typed_state
from pops.numerics.terms import Flux
from pops.solvers import DenseLU

import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_local_solve (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


def _predictor_corrector(t):
    """Spec example 5: predictor-corrector Poisson/Lorentz (electric source + Lorentz local solve)."""
    from pops.physics._facade import Model

    model = Model("predictor-corrector")
    (u,) = model.conservative_vars("u")
    model.elliptic_rhs(u)
    electric = model.source_term("electric", [-u])
    lorentz = model.local_linear_map("lorentz", [[-1]])
    P = t.Program("predictor_corrector_poisson_lorentz")
    dt = P.dt
    U_n = typed_state(P, "plasma", model=model)
    endpoint = typed_state(P, "plasma", state_name="U", model=model).next
    predictor = t.StagePoint(
        "predictor", {"main": t.TimePoint(P.clock, 1)})
    f_n = solve_field(P, U_n, name="fields_n")
    R_n = P.rhs(name="R_n", state=U_n, fields=f_n, terms=[Flux(), electric])
    U_star_rhs = P.value(
        "U_star_rhs", U_n + dt * R_n, at=predictor)
    U_star = P.solve(
        t.LocalLinear(operator=P.I - dt * P.linear_source(lorentz), rhs=U_star_rhs, fields=f_n),
        solver=DenseLU(), name="U_star",
    ).consume(action=t.FailRun())
    f_star = solve_field(P, U_star, name="fields_star")
    R_star = P.rhs(name="R_star", state=U_star, fields=f_star, terms=[Flux(), electric])
    C_star = P.apply(operator=lorentz, state=U_star, fields=f_star, name="C_star")
    Q = P.value(
        "Q", U_n + 0.5 * dt * R_n + 0.5 * dt * R_star + 0.5 * dt * C_star,
        at=endpoint.point)
    U_np1 = P.solve(
        t.LocalLinear(
            operator=P.I - 0.5 * dt * P.linear_source(lorentz), rhs=Q, fields=f_star),
        solver=DenseLU(), name="U_np1",
    ).consume(action=t.FailRun())
    solve_field(P, U_np1, name="fields_np1")
    P.commit(endpoint, U_np1)
    return P, model


def _emit(program, model):
    from pops.codegen.program_codegen import emit_cpp_program

    solve = next(value for value in program._values if value.op == "solve_fields")
    field = solve.attrs["field"]
    plan = SimpleNamespace(
        name=field.local_id,
        native_options={
            "provider_slot": field.local_id,
            "output_route": {"components": list(solve.field_context.outputs)},
            "boundary_kernel_required": False,
        },
    )
    return emit_cpp_program(
        program, model=model, field_plans={field.local_id: plan})


def test_predictor_corrector_builds(t):
    P, _ = _predictor_corrector(t)
    assert P.validate() is True, "the predictor-corrector IR must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_a_coeff_recorded_and_hashed(t):
    # operator I - a*dt*L records a as a dt-polynomial in attrs; a changes the IR hash.
    def prog(a):
        from pops.physics._facade import Model
        model = Model("coefficient")
        model.conservative_vars("u")
        linear = model.local_linear_map("lorentz", [[-1]])
        P = t.Program("scl")
        U = typed_state(P, "plasma", model=model)
        endpoint = typed_state(P, "plasma", state_name="U", model=model).next
        Q = P.value("Q", 1.0 * U, at=endpoint.point)
        op = P.I - a * P.dt * P.linear_source(linear)
        solved = P.solve(
            t.LocalLinear(operator=op, rhs=Q), solver=DenseLU(), name="W",
        ).consume(action=t.FailRun())
        P.commit(endpoint, solved)
        return P
    assert prog(1.0)._ir_hash() != prog(0.5)._ir_hash(), "a different solve coefficient must rehash"


def test_local_linear_problem_rejects_non_operator(t):
    P = t.Program("bad")
    U = typed_state(P, "plasma")
    Q = P.value("Q", 1.0 * U)
    try:  # a plain State is not a local linear operator
        P.solve(t.LocalLinear(operator=U, rhs=Q), solver=DenseLU(), name="W")
    except ValueError as exc:
        assert "local linear operators only" in str(exc)
    else:
        raise AssertionError("expected ValueError for a non-operator")


def test_local_linear_problem_requires_identity(t):
    from pops.physics._facade import Model
    model = Model("identity")
    model.conservative_vars("u")
    linear = model.local_linear_map("lorentz", [[-1]])
    P = t.Program("bad2")
    U = typed_state(P, "plasma", model=model)
    Q = P.value("Q", 1.0 * U)
    try:  # a*L without the identity I is not the I +/- a*L form
        P.solve(
            t.LocalLinear(operator=P.dt * P.linear_source(linear), rhs=Q),
            solver=DenseLU(), name="W")
    except ValueError as exc:
        assert "local linear operators only" in str(exc)
    else:
        raise AssertionError("expected ValueError for an operator without identity")


def test_source_and_apply_are_rhs_like(t):
    from pops.physics._facade import Model
    model = Model("source-apply")
    model.conservative_vars("u")
    electric = model.source_term("electric", [0])
    lorentz = model.local_linear_map("lorentz", [[-1]])
    P = t.Program("sa")
    U = typed_state(P, "plasma", model=model)
    f = solve_field(P, U)
    # This unit pins the primitive IR. Public handle resolution is covered by
    # test_operator_handle_resolution; use the explicit private selector seam here.
    S = P.source(electric, state=U, fields=f)
    LU = P.apply(lorentz, state=U, fields=f)
    assert S.vtype == "rhs" and LU.vtype == "rhs", "source/apply are dU/dt-like (RHS) values"
    endpoint = typed_state(P, "plasma", state_name="U", model=model).next
    P.commit(endpoint, P.value(
        "Un", U + P.dt * S + P.dt * LU, at=endpoint.point))
    assert P.validate() is True


def test_phase4_ops_lower_through_typed_handles(t):
    P, model = _predictor_corrector(t)
    source = _emit(P, model)
    assert "pops_install_program" in source
    assert "mat_inverse" in source


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_local_solve (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
