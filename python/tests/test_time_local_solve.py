"""Internal local-source/local-linear IR regression tests.

The public Spec 5 route is operator-first ``P.call(handle, ...)``. These tests exercise the
underscored compatibility nodes still used by older local-solve internals while that lowering is
migrated; they must not reintroduce the removed public source/local-linear methods.
"""
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
    P = t.Program("predictor_corrector_poisson_lorentz")
    dt = P.dt
    U_n = P._state_value("plasma")
    f_n = P._fields_from_state("fields_n", U_n)
    R_n = P._rate_from_transport(name="R_n", state=U_n, fields=f_n, flux=True, sources=["electric"])
    U_star_rhs = P.linear_combine("U_star_rhs", U_n + dt * R_n)
    U_star = P.solve_local_linear(name="U_star", operator=P.I - dt * P._linear_source_value("lorentz"),
                                  rhs=U_star_rhs, fields=f_n)
    f_star = P._fields_from_state("fields_star", U_star)
    R_star = P._rate_from_transport(name="R_star", state=U_star, fields=f_star, flux=True, sources=["electric"])
    C_star = P.apply(operator=P._linear_source_value("lorentz"), state=U_star, fields=f_star, name="C_star")
    Q = P.linear_combine("Q", U_n + 0.5 * dt * R_n + 0.5 * dt * R_star + 0.5 * dt * C_star)
    U_np1 = P.solve_local_linear(name="U_np1", operator=P.I - 0.5 * dt * P._linear_source_value("lorentz"),
                                 rhs=Q, fields=f_star)
    P._fields_from_state("fields_np1", U_np1)
    P.commit("plasma", U_np1)
    return P


def test_predictor_corrector_builds(t):
    P = _predictor_corrector(t)
    assert P.validate() is True, "the predictor-corrector IR must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_a_coeff_recorded_and_hashed(t):
    # operator I - a*dt*L records a as a dt-polynomial in attrs; a changes the IR hash.
    def prog(a):
        P = t.Program("scl")
        U = P._state_value("plasma")
        Q = P.linear_combine("Q", 1.0 * U)
        op = P.I - a * P.dt * P._linear_source_value("lorentz")
        P.commit("plasma", P.solve_local_linear(name="W", operator=op, rhs=Q))
        return P
    assert prog(1.0)._ir_hash() != prog(0.5)._ir_hash(), "a different solve coefficient must rehash"


def test_solve_local_linear_rejects_non_operator(t):
    P = t.Program("bad")
    U = P._state_value("plasma")
    Q = P.linear_combine("Q", 1.0 * U)
    try:  # a plain State is not a local linear operator
        P.solve_local_linear(name="W", operator=U, rhs=Q)
    except ValueError as exc:
        assert "local linear operators only" in str(exc)
    else:
        raise AssertionError("expected ValueError for a non-operator")


def test_solve_local_linear_requires_identity(t):
    P = t.Program("bad2")
    U = P._state_value("plasma")
    Q = P.linear_combine("Q", 1.0 * U)
    try:  # a*L without the identity I is not the I +/- a*L form
        P.solve_local_linear(name="W", operator=P.dt * P._linear_source_value("lorentz"), rhs=Q)
    except ValueError as exc:
        assert "local linear operators only" in str(exc)
    else:
        raise AssertionError("expected ValueError for an operator without identity")


def test_source_and_apply_are_rhs_like(t):
    P = t.Program("sa")
    U = P._state_value("plasma")
    f = P._fields_from_state(U)
    S = P._source_value("electric", state=U, fields=f)
    LU = P.apply(P._linear_source_value("lorentz"), state=U, fields=f)
    assert S.vtype == "rhs" and LU.vtype == "rhs", "source/apply are dU/dt-like (RHS) values"
    P.commit("plasma", P.linear_combine("Un", U + P.dt * S + P.dt * LU))
    assert P.validate() is True


def test_codegen_requires_model_for_internal_local_ops(t):
    # The local-source/local-linear kernels need model coefficients. Without a model, codegen must
    # refuse rather than inventing a placeholder route.
    P = _predictor_corrector(t)
    try:
        P.emit_cpp_program()
    except NotImplementedError as exc:
        msg = str(exc).lower()
        assert "op" in msg or "source" in msg or "solve_local_linear" in msg
    else:
        raise AssertionError("expected NotImplementedError without a model")


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_local_solve (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
