#!/usr/bin/env python3
"""pops.time matrix-free dynamic linear solve, end to end (epic ADC-399 / ADC-405 Phase 6b).

`emit_cpp_program` now lowers a DYNAMIC matrix-free linear solve: a ``matrix_free_operator`` whose
apply ``out <- A(in)`` is an IR sub-block (``P.set_apply``, built from ``P.laplacian`` + the affine
algebra) lowered to a C++ ``pops::ApplyFn`` lambda, and ``P.solve(LinearProblem(...), solver=...)``
lowered to the runtime context's typed ``solve_prepared_linear`` seam. The iteration is DYNAMIC and
lives C++-side, inside the loop -- the IR carries
only the apply, the rhs, the method / tolerance / iteration budget. The persistent scratch (the
Laplacian output, the solution field) is allocated ONCE at install time (a ``std::shared_ptr``
captured into the step closure), reused across every step and every Krylov iteration.

(A) Codegen (pure Python, always runs): a Helmholtz operator ``A(in) = in - alpha*Lap(in)`` solved by
    cg / bicgstab / richardson lowers to the apply lambda + ``ctx.laplacian`` +
    ``ctx.solve_prepared_linear``; the spec validation errors fire (max_iter absent /
    <= 0 -> ValueError "dynamic solver loops require max_iter"; tol <= 0 -> error; unknown method ->
    error; operator not a matrix_free_operator -> error).

(B) End-to-end parity (skips unless the full toolchain is present): a 1-variable model (rho, zero
    flux); A = matrix_free_operator with apply out = in - alpha*Lap(in) (alpha = 0.1, SPD); the
    Program solves (I - alpha*Lap) phi = U via cg (tol 1e-10, max_iter 200) and commits U = phi.
    compile_problem -> install_program -> set a smooth periodic rho0 -> step once -> get_state, vs an
    OFFLINE numpy CG on the SAME discrete periodic 5-point system. Asserts max|compiled - offline| <=
    1e-6, the solve changed the state, and the offline solve took > 1 iteration. Self-skips (exit 0)
    without numpy / _pops / install_program / a compiler / a visible Kokkos -- never fakes the engine.
"""
from tests.python.support.requirements import require_native_or_skip
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen import _compile_drivers as compile_drivers
from typed_program_support import typed_state

from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.numerics.reconstruction import FirstOrder
from pops.time import FailRun, RejectAttempt
from pops.numerics.riemann import Rusanov
from pops.runtime._system import System  # ADC-545 advanced runtime seam


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        require_native_or_skip('test_time_solve_linear (pops.time unavailable: %s)' % exc)
    return t


_ALPHA = 0.1  # Helmholtz coefficient: A = I - alpha*Lap (SPD, well-conditioned for CG)


def _krylov(method, *, max_iter, rel_tol=None, restart=None, preconditioner=None):
    """Build the exact typed Krylov descriptor selected by the test."""
    from pops.solvers import krylov
    options = {"max_iter": max_iter}
    if rel_tol is not None:
        options["rel_tol"] = rel_tol
    if restart is not None:
        options["restart"] = restart
    if preconditioner is not None:
        options["preconditioner"] = preconditioner
    return {"cg": krylov.CG, "bicgstab": krylov.BiCGStab,
            "richardson": krylov.Richardson, "gmres": krylov.GMRES}[method](**options)


def _precond(scheme):
    """Map a preconditioner name to its TYPED pops.solvers.preconditioners descriptor."""
    from pops.solvers import preconditioners
    return {"identity": preconditioners.Identity,
            "geometric_mg": preconditioners.GeometricMG}[scheme]()


def _solve_program(t, *, name="solve_lin", method="cg", tol=1e-10, max_iter=200, alpha=_ALPHA,
                   preconditioner=None, action=None, operator_uses_dt=False):
    """(I - alpha*Lap) phi = U, committed back into the 1-component block (its state == a scalar field).

    The apply ``out = in - alpha*Lap(in)`` is built with P.laplacian + the affine algebra; Program.solve
    drives the runtime Krylov loop. The Program needs no model (the apply is a pure Laplacian). An
    optional @p preconditioner (a typed descriptor) is carried by the Krylov solver."""
    P = t.Program(name)
    U = typed_state(P, "blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        coefficient = P.dt if operator_uses_dt else alpha
        return x - coefficient * lap  # out = in - coefficient*Lap(in)

    P.set_apply(A, apply)
    endpoint = typed_state(P, "blk", state_name="U").next
    rhs = P.value("rhs", U, at=endpoint.point)
    solver = _krylov(
        method, max_iter=max_iter, rel_tol=tol, preconditioner=preconditioner)
    phi = P.solve(
        LinearProblem(
            A, rhs, at=endpoint.point,
            properties=(LinearOperatorProperties.symmetric_positive_definite()
                        if method == "cg" else LinearOperatorProperties.general())),
        solver=solver,
    ).consume(action=action or FailRun())
    P.commit(endpoint, phi)
    return P


# ---- (A) codegen: pure Python, always runs ----
def test_apply_lambda_and_cg_codegen(t):
    src = emit_cpp_program(_solve_program(t, method="cg"))
    for frag in ("pops::ApplyFn apply_A", "ctx.laplacian", "ctx.solve_prepared_linear",
                 "pops::PreparedAffineLinearProblem", "pops::KrylovWorkspace",
                 "std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field"):
        assert frag in src, "the generated cg solve must contain %r\n%s" % (frag, src)


def test_reject_attempt_solve_codegen_throws_step_attempt_signal(t):
    src = emit_cpp_program(_solve_program(t, method="cg", action=RejectAttempt()))
    assert "#include <pops/runtime/program/step_transaction.hpp>" in src, src
    assert "pops::runtime::program::StepAttemptRejected" in src, src
    assert "solve_linear failed" in src, src


def test_bicgstab_codegen(t):
    src = emit_cpp_program(_solve_program(t, method="bicgstab"))
    assert "ctx.solve_prepared_linear" in src, src
    assert "PreparedLinearPreconditioner::identity()" in src, src


def test_richardson_codegen(t):
    src = emit_cpp_program(_solve_program(t, method="richardson"))
    assert "ctx.solve_prepared_linear" in src, src


# ---- (A') GeometricMG preconditioner (ADC-516): the complete non-identity route ----
def _solve_call(src):
    """The single generic context solve line of @p src."""
    return [ln for ln in src.splitlines() if "ctx.solve_prepared_linear(" in ln][0]


def test_gmres_gmg_precond_codegen(t):
    # GMRES + GeometricMG lowers to a REAL ApplyFn (one V-cycle of the wired multigrid), NOT the empty
    # identity ApplyFn. ADC-637: the precond V-cycle cache lives in a persistent
    # pops::runtime::program::GeometricMgPreconditioner (re-homed to the Schur-free coeff_elliptic_ops.hpp)
    # the named lambda forwards apply() to.
    src = emit_cpp_program(_solve_program(t, method="gmres", preconditioner=_precond("geometric_mg")))
    assert "pops::runtime::program::GeometricMgPreconditioner" in src, (
        "the MG V-cycle preconditioner state must be emitted\n%s" % src)
    assert "->apply(ctx," in src, "the MG V-cycle apply must be emitted\n%s" % src
    assert "->prepare(ctx," in src, "MG must be built before the Krylov loop\n%s" % src
    assert "pops::ApplyFn precond_mg" in src, "a named real precond ApplyFn must be emitted\n%s" % src
    assert "pops::PreparedLinearPreconditioner(precond_mg" in src, src


def test_bicgstab_gmg_precond_codegen(t):
    src = emit_cpp_program(_solve_program(t, method="bicgstab",
                         preconditioner=_precond("geometric_mg")))
    assert "pops::runtime::program::GeometricMgPreconditioner" in src, src
    assert "->apply(ctx," in src, src
    assert "pops::PreparedLinearPreconditioner(precond_mg" in src, src


def test_identity_precond_byte_identical(t):
    # The identity (default) path is unchanged: the empty ApplyFn{}, no MG apply emitted. The explicit
    # Identity() descriptor and the None default lower to the SAME source.
    src_default = emit_cpp_program(_solve_program(t, method="gmres"))
    src_identity = emit_cpp_program(_solve_program(t, method="gmres",
                                  preconditioner=_precond("identity")))
    assert src_default == src_identity, "explicit Identity() must match the None default byte-for-byte"
    assert "PreparedLinearPreconditioner::identity()" in src_default
    assert "geometric_mg_precond_apply" not in src_default, "identity emits no MG apply"


def test_cg_gmg_precond_rejected(t):
    # CG / Richardson have no preconditioner slot in the matrix-free path: a non-identity precond is an
    # honest capability limit (ValueError naming GMRES/BiCGStab), not a transitional reject.
    for method in ("cg", "richardson"):
        try:
            _solve_program(t, method=method, preconditioner=_precond("geometric_mg"))
        except ValueError as exc:
            assert (method in str(exc) and "native preconditioner slot" in str(exc)
                    and "GMRES or BiCGStab" in str(exc)), str(exc)
        else:
            raise AssertionError("%s + GeometricMG must raise ValueError" % method)


def test_unwired_preconditioners_are_not_published(t):
    from pops.solvers import preconditioners

    assert not hasattr(preconditioners, "Jacobi")
    assert not hasattr(preconditioners, "BlockJacobi")


def test_string_precond_rejected(t):
    # Spec 5 sec.7: a bare string preconditioner is rejected, naming the typed alternative.
    P = t.Program("p")
    U = typed_state(P, "blk")
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    try:
        P.solve(
            LinearProblem(A, U),
            solver=_krylov(
                "gmres", max_iter=10, preconditioner="geometric_mg"),
        )
    except TypeError as exc:
        assert "preconditioner" in str(exc) and "pops.solvers.preconditioners" in str(exc), str(exc)
    else:
        raise AssertionError("a string preconditioner must raise TypeError")


def test_gmg_precond_validates(t):
    P = _solve_program(t, method="gmres", preconditioner=_precond("geometric_mg"))
    assert P.validate() is True, "the gmres+GeometricMG Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_solve_validates(t):
    P = _solve_program(t)
    assert P.validate() is True, "the typed linear-solve Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_prepared_codegen_has_frozen_snapshot_and_no_context_algebra_in_apply(t):
    src = emit_cpp_program(_solve_program(t, method="cg", operator_uses_dt=True))
    apply_body = src.split("pops::ApplyFn apply_A", 1)[1].split("};", 1)[0]
    assert "operator_evaluation_snapshot" in src
    assert "probe_operator_evaluation" in src
    assert "->prepare(*operator_snapshot" in src
    assert "->bind(*prepared_problem" in src
    assert "pops::PureFieldAlgebra" in apply_body
    assert "ctx.axpy" not in apply_body and "ctx.lincomb" not in apply_body
    assert "(*operator_dt" in apply_body


def test_max_iter_required(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    for bad in (None, 0, -5):
        try:
            P.solve(
                LinearProblem(A, U), solver=_krylov("cg", max_iter=bad))
        except ValueError as exc:
            assert "dynamic solver loops require max_iter" in str(exc), str(exc)
        else:
            raise AssertionError("max_iter=%r must raise the dynamic-loop budget error" % (bad,))


def test_tol_positive(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    for bad in (0.0, -1e-8):
        try:
            P.solve(
                LinearProblem(A, U),
                solver=_krylov("cg", max_iter=10, rel_tol=bad),
            )
        except ValueError as exc:
            assert "tol" in str(exc), str(exc)
        else:
            raise AssertionError("tol=%r must raise (a non-positive tolerance is a config error)" % bad)


def test_string_method_rejected(t):
    # Program.solve takes a typed solver descriptor; known and unknown strings are both refused.
    P = t.Program("p")
    U = typed_state(P, "blk")
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    for bad in ("cg", "minres"):
        try:
            P.solve(LinearProblem(A, U), solver=bad)
        except TypeError as exc:
            assert "solver" in str(exc) and "typed descriptor" in str(exc), str(exc)
        else:
            raise AssertionError("a string solver=%r must raise TypeError" % (bad,))


def test_operator_must_be_matrix_free(t):
    P = t.Program("p")
    U = typed_state(P, "blk")
    try:
        P.solve(
            LinearProblem(U, U), solver=_krylov("cg", max_iter=10))
    except ValueError as exc:
        assert "operator" in str(exc), str(exc)
    else:
        raise AssertionError("operator must be a matrix_free_operator value")


def _helmholtz(P, x):
    lap = P.scalar_field("lap")
    P.laplacian(lap, x)
    return x - _ALPHA * lap


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _np_cg(apply, b, *, tol=1e-10, max_iter=2000):
    """Plain numpy CG solving A x = b from x = 0 (A = the discrete periodic Helmholtz matvec). Returns
    (x, iters). The reference for the compiled matrix-free CG."""
    import numpy as np

    x = np.zeros_like(b)
    r = b - apply(x)
    p = r.copy()
    rs_old = float(np.sum(r * r))
    bnorm = float(np.sqrt(np.sum(b * b))) or 1.0
    iters = 0
    for _ in range(max_iter):
        Ap = apply(p)
        pap = float(np.sum(p * Ap))
        if abs(pap) < 1e-300:
            break
        a = rs_old / pap
        x = x + a * p
        r = r - a * Ap
        rs_new = float(np.sum(r * r))
        iters += 1
        if np.sqrt(rs_new) <= tol * bnorm:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, iters


def _discrete_helmholtz(n, alpha):
    """The discrete periodic 5-point Helmholtz matvec A x = x - alpha*Lap(x) on an n x n unit-square
    grid (dx = dy = 1/n), matching pops::apply_laplacian's bare path with periodic ghosts."""
    import numpy as np

    h2 = (1.0 / n) ** 2

    def apply(x):
        lap = (np.roll(x, -1, 0) + np.roll(x, 1, 0) - 2 * x) / h2 + \
              (np.roll(x, -1, 1) + np.roll(x, 1, 1) - 2 * x) / h2
        return x - alpha * lap

    return apply


def _run_section_b(t):
    try:
        import numpy as np

        import pops.runtime._engine_descriptors as engine
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable in this interpreter
        require_native_or_skip('-- (B) skipped: pops/numpy unavailable: %s --' % exc)
        return None

    n = 16
    sim = System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        require_native_or_skip('-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --')
        return None

    from pops.physics._facade import Model

    # A minimal 1-variable model with NO flux and NO Poisson coupling: the Program never runs a rhs or
    # solve_fields; the block's single conservative variable (rho) doubles as the scalar field the
    # matrix-free solve writes. A complete compilable block (flux + primitive + eigenvalue).
    def passive_model(name):
        m = Model(name)
        (rho,) = m.conservative_vars("rho")
        u = m.primitive("u", 0.0 * rho)
        m.primitive_vars(rho=rho, u=u)
        m.conservative_from([rho])
        m.flux(x=[0.0 * rho], y=[0.0 * rho])
        m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
        return m

    tol = 1e-10
    try:
        compiled = compile_drivers.compile_problem(
            model=passive_model("solve_prog"),
            time=_solve_program(t, name="solve_step", method="cg", tol=tol, max_iter=200))
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        require_native_or_skip('-- (B) skipped: compile_problem could not build the .so: %s --' % str(exc)[:200])
        return None

    assert compiled.program_name == "solve_step", "handle carries the program name"

    try:
        compiled_model = passive_model("solve_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        require_native_or_skip('-- (B) skipped: model compile could not build the .so: %s --' % str(exc)[:200])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="euler"))

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    sim.step(0.05)  # dt is irrelevant: the solve is dt-free
    out = np.array(sim.get_state("blk"))[0]

    # OFFLINE reference: solve the SAME discrete system (I - alpha*Lap_periodic) phi = rho0 with a numpy
    # CG to the same tolerance. The compiled matrix-free CG must recover the same phi.
    apply = _discrete_helmholtz(n, _ALPHA)
    phi_ref, iters = _np_cg(apply, rho0, tol=tol)
    err = float(np.abs(out - phi_ref).max())
    moved = float(np.abs(out - rho0).max())
    print("  solve_linear parity: max|compiled - offline| = %.2e  offline iters = %d  max|phi - U0| "
          "= %.2e" % (err, iters, moved))
    assert err <= 1e-6, "compiled matrix-free CG == offline numpy CG (max|d| = %.2e)" % err
    assert moved > 1e-6, "the solve must change the state from U0 (max|d| = %.2e)" % moved
    assert iters > 1, "the offline (and compiled) solve must take > 1 iteration, got %d" % iters
    return (err, iters)


def _run_section_b_gmg_precond(t):
    """(B') GMRES + GeometricMG preconditioner convergence (ADC-516), Kokkos/_pops-gated.

    Solves the SAME periodic Helmholtz system as (B) -- (I - alpha*Lap) phi = U -- but with GMRES
    preconditioned by ONE GeometricMG V-cycle, and checks the compiled matrix-free solve recovers the
    SAME phi as the offline numpy CG (parity == convergence: a correctly-preconditioned GMRES converges
    to the unique solution). Self-skips (exit 0) without numpy / _pops / install_program / a compiler /
    a visible Kokkos -- the .so build needs ADC_KOKKOS_ROOT, not available host-only on this Mac, so the
    real preconditioned convergence run is confirmed on ROMEO/CI."""
    try:
        import numpy as np

        import pops.runtime._engine_descriptors as engine
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip("-- (B') skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 16
    sim = System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        require_native_or_skip("-- (B') skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None

    from pops.physics._facade import Model
    from pops.solvers import preconditioners

    def passive_model(name):
        m = Model(name)
        (rho,) = m.conservative_vars("rho")
        u = m.primitive("u", 0.0 * rho)
        m.primitive_vars(rho=rho, u=u)
        m.conservative_from([rho])
        m.flux(x=[0.0 * rho], y=[0.0 * rho])
        m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
        return m

    tol = 1e-10
    prog = _solve_program(t, name="solve_gmg", method="gmres", tol=tol, max_iter=200,
                          preconditioner=preconditioners.GeometricMG())
    try:
        compiled = compile_drivers.compile_problem(model=passive_model("solve_gmg_prog"), time=prog)
        compiled_model = passive_model("solve_gmg_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        require_native_or_skip("-- (B') skipped: compile could not build the .so: %s --" % str(exc)[:200])
        return None

    sim.add_equation("blk", compiled_model,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method="euler"))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))
    sim.install_program(compiled.so_path)
    sim.step(0.05)
    out = np.array(sim.get_state("blk"))[0]

    apply = _discrete_helmholtz(n, _ALPHA)
    phi_ref, iters = _np_cg(apply, rho0, tol=tol)
    err = float(np.abs(out - phi_ref).max())
    moved = float(np.abs(out - rho0).max())
    print("  gmres+GeometricMG parity: max|compiled - offline| = %.2e  max|phi - U0| = %.2e"
          % (err, moved))
    # Convergence: the preconditioned GMRES reaches the SAME solution (unique) as the offline CG.
    assert err <= 1e-6, "compiled gmres+GeometricMG == offline solution (max|d| = %.2e)" % err
    assert moved > 1e-6, "the preconditioned solve must change the state (max|d| = %.2e)" % moved
    return (err, iters)


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_solve_linear (A: %d checks)" % len(fns))
    _run_section_b(t)
    _run_section_b_gmg_precond(t)


if __name__ == "__main__":
    _run()
