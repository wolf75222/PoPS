#!/usr/bin/env python3
"""pops.time matrix-free dynamic linear solve, end to end (epic ADC-399 / ADC-405 Phase 6b).

`emit_cpp_program` now lowers a DYNAMIC matrix-free linear solve: a ``matrix_free_operator`` whose
apply ``out <- A(in)`` is an IR sub-block (``P.set_apply``, built from ``P.laplacian`` + the affine
algebra) lowered to a C++ ``pops::ApplyFn`` lambda, and ``P.solve_linear(operator=A, rhs=, method=...)``
lowered to a call into the runtime's Krylov loop (``pops::cg_solve`` / ``bicgstab_solve`` /
``richardson_solve``). The iteration is DYNAMIC and lives C++-side, inside the loop -- the IR carries
only the apply, the rhs, the method / tolerance / iteration budget. The persistent scratch (the
Laplacian output, the solution field) is allocated ONCE at install time (a ``std::shared_ptr``
captured into the step closure), reused across every step and every Krylov iteration.

(A) Codegen (pure Python, always runs): a Helmholtz operator ``A(in) = in - alpha*Lap(in)`` solved by
    cg / bicgstab / richardson lowers to the apply lambda + ``ctx.laplacian`` + ``pops::cg_solve`` /
    ``bicgstab_solve`` / ``richardson_solve``; the spec validation errors fire (max_iter absent /
    <= 0 -> ValueError "dynamic solver loops require max_iter"; tol <= 0 -> error; unknown method ->
    error; operator not a matrix_free_operator -> error).

(B) End-to-end parity (skips unless the full toolchain is present): a 1-variable model (rho, zero
    flux); A = matrix_free_operator with apply out = in - alpha*Lap(in) (alpha = 0.1, SPD); the
    Program solves (I - alpha*Lap) phi = U via cg (tol 1e-10, max_iter 200) and commits U = phi.
    compile_problem -> sim.install(...) -> set a smooth periodic rho0 -> step once -> get_state, vs an
    OFFLINE numpy CG on the SAME discrete periodic 5-point system. Asserts max|compiled - offline| <=
    1e-6, the solve changed the state, and the offline solve took > 1 iteration. Self-skips (exit 0)
without numpy / _pops / a compiler / a visible Kokkos -- never fakes the engine.
"""
import pytest
import sys


def _pops_time():
    try:
        import pops.time as t
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_solve_linear (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


@pytest.fixture
def t():
    return _pops_time()


_ALPHA = 0.1  # Helmholtz coefficient: A = I - alpha*Lap (SPD, well-conditioned for CG)


def _krylov(method):
    """Map a method name to its TYPED pops.solvers.krylov descriptor (Spec 5 sec.7: solve_linear
    takes a typed solver, not a string -- the test still parametrizes by the name for clarity)."""
    from pops.solvers import krylov
    return {"cg": krylov.CG, "bicgstab": krylov.BiCGStab,
            "richardson": krylov.Richardson, "gmres": krylov.GMRES}[method]()


def _precond(scheme):
    """Map a preconditioner name to its TYPED pops.solvers.preconditioners descriptor."""
    from pops.solvers import preconditioners
    return {"identity": preconditioners.Identity, "geometric_mg": preconditioners.GeometricMG}[scheme]()


def _solve_program(t, *, name="solve_lin", method="cg", tol=1e-10, max_iter=200, alpha=_ALPHA,
                   preconditioner=None):
    """(I - alpha*Lap) phi = U, committed back into the 1-component block (its state == a scalar field).

    The apply ``out = in - alpha*Lap(in)`` is built with P.laplacian + the affine algebra; solve_linear
    drives the runtime Krylov loop. The Program needs no model (the apply is a pure Laplacian). An
    optional @p preconditioner (a typed descriptor) is threaded into solve_linear."""
    P = t.Program(name)
    U = P.state("U", block="blk").n
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        lap = P.scalar_field("lap")
        P.laplacian(lap, x)
        return x - alpha * lap  # out = in - alpha*Lap(in)

    P.set_apply(A, apply)
    kw = {} if preconditioner is None else {"preconditioner": preconditioner}
    phi = P.solve_linear(operator=A, rhs=U, method=_krylov(method), tol=tol, max_iter=max_iter, **kw)
    P.commit("blk", phi)
    return P


# ---- (A) codegen: pure Python, always runs ----
def test_apply_lambda_and_cg_codegen(t):
    src = _solve_program(t, method="cg").emit_cpp_program()
    for frag in ("pops::ApplyFn apply_A", "ctx.laplacian", "pops::cg_solve",
                 "std::make_shared<pops::MultiFab>(ctx.alloc_scalar_field"):
        assert frag in src, "the generated cg solve must contain %r\n%s" % (frag, src)


def test_bicgstab_codegen(t):
    src = _solve_program(t, method="bicgstab").emit_cpp_program()
    assert "pops::bicgstab_solve" in src, src
    assert "pops::ApplyFn{}" in src, "bicgstab uses the identity (empty) preconditioner\n%s" % src


def test_richardson_codegen(t):
    src = _solve_program(t, method="richardson").emit_cpp_program()
    assert "pops::richardson_solve" in src, src


# ---- (A') GeometricMG preconditioner (ADC-516): the complete non-identity route ----
def _gmres_call(src):
    """The single ``pops::gmres_solve(...)`` line of @p src."""
    return [ln for ln in src.splitlines() if "pops::gmres_solve(" in ln][0]


def _bicgstab_call(src):
    return [ln for ln in src.splitlines() if "pops::bicgstab_solve(" in ln][0]


def test_gmres_gmg_precond_codegen(t):
    # GMRES + GeometricMG lowers to a REAL ApplyFn (one V-cycle of the wired multigrid), NOT the empty
    # identity ApplyFn. The named precond lambda forwards to ctx.geometric_mg_precond_apply.
    src = _solve_program(t, method="gmres", preconditioner=_precond("geometric_mg")).emit_cpp_program()
    assert "ctx.geometric_mg_precond_apply" in src, "the MG V-cycle apply must be emitted\n%s" % src
    assert "pops::ApplyFn precond_mg" in src, "a named real precond ApplyFn must be emitted\n%s" % src
    call = _gmres_call(src)
    assert "precond_mg" in call, "gmres_solve must take the real precond, got: %s" % call
    assert "pops::ApplyFn{}" not in call, "gmres+gmg must NOT pass the empty ApplyFn: %s" % call


def test_bicgstab_gmg_precond_codegen(t):
    src = _solve_program(t, method="bicgstab",
                         preconditioner=_precond("geometric_mg")).emit_cpp_program()
    assert "ctx.geometric_mg_precond_apply" in src, src
    call = _bicgstab_call(src)
    assert "precond_mg" in call and "pops::ApplyFn{}" not in call, call


def test_identity_precond_byte_identical(t):
    # The identity (default) path is unchanged: the empty ApplyFn{}, no MG apply emitted. The explicit
    # Identity() descriptor and the None default lower to the SAME source.
    src_default = _solve_program(t, method="gmres").emit_cpp_program()
    src_identity = _solve_program(t, method="gmres",
                                  preconditioner=_precond("identity")).emit_cpp_program()
    assert src_default == src_identity, "explicit Identity() must match the None default byte-for-byte"
    assert "pops::ApplyFn{}" in _gmres_call(src_default), "identity gmres keeps the empty ApplyFn"
    assert "geometric_mg_precond_apply" not in src_default, "identity emits no MG apply"


def test_cg_gmg_precond_rejected(t):
    # CG / Richardson have no preconditioner slot in the matrix-free path: a non-identity precond is an
    # honest capability limit (ValueError naming GMRES/BiCGStab), not a transitional reject.
    for method in ("cg", "richardson"):
        try:
            _solve_program(t, method=method, preconditioner=_precond("geometric_mg"))
        except ValueError as exc:
            assert "CG/Richardson" in str(exc) and "GMRES" in str(exc), str(exc)
        else:
            raise AssertionError("%s + GeometricMG must raise ValueError" % method)


def test_unwired_preconditioners_are_not_public(t=None):
    # Clean break: no public Jacobi()/BlockJacobi() descriptors until their native C++ kernels exist.
    from pops.solvers import preconditioners
    assert not hasattr(preconditioners, "Jacobi")
    assert not hasattr(preconditioners, "BlockJacobi")


def test_string_precond_rejected(t):
    # Spec 5 sec.7: a bare string preconditioner is rejected, naming the typed alternative.
    P = t.Program("p")
    U = P.state("U", block="blk").n
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    try:
        P.solve_linear(operator=A, rhs=U, method=_krylov("gmres"), max_iter=10,
                       preconditioner="geometric_mg")
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
    assert P.validate() is True, "the solve_linear Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_max_iter_required(t):
    P = t.Program("p")
    U = P.state("U", block="blk").n
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    for bad in (None, 0, -5):
        try:
            P.solve_linear(operator=A, rhs=U, max_iter=bad)
        except ValueError as exc:
            assert "dynamic solver loops require max_iter" in str(exc), str(exc)
        else:
            raise AssertionError("max_iter=%r must raise the dynamic-loop budget error" % (bad,))


def test_tol_positive(t):
    P = t.Program("p")
    U = P.state("U", block="blk").n
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    for bad in (0.0, -1e-8):
        try:
            P.solve_linear(operator=A, rhs=U, max_iter=10, tol=bad)
        except ValueError as exc:
            assert "tol" in str(exc), str(exc)
        else:
            raise AssertionError("tol=%r must raise (a non-positive tolerance is a config error)" % bad)


def test_string_method_rejected(t):
    # Spec 5 sec.7: solve_linear takes a TYPED pops.solvers.krylov descriptor; a bare string
    # (known or unknown) is rejected and the error names the typed alternative.
    P = t.Program("p")
    U = P.state("U", block="blk").n
    A = P.matrix_free_operator("A")
    P.set_apply(A, lambda P, out, x: _helmholtz(P, x))
    for bad in ("cg", "minres"):
        try:
            P.solve_linear(operator=A, rhs=U, max_iter=10, method=bad)
        except TypeError as exc:
            assert "method" in str(exc) and "pops.solvers.krylov" in str(exc), str(exc)
        else:
            raise AssertionError("a string method=%r must raise TypeError" % (bad,))


def test_operator_must_be_matrix_free(t):
    P = t.Program("p")
    U = P.state("U", block="blk").n
    try:
        P.solve_linear(operator=U, rhs=U, max_iter=10)  # a State is not an operator
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

        import pops
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable in this interpreter
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    from _module_models import explicit_euler, first_order_rusanov, passive_scalar_module

    n = 16
    sim = pops.System(n=n, L=1.0, periodic=True)

    def passive_model(name):
        return passive_scalar_module(name)

    tol = 1e-10
    try:
        compiled = pops.compile_problem(
            model=passive_model("solve_prog"),
            time=_solve_program(t, name="solve_step", method="cg", tol=tol, max_iter=200))
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:200])
        return None

    assert compiled.program_name == "solve_step", "handle carries the program name"

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    try:
        sim.install(compiled,
                    instances={"blk": {"model": passive_model("solve_block"),
                                       "spatial": first_order_rusanov(),
                                       "time": explicit_euler(),
                                       "initial": np.stack([rho0])}})
    except RuntimeError as exc:
        print("-- (B) skipped: install could not build the block .so: %s --" % str(exc)[:200])
        return None
    sim.step(0.05)  # dt is irrelevant: the solve is dt-free
    out = np.array(sim._get_state("blk"))[0]

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

        import pops
    except Exception as exc:  # noqa: BLE001
        print("-- (B') skipped: pops/numpy unavailable: %s --" % exc)
        return None

    from pops.solvers import preconditioners
    from _module_models import explicit_euler, first_order_rusanov, passive_scalar_module

    n = 16
    sim = pops.System(n=n, L=1.0, periodic=True)

    def passive_model(name):
        return passive_scalar_module(name)

    tol = 1e-10
    prog = _solve_program(t, name="solve_gmg", method="gmres", tol=tol, max_iter=200,
                          preconditioner=preconditioners.GeometricMG())
    try:
        compiled = pops.compile_problem(model=passive_model("solve_gmg_prog"), time=prog)
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B') skipped: compile could not build the .so: %s --" % str(exc)[:200])
        return None

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    try:
        sim.install(compiled,
                    instances={"blk": {"model": passive_model("solve_gmg_block"),
                                       "spatial": first_order_rusanov(),
                                       "time": explicit_euler(),
                                       "initial": np.stack([rho0])}})
    except RuntimeError as exc:
        print("-- (B') skipped: install could not build the block .so: %s --" % str(exc)[:200])
        return None
    sim.step(0.05)
    out = np.array(sim._get_state("blk"))[0]

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
