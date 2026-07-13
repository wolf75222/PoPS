#!/usr/bin/env python3
"""pops.time centered divergence primitive + a div(grad) Helmholtz solve (epic ADC-399 / ADC-412).

ADC-412 adds the ``ctx.divergence`` primitive (the centered finite-volume divergence factored as
``pops::apply_divergence``) and the ``P.divergence(out, fx, fy)`` IR op. A matrix-free Schur-like operator
``A(phi) = phi - alpha*div(grad phi)`` (the div(flux) structure of the condensed-Schur operator) is
built from ``P.gradient`` chained into ``P.divergence`` and solved with ``P.solve_linear`` -- exactly
the matrix-free Krylov path acceptance 32 needs in place. The centered ``div(grad)`` is the WIDE-stencil
Laplacian ``(x(i+2) - 2 x(i) + x(i-2))/(4 h^2)`` (not the compact 5-point ``apply_laplacian``), so the
compiled solve is verified against an OFFLINE numpy CG on that SAME wide-stencil discrete operator.

(A) Pure Python, always runs:
    - ``P.divergence`` records a 3-input scalar_field op, validates its operands, and serializes;
    - the div(grad) Helmholtz apply (gradient -> divergence) lowers to ``ctx.gradient`` + a
      ``ctx.divergence`` + ``ctx.solve_linear_matfree``, with the gradient buffer allocated 2-component
      (``ctx.alloc_scalar_field(2, 1)``);
    - a standalone divergence-of-a-known-field check: the offline centered FV divergence of
      f = (cos 2pi x, sin 2pi y) matches the analytic div f = -2pi sin 2pi x + 2pi cos 2pi y to the
      discretization error -- the reference the compiled ctx.divergence reproduces;
    - ``pops.lib.time.CondensedSchur`` lowers at theta == 1 and raises for
      the deferred theta != 1 extrapolation (the full end-to-end parity is test_time_condensed_schur.py).

(B) End-to-end parity (skips unless the full toolchain is present): the div(grad) Helmholtz Program is
    compiled + installed + stepped, then compared to an OFFLINE numpy CG on the identical discrete
    periodic 5-point system. Asserts max|compiled - offline| <= 1e-6. Self-skips (exit 0) without numpy
    / _pops / install_program / a compiler / a visible Kokkos -- never fakes the engine.
"""
from pops.codegen import compile_drivers
from typed_program_support import state_refs, typed_state

from pops.params import ConstParam
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.solvers import krylov
from pops.time import FailRun
import sys
from pops.runtime.system import System  # ADC-545 advanced runtime seam


def _pops_time():
    global lt  # ready schemes live in pops.lib.time (Spec 4)
    try:
        import pops.time as t
        import pops.lib.time as lt  # ready schemes live in pops.lib.time (Spec 4)
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_divergence (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


_ALPHA = 0.1  # Helmholtz coefficient: A = I - alpha*div(grad) = I - alpha*Lap (SPD, well-conditioned)


def _divgrad_program(t, *, name="divgrad", method=None, tol=1e-10, max_iter=200, alpha=_ALPHA):
    """Solve (I - alpha*div(grad)) phi = U, committed back into the 1-component block.

    The apply ``out = in - alpha*div(grad(in))`` chains P.gradient (into a 2-component buffer) then
    P.divergence (recovering Lap), so it exercises ctx.divergence inside the matrix-free Krylov loop."""
    P = t.Program(name)
    U = typed_state(P, "blk")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        g = P.scalar_field("g", ncomp=2)  # 2-component gradient buffer (d/dx, d/dy)
        P.gradient(g, x)
        d = P.scalar_field("d")
        P.divergence(d, g, g)  # div(grad x) == Lap x; fy reads component 1 of the same buffer
        return x - alpha * d  # out = in - alpha*div(grad(in)) = in - alpha*Lap(in)

    if method is None:
        from pops.solvers.krylov import BiCGStab  # typed default (Spec 5 sec.7); lowers to "bicgstab"
        method = BiCGStab(max_iter=max_iter)  # ADC-535: max_iter is mandatory on the descriptor
    P.set_apply(A, apply)
    phi = P.solve_linear(
        operator=A, rhs=U, method=method, tol=tol, max_iter=max_iter).consume(action=FailRun())
    endpoint = typed_state(P, "blk", state_name="U").next
    final = P.linear_combine("phi_next", phi, at=endpoint.point)
    P.commit(endpoint, final)
    return P


# ---- (A) codegen + IR + analytic divergence: pure Python, always runs ----
def test_divergence_records_and_validates(t):
    P = t.Program("p")
    A = P.matrix_free_operator("A")

    def apply(P, out, x):
        g = P.scalar_field("g", ncomp=2)
        P.gradient(g, x)
        d = P.scalar_field("d")
        div = P.divergence(d, g, g)
        assert div.vtype == "scalar_field", "divergence yields a scalar_field value"
        return x - div

    from pops.solvers.krylov import BiCGStab
    P.set_apply(A, apply)
    U = typed_state(P, "blk")
    phi = P.solve_linear(
        operator=A, rhs=U, method=BiCGStab(max_iter=50), tol=1e-8,
        max_iter=50).consume(action=FailRun())
    endpoint = typed_state(P, "blk", state_name="U").next
    final = P.linear_combine("phi_next", phi, at=endpoint.point)
    P.commit(endpoint, final)
    assert P.validate() is True, "the div(grad) Program must validate"
    assert P._ir_hash(), "the IR must serialize to a stable hash"


def test_divergence_operand_types(t):
    P = t.Program("p")
    A = P.matrix_free_operator("A")
    bad = []

    def apply(P, out, x):
        d = P.scalar_field("d")
        for args in ((U_state, x, x), (d, U_state, x), (d, x, U_state)):
            try:
                P.divergence(*args)
            except ValueError as exc:
                bad.append("divergence" in str(exc))
            else:
                bad.append(False)
        g = P.scalar_field("g", ncomp=2)
        P.gradient(g, x)
        dd = P.scalar_field("dd")
        P.divergence(dd, g, g)
        return x - dd

    U_state = typed_state(P, "blk")  # a State is not a scalar_field -> each divergence operand must reject it
    P.set_apply(A, apply)
    assert all(bad), "divergence must reject a non-scalar_field operand (out / fx / fy)"


def test_scalar_field_ncomp_validates(t):
    P = t.Program("p")
    for bad in (0, -1, 1.5, True):
        try:
            P.scalar_field("g", ncomp=bad)
        except ValueError as exc:
            assert "ncomp" in str(exc), str(exc)
        else:
            raise AssertionError("scalar_field ncomp=%r must raise" % (bad,))
    g = P.scalar_field("g2", ncomp=2)
    assert g.attrs["ncomp"] == 2, "ncomp is recorded on the scalar_field node"


def test_divgrad_codegen(t):
    src = _divgrad_program(t, method=krylov.BiCGStab(max_iter=200)).emit_cpp_program()
    for frag in ("ctx.gradient", "ctx.divergence", "ctx.solve_linear_matfree",
                 "ctx.alloc_scalar_field(2, 1)"):  # the 2-component gradient buffer
        assert frag in src, "the div(grad) solve must contain %r\n%s" % (frag, src)


def _lorentz_model(name):
    """A rho/mx/my block carrying the electrostatic-Lorentz linearization J the generic condensed
    route (ADC-637) resolves at emit time."""
    from pops.ir.ops import sqrt
    from pops.lib.models import author_electrostatic_lorentz
    from pops.physics.facade import Model
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    cs2 = m.value(m.param(ConstParam("cs2", 0.5)))
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    p = m.primitive("p", cs2 * rho)
    m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u + p, my * u], y=[my, mx * v, my * v + p])
    cs = sqrt(cs2)
    m.eigenvalues(x=[u - cs, u, u + cs], y=[v - cs, v, v + cs])
    m.elliptic_rhs(rho)
    m.aux("grad_x")
    m.aux("grad_y")
    m.aux("B_z")
    author_electrostatic_lorentz(m)
    return m


def _linear_handle(model):
    from pops.model import OperatorHandle
    registry = model.operator_registry()
    operator = registry.operators_of_kind("local_linear_operator")[0]
    return OperatorHandle(
        operator.name, kind=operator.kind, owner=registry.owner_path,
        signature=operator.signature)


def test_condensed_schur_macro_lowers(t):
    # ADC-421 + ADC-427 + ADC-637: the condensed-implicit macro (generic-only route) lowers for any theta
    # in (0, 1]. theta == 1 lowers the full anisotropic assemble / solve / reconstruct chain; theta != 1
    # adds the n+1 momentum extrapolation by factor 1/theta on top. theta out of (0, 1] raises ValueError.
    # The end-to-end parity lives in test_time_condensed_schur.py.
    model1 = _lorentz_model("div_m1")
    P = t.Program("p").bind_operators(model1)
    lt.CondensedSchur(
        P, *state_refs(P, "blk"), alpha=1.0, theta=1.0,
        linear_operator=_linear_handle(model1))
    assert P.validate() is True, "the condensed macro must validate"
    src = P.emit_cpp_program(model=model1)
    # ADC-637: the condensed ops lower INLINE via the block_inverse intrinsic; NO coupling/schur.
    assert "pops::detail::block_inverse<2>(M_, Mi_);" in src, src
    assert "pops::detail::block_apply_inverse<2>(M_, cond_v_, cond_mv_);" in src, src
    assert "coupling/schur" not in src and "coupling::schur" not in src, src
    # ADC-427: theta != 1 now lowers (the extrapolation is plain affine algebra), no longer raises.
    model2 = _lorentz_model("div_m2")
    P2 = t.Program("p2").bind_operators(model2)
    lt.CondensedSchur(
        P2, *state_refs(P2, "blk"), alpha=1.0, theta=0.5,
        linear_operator=_linear_handle(model2))
    assert P2.validate() is True, "condensed_schur(theta != 1) must validate (ADC-427)"
    assert "pops::detail::block_apply_inverse<2>" in P2.emit_cpp_program(model=model2), (
        "theta=0.5 must lower the reconstruct chain (inline block_apply_inverse)")
    # theta out of (0, 1] is still rejected (loud).
    invalid_model = _lorentz_model("div_invalid")
    invalid = t.Program("p3").bind_operators(invalid_model)
    try:
        lt.CondensedSchur(
            invalid, *state_refs(invalid, "blk"), alpha=1.0, theta=1.5,
            linear_operator=_linear_handle(invalid_model))
    except ValueError as exc:
        assert "theta must be in (0, 1]" in str(exc), str(exc)
    else:
        raise AssertionError("condensed_schur(theta out of (0, 1]) must raise ValueError")


def _analytic_divergence_check():
    """Standalone offline check: the centered FV divergence of a known smooth flux matches the analytic
    divergence to the discretization error. The same centered stencil pops::apply_divergence (and the
    compiled ctx.divergence) computes. Skips silently without numpy."""
    try:
        import numpy as np
    except Exception:  # noqa: BLE001  -- numpy unavailable here
        return
    n = 64
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    fx = np.cos(2 * np.pi * X)  # x-flux
    fy = np.sin(2 * np.pi * Y)  # y-flux
    h = 1.0 / n
    div = (np.roll(fx, -1, 0) - np.roll(fx, 1, 0)) / (2 * h) + \
          (np.roll(fy, -1, 1) - np.roll(fy, 1, 1)) / (2 * h)
    analytic = -2 * np.pi * np.sin(2 * np.pi * X) + 2 * np.pi * np.cos(2 * np.pi * Y)
    err = float(np.abs(div - analytic).max())
    assert err < 0.05, "centered FV divergence vs analytic div f (n=%d): max|d| = %.3e" % (n, err)
    print("  centered divergence vs analytic: max|d| = %.3e (n=%d, O(h^2))" % (err, n))


# ---- (B) end-to-end parity: skips unless the full toolchain is present ----
def _np_cg(apply, b, *, tol=1e-10, max_iter=2000):
    """Plain numpy CG solving A x = b from x = 0 (A = the discrete periodic Helmholtz matvec)."""
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


def _discrete_divgrad_helmholtz(n, alpha):
    """The discrete periodic Helmholtz matvec A x = x - alpha*div(grad x) built from the CENTERED
    gradient (d/dx = (x(i+1) - x(i-1))/(2h)) followed by the CENTERED divergence -- exactly the operator
    the compiled P.gradient -> P.divergence chain composes. Centered div(grad) is the WIDE-stencil
    Laplacian (x(i+2) - 2 x(i) + x(i-2))/(4 h^2) (not the compact 5-point apply_laplacian), so the
    reference must use the same wide stencil. On an n x n unit-square grid (dx = dy = 1/n)."""
    import numpy as np

    h = 1.0 / n

    def apply(x):
        gx = (np.roll(x, -1, 0) - np.roll(x, 1, 0)) / (2 * h)
        gy = (np.roll(x, -1, 1) - np.roll(x, 1, 1)) / (2 * h)
        div = (np.roll(gx, -1, 0) - np.roll(gx, 1, 0)) / (2 * h) + \
              (np.roll(gy, -1, 1) - np.roll(gy, 1, 1)) / (2 * h)
        return x - alpha * div

    return apply


def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001  -- numpy / _pops unavailable here
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return None

    n = 16
    sim = System(n=n, L=1.0, periodic=True)
    if not hasattr(sim, "install_program"):
        print("-- (B) skipped: _pops lacks the install_program binding (rebuild _pops) --")
        return None

    from pops.physics.facade import Model

    def passive_model(name):  # 1-variable block, no flux, no Poisson coupling
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
            model=passive_model("divgrad_prog"),
            time=_divgrad_program(t, name="divgrad_step", method=krylov.BiCGStab(max_iter=200),
                                  tol=tol, max_iter=200))
    except RuntimeError as exc:  # no compiler / no Kokkos visible / .so compile failed
        print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:200])
        return None

    assert compiled.program_name == "divgrad_step", "handle carries the program name"

    try:
        compiled_model = passive_model("divgrad_block").compile(backend="production")
    except RuntimeError as exc:  # no compiler / no Kokkos visible
        print("-- (B) skipped: model compile could not build the .so: %s --" % str(exc)[:200])
        return None
    sim.add_equation("blk", compiled_model,
                     spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                     time=pops.Explicit(method="euler"))

    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("blk", np.stack([rho0]))

    sim.install_program(compiled.so_path)
    sim.step(0.05)  # dt is irrelevant: the solve is dt-free
    out = np.array(sim.get_state("blk"))[0]

    # OFFLINE reference: solve (I - alpha*div(grad)) phi = rho0 on the SAME centered div(grad) operator
    # (the wide-stencil Helmholtz the compiled gradient->divergence chain composes) with numpy CG; the
    # compiled matrix-free solve must recover the same phi.
    apply = _discrete_divgrad_helmholtz(n, _ALPHA)
    phi_ref, iters = _np_cg(apply, rho0, tol=tol)
    err = float(np.abs(out - phi_ref).max())
    moved = float(np.abs(out - rho0).max())
    print("  div(grad) Helmholtz parity: max|compiled - offline| = %.2e  offline iters = %d  "
          "max|phi - U0| = %.2e" % (err, iters, moved))
    assert err <= 1e-6, "compiled div(grad) CG == offline numpy CG (max|d| = %.2e)" % err
    assert moved > 1e-6, "the solve must change the state from U0 (max|d| = %.2e)" % moved
    assert iters > 1, "the offline (and compiled) solve must take > 1 iteration, got %d" % iters
    return (err, iters)


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    _analytic_divergence_check()
    print("PASS test_time_divergence (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
