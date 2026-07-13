#!/usr/bin/env python3
"""pops.lib.time.rk -- generic explicit Runge-Kutta from a Butcher tableau (epic ADC-399 / ADC-423).

``pops.lib.time.rk(P, block, state, tableau=...)`` lowers an arbitrary EXPLICIT Butcher tableau
(A, b, c) to the same
stage topology the hard-coded `rk4` macro emits (solve_fields + rhs + linear_combine, no RK class):

    k_i     = R(U + dt*sum_{j<i} A[i][j]*k_j)
    U^{n+1} = U + dt*sum_i b[i]*k_i

(A) IR parity, pure Python (always runs): rk(RK4_TABLEAU) produces the same exact-literal IR as the
    rk4 macro; rk(SSPRK2_TABLEAU) produces Heun's final affine
    combination U + dt(1/2 k1 + 1/2 k2); the tableau validation rejects an implicit (non-lower-tri)
    tableau and a b that does not sum to 1.

(B) Compiled trajectory parity (skips cleanly without _pops / a compiler / a visible Kokkos): the
    compiled rk(RK4_TABLEAU) program and the compiled rk4 program step an identical System to the SAME
    state bit-for-bit after target lowering. Self-skips, never fakes the engine.

Pure-Python IR construction is always available; the compiled section gates on the full toolchain.
"""
from typed_program_support import commits_by_block, state_refs

from decimal import Decimal
from fractions import Fraction
import sys

import pytest

from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.runtime.system import System  # ADC-545 advanced runtime seam


def _pops_time():
    global lt  # ready schemes live in pops.lib.time (Spec 4)
    try:
        import pops.time as t
        import pops.lib.time as lt  # ready schemes live in pops.lib.time (Spec 4)
    except Exception as exc:  # pops not importable here -> skip, never fake
        print("skip test_time_std_rk (pops.time unavailable: %s)" % exc)
        sys.exit(0)
    return t


def _coeff(node, value):
    for v, c in zip(node.inputs, node.attrs["coeffs"], strict=True):
        if v is value:
            return c
    raise AssertionError("value %r not an input of %r" % (value, node))


# ---- (A) IR parity: pure Python, always runs ----
def test_rk_rk4_tableau_matches_rk4_macro_exactly(t):
    """The data-driven and handwritten RK4 forms retain the same exact coefficients."""
    macro = t.Program("rk4")
    lt.rk4(macro, *state_refs(macro, "plasma"))
    generic = t.Program("rk4")  # the RK4_TABLEAU is named "rk4", so the node tags match too
    lt.rk(generic, *state_refs(generic, "plasma"), tableau=lt.RK4_TABLEAU)
    macro_nodes = [
        (v.vtype, v.op, v.name, tuple(i.id for i in v.inputs),
         tuple(dict(coeff) for coeff in v.attrs.get("coeffs", ())))
        for v in macro._values
    ]
    generic_nodes = [
        (v.vtype, v.op, v.name, tuple(i.id for i in v.inputs),
         tuple(dict(coeff) for coeff in v.attrs.get("coeffs", ())))
        for v in generic._values
    ]
    assert generic_nodes == macro_nodes
    assert generic._ir_hash() == macro._ir_hash(), (
        "RK4_TABLEAU and rk4 must use the same exact rational IR")


def test_rk_ssprk2_tableau_is_heun(t):
    """rk(SSPRK2_TABLEAU) commits Heun's U + dt(1/2 k1 + 1/2 k2): two stages, two equal-weighted RHS."""
    P = t.Program("ssprk2")
    lt.rk(P, *state_refs(P, "plasma"), tableau=lt.SSPRK2_TABLEAU)
    P.validate()
    node = commits_by_block(P)["plasma"]
    assert node.op == "linear_combine"
    states = [v for v in node.inputs if v.vtype == "state"]
    rhss = [v for v in node.inputs if v.vtype == "rhs"]
    assert len(states) == 1 and len(rhss) == 2, "Heun final stage = U0 + dt(1/2 k1 + 1/2 k2)"
    state_coeff = _coeff(node, states[0])
    assert state_coeff == {0: 1}
    assert isinstance(state_coeff[0], int)
    for r in rhss:
        c = _coeff(node, r)
        assert c == {1: Fraction(1, 2)}, "each k carries dt*1/2 (got %r)" % c
        assert isinstance(c[1], Fraction)


def test_rk_accepts_raw_triple(t):
    """A raw (A, b, c) triple is accepted (wrapped in a ButcherTableau)."""
    A = [[], [1]]
    b = [Fraction(1, 2), Fraction(1, 2)]
    c = [0, 1]
    P = t.Program("raw")
    lt.rk(P, *state_refs(P, "plasma"), tableau=(A, b, c))
    assert P.validate() is True
    heun = t.Program("raw")
    lt.rk(
        heun, *state_refs(heun, "plasma"), tableau=lt.ButcherTableau(A, b, c))
    assert P._ir_hash() == heun._ir_hash(), "the raw triple == its wrapped tableau IR"


def test_tableau_rejects_implicit(t):
    try:  # an entry on/above the diagonal is implicit -> rejected (rk lowers explicit only)
        lt.ButcherTableau(A=[[0.0], [1.0, 0.5]], b=[0.5, 0.5])
    except ValueError as exc:
        assert "lower-triangular" in str(exc) or "EXPLICIT" in str(exc)
    else:
        raise AssertionError("an implicit tableau must be rejected")


def test_tableau_rejects_inconsistent_weights(t):
    try:  # b must sum to 1 for a consistent RK method
        lt.ButcherTableau(A=[[], [1.0]], b=[0.5, 0.6])
    except ValueError as exc:
        assert "sum exactly to 1" in str(exc)
    else:
        raise AssertionError("weights that do not sum to 1 must be rejected")


def test_tableau_preserves_decimal_domain_and_is_immutable(t):
    """Tableau data are exact immutable values, not float-normalized mutable lists."""
    half = Decimal("0.5")
    tableau = lt.ButcherTableau(
        A=[[], [half]], b=[half, half], c=[Decimal("0"), half], name="decimal_heun")
    assert tableau.A == ((), (half,))
    assert tableau.b == (half, half)
    assert tableau.c == (Decimal("0"), half)
    assert all(isinstance(value, Decimal) for value in tableau.b)
    with pytest.raises(AttributeError):
        tableau.name = "changed"
    with pytest.raises(TypeError):
        tableau.A[1][0] = Decimal("0.25")


def test_tableau_derives_exact_nodes_and_normalizes_full_matrix(t):
    """Omitted c is derived exactly and an explicit full matrix has one canonical form."""
    tableau = lt.ButcherTableau(
        A=[[0, 0], [Fraction(1, 3), 0]],
        b=[Fraction(1, 4), Fraction(3, 4)],
    )
    assert tableau.A == ((), (Fraction(1, 3),))
    assert tableau.c == (Fraction(0, 1), Fraction(1, 3))


def test_tableau_validates_weights_and_nodes_exactly(t):
    """Near-equality never passes through the old binary tolerance."""
    with pytest.raises(ValueError, match="sum exactly to 1"):
        lt.ButcherTableau(
            A=[[], [Decimal("0.5")]],
            b=[Decimal("0.5"), Decimal("0.5000000000001")],
        )
    with pytest.raises(ValueError, match=r"c\[1\].*exact row sum"):
        lt.ButcherTableau(
            A=[[], [Decimal("0.5")]],
            b=[Decimal("0.5"), Decimal("0.5")],
            c=[0, Decimal("0.5000000000001")],
        )


def test_tableau_rejects_non_scalar_boolean_and_non_finite_coefficients(t):
    with pytest.raises(TypeError, match="finite coefficient"):
        lt.ButcherTableau(A=[[]], b=[True])
    with pytest.raises(ValueError, match="finite real coefficient"):
        lt.ButcherTableau(A=[[]], b=[float("nan")])


# ---- (B) compiled trajectory parity: skips cleanly without the full toolchain ----
def _passive_model(name):
    """A 1-variable model (rho), ZERO flux, default LINEAR source S = c*rho (R changes every stage)."""
    from pops.physics._facade import Model
    m = Model(name)
    (rho,) = m.conservative_vars("rho")
    u = m.primitive("u", 0.0 * rho)
    m.primitive_vars(rho=rho, u=u)
    m.conservative_from([rho])
    m.flux(x=[0.0 * rho], y=[0.0 * rho])
    m.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    m.source([0.75 * rho])
    return m


def _run_section_b(t):
    try:
        import numpy as np

        import pops
    except Exception as exc:  # noqa: BLE001
        print("-- (B) skipped: pops/numpy unavailable: %s --" % exc)
        return
    if not hasattr(System(n=8, L=1.0, periodic=True), "install_program"):
        print("-- (B) skipped: _pops lacks install_program (rebuild _pops) --")
        return

    def compiled_so(build, name):
        P = t.Program(name)
        build(P)
        try:
            from pops.codegen._compile_drivers import compile_problem
            return compile_problem(model=_passive_model(name + "_m"), time=P)
        except RuntimeError as exc:
            print("-- (B) skipped: compile_problem could not build the .so: %s --" % str(exc)[:160])
            return None

    macro = compiled_so(
        lambda P: lt.rk4(P, *state_refs(P, "blk")), "rk4_macro")
    if macro is None:
        return
    generic = compiled_so(
        lambda P: lt.rk(
            P, *state_refs(P, "blk"), tableau=lt.RK4_TABLEAU),
        "rk4_macro",
    )
    if generic is None:
        return

    n = 16
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho0 = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

    def run(handle):
        sim = System(n=n, L=1.0, periodic=True)
        try:
            cm = _passive_model("rk_block").compile(backend="production")
        except RuntimeError as exc:
            print("-- (B) skipped: model compile failed: %s --" % str(exc)[:160])
            return None
        sim.add_equation("blk", cm, spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
                         time=pops.Explicit(method="euler"))
        sim.set_state("blk", np.stack([rho0]))
        sim.install_program(handle.so_path)
        for _ in range(5):
            sim.step(0.01)
        return np.array(sim.get_state("blk"))[0]

    a = run(macro)
    b = run(generic)
    if a is None or b is None:
        return
    err = float(np.abs(a - b).max())
    print("  rk(RK4) vs rk4 compiled trajectory: max|d| = %.2e" % err)
    assert err == 0.0, "rk(RK4_TABLEAU) and rk4 lower to a bit-identical trajectory"


def _run():
    t = _pops_time()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_std_rk (A: %d checks)" % len(fns))
    _run_section_b(t)


if __name__ == "__main__":
    _run()
