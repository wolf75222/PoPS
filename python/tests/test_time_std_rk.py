#!/usr/bin/env python3
"""pops.lib.time.rk -- generic explicit Runge-Kutta from a Butcher tableau.

``pops.lib.time.rk(P, block, tableau, rhs_operator=...)`` lowers an arbitrary EXPLICIT Butcher
tableau (A, b, c) to typed operator calls plus linear_combine nodes, no RK class:

    k_i     = R(U + dt*sum_{j<i} A[i][j]*k_j)
    U^{n+1} = U + dt*sum_i b[i]*k_i

IR construction is always available and is the scope of this test. Runtime integration belongs in the
clean ``compile_problem -> System.install`` integration tests, not in this historical macro unit test.
"""
import pops.model as pm
import pops.time as t
import pops.lib.time as lt  # ready schemes live in pops.lib.time (Spec 4)


def _coeff(node, value):
    for v, c in zip(node.inputs, node.attrs["coeffs"], strict=True):
        if v is value:
            return c
    raise AssertionError("value %r not an input of %r" % (value, node))


# ---- (A) IR parity: pure Python, always runs ----
def _program(t, name):
    m = pm.Module(name + "_module")
    U = m.state_space("U", ("rho",))
    rhs = m.operator(
        "rhs", signature=(U,) >> pm.Rate(U), kind="local_rate",
        capabilities={"produces_rate": True}, lowering={"flux": False, "sources": []},
        expr=0.0)
    return t.Program(name).bind_operators(m), rhs


def test_rk_rk4_tableau_matches_rk4_macro(t):
    """rk(RK4_TABLEAU) lowers to the same IR as the named rk4 macro."""
    macro, rhs = _program(t, "rk4")
    lt.rk4(macro, "plasma", rhs_operator=rhs)
    generic, rhs2 = _program(t, "rk4")
    lt.rk(generic, "plasma", lt.RK4_TABLEAU, rhs_operator=rhs2)
    assert generic._ir_hash() == macro._ir_hash(), \
        "rk(RK4_TABLEAU) must produce the same IR as the rk4 macro"


def test_rk_ssprk2_tableau_is_heun(t):
    """rk(SSPRK2_TABLEAU) commits Heun's U + dt(1/2 k1 + 1/2 k2): two stages, two equal-weighted RHS."""
    P, rhs = _program(t, "ssprk2")
    lt.rk(P, "plasma", lt.SSPRK2_TABLEAU, rhs_operator=rhs)
    P.validate()
    node = P.commits()["plasma"]
    assert node.op == "linear_combine"
    states = [v for v in node.inputs if v.vtype == "state"]
    rhss = [v for v in node.inputs if v.vtype == "rhs"]
    assert len(states) == 1 and len(rhss) == 2, "Heun final stage = U0 + dt(1/2 k1 + 1/2 k2)"
    assert _coeff(node, states[0]) == {0: 1.0}
    for r in rhss:
        c = _coeff(node, r)
        assert c == {1: 0.5}, "each k carries dt*1/2 (got %r)" % c


def test_rk_accepts_raw_triple(t):
    """A raw (A, b, c) triple is accepted (wrapped in a ButcherTableau)."""
    A = [[], [1.0]]
    b = [0.5, 0.5]
    c = [0.0, 1.0]
    P, rhs = _program(t, "raw")
    lt.rk(P, "plasma", (A, b, c), rhs_operator=rhs)
    assert P.validate() is True
    node = P.commits()["plasma"]
    rhss = [v for v in node.inputs if v.vtype == "rhs"]
    assert len(rhss) == 2


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
        assert "sum to 1" in str(exc)
    else:
        raise AssertionError("weights that do not sum to 1 must be rejected")


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(t)
        print("ok", fn.__name__)
    print("PASS test_time_std_rk (A: %d checks)" % len(fns))


if __name__ == "__main__":
    _run()
