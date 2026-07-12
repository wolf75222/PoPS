"""pops.lib.time -- standard library of time-stepping macros that LOWER to the Program IR (ADC-407).

These are Python functions that BUILD pops.time.Program IR (not separate C++ steppers): Forward Euler,
SSPRK2, SSPRK3, RK4 and a Strang-splitting combinator. They reuse the merged Phase 2a builder ops and
the affine algebra over dt, so a scheme is expressed once, without any scheme-specific class (spec
acceptance: "RK4 is expressed without a special RK4 class"). This test exercises only IR CONSTRUCTION
(no codegen, no compilation): it asserts each macro produces the expected per-input coefficient
polynomials in dt on the committed state. Parity vs the old C++ steppers needs compile_problem (Phase 2c)
and is deferred.

Run with python3 (PYTHONPATH = built pops package).
"""
from typed_program_support import commits_by_block, state_refs

from fractions import Fraction

from pops import time as adctime
import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)


def _coeff(node, value):
    for v, c in zip(node.inputs, node.attrs["coeffs"], strict=True):
        if v is value:
            return c
    raise AssertionError("value %r not an input of %r" % (value, node))


def _committed(P, block):
    P.validate()
    node = commits_by_block(P)[block]
    assert node.op == "linear_combine", node.op
    states = [v for v in node.inputs if v.vtype == "state"]
    rhss = [v for v in node.inputs if v.vtype == "rhs"]
    return node, states, rhss


def _approx(d, power, val):
    return power in d and abs(d[power] - val) < 1e-15 and len(d) == 1


def test_forward_euler():
    P = adctime.Program("fe")
    libtime.forward_euler(P, *state_refs(P, "plasma"))
    node, states, rhss = _committed(P, "plasma")
    assert len(states) == 1 and len(rhss) == 1
    assert _coeff(node, states[0]) == {0: 1}     # U
    assert _coeff(node, rhss[0]) == {1: 1}       # dt * R
    print("OK  forward_euler -> U + dt*R")


def test_ssprk2():
    P = adctime.Program("ssprk2")
    libtime.ssprk2(P, *state_refs(P, "plasma"))
    node, states, rhss = _committed(P, "plasma")
    # Canonical Butcher form: U2 = U0 + dt*(k0+k1)/2.
    assert len(states) == 1 and len(rhss) == 2
    assert _coeff(node, states[0]) == {0: 1}
    assert all(_coeff(node, rhs) == {1: Fraction(1, 2)} for rhs in rhss)
    print("OK  ssprk2 -> U0 + 0.5 dt k0 + 0.5 dt k1")


def test_ssprk3():
    P = adctime.Program("ssprk3")
    libtime.ssprk3(P, *state_refs(P, "plasma"))
    node, states, rhss = _committed(P, "plasma")
    # Canonical Butcher form of the same SSP method.
    assert len(states) == 1 and len(rhss) == 3
    assert _coeff(node, states[0]) == {0: 1}
    cs = sorted(_coeff(node, rhs)[1] for rhs in rhss)
    assert cs == [Fraction(1, 6), Fraction(1, 6), Fraction(2, 3)]
    print("OK  ssprk3 -> U0 + dt*(k0/6 + k1/6 + 2 k2/3)")


def test_rk4_no_special_class():
    P = adctime.Program("rk4")
    libtime.rk4(P, *state_refs(P, "plasma"))
    node, states, rhss = _committed(P, "plasma")
    # Unp1 = U0 + dt/6 k1 + dt/3 k2 + dt/3 k3 + dt/6 k4
    assert len(states) == 1 and len(rhss) == 4
    assert _coeff(node, states[0]) == {0: 1}
    kcoeffs = sorted(_coeff(node, r)[1] for r in rhss)
    assert kcoeffs == sorted([
        Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6),
    ])
    print("OK  rk4 (no special RK4 class) -> U0 + dt(1/6 k1 + 1/3 k2 + 1/3 k3 + 1/6 k4)")


def test_strang_combinator():
    # Strang splitting H(dt/2); S(dt); H(dt/2) as IR-building callables. Here H and S are trivial
    # affine updates so we can check the macro chains three stages and commits the last.
    P = adctime.Program("strang")

    def half_flow(prog, U, frac, *, at):
        R = prog._rhs_legacy(state=U, fields=prog.solve_fields(U), flux=True, sources=["default"])
        return prog.linear_combine(None, U + (frac * prog.dt) * R, at=at)

    def source(prog, U, frac, *, at):
        S = prog._rhs_legacy(state=U, fields=None, flux=False, sources=["default"])
        return prog.linear_combine(None, U + (frac * prog.dt) * S, at=at)

    out = libtime.strang(
        P, *state_refs(P, "plasma"), half_flow=half_flow, source=source)
    P.validate()
    assert commits_by_block(P)["plasma"] is out and out.vtype == "state"
    # three linear_combine stages were built (two half flows + one source)
    n_lc = sum(1 for v in P._values if v.op == "linear_combine")
    assert n_lc == 3, n_lc
    print("OK  strang combinator chains H(dt/2); S(dt); H(dt/2)")


def main():
    test_forward_euler()
    test_ssprk2()
    test_ssprk3()
    test_rk4_no_special_class()
    test_strang_combinator()
    print("test_time_std : tout est vert")


if __name__ == "__main__":
    main()
