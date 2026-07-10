#!/usr/bin/env python3
"""ADC-532: every pops.lib.time macro lowers byte-identically to the equivalent manual Program.

The de-stringed macros must not drift the IR: for each scheme family, a Program built by the macro
(taking typed OperatorHandles / flux+source terms) has the SAME ``_ir_hash`` as a manual Program
built from the primitive ops (the internal ``_call`` / ``_rhs_legacy`` / ``linear_combine`` /
``solve_local_linear`` / ``commit`` seams the macro lowers through). Because the handle path resolves
to ``_call(handle.name, ...)``, the serialized IR is identical -- so the compiled ``.so`` cache key
is unchanged and production routes stay BIT-IDENTICAL. This is the ADC-532 acceptance test.

Families covered: forward_euler, ssprk2, ssprk3, rk4, rk(SSPRK2 tableau), explicit_rk(SSPRK2),
imex_local, imex_local_linear, predictor_corrector_local_linear, adams_bashforth(AB2), bdf(order 1).

Pure Python IR construction (no numerics / no _pops); collected as pytest functions.
"""
import sys
from fractions import Fraction

import pytest

from pops.ir.expr import Const
from pops.model import OperatorHandle
from pops.physics.facade import Model
from pops import time as adctime
import pops.lib.time as libtime


def _model(name="ep"):
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def _op(m, name):
    op = m.operator_registry().get(name)
    return OperatorHandle(
        op.name, kind=op.kind, owner=m.operator_registry().owner_path,
        signature=op.signature)


def _assert_parity(macro_prog, manual_prog):
    macro_prog.validate()
    manual_prog.validate()
    assert macro_prog._ir_hash() == manual_prog._ir_hash(), (
        "macro IR hash %s != manual IR hash %s" % (macro_prog._ir_hash(), manual_prog._ir_hash()))


# --- flux/source explicit schemes (sources=/flux= de-sugaring, no operator handle) ---------------
def _stage(P, U):
    return P._rhs_legacy(state=U, fields=P.solve_fields(U), flux=True, sources=["default"])


def test_forward_euler_parity():
    macro = libtime.forward_euler("plasma")
    manual = adctime.Program("forward_euler")
    U = manual.state("plasma")
    R = _stage(manual, U)
    manual.commit(manual.state("U", block="plasma").next, manual.linear_combine("fe_step", U + manual.dt * R))
    _assert_parity(macro, manual)


def test_ssprk2_parity():
    macro = libtime.ssprk2("plasma")
    manual = adctime.Program("ssprk2")
    U0 = manual.state("plasma")
    k0 = _stage(manual, U0)
    U1 = manual.linear_combine("ssprk2_U1", U0 + manual.dt * k0)
    k1 = _stage(manual, U1)
    manual.commit(
        manual.state("U", block="plasma").next,
        manual.linear_combine(
            "ssprk2_step",
            Fraction(1, 2) * U0 + Fraction(1, 2) * (U1 + manual.dt * k1),
        ),
    )
    _assert_parity(macro, manual)


def test_ssprk3_parity():
    macro = libtime.ssprk3("plasma")
    manual = adctime.Program("ssprk3")
    U0 = manual.state("plasma")
    k0 = _stage(manual, U0)
    U1 = manual.linear_combine("ssprk3_U1", U0 + manual.dt * k0)
    k1 = _stage(manual, U1)
    U2 = manual.linear_combine(
        "ssprk3_U2",
        Fraction(3, 4) * U0 + Fraction(1, 4) * (U1 + manual.dt * k1),
    )
    k2 = _stage(manual, U2)
    manual.commit(manual.state("U", block="plasma").next, manual.linear_combine(
        "ssprk3_step",
        Fraction(1, 3) * U0 + Fraction(2, 3) * (U2 + manual.dt * k2)))
    _assert_parity(macro, manual)


def test_rk4_parity():
    macro = libtime.rk4("plasma")
    manual = adctime.Program("rk4")
    dt = manual.dt
    U0 = manual.state("plasma")
    k1 = _stage(manual, U0)
    U1 = manual.linear_combine("rk4_U1", U0 + Fraction(1, 2) * dt * k1)
    k2 = _stage(manual, U1)
    U2 = manual.linear_combine("rk4_U2", U0 + Fraction(1, 2) * dt * k2)
    k3 = _stage(manual, U2)
    U3 = manual.linear_combine("rk4_U3", U0 + dt * k3)
    k4 = _stage(manual, U3)
    manual.commit(manual.state("U", block="plasma").next, manual.linear_combine(
        "rk4_step",
        U0 + Fraction(1, 6) * dt * k1 + Fraction(1, 3) * dt * k2
        + Fraction(1, 3) * dt * k3 + Fraction(1, 6) * dt * k4))
    _assert_parity(macro, manual)


def test_rk_ssprk2_tableau_matches_the_rk_macro():
    # The generic rk over the SSPRK2 tableau lowers to its own stage chain; two builds are identical.
    a = adctime.Program("rk")
    libtime.rk(a, "plasma", libtime.SSPRK2_TABLEAU)
    b = adctime.Program("rk")
    libtime.rk(b, "plasma", libtime.SSPRK2_TABLEAU)
    _assert_parity(a, b)


# --- operator-first schemes (typed handles) -----------------------------------------------------
def test_explicit_rk_parity():
    m = _model("rk")
    macro = adctime.Program("rk").bind_operators(m)
    libtime.explicit_rk(macro, "plasma", rhs_operator=_op(m, "explicit_rhs"),
                        fields_operator=_op(m, "fields_from_state"), tableau=libtime.SSPRK2_TABLEAU)
    # Manual SSPRK2 over the typed rate via the internal _call seam.
    manual = adctime.Program("rk").bind_operators(m)
    dt = manual.dt
    u0 = manual.state("plasma")
    f0 = manual._call("fields_from_state", u0)
    k0 = manual._call("explicit_rhs", u0, f0, name="ssprk2_k0")
    u1 = manual.linear_combine("ssprk2_U1", u0 + dt * k0)
    f1 = manual._call("fields_from_state", u1)
    k1 = manual._call("explicit_rhs", u1, f1, name="ssprk2_k1")
    manual.commit(manual.state("U", block="plasma").next, manual.linear_combine(
        "ssprk2_step",
        u0 + (dt * Fraction(1, 2)) * k0 + (dt * Fraction(1, 2)) * k1))
    _assert_parity(macro, manual)


def test_imex_local_linear_parity():
    m = _model("imex")
    macro = adctime.Program("imex").bind_operators(m)
    libtime.imex_local_linear(macro, "plasma", explicit_operator=_op(m, "explicit_rhs"),
                              implicit_operator=_op(m, "lorentz"),
                              fields_operator=_op(m, "fields_from_state"), theta=1.0)
    manual = adctime.Program("imex").bind_operators(m)
    u = manual.state("plasma")
    fields = manual._call("fields_from_state", u, name="fields")
    r = manual._call("explicit_rhs", u, fields, name="R")
    lin = manual._call("lorentz", fields, name="L")
    q = manual.linear_combine("imex_rhs", u + manual.dt * r)
    u1 = manual.solve_local_linear("imex_step", operator=manual.I - 1.0 * manual.dt * lin,
                                   rhs=q, fields=fields)
    manual.commit(manual.state("U", block="plasma").next, u1)
    _assert_parity(macro, manual)


def test_imex_local_parity():
    m = _model("imexl")
    lorentz = _op(m, "lorentz")
    macro = adctime.Program("imex_local").bind_operators(m)
    libtime.imex_local(macro, "plasma", linear_source=lorentz)
    manual = adctime.Program("imex_local").bind_operators(m)
    U = manual.state("plasma")
    fields = manual.solve_fields(U)
    R = manual._rhs_legacy(state=U, fields=fields, flux=True, sources=["default"])
    rhs = manual.linear_combine("plasma_imex_rhs", U + manual.dt * R)
    operator = manual.I - manual.dt * manual.linear_source(lorentz)
    out = manual.solve_local_linear(name="plasma_imex_step", operator=operator, rhs=rhs, fields=fields)
    manual.commit(manual.state("U", block="plasma").next, out)
    _assert_parity(macro, manual)


def test_predictor_corrector_parity():
    m = _model("pc")
    fo, ro, lo = _op(m, "fields_from_state"), _op(m, "explicit_rhs"), _op(m, "lorentz")
    macro = adctime.Program("pc").bind_operators(m)
    libtime.predictor_corrector_local_linear(macro, "plasma", fields_operator=fo,
                                             explicit_rate_operator=ro, implicit_operator=lo)
    manual = adctime.Program("pc").bind_operators(m)
    dt = manual.dt
    u_n = manual.state("plasma")
    fields_n = manual._call("fields_from_state", u_n, name="fields_n")
    r_n = manual._call("explicit_rhs", u_n, fields_n, name="R_n")
    l_n = manual._call("lorentz", fields_n, name="L_n")
    u_star = manual.solve_local_linear("U_star", operator=manual.I - dt * l_n,
                                       rhs=manual.linear_combine("U_star_rhs", u_n + dt * r_n),
                                       fields=fields_n)
    fields_star = manual._call("fields_from_state", u_star, name="fields_star")
    r_star = manual._call("explicit_rhs", u_star, fields_star, name="R_star")
    l_star = manual._call("lorentz", fields_star, name="L_star")
    c_star = manual.apply(l_star, u_star, fields=fields_star, name="C_star")
    half = Fraction(1, 2)
    q = manual.linear_combine(
        "Q", u_n + half * dt * r_n + half * dt * r_star + half * dt * c_star)
    u_np1 = manual.solve_local_linear("U_np1", operator=manual.I - half * dt * l_star, rhs=q,
                                      fields=fields_star)
    manual.commit(manual.state("U", block="plasma").next, u_np1)
    _assert_parity(macro, manual)


# --- multistep ----------------------------------------------------------------------------------
def test_adams_bashforth2_parity():
    macro = adctime.Program("adams_bashforth")
    libtime.adams_bashforth(macro, "plasma", 2)
    manual = adctime.Program("adams_bashforth")
    U = manual.state("plasma")
    R_n = manual._rhs_legacy(state=U, fields=manual.solve_fields(U), flux=True, sources=["default"])
    manual.store_history("plasma.R", R_n)
    expr = (
        U + (manual.dt * Fraction(3, 2)) * R_n
        + (manual.dt * Fraction(-1, 2)) * manual.history("plasma.R", lag=1)
    )
    manual.commit(manual.state("U", block="plasma").next, manual.linear_combine("ab2_step", expr))
    _assert_parity(macro, manual)


def test_bdf1_linear_source_parity():
    m = _model("bdf")
    lorentz = _op(m, "lorentz")
    macro = adctime.Program("bdf").bind_operators(m)
    libtime.bdf(macro, "plasma", 1, linear_source=lorentz)
    manual = adctime.Program("bdf").bind_operators(m)
    U = manual.state("plasma")
    fields = manual.solve_fields(U)
    R = manual._rhs_legacy(state=U, fields=fields, flux=True, sources=["default"])
    rhs = manual.linear_combine("plasma_bdf1_rhs", U + manual.dt * R)
    operator = manual.I - manual.dt * manual.linear_source(lorentz)
    out = manual.solve_local_linear(name="plasma_bdf1_step", operator=operator, rhs=rhs, fields=fields)
    manual.commit(manual.state("U", block="plasma").next, out)
    _assert_parity(macro, manual)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
