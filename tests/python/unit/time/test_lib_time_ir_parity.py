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
from decimal import Decimal
from fractions import Fraction

import pytest

from pops.ir.expr import Const
from pops.model import OperatorHandle
from pops.physics.facade import Model
from pops.problem import Problem
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


def _refs(model):
    # Use the facade's real Module declaration graph.  A synthetic DeclarationIndex sharing the
    # model OwnerPath but omitting its operator declarations is not an equivalent model: it gives
    # the same owner two competing structural fingerprints and makes the first canonical IR
    # projection depend on serialization order.
    module = model.module
    block = Problem(name="parity_case").add_block("plasma", model)
    state = module.state_handle(module.state_spaces()["U"])
    return block, state


def _assert_parity(macro_prog, manual_prog):
    macro_prog.validate()
    manual_prog.validate()
    assert macro_prog._ir_hash() == manual_prog._ir_hash(), (
        "macro IR hash %s != manual IR hash %s" % (macro_prog._ir_hash(), manual_prog._ir_hash()))


# --- flux/source explicit schemes (sources=/flux= de-sugaring, no operator handle) ---------------
def _point(P, name, offset=0, *, partitions=None):
    if partitions is None:
        partitions = {"main": offset}
    return adctime.StagePoint(name, {
        partition: adctime.TimePoint(P.clock, coordinate)
        for partition, coordinate in partitions.items()
    })


def _at(P, value, point):
    return P._replace_value(value, point=point)


def _stage(P, U, name, offset):
    point = _point(P, name, offset)
    fields = _at(P, P.solve_fields(U), point)
    return _at(
        P, P._rhs_legacy(state=U, fields=fields, flux=True, sources=["default"]), point)


def test_forward_euler_parity():
    block, state_handle = _refs(_model("fe"))
    macro = libtime.forward_euler(block, state_handle)
    manual = adctime.Program("forward_euler")
    state = manual.state(block, state_handle)
    U = state.n
    R = _stage(manual, U, "forward_euler", 0)
    manual.commit(state.next, manual.linear_combine(
        "fe_step", U + manual.dt * R, at=state.next.point))
    _assert_parity(macro, manual)


def test_ssprk2_parity():
    block, state_handle = _refs(_model("ssprk2"))
    macro = libtime.ssprk2(block, state_handle)
    manual = adctime.Program("ssprk2")
    libtime.rk(manual, block, state_handle, tableau=libtime.SSPRK2_TABLEAU)
    _assert_parity(macro, manual)
    assert macro.to_graph().graph_hash == manual.to_graph().graph_hash


def test_ssprk3_parity():
    block, state_handle = _refs(_model("ssprk3"))
    macro = libtime.ssprk3(block, state_handle)
    manual = adctime.Program("ssprk3")
    from pops.lib.time.ssprk import SSPRK3_TABLEAU
    libtime.rk(manual, block, state_handle, tableau=SSPRK3_TABLEAU)
    _assert_parity(macro, manual)
    assert macro.to_graph().graph_hash == manual.to_graph().graph_hash


def test_rk4_parity():
    block, state_handle = _refs(_model("rk4"))
    macro = libtime.rk4(block, state_handle)
    manual = adctime.Program("rk4")
    dt = manual.dt
    state = manual.state(block, state_handle)
    U0 = state.n
    k1 = _stage(manual, U0, "rk4_stage_0", 0)
    U1 = manual.linear_combine(
        "rk4_U1", U0 + Fraction(1, 2) * dt * k1,
        at=_point(manual, "rk4_stage_1", Fraction(1, 2)))
    k2 = _stage(manual, U1, "rk4_stage_1", Fraction(1, 2))
    U2 = manual.linear_combine(
        "rk4_U2", U0 + Fraction(1, 2) * dt * k2,
        at=_point(manual, "rk4_stage_2", Fraction(1, 2)))
    k3 = _stage(manual, U2, "rk4_stage_2", Fraction(1, 2))
    U3 = manual.linear_combine(
        "rk4_U3", U0 + dt * k3, at=_point(manual, "rk4_stage_3", 1))
    k4 = _stage(manual, U3, "rk4_stage_3", 1)
    manual.commit(state.next, manual.linear_combine(
        "rk4_step",
        U0 + Fraction(1, 6) * dt * k1 + Fraction(1, 3) * dt * k2
        + Fraction(1, 3) * dt * k3 + Fraction(1, 6) * dt * k4,
        at=state.next.point))
    _assert_parity(macro, manual)


def test_rk_ssprk2_tableau_matches_the_rk_macro():
    # The generic rk over the SSPRK2 tableau lowers to its own stage chain; two builds are identical.
    block, state_handle = _refs(_model("rk-tableau"))
    a = adctime.Program("rk")
    libtime.rk(a, block, state_handle, tableau=libtime.SSPRK2_TABLEAU)
    b = adctime.Program("rk")
    libtime.rk(b, block, state_handle, tableau=libtime.SSPRK2_TABLEAU)
    _assert_parity(a, b)


def test_explicit_presets_put_every_rhs_at_its_exact_stage_abscissa():
    cases = (
        ("fe-points", libtime.forward_euler, (0,)),
        ("ssprk2-points", libtime.ssprk2, (0, 1)),
        ("ssprk3-points", libtime.ssprk3, (0, 1, Fraction(1, 2))),
        ("rk4-points", libtime.rk4, (0, Fraction(1, 2), Fraction(1, 2), 1)),
    )
    for model_name, preset, expected in cases:
        block, state_handle = _refs(_model(model_name))
        program = preset(block, state_handle)
        rhs_values = [value for value in program._values if value.op == "rhs"]
        assert [value.point.time.offset.to_python() for value in rhs_values] == list(expected)


def test_generic_tableau_c_is_the_exact_rhs_coordinate_not_documentation_only():
    tableau = libtime.ButcherTableau(
        A=[[], [Decimal("0.25")]],
        b=[Fraction(1, 2), Fraction(1, 2)],
        c=[0, Decimal("0.25")],
        name="decimal-c",
    )
    block, state_handle = _refs(_model("tableau-exact-c"))
    program = adctime.Program("rk")
    libtime.rk(program, block, state_handle, tableau=tableau)

    rhs_values = [value for value in program._values if value.op == "rhs"]
    assert [value.point.time.offset.to_python() for value in rhs_values] == [0, Decimal("0.25")]
    assert rhs_values[1].point.time.offset.to_data() == {
        "kind": "decimal", "value": "0.25"}
    stage = next(value for value in program._values if value.name == "decimal-c_U1")
    assert stage.point == rhs_values[1].point


def test_split_callbacks_must_materialize_the_declared_exact_endpoint():
    block, state_handle = _refs(_model("split-point-guard"))
    program = adctime.Program("split-point-guard")

    def ignores_endpoint(_program, state, _fraction, *, at):  # noqa: ARG001
        return state

    with pytest.raises(ValueError, match="declared split endpoint.*linear_combine"):
        libtime.lie(
            program,
            block,
            state_handle,
            half_flow=ignores_endpoint,
            source=ignores_endpoint,
        )


# --- operator-first schemes (typed handles) -----------------------------------------------------
def test_explicit_rk_parity():
    m = _model("rk")
    block, state_handle = _refs(m)
    macro = adctime.Program("rk").bind_operators(m)
    libtime.explicit_rk(macro, block, state_handle, rhs_operator=_op(m, "explicit_rhs"),
                        fields_operator=_op(m, "fields_from_state"), tableau=libtime.SSPRK2_TABLEAU)
    # Manual SSPRK2 over the typed rate via the internal _call seam.
    manual = adctime.Program("rk").bind_operators(m)
    dt = manual.dt
    state = manual.state(block, state_handle)
    u0 = state.n
    point0 = _point(manual, "ssprk2_stage_0", 0)
    f0 = _at(manual, manual.call(_op(m, "fields_from_state"), u0), point0)
    k0 = _at(
        manual, manual.call(_op(m, "explicit_rhs"), u0, f0, name="ssprk2_k0"), point0)
    point1 = _point(manual, "ssprk2_stage_1", 1)
    u1 = manual.linear_combine("ssprk2_U1", u0 + dt * k0, at=point1)
    f1 = _at(manual, manual.call(_op(m, "fields_from_state"), u1), point1)
    k1 = _at(
        manual, manual.call(_op(m, "explicit_rhs"), u1, f1, name="ssprk2_k1"), point1)
    manual.commit(state.next, manual.linear_combine(
        "ssprk2_step",
        u0 + (dt * Fraction(1, 2)) * k0 + (dt * Fraction(1, 2)) * k1,
        at=state.next.point))
    _assert_parity(macro, manual)


def test_imex_local_linear_parity():
    m = _model("imex")
    block, state_handle = _refs(m)
    macro = adctime.Program("imex").bind_operators(m)
    libtime.imex_local_linear(
        macro, block, state_handle, explicit_operator=_op(m, "explicit_rhs"),
                              implicit_operator=_op(m, "lorentz"),
                              fields_operator=_op(m, "fields_from_state"), theta=1.0)
    manual = adctime.Program("imex").bind_operators(m)
    state = manual.state(block, state_handle)
    u = state.n
    point = _point(
        manual, "imex", partitions={"explicit": 0, "implicit": 1.0})
    fields = _at(
        manual, manual.call(_op(m, "fields_from_state"), u, name="fields"), point)
    r = _at(manual, manual.call(_op(m, "explicit_rhs"), u, fields, name="R"), point)
    lin = _at(manual, manual.call(_op(m, "lorentz"), fields, name="L"), point)
    q = manual.linear_combine("imex_rhs", u + manual.dt * r, at=state.next.point)
    u1 = manual.solve_local_linear(
        "imex_step", operator=manual.I - 1.0 * manual.dt * lin, rhs=q, fields=fields)
    manual.commit(state.next, u1)
    _assert_parity(macro, manual)


def test_imex_keeps_distinct_explicit_and_implicit_abscissae():
    m = _model("imex-points")
    block, state_handle = _refs(m)
    program = adctime.Program("imex-points").bind_operators(m)
    theta = Fraction(2, 3)
    libtime.imex_local_linear(
        program,
        block,
        state_handle,
        explicit_operator=_op(m, "explicit_rhs"),
        implicit_operator=_op(m, "lorentz"),
        fields_operator=_op(m, "fields_from_state"),
        theta=theta,
    )

    for name in ("R", "lorentz"):
        value = next(item for item in program._values if item.name == name)
        assert value.point.time_for("explicit").offset.to_python() == 0
        assert value.point.time_for("implicit").offset.to_python() == theta
        with pytest.raises(ValueError, match="ambiguous partition times"):
            _ = value.point.time
    step = next(item for item in program._values if item.name == "imex_step")
    assert step.point == next(iter(program._time_states.values())).next.point


def test_imex_local_parity():
    m = _model("imexl")
    block, state_handle = _refs(m)
    lorentz = _op(m, "lorentz")
    macro = adctime.Program("imex_local").bind_operators(m)
    libtime.imex_local(macro, block, state_handle, linear_source=lorentz)
    manual = adctime.Program("imex_local").bind_operators(m)
    state = manual.state(block, state_handle)
    U = state.n
    point = _point(
        manual, "plasma_imex", partitions={"explicit": 0, "implicit": 1})
    fields = _at(manual, manual.solve_fields(U), point)
    R = _at(
        manual,
        manual._rhs_legacy(state=U, fields=fields, flux=True, sources=["default"]),
        point,
    )
    rhs = manual.linear_combine(
        "plasma_imex_rhs", U + manual.dt * R, at=state.next.point)
    linear = _at(manual, manual.linear_source(lorentz), point)
    operator = manual.I - manual.dt * linear
    out = manual.solve_local_linear(
        name="plasma_imex_step", operator=operator, rhs=rhs, fields=fields)
    manual.commit(state.next, out)
    _assert_parity(macro, manual)


def test_predictor_corrector_parity():
    m = _model("pc")
    block, state_handle = _refs(m)
    fo, ro, lo = _op(m, "fields_from_state"), _op(m, "explicit_rhs"), _op(m, "lorentz")
    macro = adctime.Program("pc").bind_operators(m)
    libtime.predictor_corrector_local_linear(
        macro, block, state_handle, fields_operator=fo,
                                             explicit_rate_operator=ro, implicit_operator=lo)
    manual = adctime.Program("pc").bind_operators(m)
    dt = manual.dt
    state = manual.state(block, state_handle)
    u_n = state.n
    predictor = _point(
        manual, "predictor", partitions={"explicit": 0, "implicit": 1})
    fields_n = _at(manual, manual.call(fo, u_n, name="fields_n"), predictor)
    r_n = _at(manual, manual.call(ro, u_n, fields_n, name="R_n"), predictor)
    l_n = _at(manual, manual.call(lo, fields_n, name="L_n"), predictor)
    u_star = _at(manual, manual.solve_local_linear(
        "U_star", operator=manual.I - dt * l_n,
        rhs=manual.linear_combine("U_star_rhs", u_n + dt * r_n, at=predictor),
        fields=fields_n), predictor)
    corrector = _point(
        manual, "corrector", partitions={"explicit": 1, "implicit": 1})
    fields_star = _at(
        manual, manual.call(fo, u_star, name="fields_star"), corrector)
    r_star = _at(manual, manual.call(ro, u_star, fields_star, name="R_star"), corrector)
    l_star = _at(manual, manual.call(lo, fields_star, name="L_star"), corrector)
    c_star = _at(
        manual, manual.apply(l_star, u_star, fields=fields_star, name="C_star"), corrector)
    half = Fraction(1, 2)
    q = manual.linear_combine(
        "Q", u_n + half * dt * r_n + half * dt * r_star + half * dt * c_star,
        at=state.next.point)
    u_np1 = manual.solve_local_linear(
        "U_np1", operator=manual.I - half * dt * l_star, rhs=q, fields=fields_star)
    manual.commit(state.next, u_np1)
    _assert_parity(macro, manual)


def test_predictor_corrector_preserves_partition_abscissae_per_stage():
    m = _model("pc-points")
    block, state_handle = _refs(m)
    fo, ro, lo = _op(m, "fields_from_state"), _op(m, "explicit_rhs"), _op(m, "lorentz")
    program = adctime.Program("pc-points").bind_operators(m)
    libtime.predictor_corrector_local_linear(
        program,
        block,
        state_handle,
        fields_operator=fo,
        explicit_rate_operator=ro,
        implicit_operator=lo,
    )

    predictor = next(item for item in program._values if item.name == "R_n").point
    corrector = next(item for item in program._values if item.name == "R_star").point
    assert predictor.time_for("explicit").offset.to_python() == 0
    assert predictor.time_for("implicit").offset.to_python() == 1
    assert corrector.time_for("explicit").offset.to_python() == 1
    assert corrector.time_for("implicit").offset.to_python() == 1


# --- multistep ----------------------------------------------------------------------------------
def test_adams_bashforth2_parity():
    block, state_handle = _refs(_model("ab2"))
    macro = adctime.Program("adams_bashforth")
    libtime.adams_bashforth(macro, block, state_handle, order=2)
    manual = adctime.Program("adams_bashforth")
    state = manual.state(block, state_handle)
    U = state.n
    R_n = _stage(manual, U, "ab_current", 0)
    manual.store_history("plasma.R", R_n)
    expr = (
        U + (manual.dt * Fraction(3, 2)) * R_n
        + (manual.dt * Fraction(-1, 2)) * manual.history(
            "plasma.R", lag=1, space=R_n.space, block=block, state_ref=state.state)
    )
    manual.commit(state.next, manual.linear_combine(
        "ab2_step", expr, at=state.next.point))
    _assert_parity(macro, manual)


def test_bdf1_linear_source_parity():
    m = _model("bdf")
    block, state_handle = _refs(m)
    lorentz = _op(m, "lorentz")
    macro = adctime.Program("bdf").bind_operators(m)
    libtime.bdf(macro, block, state_handle, order=1, linear_source=lorentz)
    manual = adctime.Program("bdf").bind_operators(m)
    state = manual.state(block, state_handle)
    U = state.n
    fields = manual.solve_fields(U)
    R = manual._rhs_legacy(state=U, fields=fields, flux=True, sources=["default"])
    rhs = manual.linear_combine(
        "plasma_bdf1_rhs", U + manual.dt * R, at=state.next.point)
    operator = manual.I - manual.dt * manual.linear_source(lorentz)
    out = manual.solve_local_linear(name="plasma_bdf1_step", operator=operator, rhs=rhs, fields=fields)
    manual.commit(state.next, out)
    _assert_parity(macro, manual)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
