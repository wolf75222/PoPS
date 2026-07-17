#!/usr/bin/env python3
"""pops.time typed temporal-version handles (Spec 5 sec.5.3.1, ADC-485).

The handle layer (``typed_state(P, "plasma", state_name="U")`` -> a :class:`TimeState` with ``.n`` /
``.stage`` / ``.next`` / ``.prev``, plus ``T.value`` / ``T.commit`` / ``T.keep_history``)
is SUGAR over the existing SSA IR: it lowers to the SAME ``state`` / ``linear_combine`` /
``commit`` / ``history`` / ``store_history`` ops the positional ``P.state`` style builds.

These checks are pure Python (no compilation): they exercise the handle algebra, the SSA /
read-only / history-policy guards, executable C++ parity between SSPRK3 handle and positional
authoring (their distinct debug names intentionally give distinct hashes), and no ndarray payload.

Run with python3 (PYTHONPATH = built pops package); falls back to pytest from the runner.
"""
from typed_program_support import solve_field, typed_state

from fractions import Fraction
import sys

import pytest

from pops import time as adctime
from pops.numerics.terms import DefaultSource, Flux


def _stage(state, name, offset):
    return state.stage(
        name,
        point=adctime.StagePoint(
            name, {"main": adctime.TimePoint(state.clock, offset)}),
    )


def _expect_value_error(fn, needle):
    """Call ``fn`` and assert it raises a ValueError whose message contains ``needle``."""
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), "wrong message: %r (wanted %r)" % (str(exc), needle)
    else:
        raise AssertionError("expected ValueError containing %r" % (needle,))


def test_current_state_is_read_only():
    P = adctime.Program("ro")
    U = typed_state(P, "plasma", state_name="U")
    _expect_value_error(lambda: P.value(U.n, U.n + P.dt * U.n),
                        "current state is read-only in Program")


def test_define_prev_rejected():
    P = adctime.Program("prev_def")
    U = typed_state(P, "plasma", state_name="U")
    _expect_value_error(lambda: P.value(U.prev, U.n),
                        "history is produced by the history policy")


def test_use_before_define_raises():
    P = adctime.Program("ubd")
    U = typed_state(P, "plasma", state_name="U")
    s1 = _stage(U, "predictor", 1)
    _expect_value_error(lambda: s1 + P.dt * s1,
                        "stage 'predictor' is undefined (materialize it with T.value first)")
    with pytest.raises(TypeError, match="StateEndpointHandle"):
        P.commit(s1, U.n)


def test_double_define_rejected():
    P = adctime.Program("dd")
    U = typed_state(P, "plasma", state_name="U")
    k0 = P.rhs(
        state=U.n, fields=solve_field(P, U.n), terms=[Flux(), DefaultSource()])
    stage = _stage(U, "predictor", 1)
    P.value(stage, U.n + P.dt * k0)
    _expect_value_error(lambda: P.value(stage, U.n + P.dt * k0),
                        "SSA stage already defined")


def test_prev_without_keep_history_raises():
    P = adctime.Program("noh")
    U = typed_state(P, "plasma", state_name="U")
    _expect_value_error(lambda: U.prev(1), "requires keep_history first")
    _expect_value_error(lambda: U.n + P.dt * U.prev, "requires keep_history first")


def test_keep_history_then_prev_reads_history():
    P = adctime.Program("hist")
    U = typed_state(P, "plasma", state_name="U")
    P.keep_history(U, depth=2)
    p1 = U.prev(1)
    assert isinstance(p1, adctime.HistoryHandle) and p1 is U.prev
    p1_value = p1.value
    assert p1_value.vtype == "state" and p1_value.op == "history"
    assert p1_value.attrs["lag"] == 1 and p1_value.attrs["history"] == "plasma.U"
    p2 = U.prev(2)
    assert isinstance(p2, adctime.HistoryHandle) and p2.value.attrs["lag"] == 2
    # bare U.prev behaves as lag 1: the affine proxy reads the lag-1 history ProgramValue
    bare = U.n + P.dt * U.prev  # forces the lag-1 affine proxy
    hist_terms = [v for v, _ in bare.terms if v.op == "history"]
    assert hist_terms and hist_terms[0].attrs["lag"] == 1
    _expect_value_error(lambda: U.prev(3), "exceeds the kept history depth")


def _ssprk3_values(P, block):
    """SSPRK3 built from typed current-state values and explicit combines."""
    U0 = typed_state(P, block)
    f0 = solve_field(P, U0)
    k0 = P.rhs(state=U0, fields=f0, terms=[Flux(), DefaultSource()])
    state = typed_state(P, block, state_name="U")
    stage1 = _stage(state, "stage1", 1)
    stage2 = _stage(state, "stage2", Fraction(1, 2))
    U1 = P.value("ssprk3_U1", U0 + P.dt * k0, at=stage1.point)
    f1 = solve_field(P, U1)
    k1 = P.rhs(state=U1, fields=f1, terms=[Flux(), DefaultSource()])
    U2 = P.value(
        "ssprk3_U2", 0.75 * U0 + 0.25 * (U1 + P.dt * k1), at=stage2.point)
    f2 = solve_field(P, U2)
    k2 = P.rhs(state=U2, fields=f2, terms=[Flux(), DefaultSource()])
    U_next = P.value(
        "ssprk3_step", (1.0 / 3.0) * U0 + (2.0 / 3.0) * (U2 + P.dt * k2),
        at=state.next.point)
    P.commit(state.next, U_next)


def _ssprk3_handles(P, block):
    """The SAME SSPRK3, written with the typed temporal-version handles."""
    U = typed_state(P, block, state_name="U")
    f0 = solve_field(P, U.n)
    k0 = P.rhs(state=U.n, fields=f0, terms=[Flux(), DefaultSource()])
    stage1 = _stage(U, "stage1", 1)
    stage2 = _stage(U, "stage2", Fraction(1, 2))
    P.value(stage1, U.n + P.dt * k0)
    f1 = solve_field(P, stage1.value)
    k1 = P.rhs(state=stage1, fields=f1, terms=[Flux(), DefaultSource()])
    P.value(stage2, 0.75 * U.n + 0.25 * (stage1 + P.dt * k1))
    f2 = solve_field(P, stage2.value)
    k2 = P.rhs(state=stage2, fields=f2, terms=[Flux(), DefaultSource()])
    U_next = P.value(
        "ssprk3_step",
        (1.0 / 3.0) * U.n + (2.0 / 3.0) * (stage2 + P.dt * k2),
        at=U.next.point,
    )
    P.commit(U.next, U_next)


def test_ssprk3_handles_keep_numerical_ir_parity_with_value_authoring():
    value_style = adctime.Program("ssprk3")
    _ssprk3_values(value_style, "plasma")
    value_style.validate()
    handles = adctime.Program("ssprk3")
    _ssprk3_handles(handles, "plasma")
    handles.validate()
    def numerical_ir(program):
        # Provenance records the authoring door (free value vs named stage) and is intentionally
        # different. Numerical parity retains exact point/clock metadata while excluding only that
        # non-numerical lineage and the debug labels.
        data = program._serialize(include_provenance=False)
        for node in data["nodes"]:
            node.pop("name")
        return data

    assert numerical_ir(value_style) == numerical_ir(handles), (
        "SSPRK3 via handles must retain the same numerical IR after debug names are removed")
    assert value_style._ir_hash() != handles._ir_hash(), (
        "distinct authoring names are intentionally part of Program IR identity")


def test_handles_carry_no_ndarray():
    P = adctime.Program("nodata")
    U = typed_state(P, "plasma", state_name="U")
    s1 = _stage(U, "predictor", 1)
    nxt = U.next
    prev = U.prev
    # No handle (including the slots-only StateEndpointHandle) owns a numpy array.
    for handle in (U, s1, nxt, prev):
        attributes = dict(getattr(handle, "__dict__", {}))
        for attr in ("owner_path", "local_id", "kind", "schema_version", "block", "state_name"):
            if hasattr(handle, attr):
                attributes[attr] = getattr(handle, attr)
        for attr, val in attributes.items():
            assert type(val).__module__ != "numpy", (
                "%r.%s is a numpy object (%r); handles must carry no runtime data"
                % (handle, attr, type(val)))
            assert not (hasattr(val, "shape") and hasattr(val, "dtype")), (
                "%r.%s looks like an ndarray; handles must carry no runtime data"
                % (handle, attr))


def main():
    test_current_state_is_read_only()
    test_define_prev_rejected()
    test_use_before_define_raises()
    test_double_define_rejected()
    test_prev_without_keep_history_raises()
    test_keep_history_then_prev_reads_history()
    test_ssprk3_handles_keep_numerical_ir_parity_with_value_authoring()
    test_handles_carry_no_ndarray()
    print("test_time_handles : tout est vert")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
