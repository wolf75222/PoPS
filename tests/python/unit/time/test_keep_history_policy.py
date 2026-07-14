#!/usr/bin/env python3
"""ADC-626: keep_history accepts the typed persistence policy; compile-time coherence + determinism.

The NotImplementedError gate is REMOVED: ``T.keep_history(U, depth, checkpoint_policy=...)`` accepts a
typed history-persistence descriptor, records it in the Program-owned history table, validates coherence at
author time, and the compile-time gate (Program.validate) refuses a non-Dense policy whose replay would
reach a non-deterministic op. Pure Python IR construction (no numerics / no _pops).
"""
from typed_program_support import typed_state

import sys

import pytest

from pops import time as adctime
from pops.time._history.persistence import Dense, Interval, Revolve
from pops.time._history.validation import check_program


def _expect(exc_type, fn, needle):
    try:
        fn()
    except exc_type as exc:
        assert needle in str(exc), "wrong message: %r (wanted %r)" % (str(exc), needle)
    else:
        raise AssertionError("expected %s containing %r" % (exc_type.__name__, needle))


# --- acceptance: the gate is gone ---------------------------------------------------------------
def test_keep_history_no_longer_raises_not_implemented():
    P = adctime.Program("k")
    U = typed_state(P, "plasma", state_name="U")
    # This exact call raised NotImplementedError before ADC-626; now it is accepted.
    node = P.keep_history(U, depth=4, checkpoint_policy=Interval(3))
    assert node.op == "store_history"


def test_keep_history_records_depth_and_policy_on_program():
    P = adctime.Program("k")
    U = typed_state(P, "plasma", state_name="U")
    P.keep_history(U, depth=5, checkpoint_policy=Revolve(3))
    depth, policy = P._history_persistence["plasma.U"]
    assert depth == 5 and isinstance(policy, Revolve)
    assert policy.stored_slots(5) == (0, 2, 4)


def test_multiple_histories_distinct_policies_one_program():
    """GENERIC: several rings with DIFFERENT policies coexist in one Program (owner constraint)."""
    P = adctime.Program("multi")
    U = typed_state(P, "plasma", state_name="U")
    W = typed_state(P, "neutral", state_name="W")
    P.keep_history(U, depth=4, checkpoint_policy=Interval(3))
    P.keep_history(W, depth=5, checkpoint_policy=Revolve(3))
    assert isinstance(P._history_persistence["plasma.U"][1], Interval)
    assert isinstance(P._history_persistence["neutral.W"][1], Revolve)


# --- author-time coherence refusals -------------------------------------------------------------
def test_incoherent_interval_refused_at_author_time():
    P = adctime.Program("k")
    U = typed_state(P, "plasma", state_name="U")
    _expect(ValueError, lambda: P.keep_history(U, depth=4, checkpoint_policy=Interval(2)),
            "oldest slot")


def test_oversized_revolve_refused_at_author_time():
    P = adctime.Program("k")
    U = typed_state(P, "plasma", state_name="U")
    _expect(ValueError, lambda: P.keep_history(U, depth=3, checkpoint_policy=Revolve(5)),
            "exceeds ring depth")


# --- compile-time determinism gate --------------------------------------------------------------
def test_deterministic_program_with_non_dense_policy_passes_compile_gate():
    P = adctime.Program("det")
    U = typed_state(P, "plasma", state_name="U")
    P.keep_history(U, depth=4, checkpoint_policy=Interval(3))
    # A deterministic combine reading the ring, committed as the new state (a valid State value).
    nxt = P.value(
        "U_next", 1.0 * U.n + 0.5 * U.prev(1), at=U.next.point)
    P.commit(U.next, nxt)
    check_program(P)  # no refusal: every op is on the vetted deterministic allow-list


def test_dense_policy_never_refused_even_with_unknown_op():
    """Dense needs no replay, so the determinism scan never refuses it."""
    from pops._report import ReportTree
    from pops.time._history.validation import validate_history_persistence

    class FakeOp:
        op = "some_future_stochastic_op"
        attrs = {}

    class FakeProg:
        _history_persistence = {"plasma.U": (4, Dense())}
        _values = [FakeOp()]

    root = ReportTree(
        phase="validation", severity="info", code="validation.history_persistence.report")
    report = validate_history_persistence(FakeProg(), root)
    assert report.ok, str(report)


def test_non_deterministic_op_refuses_non_dense_policy_verbatim():
    """A non-Dense policy whose replay reaches an op OUTSIDE the vetted allow-list is refused loud."""
    from pops._report import ReportTree
    from pops.time._history.validation import validate_history_persistence

    class FakeOp:
        op = "rng_source"
        attrs = {}

    class FakeProg:
        _history_persistence = {"plasma.U": (4, Interval(3))}
        _values = [FakeOp()]

    root = ReportTree(
        phase="validation", severity="info", code="validation.history_persistence.report")
    report = validate_history_persistence(FakeProg(), root)
    assert not report.ok
    message = str(report)
    assert "non-deterministic" in message
    assert "deterministic replay" in message
    assert "'plasma.U'" in message


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
