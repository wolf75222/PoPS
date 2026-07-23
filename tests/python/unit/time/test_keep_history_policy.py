#!/usr/bin/env python3
"""ADC-626: keep_history persistence coherence and exact selective-replay eligibility.

The NotImplementedError gate is REMOVED: ``T.keep_history(U, depth, checkpoint_policy=...)`` accepts a
typed history-persistence descriptor, records it in the Program-owned history table, validates coherence at
author time. The compile-time gate refuses a non-Dense policy outside the proven primary-clock,
strictly owner-affine replay class. Pure Python IR construction (no numerics / no _pops).
"""
from typed_program_support import state_refs, typed_state

import sys

import pytest

from pops import time as adctime
from pops.time._history.persistence import Dense, Interval, Revolve
from pops.time._history.validation import check_program, validate_history_persistence
from pops.time.points import Clock


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
    node = P.keep_history(U, depth=3, checkpoint_policy=Interval(3))
    assert node.op == "store_history"


def test_keep_history_records_depth_and_policy_on_program():
    P = adctime.Program("k")
    U = typed_state(P, "plasma", state_name="U")
    P.keep_history(U, depth=5, checkpoint_policy=Revolve(3))
    ring_slots, policy = P._history_persistence["plasma.U"]
    assert ring_slots == 6 and isinstance(policy, Revolve)
    assert policy.stored_slots(ring_slots) == (0, 2, 5)


def test_manual_selective_history_fails_closed_without_state_phase_provenance():
    from pops._report import DiagnosticError
    from pops.numerics.terms import DefaultSource

    P = adctime.Program("manual_interval_replay")
    U = typed_state(P, "blk")
    R = P.rhs(state=U, terms=[DefaultSource()])
    P.store_history("blk.R", R, depth=3, checkpoint_policy=Interval(3))
    endpoint = typed_state(P, "blk", state_name="U").next
    P.commit(endpoint, P.value("same_state", 1.0 * U, at=endpoint.point))
    with pytest.raises(DiagnosticError) as exc:
        check_program(P)
    message = str(exc.value)
    assert "was not declared by keep_history" in message
    assert "outgoing-dt" in message


def test_multiple_histories_distinct_policies_one_program():
    """GENERIC: several rings with DIFFERENT policies coexist in one Program (owner constraint)."""
    P = adctime.Program("multi")
    U = typed_state(P, "plasma", state_name="U")
    W = typed_state(P, "neutral", state_name="W")
    P.keep_history(U, depth=3, checkpoint_policy=Interval(3))
    P.keep_history(W, depth=5, checkpoint_policy=Revolve(3))
    assert isinstance(P._history_persistence["plasma.U"][1], Interval)
    assert isinstance(P._history_persistence["neutral.W"][1], Revolve)


def _persistence_report(program):
    from pops._report import ReportTree

    root = ReportTree(
        phase="validation",
        severity="info",
        code="validation.history_persistence.report",
        source="history_persistence",
        owner=program,
    )
    return validate_history_persistence(program, root)


def test_compile_gate_refuses_a_ring_without_its_compiled_policy():
    P = adctime.Program("missing_policy")
    U = typed_state(P, "plasma", state_name="U")
    P.keep_history(U, depth=2)
    del P._history_persistence["plasma.U"]
    report = _persistence_report(P)
    assert not report.ok
    assert any(issue.code.endswith("missing_policy") for issue in report.issues)


def test_compile_gate_refuses_an_orphan_policy_and_a_depth_mismatch():
    orphan = adctime.Program("orphan_policy")
    orphan._history_persistence["ghost.U"] = (1, Dense())
    orphan_report = _persistence_report(orphan)
    assert not orphan_report.ok
    assert any(issue.code.endswith("orphan_policy") for issue in orphan_report.issues)

    mismatch = adctime.Program("depth_mismatch")
    state = typed_state(mismatch, "plasma", state_name="U")
    mismatch.keep_history(state, depth=3)
    mismatch._history_persistence["plasma.U"] = (2, Dense())
    mismatch_report = _persistence_report(mismatch)
    assert not mismatch_report.ok
    assert any(issue.code.endswith("depth_mismatch") for issue in mismatch_report.issues)


# --- author-time coherence refusals -------------------------------------------------------------
def test_incoherent_interval_refused_at_author_time():
    P = adctime.Program("k")
    U = typed_state(P, "plasma", state_name="U")
    _expect(ValueError, lambda: P.keep_history(U, depth=3, checkpoint_policy=Interval(2)),
            "oldest slot")


def test_oversized_revolve_refused_at_author_time():
    P = adctime.Program("k")
    U = typed_state(P, "plasma", state_name="U")
    _expect(ValueError, lambda: P.keep_history(U, depth=3, checkpoint_policy=Revolve(5)),
            "exceeds ring depth")


# --- compile-time exact-replay gate -------------------------------------------------------------
def test_primary_clock_affine_transition_with_dt_and_zero_weight_lag_passes():
    P = adctime.Program("affine")
    U = typed_state(P, "plasma", state_name="U")
    P.keep_history(U, depth=3, checkpoint_policy=Interval(3))
    # The exact-zero coefficient is canonicalized out of the effective dependency graph. It may
    # declare/read the physical lag without making the dt-dependent transition multistep.
    nxt = P.value(
        "U_next", U.n + 0.25 * P.dt * U.n + 0.0 * U.prev(3), at=U.next.point)
    P.commit(U.next, nxt)
    check_program(P)
    from pops.time._program.detach import detach_compiled_program
    check_program(detach_compiled_program(P))


def test_load_bearing_lag_refuses_single_anchor_replay():
    P = adctime.Program("multistep")
    U = typed_state(P, "plasma", state_name="U")
    P.keep_history(U, depth=3, checkpoint_policy=Interval(3))
    nxt = P.value(
        "U_next", 1.0 * U.n + 0.5 * U.prev(1), at=U.next.point)
    P.commit(U.next, nxt)
    report = _persistence_report(P)
    assert not report.ok
    assert any(issue.code.endswith("non_affine_replay") for issue in report.issues)
    assert "depends on lagged history" in str(report)


def test_rhs_transition_refuses_single_anchor_replay():
    from pops.numerics.terms import DefaultSource

    P = adctime.Program("rhs_transition")
    U = typed_state(P, "plasma", state_name="U")
    P.keep_history(U, depth=3, checkpoint_policy=Interval(3))
    rate = P.rhs(state=U.n, terms=[DefaultSource()])
    P.commit(
        U.next,
        P.value("U_next", U.n + P.dt * rate + 0.0 * U.prev(3), at=U.next.point),
    )
    report = _persistence_report(P)
    assert not report.ok
    assert any(issue.code.endswith("non_affine_replay") for issue in report.issues)
    assert "unproved replay op 'rhs'" in str(report)
    assert any(issue.code.endswith("unrestored_replay_context") for issue in report.issues)
    assert "non-affine/context-dependent op 'rhs'" in str(report)


def test_child_clock_history_refuses_primary_step_replay():
    P = adctime.Program("child_history")
    block, declaration = state_refs(P, "plasma", state_name="U")
    fast = Clock("fast", owner=P.owner_path)
    U = P.state(block[declaration], clock=fast)
    P.keep_history(U, depth=3, checkpoint_policy=Interval(3))
    P.commit(U.next, P.value("U_next", 1.0 * U.n, at=U.next.point))
    report = _persistence_report(P)
    assert not report.ok
    assert any(issue.code.endswith("non_primary_clock_replay") for issue in report.issues)
    assert "child clock" in str(report)


def test_context_side_effect_refuses_single_anchor_replay():
    P = adctime.Program("context_history")
    U = typed_state(P, "plasma", state_name="U")
    P.keep_history(U, depth=3, checkpoint_policy=Interval(3))
    P.record_scalar("state_sum", P.sum(U.n))
    P.commit(U.next, P.value("U_next", 1.0 * U.n, at=U.next.point))
    report = _persistence_report(P)
    assert not report.ok
    assert any(issue.code.endswith("unrestored_replay_context") for issue in report.issues)
    assert "non-affine/context-dependent op 'reduce'" in str(report)


def test_cross_block_commit_is_rejected_before_replay_validation():
    P = adctime.Program("cross_block_history")
    U = typed_state(P, "plasma", state_name="U")
    W = typed_state(P, "neutral", state_name="W")
    P.keep_history(U, depth=3, checkpoint_policy=Interval(3))
    with pytest.raises(ValueError, match="cross-block write"):
        P.commit(U.next, P.value("U_next", 1.0 * W.n, at=U.next.point))


def test_dense_policy_never_refused_even_with_unknown_op():
    """Dense needs no replay, so the determinism scan never refuses it."""
    from pops._report import ReportTree
    from pops.time._history.validation import validate_history_persistence

    class FakeOp:
        op = "some_future_stochastic_op"
        attrs = {}

    class FakeProg:
        _histories = {"plasma.U": 4}
        _history_persistence = {"plasma.U": (5, Dense())}
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
        _histories = {"plasma.U": 4}
        _history_persistence = {"plasma.U": (5, Interval(2))}
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
