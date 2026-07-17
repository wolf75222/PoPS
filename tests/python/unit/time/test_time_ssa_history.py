#!/usr/bin/env python3
"""ADC-531: SSA history + bounded control-flow contracts (the remaining gaps).

Complements test_time_handles.py / test_time_control_flow.py with the ADC-531 acceptance points not
covered there:

  - keep_history carries a typed history-persistence checkpoint_policy (ADC-626); the default resolves
    to Dense and a bare-string policy is refused (only a typed descriptor is accepted);
  - a bounded loop is MANDATORY-BOUND: P.range / P.static_range require a Python-int count (the loop
    bound), and a non-int / runtime-scalar / negative count is refused;
  - commit_many is ATOMIC: a group with a double-commit or a foreign value is rejected as a unit,
    leaving no block half-committed.

Pure Python IR construction (no numerics / no _pops); collected as pytest functions.
"""
from typed_program_support import commits_by_block, typed_state

import sys

import pytest

from pops import time as adctime
from pops.time._history.policy import CopyCurrent
from pops.time._history.persistence import Dense, Interval


def _expect(exc_type, fn, needle):
    try:
        fn()
    except exc_type as exc:
        assert needle in str(exc), "wrong message: %r (wanted %r)" % (str(exc), needle)
    else:
        raise AssertionError("expected %s containing %r" % (exc_type.__name__, needle))


# --- keep_history checkpoint policy (ADC-626) ---------------------------------------------------
def test_keep_history_default_resolves_to_dense():
    P = adctime.Program("h")
    U = typed_state(P, "plasma", state_name="U")
    node = P.keep_history(U, depth=2)
    assert node.op == "store_history"
    # The historical whole-ring behaviour: cold start defaults to CopyCurrent, persistence to Dense.
    _, cold_start, configured_policy = P._time_history_configs[U]
    assert isinstance(cold_start, CopyCurrent)
    assert isinstance(configured_policy, Dense)
    # The resolved policy is recorded against the physical slot count (max lag + current slot).
    ring_slots, policy = P._history_persistence["plasma.U"]
    assert ring_slots == 3 and isinstance(policy, Dense)


def test_keep_history_accepts_typed_policy():
    P = adctime.Program("h")
    U = typed_state(P, "plasma", state_name="U")
    # max lag 3 creates four slots; Interval(3) stores both anchors {0, 3}.
    node = P.keep_history(U, depth=3, checkpoint_policy=Interval(3))
    assert node.op == "store_history"
    _, _, configured_policy = P._time_history_configs[U]
    assert isinstance(configured_policy, Interval) and configured_policy.k == 3
    ring_slots, policy = P._history_persistence["plasma.U"]
    assert ring_slots == 4 and policy.stored_slots(ring_slots) == (0, 3)


def test_keep_history_bad_string_policy_refused():
    P = adctime.Program("h")
    U = typed_state(P, "plasma", state_name="U")
    # A bare string is NOT a typed policy: refused with a TypeError (only the descriptors are accepted).
    _expect(TypeError, lambda: P.keep_history(U, depth=2, checkpoint_policy="disk"), "typed policy")


def test_keep_history_incoherent_policy_refused_at_author_time():
    P = adctime.Program("h")
    U = typed_state(P, "plasma", state_name="U")
    # Max lag 3 creates four slots; Interval(2) misses oldest slot 3.
    _expect(ValueError, lambda: P.keep_history(U, depth=3, checkpoint_policy=Interval(2)),
            "oldest slot")


# --- bounded loops: the count is a MANDATORY bound ----------------------------------------------
def test_static_range_requires_int_bound():
    P = adctime.Program("sr")
    U = typed_state(P, "plasma")

    def body(prog, x):
        return prog.value("x1", 1.0 * x)

    # a valid bound unrolls the body 'count' times
    out = P.static_range(U, 3, body)
    assert out.vtype == "state"
    # a non-int (float) bound is refused: the loop bound must be a compile-time count
    _expect(TypeError, lambda: P.static_range(U, 3.0, body), "int")
    # a bool is not an int bound
    _expect(TypeError, lambda: P.static_range(U, True, body), "int")


def test_range_requires_int_bound_and_refuses_runtime_scalar():
    P = adctime.Program("rg")
    U = typed_state(P, "plasma")

    def body(prog, x):
        return prog.value("x1", 1.0 * x)

    out = P.range(U, 4, body)
    assert out.op == "range" and out.attrs["count"] == 4
    # a float bound is refused
    _expect(TypeError, lambda: P.range(U, 2.5, body), "int")
    # a runtime Scalar as the bound is a later phase -> refused loudly (a bounded loop needs a count)
    scal = P.norm2(U)
    _expect(NotImplementedError, lambda: P.range(U, scal, body), "runtime Scalar")


def test_range_negative_bound_rejected():
    P = adctime.Program("neg")
    U = typed_state(P, "plasma")
    _expect(ValueError, lambda: P.range(U, -1, lambda prog, x: x), "non-negative")


# --- commit_many atomicity ----------------------------------------------------------------------
def test_commit_many_atomic_double_commit_rejected():
    P = adctime.Program("cm")
    Ua = typed_state(P, "a")
    Ub = typed_state(P, "b")
    a_next = typed_state(P, "a", state_name="U").next
    b_next = typed_state(P, "b", state_name="U").next
    a1 = P.value("a1", 1.0 * Ua, at=a_next.point)
    b1 = P.value("b1", 1.0 * Ub, at=b_next.point)
    P.commit(a_next, a1)  # 'a' already committed
    # commit_many of {a, b} must be rejected as a UNIT (a is double), and b must NOT be committed.
    _expect(
        ValueError,
        lambda: P.commit_many({a_next: a1, b_next: b1}),
        "committed more than once",
    )
    assert "b" not in P.commits(), "commit_many must be atomic: no partial commit of the group"


def test_commit_many_foreign_value_rejected_atomically():
    P = adctime.Program("cm2")
    other = adctime.Program("other")
    Ua = typed_state(P, "a")
    a_next = typed_state(P, "a", state_name="U").next
    a1 = P.value("a1", 1.0 * Ua, at=a_next.point)
    foreign = other.value("z", 1.0 * typed_state(other, "a"))
    z_next = typed_state(P, "z", state_name="U").next
    _expect(
        ValueError,
        lambda: P.commit_many({a_next: a1, z_next: foreign}),
        "different Program",
    )
    assert P.commits() == {}, "no block committed when the group is rejected"


def test_commit_many_success_commits_all():
    P = adctime.Program("cm3")
    Ua = typed_state(P, "a")
    Ub = typed_state(P, "b")
    a_next = typed_state(P, "a", state_name="U").next
    b_next = typed_state(P, "b", state_name="U").next
    a1 = P.value("a1", 1.0 * Ua, at=a_next.point)
    b1 = P.value("b1", 1.0 * Ub, at=b_next.point)
    P.commit_many({a_next: a1, b_next: b1})
    assert commits_by_block(P) == {"a": a1, "b": b1}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
