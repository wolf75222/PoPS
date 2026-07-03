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
import sys

import pytest

from pops import time as adctime
from pops.time.history import CopyCurrent
from pops.time.history_persistence import Dense, Interval


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
    U = P.state("U", block="plasma")
    node = P.keep_history(U, depth=2)
    assert node.op == "store_history"
    # The historical whole-ring behaviour: cold start defaults to CopyCurrent, persistence to Dense.
    assert isinstance(U._cold_start, CopyCurrent)
    assert isinstance(U._checkpoint_policy, Dense)
    # The resolved policy is recorded on the Program keyed by the ring name (depth, policy).
    depth, policy = P._history_persistence["plasma.U"]
    assert depth == 2 and isinstance(policy, Dense)


def test_keep_history_accepts_typed_policy():
    P = adctime.Program("h")
    U = P.state("U", block="plasma")
    # depth 4 with Interval(3): (depth-1)=3 divisible by 3 -> stores {0, 3}, coherent.
    node = P.keep_history(U, depth=4, checkpoint_policy=Interval(3))
    assert node.op == "store_history"
    assert isinstance(U._checkpoint_policy, Interval) and U._checkpoint_policy.k == 3
    depth, policy = P._history_persistence["plasma.U"]
    assert depth == 4 and policy.stored_slots(4) == (0, 3)


def test_keep_history_bad_string_policy_refused():
    P = adctime.Program("h")
    U = P.state("U", block="plasma")
    # A bare string is NOT a typed policy: refused with a TypeError (only the descriptors are accepted).
    _expect(TypeError, lambda: P.keep_history(U, depth=2, checkpoint_policy="disk"), "typed policy")


def test_keep_history_incoherent_policy_refused_at_author_time():
    P = adctime.Program("h")
    U = P.state("U", block="plasma")
    # Interval(2) on depth 4: (depth-1)=3 not divisible by 2 -> the oldest lag is unreconstructable.
    _expect(ValueError, lambda: P.keep_history(U, depth=4, checkpoint_policy=Interval(2)),
            "oldest slot")


# --- bounded loops: the count is a MANDATORY bound ----------------------------------------------
def test_static_range_requires_int_bound():
    P = adctime.Program("sr")
    U = P.state("plasma")

    def body(prog, x):
        return prog.linear_combine("x1", 1.0 * x)

    # a valid bound unrolls the body 'count' times
    out = P.static_range(U, 3, body)
    assert out.vtype == "state"
    # a non-int (float) bound is refused: the loop bound must be a compile-time count
    _expect(TypeError, lambda: P.static_range(U, 3.0, body), "int")
    # a bool is not an int bound
    _expect(TypeError, lambda: P.static_range(U, True, body), "int")


def test_range_requires_int_bound_and_refuses_runtime_scalar():
    P = adctime.Program("rg")
    U = P.state("plasma")

    def body(prog, x):
        return prog.linear_combine("x1", 1.0 * x)

    out = P.range(U, 4, body)
    assert out.op == "range" and out.attrs["count"] == 4
    # a float bound is refused
    _expect(TypeError, lambda: P.range(U, 2.5, body), "int")
    # a runtime Scalar as the bound is a later phase -> refused loudly (a bounded loop needs a count)
    scal = P.norm2(U)
    _expect(NotImplementedError, lambda: P.range(U, scal, body), "runtime Scalar")


def test_range_negative_bound_rejected():
    P = adctime.Program("neg")
    U = P.state("plasma")
    _expect(ValueError, lambda: P.range(U, -1, lambda prog, x: x), "non-negative")


# --- commit_many atomicity ----------------------------------------------------------------------
def test_commit_many_atomic_double_commit_rejected():
    P = adctime.Program("cm")
    Ua = P.state("a")
    Ub = P.state("b")
    a1 = P.linear_combine("a1", 1.0 * Ua)
    b1 = P.linear_combine("b1", 1.0 * Ub)
    P.commit("a", a1)  # 'a' already committed
    # commit_many of {a, b} must be rejected as a UNIT (a is double), and b must NOT be committed.
    _expect(ValueError, lambda: P.commit_many({"a": a1, "b": b1}), "committed more than once")
    assert "b" not in P.commits(), "commit_many must be atomic: no partial commit of the group"


def test_commit_many_foreign_value_rejected_atomically():
    P = adctime.Program("cm2")
    other = adctime.Program("other")
    Ua = P.state("a")
    a1 = P.linear_combine("a1", 1.0 * Ua)
    foreign = other.linear_combine("z", 1.0 * other.state("a"))
    _expect(ValueError, lambda: P.commit_many({"a": a1, "z": foreign}), "different Program")
    assert P.commits() == {}, "no block committed when the group is rejected"


def test_commit_many_success_commits_all():
    P = adctime.Program("cm3")
    Ua = P.state("a")
    Ub = P.state("b")
    a1 = P.linear_combine("a1", 1.0 * Ua)
    b1 = P.linear_combine("b1", 1.0 * Ub)
    P.commit_many({"a": a1, "b": b1})
    assert P.commits() == {"a": a1, "b": b1}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
