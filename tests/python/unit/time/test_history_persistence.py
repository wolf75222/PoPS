#!/usr/bin/env python3
"""ADC-626: history-persistence descriptor validation, schedule placement, manifest round-trip.

Pure Python (no numerics / no _pops beyond the descriptor import): the three policies
(Dense / Interval(k) / Revolve(snapshots)) declare which ring slots a checkpoint stores, and this
suite proves the generic invariants the writer / reader / native replay all rely on:

  - stored_slots invariants (subset, count, both endpoints) over a (depth, param) SWEEP, not a table;
  - the min-max-gap OPTIMALITY of the Revolve placement (equispaced, both endpoints forced);
  - the coherence REFUSALS (verbatim messages) at validate_for;
  - the manifest round-trip (to_manifest -> from_manifest == identity) and the fail-loud unknown kind;
  - freeze() sealing (ADC-563) and deterministic inspect() / __str__ (ADC-591).
"""
import math
import sys

import pytest

from pops.time.history_persistence import (
    Dense, HistoryPersistence, Interval, Revolve,
    DEFAULT_HISTORY_PERSISTENCE, resolve_history_persistence,
)


def _expect(exc_type, fn, needle):
    try:
        fn()
    except exc_type as exc:
        assert needle in str(exc), "wrong message: %r (wanted %r)" % (str(exc), needle)
    else:
        raise AssertionError("expected %s containing %r" % (exc_type.__name__, needle))


# --- Dense --------------------------------------------------------------------------------------
def test_dense_stores_every_slot():
    for depth in range(1, 33):
        assert Dense().stored_slots(depth) == tuple(range(depth))
        assert Dense().recomputed_slots(depth) == ()
        assert Dense().degenerate_to_dense(depth)


def test_dense_never_refused():
    for depth in (1, 2, 8, 64):
        assert Dense().validate_for(depth) == depth


# --- Interval -----------------------------------------------------------------------------------
def test_interval_one_equals_dense():
    for depth in range(1, 17):
        assert Interval(1).stored_slots(depth) == Dense().stored_slots(depth)


def test_interval_stores_stride_and_newest():
    assert Interval(2).stored_slots(5) == (0, 2, 4)
    assert Interval(3).stored_slots(7) == (0, 3, 6)
    # slot 0 is ALWAYS stored, even when it is not a multiple of k.
    assert 0 in Interval(3).stored_slots(5)


def test_interval_requires_positive_int_k():
    _expect(ValueError, lambda: Interval(0), ">= 1")
    _expect(ValueError, lambda: Interval(-2), ">= 1")
    _expect(ValueError, lambda: Interval(True), ">= 1")


def test_interval_refuses_when_stride_misses_oldest_slot():
    # (depth-1) % k != 0 -> the oldest lag is unreconstructable; verbatim refusal.
    _expect(ValueError, lambda: Interval(2).validate_for(4), "oldest slot")
    _expect(ValueError, lambda: Interval(3).validate_for(5), "oldest slot")


def test_interval_refuses_k_ge_depth():
    _expect(ValueError, lambda: Interval(4).validate_for(4), "stores only the newest")
    _expect(ValueError, lambda: Interval(5).validate_for(4), "stores only the newest")


def test_interval_valid_when_stride_divides_depth_minus_one():
    for depth, k in ((4, 3), (5, 2), (7, 3), (7, 2), (9, 4), (9, 2)):
        assert Interval(k).validate_for(depth) == depth
        assert (depth - 1) in Interval(k).stored_slots(depth), (depth, k)


def test_interval_depth_one_normalises_to_dense():
    assert Interval(3).validate_for(1) == 1
    assert Interval(3).stored_slots(1) == (0,)
    assert Interval(3).degenerate_to_dense(1)


# --- Revolve ------------------------------------------------------------------------------------
def test_revolve_requires_two_snapshots():
    _expect(ValueError, lambda: Revolve(1), ">= 2")
    _expect(ValueError, lambda: Revolve(0), ">= 2")
    _expect(ValueError, lambda: Revolve(True), ">= 2")


def test_revolve_refuses_budget_exceeding_depth():
    _expect(ValueError, lambda: Revolve(6).validate_for(5), "exceeds ring depth")


def test_revolve_depth_one_normalises_to_dense():
    assert Revolve(2).validate_for(1) == 1
    assert Revolve(2).stored_slots(1) == (0,)


def test_revolve_equispaced_placement():
    assert Revolve(2).stored_slots(5) == (0, 4)
    assert Revolve(3).stored_slots(5) == (0, 2, 4)
    assert Revolve(3).stored_slots(9) == (0, 4, 8)


# --- the (depth, param) property SWEEP (owner: sweep, not cherry-picked) -------------------------
def test_revolve_placement_property_sweep():
    """Over d in 2..64, s in 2..d: subset, exact count, BOTH endpoints, reconstructability, and the
    min-max-gap optimum g* = ceil((d-1)/(s-1))."""
    for d in range(2, 65):
        for s in range(2, d + 1):
            slots = Revolve(s).stored_slots(d)
            ss = set(slots)
            assert ss <= set(range(d)), (d, s, slots)
            assert len(ss) == min(s, d), ("count", d, s, slots)
            # BOTH endpoints forced: newest (0) is free, oldest (d-1) is mandatory (nothing older).
            assert 0 in ss and (d - 1) in ss, ("endpoints", d, s, slots)
            # every missing slot has a strictly-OLDER (larger index) stored anchor to replay from.
            for k in range(d):
                if k not in ss:
                    assert any(a > k for a in ss), ("older-anchor", d, s, k, slots)
            # optimality: the largest gap does not exceed the closed-form equispaced optimum.
            gaps = [hi - lo for lo, hi in zip(slots, slots[1:])]
            assert max(gaps) <= math.ceil((d - 1) / (s - 1)), ("maxgap", d, s, slots)


def test_interval_divisibility_equals_oldest_stored_sweep():
    """Interval(k) is valid <=> its stored set contains the oldest slot <=> (d-1) % k == 0."""
    for d in range(2, 33):
        for k in range(1, d):
            stored = Interval(k).stored_slots(d)
            oldest_stored = (d - 1) in stored
            divides = (d - 1) % k == 0
            assert oldest_stored == divides, (d, k, stored)
            if divides:
                assert Interval(k).validate_for(d) == d
            else:
                _expect(ValueError, lambda kk=k, dd=d: Interval(kk).validate_for(dd), "oldest slot")


# --- manifest round-trip + reader dispatch ------------------------------------------------------
def test_manifest_round_trip_identity():
    for policy in (Dense(), Interval(3), Revolve(4)):
        manifest = policy.to_manifest()
        back = HistoryPersistence.from_manifest(manifest)
        assert type(back) is type(policy)
        assert back.to_manifest() == manifest


def test_manifest_tags_are_the_kind():
    assert Dense().to_manifest() == {"kind": "dense"}
    assert Interval(3).to_manifest() == {"kind": "interval", "k": 3}
    assert Revolve(4).to_manifest() == {"kind": "revolve", "snapshots": 4}


def test_manifest_unknown_kind_fails_loud():
    _expect(ValueError, lambda: HistoryPersistence.from_manifest({"kind": "brand_new"}), "unknown")
    _expect(ValueError, lambda: HistoryPersistence.from_manifest({"no_kind": 1}), "kind")


# --- resolve default ----------------------------------------------------------------------------
def test_resolve_none_is_dense():
    assert DEFAULT_HISTORY_PERSISTENCE is None
    assert isinstance(resolve_history_persistence(None), Dense)


def test_resolve_passthrough_and_string_refusal():
    p = Interval(2)
    assert resolve_history_persistence(p) is p
    _expect(TypeError, lambda: resolve_history_persistence("interval"), "typed policy")
    _expect(TypeError, lambda: resolve_history_persistence(3), "typed policy")


# --- descriptor discipline (freeze / inspect / str) ---------------------------------------------
def test_freeze_seals_mutation():
    p = Interval(2)
    p.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        p.k = 3


def test_category_is_history_persistence_not_checkpoint_policy():
    # The category must NOT collide with pops.output.CheckpointPolicy's "checkpoint_policy".
    for policy in (Dense(), Interval(2), Revolve(2)):
        assert policy.category == "history_persistence"


def test_inspect_and_str_deterministic():
    p = Revolve(4)
    info = p.inspect()
    assert info["kind"] == "revolve" and info["options"] == {"snapshots": 4}
    assert str(p) == str(Revolve(4))  # deterministic


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
