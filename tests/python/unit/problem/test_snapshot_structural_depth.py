"""Authoring snapshot authority survives deep, finite public metadata."""
import json
from types import MappingProxyType

import pytest

from pops._structural_data import canonical_json, equal
from pops.problem._snapshot import AuthoringSnapshot, validate_problem_snapshot
from pops.problem._snapshot_canonical import _canonical


def test_deep_snapshot_roundtrip_authenticates_all_three_identities():
    payload = {"leaf": 7}
    for _ in range(1024):
        payload = {"node": (payload,)}
    snapshot = AuthoringSnapshot(payload)
    assert validate_problem_snapshot(snapshot) == snapshot.hash
    assert equal(snapshot.to_dict(), snapshot.to_dict())
    assert snapshot.semantic_to_dict() is not snapshot.semantic_to_dict()
    assert snapshot.artifact_to_dict() is not snapshot.artifact_to_dict()


def test_container_order_and_tags_preserve_exact_snapshot_bytes():
    payload = {"z": [1, (2,)], "a": MappingProxyType({"second": 2, "first": 1})}
    canonical = _canonical(payload)
    assert list(canonical) == ["z", "a"]
    assert canonical["z"][1] == {"$tuple": [{"$scalar": {"kind": "integer", "value": "2"}}]}
    assert [entry[0] for entry in canonical["a"]["$mapping"]["entries"]] == ["second", "first"]
    assert canonical_json(canonical, sort_keys=False, allow_nan=False) == json.dumps(
        canonical, sort_keys=False, separators=(",", ":"), allow_nan=False)


def test_deep_snapshot_cycle_rejection_cleans_shared_active_context():
    value = []
    value.append(value)
    active = set()
    with pytest.raises(ValueError, match="reference cycle"):
        _canonical({"loop": value}, active=active)
    assert active == set()
    assert _canonical({"valid": 3}, active=active)["valid"]["$scalar"]["value"] == "3"


@pytest.mark.parametrize("slot", ("_canonical_json", "_artifact_canonical_json"))
def test_snapshot_validation_still_authenticates_decoded_bytes(slot):
    snapshot = AuthoringSnapshot({"value": 7})
    original = getattr(snapshot, slot)
    object.__setattr__(snapshot, slot, original.replace('"7"', '"8"'))
    with pytest.raises(ValueError, match="does not match"):
        validate_problem_snapshot(snapshot)


def test_snapshot_validation_rejects_noncanonical_numeric_payload():
    snapshot = AuthoringSnapshot({"value": 7})
    object.__setattr__(snapshot, "_canonical_json", '{"schema_version":11,"bad":NaN}')
    with pytest.raises((TypeError, ValueError)):
        validate_problem_snapshot(snapshot)



def test_constructed_receipt_compares_closed_preimages_without_reencoding(monkeypatch):
    import pops.problem._snapshot as implementation

    payload = {"leaf": [0, False, None, -0.0, 0.125, "é"]}
    for _ in range(1024):
        payload = {"node": payload}
    snapshot = AuthoringSnapshot(payload)
    assert implementation._has_exact_snapshot_receipt(snapshot)
    # The independently callable full validator agrees before its counter is installed.
    assert implementation._validate_problem_snapshot_full(snapshot) == snapshot.hash

    def unexpected(_):
        raise AssertionError("unchanged builtin preimage was redundantly encoded")

    monkeypatch.setattr(implementation, "_validate_problem_snapshot_full", unexpected)
    assert validate_problem_snapshot(snapshot) == snapshot.hash
    assert validate_problem_snapshot(snapshot) == snapshot.hash


@pytest.mark.parametrize("slot", (
    "_canonical_json", "_hash", "_semantic_canonical_json",
    "_artifact_canonical_json", "_artifact_hash",
))
def test_receipt_does_not_authorize_mutated_slot(slot):
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    original = getattr(snapshot, slot)
    replacement = ("0" if original[0] != "0" else "1") + original[1:]
    object.__setattr__(snapshot, slot, replacement)
    assert not implementation._has_exact_snapshot_receipt(snapshot)
    with pytest.raises((ValueError, TypeError)):
        validate_problem_snapshot(snapshot)


@pytest.mark.parametrize("field,replacement", (
    ("domain", "artifact"), ("schema_version", 999),
    ("algorithm", "other"), ("digest", b"x" * 32),
))
def test_receipt_copies_identity_scalars_instead_of_retaining_identity(field, replacement):
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    object.__setattr__(snapshot.semantic_identity, field, replacement)
    assert not implementation._has_exact_snapshot_receipt(snapshot)
    with pytest.raises((ValueError, TypeError)):
        validate_problem_snapshot(snapshot)


@pytest.mark.parametrize("field,wrap", (("_canonical_json", str), ("_hash", str)))
def test_receipt_does_not_trust_scalar_subclass_equality(field, wrap):
    import pops.problem._snapshot as implementation

    class Forged(wrap):
        def __eq__(self, other):
            raise AssertionError("untrusted equality executed by receipt")

    snapshot = AuthoringSnapshot({"value": 7})
    object.__setattr__(snapshot, field, Forged(getattr(snapshot, field)))
    assert not implementation._has_exact_snapshot_receipt(snapshot)


def test_receipt_checks_identity_scalar_types():
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    object.__setattr__(snapshot.semantic_identity, "schema_version", float(snapshot.semantic_identity.schema_version))
    assert not implementation._has_exact_snapshot_receipt(snapshot)


def test_receipt_schema_change_takes_complete_validator(monkeypatch):
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    monkeypatch.setattr(AuthoringSnapshot, "schema_version", snapshot.schema_version + 1)
    assert not implementation._has_exact_snapshot_receipt(snapshot)
    with pytest.raises(ValueError, match="different snapshot schema"):
        validate_problem_snapshot(snapshot)


def test_receipt_semantic_version_change_takes_complete_validator(monkeypatch):
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    monkeypatch.setattr(implementation._semantic_module, "SEMANTIC_SCHEMA_VERSION", 999)
    assert not implementation._has_exact_snapshot_receipt(snapshot)
    with pytest.raises(ValueError, match="semantic payload"):
        validate_problem_snapshot(snapshot)


def test_receipt_rejects_changed_public_hook(monkeypatch):
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    monkeypatch.setattr(AuthoringSnapshot, "to_dict", lambda self: {"changed": True})
    assert not implementation._has_exact_snapshot_receipt(snapshot)
    with pytest.raises(ValueError, match="canonical payload"):
        validate_problem_snapshot(snapshot)


def test_receipt_rejects_same_function_with_replaced_code():
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    function = AuthoringSnapshot.to_dict
    original = function.__code__
    try:
        function.__code__ = (lambda self: {"changed": True}).__code__
        assert not implementation._has_exact_snapshot_receipt(snapshot)
        with pytest.raises(ValueError, match="canonical payload"):
            validate_problem_snapshot(snapshot)
    finally:
        function.__code__ = original


def test_unreceipted_equal_clone_is_fully_validated(monkeypatch):
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    clone = object.__new__(AuthoringSnapshot)
    for name in implementation._SNAPSHOT_SLOTS:
        object.__setattr__(clone, name, getattr(snapshot, name))
    assert clone == snapshot
    assert not implementation._has_exact_snapshot_receipt(clone)
    calls = []
    full = implementation._validate_problem_snapshot_full

    def counted(value):
        calls.append(value)
        return full(value)

    monkeypatch.setattr(implementation, "_validate_problem_snapshot_full", counted)
    assert validate_problem_snapshot(clone) == snapshot.hash
    assert calls == [clone]


def test_receipt_does_not_extend_snapshot_or_identity_lifetime():
    import gc
    import weakref
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    key = id(snapshot)
    reference = weakref.ref(snapshot)
    assert key in implementation._SNAPSHOT_RECEIPTS
    del snapshot
    gc.collect()
    assert reference() is None
    assert key not in implementation._SNAPSHOT_RECEIPTS


def test_closed_builtin_proof_refuses_nonjson_types_and_cycles():
    import pops.problem._snapshot as implementation

    class Text(str):
        pass

    class Number(int):
        pass

    bad = [Text("x"), Number(7), {1: "value"}, (1,), b"bytes", float("nan"), float("inf")]
    cycle = []
    cycle.append(cycle)
    for value in [*bad, cycle]:
        assert not implementation._plain_json_data(value)
    shared = [1, {"x": "-0x0.0p+0"}]
    assert implementation._plain_json_data({"a": shared, "b": shared})



def test_receipt_accepts_equal_rematerialized_identity_and_preimage_bytes():
    from pops.identity import Identity
    import pops.problem._snapshot as implementation

    snapshot = AuthoringSnapshot({"value": 7})
    original = snapshot.semantic_identity
    object.__setattr__(snapshot, "_semantic_identity", Identity.from_data(original.to_data()))
    object.__setattr__(snapshot, "_canonical_json", snapshot._canonical_json.encode().decode())
    assert snapshot.semantic_identity is not original
    assert implementation._has_exact_snapshot_receipt(snapshot)
    assert validate_problem_snapshot(snapshot) == snapshot.hash


def test_snapshot_subclass_does_not_receive_or_use_constructor_authority():
    import pops.problem._snapshot as implementation

    class Derived(AuthoringSnapshot):
        pass

    snapshot = Derived({"value": 7})
    assert id(snapshot) not in implementation._SNAPSHOT_RECEIPTS
    with pytest.raises(TypeError, match="AuthoringSnapshot"):
        validate_problem_snapshot(snapshot)


def test_closed_preimage_json_roundtrip_preserves_canonical_bytes():
    from pops._structural_data import json_containers
    from pops.identity import canonical_sha256
    import pops.problem._snapshot as implementation

    shared = [-0.0, float.fromhex("0x0.0000000000001p-1022"), float.fromhex("0x1.fffffffffffffp+1023")]
    value = {"a": shared, "b": shared, "integer": 2 ** 2048,
             "text": "é\"\\\n", "booleans": [True, False, None]}
    value = _canonical(value)
    assert implementation._plain_json_data(value)
    decoded = json_containers(canonical_json(value, sort_keys=False, allow_nan=False))
    assert canonical_sha256(value) == canonical_sha256(decoded)


def test_replaced_encoder_cannot_issue_constructor_receipt(monkeypatch):
    import pops.problem._snapshot as implementation

    monkeypatch.setattr(implementation, "_encode_json", lambda *args, **kwargs: "{}")
    snapshot = AuthoringSnapshot({"value": 7})
    assert not implementation._has_exact_snapshot_receipt(snapshot)
    with pytest.raises(ValueError, match="canonical payload"):
        validate_problem_snapshot(snapshot)



def test_changed_hook_tuple_is_normalized_by_complete_fallback(monkeypatch):
    snapshot = AuthoringSnapshot({"value": [7]})
    data = snapshot.to_dict()
    data["value"] = tuple(data["value"])
    monkeypatch.setattr(AuthoringSnapshot, "to_dict", lambda self: data)
    assert validate_problem_snapshot(snapshot) == snapshot.hash


def test_changed_hook_bytes_are_rejected_by_complete_fallback(monkeypatch):
    snapshot = AuthoringSnapshot({"value": 7})
    monkeypatch.setattr(AuthoringSnapshot, "to_dict", lambda self: {"invalid": b"bytes"})
    with pytest.raises(TypeError):
        validate_problem_snapshot(snapshot)
