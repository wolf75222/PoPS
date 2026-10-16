"""Exact snapshot regressions; source-only, no native-runtime qualification."""
import sys
from types import MappingProxyType

import pytest

from examples.api_v02.finite_structural_depth import build_finite_structural_depth
from pops.problem import _snapshot as snapshot
from pops.problem import _snapshot_canonical as canonical
from pops.problem import _snapshot_payload as payload
from pops.time._program.api import Program

def legacy_time_value(program, *, artifact=False):
    if program is None:
        return None
    return {"type": type(program).__module__ + "." + type(program).__qualname__,
            "ir": program._serialize(include_provenance=not artifact),
            "hash": program._ir_hash()}


def serialize_count(callback):
    count = 0
    code = Program._serialize.__wrapped__.__code__
    def profile(frame, event, arg):
        nonlocal count
        if event == "call" and frame.f_code is code:
            count += 1
    sys.setprofile(profile)
    try:
        result = callback()
    finally:
        sys.setprofile(None)
    return result, count


def test_artifact_keeps_both_projection_protocols_and_identical_hash():
    p = build_finite_structural_depth(depth=8).case._time_registry.program
    expected, before = serialize_count(lambda: legacy_time_value(p, artifact=True))
    actual, after = serialize_count(lambda: payload._time_snapshot_value(p, artifact=True))
    assert actual == expected
    assert (before, after) == (2, 2)
    assert actual["hash"] == p._ir_hash()


@pytest.mark.parametrize("artifact", [False, True])
def test_custom_projection_error_and_order_are_not_elided(artifact):
    p = build_finite_structural_depth(depth=2).case._time_registry.program
    node = p._values[0]
    calls = []
    class Projection:
        def to_data(self):
            calls.append("project")
            if len(calls) == 2:
                raise ValueError("second exact projection refused")
            return {"version": 1}
    object.__setattr__(node, "attrs", MappingProxyType({**node.attrs, "extension": Projection()}))
    failures = []
    for operation in (legacy_time_value, payload._time_snapshot_value):
        calls.clear()
        with pytest.raises(ValueError) as exc:
            operation(p, artifact=artifact)
        failures.append((str(exc.value), tuple(calls)))
    assert failures[0] == failures[1] == ("second exact projection refused", ("project", "project"))


def test_foreign_hash_and_serialization_protocols_keep_their_calls(monkeypatch):
    p = build_finite_structural_depth(depth=2).case._time_registry.program
    calls = []
    def changed(self):
        calls.append("custom-hash")
        return "changed"
    monkeypatch.setattr(Program, "_ir_hash", changed)
    assert payload._time_snapshot_value(p, artifact=True) == legacy_time_value(p, artifact=True)
    assert calls == ["custom-hash", "custom-hash"]


@pytest.mark.parametrize("kind", ["cycle", "nonfinite", "opaque"])
def test_diagnostic_paths_and_cycle_cleanup_are_exact(kind):
    leaf = [] if kind == "cycle" else float("inf") if kind == "nonfinite" else object()
    if kind == "cycle":
        leaf.append(leaf)
    value = leaf
    for _ in range(180):
        value = {"level": [value]}
    active = set()
    prefix = "$.named" + ".level[0]" * 180
    with pytest.raises((ValueError, TypeError)) as shallow:
        canonical._canonical(leaf, path=prefix)
    with pytest.raises(type(shallow.value)) as deep:
        canonical._canonical(value, path="$.named", active=active)
    assert str(deep.value) == str(shallow.value)
    assert active == set()


def test_successful_scalar_forest_does_not_materialize_diagnostic_spines(monkeypatch):
    value = 7
    for _ in range(180):
        value = {"level": [value]}
    expected = {"$scalar": {"kind": "integer", "value": "7"}}
    for _ in range(180):
        expected = {"level": [expected]}
    def forbidden(self):
        raise AssertionError("a successful closed scalar projection does not need an error path")
    monkeypatch.setattr(canonical._ContainerPath, "__str__", forbidden)
    assert canonical._canonical(value) == expected


def test_complete_snapshot_provenance_and_all_identities_are_unchanged(monkeypatch):
    spec = build_finite_structural_depth(depth=4)
    def build():
        return snapshot.build_authoring_snapshot(spec.case, layout=spec.layout,
                                                 time=spec.case._time_registry.program)
    with monkeypatch.context() as prior:
        prior.setattr(payload, "_time_snapshot_value", legacy_time_value)
        expected = build()
    actual = build()
    for name in ("_canonical_json", "_artifact_canonical_json", "_semantic_canonical_json",
                 "_hash", "_artifact_hash", "_semantic_identity"):
        assert getattr(actual, name) == getattr(expected, name)
    assert "provenance" in actual._canonical_json
    assert snapshot.validate_problem_snapshot(actual) == expected.hash


@pytest.mark.parametrize("sort_keys", [False, True])
def test_deep_builtin_json_chunks_preserve_exact_bytes_and_occurrence_copies(sort_keys):
    import hashlib
    from pops._structural_data import (
        canonical_json_chunks, canonical_json_sha256, _canonical_json_protocol_chunks,
    )
    leaf = {"unicode": "é\ud800", "float": -0.0, "integer": 2**100, "bool": True}
    value = {"right": leaf, "left": (None, leaf)}
    for _ in range(180):
        value = {"layer": [value]}
    expected = "".join(_canonical_json_protocol_chunks(value, sort_keys=sort_keys))
    actual = "".join(canonical_json_chunks(value, sort_keys=sort_keys))
    assert actual == expected
    assert canonical_json_sha256(value, sort_keys=sort_keys) == hashlib.sha256(expected.encode()).hexdigest()


@pytest.mark.parametrize("kind", ["cycle", "nonfinite", "opaque"])
def test_chunk_fallback_preserves_exact_failure(kind):
    from pops._structural_data import canonical_json_chunks, _canonical_json_protocol_chunks
    leaf = [] if kind == "cycle" else float("inf") if kind == "nonfinite" else object()
    if kind == "cycle":
        leaf.append(leaf)
    value = leaf
    for _ in range(180):
        value = {"layer": [value]}
    errors = []
    for encoder in (_canonical_json_protocol_chunks, canonical_json_chunks):
        with pytest.raises((TypeError, ValueError)) as failure:
            "".join(encoder(value, allow_nan=False))
        errors.append((type(failure.value), str(failure.value)))
    assert errors[0] == errors[1]


def test_json_chunk_probe_does_not_invoke_mapping_protocol_twice():
    from collections import UserDict
    from pops._structural_data import canonical_json_chunks, _canonical_json_protocol_chunks
    calls = []
    class Observed(UserDict):
        def items(self):
            calls.append("items")
            return super().items()
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()
    shared = Observed({"value": 4})
    observed = []
    for encoder in (_canonical_json_protocol_chunks, canonical_json_chunks):
        calls.clear()
        observed.append(("".join(encoder([shared, shared])), tuple(calls)))
    assert observed[0] == observed[1]


def test_semantic_diagnostic_spines_are_lazy_and_exact(monkeypatch):
    from pops._structural_data import StructuralPath
    from pops.identity.semantic import semantic_value
    value = 1
    for _ in range(180):
        value = {"level": [value]}
    def forbidden(self):
        raise AssertionError("successful semantic projection must not render diagnostic paths")
    with monkeypatch.context() as context:
        context.setattr(StructuralPath, "__str__", forbidden)
        semantic_value(value, where="root")
    invalid = object()
    for _ in range(180):
        invalid = {"level": [invalid]}
    with pytest.raises(TypeError) as failure:
        semantic_value(invalid, where="root")
    assert str(failure.value) == "root" + ".level[0]"*180 + " contains unsupported semantic value object"
