"""Closed-data acceleration preserves protocols, detach, diagnostics and bytes."""

from collections import UserDict
from types import MappingProxyType
from unittest.mock import patch
import pytest
from pops._structural_data import equal, canonical_json
from pops.problem import _snapshot as snapshot
from pops.problem import _snapshot_canonical as canonical
from pops.identity import semantic
from examples.api_v02.finite_structural_depth import build_finite_structural_depth


def reference_canonical(value, **kwargs):
    # The unchanged iterative projector remains the fallback authority.
    with patch.object(canonical, "_canonical_builtin", return_value=(False, None)):
        return canonical._canonical(value, **kwargs)


_native_semantic_value = semantic.semantic_value


def reference_semantic(value, *, where):
    # Disable only this call's closed-data optimization. No protocol is skipped.
    with patch("pops._structural_data._plain_json_data", return_value=False):
        return _native_semantic_value(value, where=where)


@pytest.mark.parametrize("projection", ["canonical", "semantic"])
def test_exact_scalar_and_occurrence_copies(projection):
    leaf = {"x": [None, True, False, 0, -(2**53), 2**80, -0.0, 1.0e-300, 1.0e300, "é\ud800"]}
    value = {"right": leaf, "left": leaf}
    functions = (
        (reference_canonical, canonical._canonical)
        if projection == "canonical"
        else (
            lambda v: reference_semantic(v, where="root"),
            lambda v: semantic.semantic_value(v, where="root"),
        )
    )
    expected, actual = [f(value) for f in functions]
    assert equal(expected, actual)
    assert canonical_json(actual) == canonical_json(expected)
    assert actual["left"] is not actual["right"]
    actual["left"]["x"].append("changed")
    assert leaf["x"][-1] == "é\ud800"
    assert actual["right"]["x"][-1] == "é\ud800"


@pytest.mark.parametrize("depth", [0, 180, 1200])
@pytest.mark.parametrize("projection", ["canonical", "semantic"])
def test_unbounded_depth_and_failure_paths(depth, projection):
    functions = (
        (reference_canonical, canonical._canonical)
        if projection == "canonical"
        else (
            lambda v: reference_semantic(v, where="root"),
            lambda v: semantic.semantic_value(v, where="root"),
        )
    )
    for leaf in (7, object(), float("inf")):
        value = leaf
        for _ in range(depth):
            value = {"layer": [value]}
        outcomes = []
        for function in functions:
            try:
                outcomes.append(("value", function(value)))
            except (TypeError, ValueError, RecursionError) as error:
                outcomes.append((type(error), str(error)))
        assert outcomes[0][0] == outcomes[1][0]
        assert equal(outcomes[0][1], outcomes[1][1])


@pytest.mark.parametrize("projection", ["canonical", "semantic"])
def test_cycle_cleanup_and_active_ancestor(projection):
    value = {"self": None}
    value["self"] = value
    functions = (
        (reference_canonical, canonical._canonical)
        if projection == "canonical"
        else (
            lambda v: reference_semantic(v, where="root"),
            lambda v: semantic.semantic_value(v, where="root"),
        )
    )
    failures = []
    for function in functions:
        with pytest.raises(ValueError) as error:
            function(value)
        failures.append(str(error.value))
    assert failures[0] == failures[1]
    if projection == "canonical":
        failures.clear()
        for function in functions:
            active = {id(value)}
            with pytest.raises(ValueError) as error:
                function(value, active=active)
            failures.append(str(error.value))
            assert active == {id(value)}
        assert failures[0] == failures[1]


@pytest.mark.parametrize("projection", ["canonical", "semantic"])
def test_subclasses_mapping_protocols_and_mutation_order(projection):
    calls = []
    shared = {"number": 3}

    class Number(float):
        def hex(self):
            calls.append("hex")
            shared["number"] += 1
            return super().hex()

    class Observed(UserDict):
        def items(self):
            calls.append("items")
            return super().items()

        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

    value = [shared, Observed({"number": Number(-0.0)}), shared]
    functions = (
        (reference_canonical, canonical._canonical)
        if projection == "canonical"
        else (
            lambda v: reference_semantic(v, where="root"),
            lambda v: semantic.semantic_value(v, where="root"),
        )
    )
    results = []
    for function in functions:
        calls.clear()
        shared["number"] = 3
        results.append((function(value), tuple(calls)))
    assert equal(results[0], results[1])
    assert results[0][1]


def test_tuple_proxy_and_string_subclass_fallback():
    class Text(str):
        pass

    for value in ((1, {"nested": 2}), MappingProxyType({"value": [3]}), {Text("key"): -0.0}):
        assert equal(reference_canonical(value), canonical._canonical(value))
        assert equal(
            reference_semantic(value, where="root"), semantic.semantic_value(value, where="root")
        )


def test_real_snapshot_provenance_hashes_and_protocol_order(monkeypatch):
    from pops.time._program.api import Program

    spec = build_finite_structural_depth(depth=6)
    events = []
    original = Program.cadence_contract

    def observed(self):
        events.append("cadence")
        return original(self)

    monkeypatch.setattr(Program, "cadence_contract", observed)

    def build():
        return snapshot.build_authoring_snapshot(
            spec.case, layout=spec.layout, time=spec.case._time_registry.program
        )

    with monkeypatch.context() as legacy:
        legacy.setattr(snapshot, "_canonical", reference_canonical)
        legacy.setattr(semantic, "semantic_value", reference_semantic)
        legacy.setattr(snapshot, "semantic_value", reference_semantic)
        expected = build()
        before = tuple(events)
    events.clear()
    actual = build()
    assert tuple(events) == before
    for name in (
        "_canonical_json",
        "_artifact_canonical_json",
        "_semantic_canonical_json",
        "_hash",
        "_artifact_hash",
        "_semantic_identity",
    ):
        assert getattr(actual, name) == getattr(expected, name)


def test_float_hook_result_is_revalidated_on_the_next_projection():
    calls = []
    opaque = object()

    class Number(float):
        def hex(self):
            calls.append("hex")
            return opaque

    failures = []
    for function in (reference_semantic, semantic.semantic_value):
        calls.clear()
        first = function({"value": Number(1)}, where="first")
        assert first["value"]["binary64"] is opaque
        with pytest.raises(TypeError) as error:
            function(first, where="second")
        failures.append((str(error.value), tuple(calls)))
    assert failures[0] == failures[1]


def test_opaque_metaclass_equality_is_not_added_by_preflight():
    events = []

    class Meta(type):
        def __eq__(cls, other):
            events.append(other)
            if other is dict or other is list:
                raise RuntimeError("container equality is not a projection protocol")
            return cls is other

        __hash__ = type.__hash__

    class Opaque(float, metaclass=Meta):
        pass

    # Warm Mapping ABC caches equally; their weakref equality is pre-existing.
    reference_canonical({"opaque": Opaque(1.5)})
    outcomes = []
    for function in (reference_canonical, canonical._canonical):
        events.clear()
        try:
            result = function({"opaque": Opaque(1.5)})
        except (TypeError, ValueError, RuntimeError) as error:
            result = (type(error), str(error))
        outcomes.append((result, tuple(events)))
    assert equal(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1] == outcomes[1][1]


def test_scalar_check_hook_order_is_preserved(monkeypatch):
    events = []
    original = canonical.math.isfinite

    def observed(value):
        events.append(value)
        return original(value)

    monkeypatch.setattr(canonical.math, "isfinite", observed)
    value = {"first": [1.25, -2.5], "second": 3.75}
    outcomes = []
    for function in (reference_canonical, canonical._canonical):
        events.clear()
        outcomes.append((function(value), tuple(events)))
    assert equal(outcomes[0], outcomes[1])
    assert outcomes[0][1] == (1.25, -2.5, 3.75)
    events.clear()
    assert equal(reference_semantic(value, where="root"),
                 semantic.semantic_value(value, where="root"))
    assert events == []


def test_json_depth_preflight_does_not_compare_user_metaclass():
    from pops._structural_data import _plain_json_depths, canonical_json_chunks

    class Meta(type):
        def __eq__(cls, other):
            raise RuntimeError("unexpected type equality")

        __hash__ = type.__hash__

    class Text(str, metaclass=Meta):
        pass

    value = {"key": Text("text")}
    assert _plain_json_depths(value) is None
    assert "".join(canonical_json_chunks(value)) == '{"key":"text"}'
