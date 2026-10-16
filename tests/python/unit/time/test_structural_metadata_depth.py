"""Finite types deeper than the interpreter stack keep exact identities and authority."""
import json

import pytest

from pops._structural_data import canonical_json, equal
from pops.time.canonical_data import CanonicalData, _json_ready
from pops.time.value_metadata import _freeze_attr
from pops.time.variant_types import ValueType, instantiate


def nested(depth, *, alternating=False):
    value = {"vtype": "scalar"}
    for i in range(depth):
        kind = "variant" if alternating and i % 2 else "product"
        value = {"vtype": kind, "members": (("value", value),)}
    return value


@pytest.mark.parametrize("alternating", (False, True))
def test_deep_type_roundtrip_instantiation_and_metadata(alternating):
    data = nested(2048, alternating=alternating)
    original = ValueType.from_data(data)
    recovered = ValueType.from_data(original.to_data())
    assert original == recovered
    assert instantiate(original, None) == original
    frozen = _freeze_attr({"signature": original.signature})
    assert equal(_json_ready(frozen)["signature"], original.to_data())
    canonical = CanonicalData(original.to_data())
    assert canonical == CanonicalData(original.to_data())
    assert hash(canonical) == hash(CanonicalData(original.to_data()))
    assert canonical_json(canonical.to_data()) == canonical._json


def test_canonical_encoding_preserves_existing_json_bytes():
    value = {"é": [None, True, False, 1, -0.0, 1.25], "x": {"z": []}}
    assert canonical_json(value) == json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_shared_signatures_remain_shared_but_cyclic_authority_is_rejected():
    child = nested(1024)
    value = ValueType({"vtype": "product", "members": (("a", child), ("b", child))})
    assert value.signature["members"][0][1] is value.signature["members"][1][1]
    cyclic = {"vtype": "product"}
    cyclic["members"] = (("self", cyclic),)
    with pytest.raises(TypeError, match="cyclic"):
        ValueType(cyclic)
    raw = []
    raw.append(raw)
    with pytest.raises(ValueError, match="cycle"):
        _json_ready(raw)


def test_flat_cpp_storage_has_no_template_or_expression_depth_from_type():
    from pops.codegen.program_emit_finite_arrays import _packed, _unpacked, finite_cpp_type
    signature = ValueType.from_data(nested(1024, alternating=True)).signature
    assert finite_cpp_type(signature) == "pops_algebra_finite::Record"
    tokens = _unpacked(signature, "input")
    lines = []
    result = _packed(signature, tokens, lines, "value")
    assert result == "value_record"
    assert not any("std::get<" in line or "std::tuple<" in line for line in lines)
    assert max(map(len, lines)) < 250


@pytest.mark.parametrize("sort_keys", (False, True))
def test_streaming_json_and_hash_preserve_standard_tokens(sort_keys):
    import hashlib
    from enum import IntEnum
    from pops._structural_data import canonical_json_chunks, canonical_json_sha256
    class Number(IntEnum):
        VALUE = 3
    class NumberFloat(float):
        def __repr__(self):
            return "wrong"
    value = {"é": [None, True, False, Number.VALUE, NumberFloat(-0.),
                   1.e-300, float("nan"), float("inf"), -float("inf")],
             "a": "\u0000\n\U0001f642", "nested": [{"x": [1, 2, 3]}]}
    expected = json.dumps(value, sort_keys=sort_keys, separators=(",", ":"))
    assert "".join(canonical_json_chunks(value, sort_keys=sort_keys)) == expected
    assert canonical_json_sha256(value, sort_keys=sort_keys) == hashlib.sha256(expected.encode()).hexdigest()
    with pytest.raises(ValueError, match="Out of range"):
        canonical_json_sha256(value, allow_nan=False)


def test_streaming_json_deep_hash_and_cycle():
    import hashlib
    from pops._structural_data import canonical_json_sha256
    value = [7]
    for _ in range(2048):
        value = [value]
    expected = "["*2049+"7"+"]"*2049
    assert canonical_json_sha256(value) == hashlib.sha256(expected.encode()).hexdigest()
    value.append(value)
    with pytest.raises(ValueError, match="Circular reference"):
        canonical_json_sha256(value)


@pytest.mark.parametrize("sort_keys", (False, True))
def test_fast_json_matches_iterative_bytes_without_coercing_keys_or_hooks(sort_keys):
    from collections import UserDict
    from pops._structural_data import canonical_json_chunks

    class Observed(UserDict):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

    value = {"z": [{"é": -0.0}, None], "a": (True, False, 1e-310)}
    expected = "".join(canonical_json_chunks(value, sort_keys=sort_keys))
    assert canonical_json(value, sort_keys=sort_keys) == expected
    with pytest.raises(TypeError, match="string mapping keys"):
        canonical_json({1: "not-a-string"})
    calls = []
    observed = Observed({"key": [1]})
    expected = "".join(canonical_json_chunks(observed, sort_keys=sort_keys))
    expected_calls = list(calls)
    calls.clear()
    assert canonical_json(observed, sort_keys=sort_keys) == expected
    assert calls == expected_calls


def test_fast_json_falls_back_for_deep_data_and_rejects_cycles():
    value = [7]
    for _ in range(2048):
        value = [value]
    assert canonical_json(value) == "[" * 2049 + "7" + "]" * 2049
    value.append(value)
    with pytest.raises(ValueError, match="Circular reference"):
        canonical_json(value)


def test_json_projection_fast_copy_preserves_occurrences_scalars_and_key_order():
    import math
    shared = {"é": (-0.0, 1e-310, 2**80), "last": [True, None]}
    value = {"z": shared, "a": shared}
    result = _json_ready(value)
    assert list(result) == ["z", "a"]
    assert result["z"] == result["a"] == {
        "é": [-0.0, 1e-310, 2**80], "last": [True, None]}
    assert math.copysign(1, result["z"]["é"][0]) == -1
    result["z"]["last"].append(7)
    assert result["a"]["last"] == shared["last"] == [True, None]


def test_json_projection_keeps_observable_hooks_and_noncanonical_keys():
    calls = []
    class Observed:
        def to_data(self):
            calls.append(len(calls))
            return {"occurrence": len(calls)}
    shared = Observed()
    assert _json_ready([shared, shared]) == [{"occurrence": 1}, {"occurrence": 2}]
    assert calls == [0, 1]
    assert _json_ready({"": 1}) == {"mapping_entries": [["", 1]]}
    assert _json_ready({2: "two"}) == {"mapping_entries": [[2, "two"]]}


def test_json_projection_fast_codec_does_not_add_integer_or_depth_limits():
    from pops._structural_data import equal, json_containers
    value = [7]
    for _ in range(2048):
        value = [value]
    projected = _json_ready(value)
    assert equal(projected, value)
    assert equal(json_containers("[" * 2049 + "7" + "]" * 2049), value)
    huge = 10**5000
    assert _json_ready([huge]) == [huge]
    value.append(value)
    with pytest.raises(ValueError, match="cycle"):
        _json_ready(value)


def test_canonical_snapshot_fast_codec_preserves_order_scalars_and_isolation():
    from pops.identity.scalar import scalar_data
    shared = {"é": (-0.0, 1e-310, 2**80), "last": [True, None]}
    source = {"z": shared, "a": shared}
    snapshot = CanonicalData(source)
    first = snapshot.to_data()
    assert list(first) == ["z", "a"]
    assert first["z"]["é"] == [{"scalar": scalar_data(x)} for x in shared["é"]]
    first["z"]["last"].append(7)
    shared["last"].append(8)
    assert snapshot.to_data()["z"]["last"] == [True, None]
    assert first["a"]["last"] == [True, None]


def test_canonical_snapshot_retains_each_observable_protocol_occurrence():
    calls = []
    class Observed:
        __pops_ir_immutable__ = True
        def to_data(self):
            calls.append(len(calls))
            return {"occurrence": len(calls)}
    observed = Observed()
    snapshot = CanonicalData([observed, observed])
    from pops.identity.scalar import scalar_data
    assert snapshot.to_data() == [
        {"occurrence": {"scalar": scalar_data(1)}},
        {"occurrence": {"scalar": scalar_data(2)}}]
    assert calls == [0, 1]


def test_streaming_chunks_bound_occurrence_expansion_and_keep_exact_bytes():
    import hashlib
    from pops._structural_data import canonical_json_chunks, canonical_json_sha256
    shared = {"z": "\U0001f642" * 10, "a": [-0.0, True, None]}
    value = [shared] * 10000
    chunks = list(canonical_json_chunks(value))
    expected = json.dumps(value, sort_keys=True, separators=(",", ":"))
    assert "".join(chunks) == expected
    assert max(map(len, chunks)) <= 65536
    assert canonical_json_sha256(value) == hashlib.sha256(expected.encode()).hexdigest()
