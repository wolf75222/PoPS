"""Deep closed metadata reaches actual Program lowering without a recursion override."""
import hashlib
import json

import pytest

from pops._structural_data import canonical_json, equal, json_containers
from pops.graph_functions import GraphFunction, _Codec, graph
from pops.identity.semantic import _drop_program_presentation, semantic_value
from pops.model.hash_data import canonical_hash_data
from pops.time.variant_types import Float64, Product, ValueType


def _nested(depth):
    value = {"vtype": "scalar"}
    for _ in range(depth):
        value = {"vtype": "product", "members": (("value", value),)}
    return ValueType.from_data(value)


def test_deep_closed_function_snapshot_and_call():
    nested = _nested(1024)
    identity = graph(inputs=Product(x=nested), output=nested)(lambda p, a: a["x"])
    loaded = GraphFunction.from_data(identity.to_data())
    assert loaded.identity == identity.identity
    assert equal(loaded.to_data(), identity.to_data())
    # Mutating the returned snapshot cannot modify the immutable closure.
    data = loaded.to_data()
    data["effects"].append("collective")
    assert loaded.effects == frozenset()


def test_deep_function_call_reaches_real_program_cpp_lowering(request):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.time import Program
    from typed_program_support import typed_state

    nested = _nested(1024)
    identity = graph(inputs=Product(x=nested), output=nested)(lambda p, a: a["x"])
    program = Program("deep_closed_call")
    q = typed_state(program, "q", state_name="U")
    value = program.float64(1.)
    for _ in range(1024):
        value = program.product(value=value)
    result = identity(program, x=value)
    assert result.vtype == "product"
    program.commit(q.next, program.value("accepted", q.n, at=q.next.point))
    source = emit_cpp_program(program)
    # This identity lowers to aliases. Storage allocation is not required, but
    # every original product must remain covered by authenticated source maps.
    assert source.count("// pops-source-map@1 begin ") >= 1024
    assert "// pops-source-map@1 inventory " in source
    assert "graph_argument" not in source and "PyObject" not in source
    assert "std::get<" not in source
    request.node.user_properties.append(("cpp_bytes", len(source.encode())))
    request.node.user_properties.append(("cpp_sha256", hashlib.sha256(source.encode()).hexdigest()))


def test_closed_graph_snapshot_preserves_existing_json_identity_bytes():
    function = graph(inputs=Product(x=Float64), output=Float64)(lambda p, a: a["x"] * 2)
    payload = json.dumps(function.to_data(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert function.identity == hashlib.sha256(payload.encode()).hexdigest()


def test_deep_capture_roundtrip_and_cycle_rejection():
    capture = {"number": 1.25}
    for _ in range(1024):
        capture = {"child": (capture,)}
    codec = _Codec()
    assert equal(codec.decode(codec.encode(capture)), capture)
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cycle"):
        codec.encode(cyclic)
    encoded = {"$tuple": []}
    encoded["$tuple"].append(encoded)
    with pytest.raises(ValueError, match="cycle"):
        codec.decode(encoded)
    with pytest.raises(ValueError, match="map entry"):
        codec.decode({"$map": [[1, 2, 3]]})
    assert equal(json_containers(canonical_json(capture)), capture)


@pytest.mark.parametrize("payload", (
    '{}', '[]', '[{},[],null,true,false,1,-2,3.25]',
    '{"escaped":"a\\\\b\\n\\u00e9","nested":{"x":[1,2]}}',
    ' { "x": 1, "x": 2 } ',
))
def test_iterative_json_metadata_preserves_stdlib_decode(payload):
    assert equal(json_containers(payload), json.loads(payload))


@pytest.mark.parametrize("payload", ('', '{', '[', '[1,]', '{"a":1,}',
                                     '{"a" 1}', '{"a":}', '[1 2]', 'true false'))
def test_iterative_json_metadata_refuses_malformed_containers(payload):
    with pytest.raises(json.JSONDecodeError):
        json_containers(payload)


def test_deep_metadata_identity_paths_and_hook_occurrences():
    class Hook:
        calls = 0

        def to_data(self):
            self.calls += 1
            return {"x": 1}

    hook = Hook()
    assert canonical_hash_data([hook, hook]) == [{"x": 1}, {"x": 1}]
    assert hook.calls == 2
    raw = {"number": 1.25}
    for _ in range(1024):
        raw = {"child": [raw]}
    assert canonical_json(canonical_hash_data(raw)) == canonical_json(raw)
    assert equal(_drop_program_presentation(raw, None), raw)
    assert len(canonical_json(semantic_value(raw, where="root"))) > 10000
    with pytest.raises(TypeError, match=r"root.child\[0\].bad"):
        semantic_value({"child": [{"bad": object()}]}, where="root")

    class Cycle:
        def to_data(self):
            return self

    with pytest.raises(ValueError, match="cycle"):
        canonical_hash_data(Cycle())


def test_deep_type_witness_cannot_replace_a_leaf_contract():
    nested = _nested(1024)
    identity = graph(inputs=Product(x=nested), output=nested)(lambda p, a: a["x"])
    data = identity.to_data()
    leaf = data["output"]
    for _ in range(1024):
        leaf = leaf["members"][0][1]
    leaf["vtype"] = "int64"
    with pytest.raises(TypeError, match="type witnesses"):
        GraphFunction.from_data(data, inputs=identity.inputs, output=identity.output)
