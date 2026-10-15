"""Storage-schema sharing never merges occurrences or scientific authorities."""

import re

import pytest

from pops.codegen.program_uniform_finite import (
    _SchemaRegistry,
    uniform_definitions,
    uniform_element,
    uniform_scope,
)


def array(element, *, family=1, capacity=2, ownership="replicated"):
    return dict(
        vtype="array",
        element=element,
        family=family,
        capacity=capacity,
        ownership=ownership,
    )


def nested(value):
    return array(
        {
            "vtype": "variant",
            "members": (
                ("active", {"vtype": "product", "members": (("value", value),)}),
                ("inactive", {"vtype": "int64"}),
            ),
        }
    )


@uniform_scope
def lower(signature, tokens):
    lines = []
    for token in tokens:
        uniform_element(signature, token, lines)
    return uniform_definitions(), "\n".join(lines)


def test_same_schema_shares_code_and_preserves_every_validation_invocation():
    signature = nested(nested({"vtype": "int64"}))
    definitions, calls = lower(signature, ("first", "second", "first"))
    assert definitions.count("while (top != 0)") == 1
    assert "schemas=7" in definitions
    rows = re.findall(
        r"pops_validate_uniform_structural_array\(ctx, (\w+), (\d+), (\d+)\);", calls
    )
    assert [row[0] for row in rows] == ["first", "second", "first"]
    assert len({row[1:] for row in rows}) == 1
    assert rows[0][2] == "7"
    assert "template<class Context>" in definitions
    assert "static void pops_validate_uniform_structural_array" in definitions
    assert definitions.index(
        "finite array allocation capacity exhausted"
    ) < definitions.index("while (top != 0)")


def test_repeated_child_types_keep_distinct_record_cells_and_variant_arms():
    child = array({"vtype": "scalar"})
    variant = {"vtype": "variant", "members": (("left", child), ("right", child))}
    product = {
        "vtype": "product",
        "members": (("a", child), ("b", child), ("choice", variant)),
    }
    registry = _SchemaRegistry()
    root, depth = registry.intern(array(product))
    product_node = registry.nodes[registry.nodes[root]["children"][0]]
    assert product_node["width"] == 5
    assert product_node["offsets"] == [0, 1, 2]
    assert product_node["children"][0] == product_node["children"][1]
    choice = registry.nodes[product_node["children"][2]]
    assert choice["children"][0] == choice["children"][1]
    assert choice["width"] == 3 and choice["offsets"] == [1, 2]
    assert depth == 5
    definitions, _ = lower(array(product), ("values",))
    assert definitions.index("replicated Variant tag differs") < definitions.index(
        "switch (*tag)"
    )
    assert "const auto offset = frame.offset + 1;" in definitions
    assert "const auto offset = frame.offset + 2;" in definitions
    assert (
        "std::any_cast<std::int64_t>(&frame.record->values[frame.offset])"
        in definitions
    )


def test_equivalent_data_shares_but_all_domain_and_scientific_authorities_are_keys():
    registry = _SchemaRegistry()
    leaf = {"vtype": "scalar", "point": {"clock": "first"}}
    base = registry.intern(array(leaf))[0]
    assert registry.intern(array(dict(leaf)))[0] == base
    mutations = [
        array(leaf, family=2),
        array(leaf, capacity=3),
        array({"vtype": "scalar", "point": {"clock": "second"}}),
    ]
    altered = {registry.intern(signature)[0] for signature in mutations}
    assert len(altered) == 3 and base not in altered
    for root in (base, *altered):
        assert registry.nodes[root]["signature"]["ownership"] == "replicated"


def test_distributed_payload_schema_and_ownership_are_never_merged():
    registry = _SchemaRegistry()
    distributed = array(
        {"vtype": "scalar"}, family=8, ownership="contiguous_distributed"
    )
    left = registry.intern(array(distributed))[0]
    other_payload = dict(distributed, element={"vtype": "int64"})
    right = registry.intern(array(other_payload))[0]
    replicated = registry.intern(array(dict(distributed, ownership="replicated")))[0]
    assert len({left, right, replicated}) == 3
    definitions, _ = lower(array(distributed), ("values",))
    assert "ctx.distributed_registry()" in definitions
    assert "::resolve_array<pops::kNativeDimension>" in definitions
    assert "reference_registry.common(reference->domain)" in definitions


def test_scope_restores_on_failure_and_definition_sealing_refuses_late_registration():
    @uniform_scope
    def bad():
        uniform_element(array({"vtype": "scalar"}), "discarded", [])
        raise ValueError("abort")

    with pytest.raises(ValueError, match="abort"):
        bad()
    with pytest.raises(RuntimeError, match="emission scope"):
        uniform_definitions()
    expected = lower(array({"vtype": "int64"}), ("value",))
    assert expected == lower(array({"vtype": "int64"}), ("value",))

    @uniform_scope
    def late():
        uniform_definitions()
        uniform_element(array({"vtype": "int64"}), "value", [])

    with pytest.raises(RuntimeError, match="after its definitions"):
        late()


def test_cycle_rejection_does_not_publish_partial_schema_definitions():
    registry = _SchemaRegistry()
    registry.intern(array({"vtype": "int64"}))
    count = len(registry.nodes)
    bad = array(None)
    bad["element"] = bad
    with pytest.raises(ValueError, match="cyclic"):
        registry.intern(bad)
    assert len(registry.nodes) == count


def test_all_prefixes_have_linear_generated_storage_validation_code():
    @uniform_scope
    def prefixes(count):
        signature = {"vtype": "int64"}
        calls = []
        for i in range(count):
            signature = nested(signature)
            uniform_element(signature, "array_%d" % i, calls)
        return uniform_definitions(), "\n".join(calls)

    small, small_calls = prefixes(32)
    large, large_calls = prefixes(64)
    assert "schemas=97" in small and "schemas=193" in large
    assert small.count("while (top != 0)") == large.count("while (top != 0)") == 1
    assert small_calls.count("pops_validate_uniform_structural_array(ctx,") == 32
    assert large_calls.count("pops_validate_uniform_structural_array(ctx,") == 64
    assert len(large) < 2.15 * len(small)
