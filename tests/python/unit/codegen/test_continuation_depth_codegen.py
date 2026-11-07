"""Deep authoring/schema checks; native serialization has separate C++ receipts."""
from __future__ import annotations

import re

import pytest

from pops.codegen.program_emit_versioned import schema_cpp


def test_schema_construction_keeps_runtime_depth_out_of_cpp_expression_depth():
    signature = {"vtype": "int64"}
    depth = 1500
    for _ in range(depth):
        signature = {"vtype": "product", "members": (("value", signature),)}
    source = schema_cpp(signature, {})
    assert source.count("schema_nodes.emplace_back(") == depth + 1
    assert source.count("ContinuationKind::Int64") == 1
    assert source.count("ContinuationKind::Product") == depth
    # Every move targets a completed child; there are no recursively nested
    # constructor expressions or a compiler template depth proportional to data.
    moves = [int(index) for index in re.findall(r"std::move\(schema_nodes\[(\d+)\]\)", source)]
    assert moves == list(range(depth))
    assert "return std::move(schema_nodes.back());" in source
    nesting = maximum = 0
    for character in source:
        nesting += (character == "{") - (character == "}")
        maximum = max(maximum, nesting)
    assert nesting == 0 and maximum <= 3


def test_schema_postorder_preserves_variant_order_and_distributed_domain_contract():
    signature = {
        "vtype": "variant",
        "members": (
            ("empty", {"vtype": "bool"}),
            ("values", {"vtype": "array", "ownership": "contiguous_distributed",
                        "family": 91, "capacity": 37, "element": {"vtype": "int64"}}),
        ),
    }
    source = schema_cpp(signature, {})
    kinds = re.findall(r"emplace_back\([^\n]*ContinuationKind::(\w+)", source)
    assert kinds == ["Boolean", "Int64", "DistributedArray", "Variant"]
    assert 'std::vector<std::string>{"empty","values"}' in source
    assert ",91,37,-1,0," in source
    moves = [int(index) for index in re.findall(r"std::move\(schema_nodes\[(\d+)\]\)", source)]
    assert moves == [1, 0, 2]


def test_schema_cycle_is_rejected_and_an_acyclic_alias_is_allowed():
    leaf = {"vtype": "int64"}
    source = schema_cpp({"vtype": "product", "members": (("left", leaf), ("right", leaf))}, {})
    assert source.count("ContinuationKind::Int64") == 2
    cyclic = {"vtype": "product"}
    cyclic["members"] = (("self", cyclic),)
    with pytest.raises(ValueError, match="cyclic"):
        schema_cpp(cyclic, {})


def test_marshalling_and_distributed_projection_keep_deep_values_out_of_cpp_nesting():
    from pops.codegen.program_emit_distributed_arrays import unpack
    from pops.codegen.program_emit_versioned import pack_local, _unpack

    signature, tokens = {"vtype": "int64"}, "INT64_C(9007199254740997)"
    for _ in range(1500):
        signature = {"vtype": "product", "members": (("value", signature),)}
        tokens = {"value": tokens}
    lines = []
    packed = pack_local(signature, tokens, lines, "deep", {}, capacity=8388608)
    restored = _unpack(signature, "source", lines, "restored", capacity=8388608)
    cpp = "\n".join([*lines, packed, restored])
    assert "finite_structural::pack<" in cpp and "finite_structural::unpack<" in cpp
    assert "Limits{8388608,8388608,8388608,&pops_finite_residency}" in cpp
    assert "std::tuple<" not in cpp and "std::variant<" not in cpp
    nesting = maximum = 0
    for character in cpp:
        nesting += (character == "{") - (character == "}")
        maximum = max(maximum, nesting)
    assert nesting == 0 and maximum <= 4
    projected = unpack(signature, "source", capacity=8388608)
    for _ in range(1500):
        projected = projected["value"]
    assert projected.count("finite_structural::project<") == 1
    assert "->members()" not in projected


def test_requests_ingress_borrows_the_token_without_a_prebudget_vector_copy():
    from pops.codegen.program_emit_versioned import pack_local
    signature = {"vtype": "array", "ownership": "contiguous_distributed", "family": 1,
                 "capacity": 100, "element": {"vtype": "index", "family": 2,
                 "capacity": 100, "ownership": "contiguous_distributed"}}
    lines = []
    source = pack_local(signature, "requests_token", lines, "borrow", {}, capacity=4096)
    assert "finite_structural::source_cell(requests_token)" in source
    assert "Requests(requests_token)" not in source
    assert "Limits{4096,4096,4096,&pops_finite_residency}" in source


def test_distributed_capture_discovery_is_iterative_for_deep_products():
    from pops.codegen.program_emit_distributed_arrays import _has_finite_capture
    finite = {"vtype": "index_set", "ownership": "replicated"}
    plain = {"vtype": "int64"}
    for _ in range(1500):
        finite = {"vtype": "product", "members": (("value", finite),)}
        plain = {"vtype": "product", "members": (("value", plain),)}
    assert _has_finite_capture(finite)
    assert not _has_finite_capture(plain)
