"""Structural validation has finite native stack storage, not compiler nesting."""
import pytest
from pops.codegen.program_emit_finite_arrays import _uniform_element
from pops.codegen.program_uniform_finite import _layout


def _array(element, family=1):
    return {"vtype": "array", "element": element, "ownership": "replicated",
            "capacity": 2, "family": family}


def _source(signature, token="values"):
    lines = []
    _uniform_element(signature, token, lines)
    return "\n".join(lines)


def _nesting(source):
    depth = maximum = 0
    for character in source:
        depth += (character == "{")-(character == "}")
        maximum = max(maximum, depth)
    assert depth == 0
    return maximum


def test_nested_array_types_do_not_use_recursive_python_or_cpp_loops():
    signature = {"vtype": "scalar"}
    for family in range(1500):
        signature = _array(signature, family)
    source = _source(signature)
    assert "std::vector<UniformFrame>(1501)" in source
    assert source.count("while (top != 0)") == 1
    assert "element_i_" not in source
    assert _nesting(source) < 12
    assert source.index("finite array allocation capacity exhausted") < source.index("while (top != 0)")
    assert "std::bit_cast<std::int64_t>" in source
    assert "replicated array has a nonfinite element" in source


def test_record_layout_slots_preserve_variant_arms_and_array_boundaries():
    leaf = {"vtype": "int64"}
    signature = _array({"vtype": "product", "members": (
        ("v", {"vtype": "variant", "members": (("first", leaf), ("array", _array(leaf)))}),
        ("tail", leaf))})
    nodes, depth = _layout(signature)
    record = nodes[1]
    variant = nodes[record["children"][0]]
    assert record["width"] == 4 and record["offsets"] == [0, 3]
    assert variant["width"] == 3 and variant["offsets"] == [1, 2]
    assert depth == 5
    source = _source(signature)
    assert "switch (*tag)" in source
    assert source.index("replicated Variant tag differs") < source.index("switch (*tag)")
    domain_guard = ("finite_collective::require_domain(frame.array->domain, 2, 1, "
                    "ctx.prepared_execution_lane())")
    assert source.index(domain_guard) < source.index("frame.cursor < frame.array->values->size()")
    assert "const std::any* child" in source and "std::any(*" not in source
    assert _nesting(source) < 12


def test_deep_mixed_records_keep_scope_depth_bounded_and_do_not_read_inactive_arms():
    signature = {"vtype": "int64"}
    for _ in range(700):
        signature = {"vtype": "variant", "members": (
            ("empty", {"vtype": "product", "members": ()}),
            ("next", {"vtype": "product", "members": (("value", signature),)}))}
    source = _source(_array(signature))
    assert source.count("switch (*tag)") == 700
    assert "std::vector<UniformFrame>(1402)" in source
    assert _nesting(source) < 12
    # Only the chosen child is pushed; inactive Record cells can remain empty.
    assert "stack[top++] = UniformFrame" in source
    assert "std::any_cast<std::int64_t>(&frame.record->values[frame.offset])" in source


def test_top_level_variants_use_short_circuit_activation_instead_of_nested_scopes():
    signature, tokens = {"vtype": "scalar"}, "leaf"
    for i in range(1500):
        signature = {"vtype": "variant", "members": (("value", signature),)}
        tokens = {"tag": "tag_%d" % i, "payloads": {"value": tokens}}
    source = _source(signature, tokens)
    assert source.count("const bool uniform_active_") == 1500
    assert "uniform_active_0 && (tag_1498)" in source
    assert _nesting(source) <= 3


def test_domains_and_indices_keep_every_original_mpi_witness_check():
    index = {"vtype": "index", "ownership": "replicated", "capacity": 8, "family": 9}
    source = _source(_array(index))
    for key in ("->visible()", ".offset", ".domain->family", ".domain->version", ".domain->size"):
        assert key in source
    assert "replicated Index instance differs" in source
    assert "pops::all_reduce_min" in source and "pops::all_reduce_max" in source


def test_structural_cycles_fail_before_partial_source_and_aliases_remain_legal():
    leaf = {"vtype": "scalar"}
    signature = {"vtype": "product", "members": (("a", leaf), ("b", leaf))}
    source = _source(signature, {"a": "left", "b": "right"})
    assert source.count("replicated array Float64 bits differ") == 2
    cyclic = _array(leaf)
    cyclic["element"] = cyclic
    lines = []
    with pytest.raises(ValueError, match="cyclic"):
        _uniform_element(cyclic, "values", lines)
    assert not lines


def test_public_nested_compositions_reach_flat_native_validation():
    import pops
    from examples.api_v02.finite_structural_depth import build_finite_structural_depth
    from pops.codegen.program_codegen import emit_cpp_program
    spec = build_finite_structural_depth(depth=4)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(plan.time)
    assert "while (top != 0)" in source
    assert "pops_algebra_finite::Record" in source
    assert "std::any_cast" in source and "replicated Variant tag differs" in source
    assert "PyObject" not in source
