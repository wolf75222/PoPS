"""Finite arrays are typed native regions, not callbacks evaluated by the runtime."""
from __future__ import annotations

import pytest

from pops.time import Program, TimePoint
from pops.codegen.program_codegen import emit_cpp_program
from pops.time._program.detach import detach_compiled_program
from typed_program_support import typed_state


def _fixture(name="finite_arrays"):
    p = Program(name)
    q = typed_state(p, "left", state_name="U")
    q.n
    return p, q


def _source(p, q):
    p.commit(q.next, p.value("accepted", q.n, at=q.next.point))
    assert p.validate()
    assert p._ir_hash() == detach_compiled_program(p)._ir_hash()
    assert p.to_graph().to_data() == detach_compiled_program(p).to_graph().to_data()
    return emit_cpp_program(p)


def test_arrays_map_scan_reduce_and_loop_carry_are_native_and_body_is_recorded_once():
    p, q = _fixture()
    d = p.range_set(p.int64((1 << 53)+3) - (1 << 53), capacity=4, ownership="replicated")
    calls = []
    def body(builder, index):
        calls.append(index)
        return builder.product(i=builder.index_value(index), x=builder.float64(builder.index_value(index)))
    a = p.map(d, body)
    assert len(calls) == 1
    a = p.fold(a, 2, lambda b, old, k: b.map(b.domain(old),
        lambda b, i: b.product(i=old[i]["i"], x=old[i]["x"]+1.)))
    scanned = p.scan(d, p.float64(0), lambda b, s, i:
        b.product(state=s+a[i]["x"], value=s+a[i]["x"]), order="ascending")
    p.reduce(scanned["values"], p.float64(0), lambda b, s, x: s+x, order="ascending")
    source = _source(p, q)
    assert "std::int64_t(9007199254740995)" in source
    assert "std::bit_cast<std::int64_t>" in source
    assert "pops_algebra_finite::make_values" in source
    assert "pops/runtime/program/finite_structural_storage.hpp" in source
    assert "finite array allocation capacity exhausted" in source
    assert "for (std::int64_t finite_" in source
    assert "pops_finite_instance_version" not in source
    assert "ctx.continuation_new_domain(" in source
    assert len(calls) == 1


def test_index_has_domain_family_and_native_instance_guards():
    p, q = _fixture("indices")
    outer = p.range_set(2, capacity=2, ownership="replicated")
    indices = p.map(outer, lambda b, i:
        b.index(b.range_set(1, capacity=1, ownership="replicated"), 0))
    first, second = indices[p.index(outer, 0)], indices[p.index(outer, 1)]
    values = p.map(p.domain(first), lambda b, i: b.int64(7))
    p.capture(lambda b: values[second])
    with pytest.raises(ValueError, match="different finite domain family"):
        values[p.index(outer, 0)]
    with pytest.raises(TypeError, match="checked Index"):
        values[p.int64(0)]
    source = _source(p, q)
    assert "same live domain instance" in source
    assert "ctx.continuation_new_domain(" in source
    assert "->visible()" in source


def test_empty_scan_and_gather_have_explicit_order_and_ownership():
    p, q = _fixture("empty")
    with pytest.raises(NotImplementedError, match="replicated"):
        p.range_set(3, capacity=3, ownership="distributed")
    empty = p.range_set(0, capacity=0, ownership="replicated")
    out = p.scan(empty, p.int64(17), lambda b, s, i:
        b.product(state=s+1, value=b.index_value(i)), order="ascending")
    assert out["state"].vtype == "int64"
    p.reduce(out["values"], p.int64(0), lambda b, s, x: s+x, order="ascending")
    d = p.range_set(3, capacity=3, ownership="replicated")
    values = p.map(d, lambda b, i: b.index_value(i)*10)
    indices = p.map(d, lambda b, i: b.index(d, 2-b.index_value(i)))
    assert p.gather(values, indices).vtype == "array"
    with pytest.raises(NotImplementedError, match="ascending"):
        p.scan(d, p.int64(0), lambda b, s, i: s, order="unordered")
    _source(p, q)


def test_authoring_failure_rolls_back_domain_issuance_and_rejects_illegal_elements():
    p, q = _fixture("rollback")
    before = p._ir_hash()
    def invalid(builder):
        d = builder.range_set(1, capacity=1, ownership="replicated")
        return builder.map(d, lambda b, i: b.success(q.n))
    with pytest.raises(TypeError, match="array elements"):
        p.capture(invalid)
    assert p._ir_hash() == before
    assert p._next_finite_domain_identity == 0
    d = p.range_set(1, capacity=1, ownership="replicated")
    assert d.attrs["signature"]["family"] == 0
    _source(p, q)


def test_map_captures_ancestor_constant_but_refuses_sibling_and_preserves_branch_type():
    p, q = _fixture("capture_constant")
    d = p.range_set(1, capacity=1, ownership="replicated")
    constant = p.int64(7)
    left = p.map(d,lambda b,i:constant)
    right = p.map(d,lambda b,i:b.int64(9))
    selected = p.branch(p.boolean(True),lambda b:left,lambda b:right)
    assert selected.attrs["signature"] == left.attrs["signature"]
    sibling=[]
    p.map(d,lambda b,i:(sibling.append(b.int64(11)) or sibling[-1]))
    with pytest.raises(ValueError,match="region|escape|sub-block"):
        p.map(d,lambda b,i:sibling[0])
    source = _source(p,q)
    assert "branch_choice_" in source


def test_amr_array_invocation_uses_the_native_level_scope():
    p,q = _fixture("amr_finite")
    d = p.range_set(3,capacity=3,ownership="replicated")
    values = p.map(d, lambda b, i: b.float64(b.index_value(i)))
    total = p.reduce(values, p.float64(0), lambda b, s, x: s+x, order="ascending")
    p.record_scalar("sum", total)
    p.commit(q.next,p.value("accepted",q.n,at=q.next.point))
    source = emit_cpp_program(p, target="amr_system")
    assert "ctx.continuation_new_domain(" in source
    assert "auto _finite_value_scope = ctx.finite_value_scope();" in source
    assert "ctx.advance_hierarchy(" in source


def test_array_projection_retains_the_exact_element_point():
    p,q = _fixture("array_point")
    later = p.value("later",q.n,at=TimePoint(p.clock,step=1))
    sample = p.norm2(later)
    d = p.range_set(1,capacity=1,ownership="replicated")
    a = p.map(d,lambda b,i:sample)
    assert a[p.index(d,0)].point == sample.point
    assert sample.point != d.point
    _source(p,q)


def test_domain_index_and_array_reads_check_the_native_realm_before_collectives():
    p, q = _fixture("live_realm")
    domain = p.range_set(2, capacity=3, ownership="replicated")
    array = p.map(domain, lambda b, i: b.index_value(i))
    p.domain(array)
    array[p.index(domain, 1)]
    source = _source(p, q)
    # Independent public paths must retain the native guard, including direct array indexing.
    assert "->visible()" in source
    assert ".domain->visible()" in source
    assert "finite domain instance" in source
    assert "same live domain instance" in source


@pytest.mark.parametrize("capacity", (2**31, 2**53+3, 2**63-1))
def test_int64_capacity_is_exact_metadata_for_small_runtime_arrays(capacity):
    from pops.time._program.finite_arrays import IndexFamily
    p, q = _fixture("int64_capacity")
    family = IndexFamily(capacity)
    domain = p.range_set(3, family=family)
    empty = p.range_set(0, family=domain)
    values = p.map(domain, lambda b, i: b.index_value(i)+1)
    p.reduce(values, p.int64(0), lambda b, s, x: s+x, order="ascending")
    p.map(empty, lambda b, i: b.int64(1)//b.int64(0))
    assert domain.attrs["signature"]["capacity"] == capacity
    assert empty.attrs["signature"]["capacity"] == capacity
    source = _source(p, q)
    assert str(capacity) in source
    assert "std::size_t((" in source  # allocation follows domain->size, never capacity


@pytest.mark.parametrize("capacity", (-1, True, 2**63, 1.0))
def test_domain_capacity_outside_int64_is_rejected_without_authoring_mutation(capacity):
    from pops.time._program.finite_arrays import IndexFamily
    p, _ = _fixture("invalid_int64_capacity")
    before = p._ir_hash()
    with pytest.raises(ValueError, match="exact Int64"):
        p.range_set(0, capacity=capacity, ownership="replicated")
    with pytest.raises(ValueError, match="exact Int64"):
        IndexFamily(capacity)
    assert p._ir_hash() == before
