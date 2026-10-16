"""A sum's envelope never changes the exact temporal authorities of its leaves."""
import pytest

from pops.time import Program
from pops.time.points import TimePoint
from pops.time.variant_types import Int64, Product, Variant, type_of
from pops.time.value_signatures import value_signature
from typed_program_support import typed_state


def test_alternatives_share_an_envelope_but_keep_distinct_field_points():
    p = Program("variant_temporal_envelope")
    q = typed_state(p, "q", state_name="U")
    later = p.value("later", q.n, at=q.next.point)
    schema = Variant(old=type_of(q.n), new=type_of(later))
    old = p.variant(schema, "old", q.n)
    new = p.variant(schema, "new", later)
    assert old.point == new.point == TimePoint(p.clock)
    assert value_signature(old) == value_signature(new)
    selected = p.branch(p.boolean(True), lambda b: old, lambda b: new)
    observed = []
    def inspect(b, field):
        observed.append(field.point)
        return b.int64(0)
    p.match(selected, old=inspect, new=inspect)
    assert set(observed) == {q.n.point, q.next.point}
    with pytest.raises(TypeError, match="exact structural type"):
        p.variant(schema, "old", later)


def test_nested_sum_keeps_old_field_leaf_and_unselected_scalar_point():
    p = Program("nested_temporal_envelope")
    q = typed_state(p, "q", state_name="U")
    later = p.value("later", q.n, at=q.next.point)
    inner = Variant(old=type_of(q.n), new=Product(field=type_of(later), count=Int64))
    payload = p.variant(inner, "old", q.n)
    outer = p.variant(Variant(record=Product(choice=type_of(payload))), "record",
                      p.product(choice=payload))
    projected = []
    def inspect(b, record):
        def old(child, field):
            projected.append(field.point)
            return child.int64(0)
        def new(child, data):
            projected.extend((data["field"].point, data["count"].point))
            return child.int64(0)
        return b.match(record["choice"], old=old, new=new)
    p.match(outer, record=inspect)
    assert q.n.point in projected and q.next.point in projected
    assert projected.count(TimePoint(p.clock)) == 2


def test_outer_point_cannot_make_a_foreign_field_authority_compatible():
    p = Program("variant_foreign_authority")
    left = typed_state(p, "left", state_name="U")
    right = typed_state(p, "right", state_name="U")
    schema = Variant(sample=type_of(left.n))
    foreign = right.n
    before = p._ir_hash(), p._next_id
    with pytest.raises(TypeError, match="exact structural type"):
        p.variant(schema, "sample", foreign)
    assert (p._ir_hash(), p._next_id) == before


def test_unbound_scalar_result_keeps_its_noninitial_evaluation_point():
    from pops.time.variant_types import Float64, Result, instantiate
    from pops.time._program.structured_control import new_typed

    p = Program("variant_scalar_evaluation")
    point = TimePoint(p.clock, step=1)
    scalar = new_typed(p, instantiate(Float64, point).signature, "literal", (),
                       {"value": 2.0}, point=point)
    result = p.variant(Result(Float64, Int64), "ok", scalar)
    assert result.point == TimePoint(p.clock)
    members = dict(value_signature(result)["members"])
    assert members["ok"]["point"] == members["err"]["point"] == point
    assert p.require_success(result).point == point
    with pytest.raises(TypeError, match="exact structural type"):
        p.variant(Result(type_of(p.float64(0)), Int64), "ok", scalar)


def test_nested_scalar_result_infers_leaf_point_instead_of_envelope():
    from pops.time.variant_types import Float64, Result, instantiate
    from pops.time._program.structured_control import new_typed

    p = Program("nested_scalar_parameter")
    point = TimePoint(p.clock, step=1)
    scalar = new_typed(p, instantiate(Float64, point).signature, "literal", (),
                       {"value": 2.0}, point=point)
    schema = Result(Float64, Int64)
    nested = p.variant(schema, "ok", scalar)
    outer = p.variant(Variant(nested=schema, missing=Int64), "nested", nested)
    actual = dict(value_signature(outer)["members"])["nested"]
    assert outer.point == nested.point == TimePoint(p.clock)
    assert dict(actual["members"])["ok"]["point"] == point
    incompatible = p.product(old=p.float64(0), new=scalar)
    before = p._ir_hash(), p._next_id
    with pytest.raises(TypeError, match="incompatible unbound scalar"):
        p.variant(Variant(row=Product(old=Float64, new=Float64)), "row", incompatible)
    assert (p._ir_hash(), p._next_id) == before
