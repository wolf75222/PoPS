"""Closed user sums through actual Program authoring/serialization/C++ lowering (no runtime mock)."""
import json

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.time import Program
from pops.time._program.detach import detach_compiled_program
from pops.time.variant_types import (Bool, Failure, Float64, Int64, Product, Result,
                                     RuntimeFailure, ValueType, Variant, type_of)
from typed_program_support import typed_state


def fixture(name="variants"):
    p = Program(name)
    u = typed_state(p, "u", state_name="U")
    u.n
    return p, u


def commit(p, u, value):
    p.commit(u.next, p.value("accepted", value, at=u.next.point))


def test_type_roundtrip_and_exact_product_payload():
    schema = Variant(advance=Product(dt=Float64, count=Int64), stop=Product(), retry=Bool)
    assert ValueType.from_data(json.loads(json.dumps(schema.to_data()))) == schema
    with pytest.raises(TypeError):
        schema.signature["vtype"] = "scalar"
    p, u = fixture()
    assert type_of(u.n).matches(u.n)
    with pytest.raises(ValueError, match="authenticated"):
        ValueType.from_data(type_of(u.n).to_data())
    with pytest.raises(ValueError, match="at least one"):
        Variant()
    with pytest.raises(TypeError, match="exact structural"):
        p.variant(Variant(integer=Int64), "integer", p.float64(1))


def test_three_tags_lazy_inactive_singularity_and_exact_large_integer():
    p, u = fixture("three-tags")
    errors = Variant(stop=Product(), retry=Product(code=Int64, residual=Float64), success=Int64)
    result = p.variant(errors, "retry", p.product(code=p.int64((1 << 60)+17), residual=p.float64(.125)))
    calls = []
    def arm(t, payload):
        calls.append("retry")
        return t.branch(payload["code"] == (1 << 60)+17, lambda q: u.n,
                        lambda q: q.value("wrong", 2*u.n))
    output = p.match(result, retry=arm, stop=lambda t, e: t.value("stopped", 3*u.n),
                     success=lambda t, n: t.value("singular", (t.float64(1)/t.float64(0))*u.n))
    assert calls == ["retry"]
    commit(p, u, output)
    source = emit_cpp_program(p)
    assert "std::int64_t(1152921504606846993)" in source
    assert "structured branch differs across execution lane" in source
    assert source.index("branch_choice_") < source.index("singular") if "singular" in source else True
    assert p._ir_hash() == detach_compiled_program(p)._ir_hash()
    assert p.to_graph().to_data() == detach_compiled_program(p).to_graph().to_data()


def test_exhaustivity_type_errors_and_region_escape_are_atomic():
    p, u = fixture("variant-errors")
    schema = Variant(ready=Int64, missing=Product())
    value = p.variant(schema, "ready", p.int64(4))
    before = p._ir_hash()
    with pytest.raises(TypeError, match="exactly every"):
        p.match(value, ready=lambda t, x: x)
    assert p._ir_hash() == before
    with pytest.raises(TypeError, match="same structural"):
        p.match(value, ready=lambda t, x: x, missing=lambda t, x: t.float64(0))
    assert p._ir_hash() == before
    escaped = []
    def ready(t, x):
        escaped.append(x)
        return u.n
    selected = p.match(value, ready=ready, missing=lambda t, x: u.n)
    with pytest.raises(ValueError, match="region|escape|sub-block"):
        p.product(escaped=escaped[0])
    with pytest.raises(TypeError, match="opaque"):
        value["ready"]
    commit(p, u, selected)
    assert emit_cpp_program(p)


def test_nested_typed_capture_preserves_complete_error_and_native_runtime_arm():
    p, u = fixture("typed-error")
    Error = Failure(retry=Product(code=Int64, residual=Float64, checkpoint=type_of(u.n)))
    Outcome = Result(type_of(u.n), Error)
    def body(t):
        payload = t.product(code=t.int64(-(1 << 60)+5), residual=t.float64(.25), checkpoint=u.n)
        return t.require_success(t.variant(Outcome, "err", t.variant(Error, "retry", payload)))
    inner = p.capture(body)
    outer = p.capture(lambda t: t.require_success(inner))
    assert Outcome.matches(outer)
    def error(t, e):
        return t.match(e,
                       retry=lambda q, x: q.branch(x["code"] == -(1 << 60)+5,
                                                  lambda r: x["checkpoint"], lambda r: r.value("wrong", 2*u.n)),
                       runtime=lambda q, x: q.value("runtime", 3*u.n))
    commit(p, u, p.match(outer, ok=lambda t, x: x, err=error))
    source = emit_cpp_program(p)
    assert "typed_capture_done_" in source and "goto typed_capture_done_" in source
    assert "std::int64_t(-1152921504606846971)" in source
    assert "failure.code" in source and "failure.origin" in source
    assert p._ir_hash() == detach_compiled_program(p)._ir_hash()
    assert p.to_graph().to_data() == detach_compiled_program(p).to_graph().to_data()


def test_conflicting_error_tag_cannot_erase_payload():
    p, u = fixture("error-union")
    First = Result(type_of(u.n), Variant(user=Int64))
    Second = Result(type_of(u.n), Variant(user=Float64))
    def body(t):
        t.require_success(t.variant(First, "err", t.variant(Variant(user=Int64), "user", t.int64(1))))
        return t.require_success(t.variant(Second, "err", t.variant(Variant(user=Float64), "user", t.float64(1))))
    before = p._ir_hash()
    with pytest.raises(TypeError, match="incompatible payload"):
        p.capture(body)
    assert p._ir_hash() == before


def test_product_and_fold_may_carry_variant_with_field_payload():
    p, u = fixture("variant-carry")
    Schema = Variant(old=type_of(u.n), new=Product(field=type_of(u.n), count=Int64))
    initial = p.product(choice=p.variant(Schema, "old", u.n), count=p.int64(0))
    def body(t, carried, i):
        new = t.match(carried["choice"],
                      old=lambda q, x: q.variant(Schema, "new", q.product(field=q.value("twice", 2*x), count=i)),
                      new=lambda q, x: q.variant(Schema, "old", x["field"]))
        return t.product(choice=new, count=i+1)
    final = p.fold(initial, 3, body)
    output = p.match(final["choice"], old=lambda t, x: x, new=lambda t, x: x["field"])
    commit(p, u, output)
    source = emit_cpp_program(p)
    assert "switch (" in source and "ctx.scratch_state" in source
    assert "structured control field copy" in source
    assert p.to_graph().to_data() == detach_compiled_program(p).to_graph().to_data()


def test_iteration_propagates_typed_payload_before_copying_failed_state():
    p, u = fixture("variant-iterate")
    initial = p.product(field=u.n, count=p.int64(0))
    Errors = Failure(retry=Product(attempt=Int64, defect=Float64))
    Step = Result(type_of(initial), Errors)
    def step(t, x):
        return t.branch(x["count"] < 1,
                        lambda q: q.variant(Step, "ok", q.product(field=q.value("advance", 2*x["field"]), count=x["count"]+1)),
                        lambda q: q.variant(Step, "err", q.variant(Errors, "retry", q.product(attempt=x["count"], defect=q.float64(.75)))))
    result = p.iterate(initial, step, lambda t, x: x["count"] >= 3, max_steps=4)
    assert result.vtype == "variant"
    recovered = p.match(result, ok=lambda t, x: x["state"]["field"],
                        err=lambda t, e: t.match(e, retry=lambda q, x: u.n,
                                                runtime=lambda q, x: q.value("runtime", 3*u.n)))
    commit(p, u, recovered)
    source = emit_cpp_program(p)
    jump = source.index("goto typed_iteration_done_")
    assert jump < source.index("copy(next_", jump)
    assert p._ir_hash() == detach_compiled_program(p)._ir_hash()
    assert p.to_graph().to_data() == detach_compiled_program(p).to_graph().to_data()


def test_lowering_rejects_a_forged_unguarded_variant_projection():
    from pops.time._program.structured_control import new_typed
    p, u = fixture("forged-projection")
    value = p.variant(Variant(ok=type_of(u.n), rejected=Product()), "rejected", p.product())
    forged = new_typed(p, type_of(u.n).signature, "variant_payload", (value,), {"tag": "ok"})
    commit(p, u, forged)
    with pytest.raises(ValueError, match="exact guarded match"):
        emit_cpp_program(p)


@pytest.mark.parametrize("kind", ("inactive_singular", "full_payload", "sum_field_loop",
                                  "iteration_payload", "divergent_choice", "array_variants"))
def test_native_reception_builders_resolve_and_emit_only(kind):
    import pops
    from examples.api_v02.variant_checks import build_variant_case
    from pops.codegen.program_models import ProgramModelGraph
    spec = build_variant_case(kind)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert plan.resolved_dimension == 2
    source = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "Variant" in source
    assert "prepared_execution_lane" in source


def test_array_of_variants_snapshots_fields_and_preserves_exact_instance():
    p, u = fixture("array-of-variants")
    d = p.range_set(3, capacity=3, ownership="replicated")
    Choice = Variant(number=Int64, field=type_of(u.n))
    def item(t, i):
        n = t.index_value(i)
        return t.branch(n % 2 == 0,
                        lambda q: q.variant(Choice, "field", q.value("field_at_index", q.float64(n+1)*u.n)),
                        lambda q: q.variant(Choice, "number", n))
    values = p.map(d, item)
    projected = values[p.index(d, 0)]
    with pytest.raises(ValueError, match="different finite domain family"):
        values[p.index(p.range_set(3, capacity=3, ownership="replicated"), 0)]
    commit(p, u, p.match(projected, field=lambda t, x: x,
                        number=lambda t, x: t.value("numeric", t.float64(x)*u.n)))
    source = emit_cpp_program(p)
    assert "std::variant<" in source
    assert "_snapshot" in source and "ctx.scratch_state_like" in source
    assert "array_read_" in source
    assert "emplace<" in source and "std::get<" in source
    assert p._ir_hash() == detach_compiled_program(p)._ir_hash()


@pytest.mark.parametrize("operation", ("map", "fold", "scan"))
def test_typed_failure_in_finite_region_exits_before_publication(operation):
    p, u = fixture("array-failure-"+operation)
    d = p.range_set(3, capacity=3, ownership="replicated")
    Errors = Failure(invalid=Product(index=Int64, defect=Float64))
    Outcome = Result(Int64, Errors)
    def checked(t, index):
        k = t.index_value(index)
        return t.require_success(t.branch(k == 1,
            lambda q: q.variant(Outcome, "err", q.variant(Errors, "invalid", q.product(index=k, defect=q.float64(.25)))),
            lambda q: q.variant(Outcome, "ok", k)))
    def attempt(t):
        if operation == "map":
            return t.map(d, checked)
        if operation == "scan":
            return t.scan(d, t.int64(0), lambda q, s, i:
                          q.product(state=s+checked(q,i), value=s), order="ascending")
        return t.fold(t.int64(0), 3, lambda q, s, i: s+checked(q,q.index(d,i)))
    result = p.capture(attempt)
    assert result.vtype == "variant"
    commit(p, u, p.match(result, ok=lambda t, x: t.value("unexpected", 0*u.n),
                        err=lambda t, e: t.match(e, invalid=lambda q,x:u.n,
                                                runtime=lambda q,x:q.value("wrong", 2*u.n))))
    source = emit_cpp_program(p)
    if operation != "fold":
        assert source.index("goto typed_capture_done_") < source.index("_array{", source.index("goto typed_capture_done_"))
    assert p._ir_hash() == detach_compiled_program(p)._ir_hash()


def test_nested_arrays_and_variants_use_distinct_native_loop_indices():
    p, u = fixture("nested-arrays")
    d = p.range_set(2, capacity=2, ownership="replicated")
    inner = p.map(d, lambda t,i:t.index_value(i))
    Choice = Variant(values=type_of(inner), empty=Product())
    middle = p.map(d, lambda t,i:t.variant(Choice, "values", inner))
    outer = p.map(d, lambda t,i:middle)
    p.at(outer,p.index(d,0))
    commit(p,u,u.n)
    source = emit_cpp_program(p)
    assert "element_i_0" in source and "element_i_1" in source
    assert p.to_graph().to_data() == detach_compiled_program(p).to_graph().to_data()
