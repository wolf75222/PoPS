"""Typed nested regions through the real Program IR, code generator and PoPS context calls."""
from __future__ import annotations

import pytest

from pops.time import Program, TimePoint
from pops.codegen.program_codegen import emit_cpp_program
from pops.time._program.detach import detach_compiled_program
from typed_program_support import typed_state


def _fixture(name="structured"):
    program = Program(name)
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")
    # Declare state inputs in the enclosing graph before any closed control region reads them.
    left.n
    right.n
    return program, left, right


def _commit(program, temporal, value):
    endpoint = temporal.next
    program.commit(endpoint, program.value("accepted", value, at=endpoint.point))


def test_nested_product_fold_preserves_two_blocks_and_simultaneous_swaps():
    p, a, b = _fixture()
    initial = p.product(a=a.n, b=b.n, meta=p.product(k=p.int64(0), norm=p.norm2(a.n)))

    def outer(t, x, i):
        def inner(q, y, j):
            return q.product(a=y["b"], b=y["a"], meta=q.product(k=y["meta"]["k"] + 1,
                                                                 norm=y["meta"]["norm"]))
        # Cross-owner swaps are not type preserving; values must retain their declared authority.
        return t.fold(x, 2, inner)

    with pytest.raises(TypeError, match="preserve every carried"):
        p.fold(initial, 3, outer)

    def valid_outer(t, x, i):
        return t.fold(x, 2, lambda q, y, j: q.product(
            a=q.value("a_update", 2 * y["a"]), b=q.value("b_update", 3 * y["b"]),
            meta=q.product(k=y["meta"]["k"] + 1, norm=y["meta"]["norm"])))

    result = p.fold(initial, 3, valid_outer)
    _commit(p, a, result["a"])
    _commit(p, b, result["b"])
    assert p.validate()
    source = emit_cpp_program(p)
    assert source.count("for (; iteration_") == 2
    assert "ctx.scratch_state" in source and "ctx.copy_with_provenance" in source
    assert "std::int64_t" in source
    assert p.to_graph().to_data() == detach_compiled_program(p).to_graph().to_data()


def test_swap_same_authority_stages_uses_two_phase_copy():
    p, a, _ = _fixture("swap")
    initial = p.product(x=a.n, y=p.value("twice", 2 * a.n))
    result = p.fold(initial, 1, lambda t, x, i: t.product(x=x["y"], y=x["x"]))
    _commit(p, a, result["x"])
    source = emit_cpp_program(p)
    line = source.index("for (; iteration_")
    tail = source[line:]
    assert tail.index("copy_with_provenance(next_") < tail.index("copy_with_provenance(carry_")
    assert tail.count("copy_with_provenance(next_") == 2
    assert "structured control field allocation" in source
    assert "structured control field copy" in source
    assert source.index("structured control field allocation") < source.index("auto& carry_")


def test_iterate_requires_guarded_projection_and_preserves_budget_in_graph():
    p, a, _ = _fixture("iterate")
    initial = p.product(u=a.n, k=p.int64(0))
    result = p.iterate(initial,
                       lambda t, x: t.product(u=t.value("next_u", 2*x["u"]), k=x["k"]+1),
                       lambda t, x: x["k"] >= 3, max_steps=4)
    with pytest.raises(TypeError, match="opaque"):
        result["state"]
    success = p.require_success(result)
    _commit(p, a, success["state"]["u"])
    source = emit_cpp_program(p)
    assert "iteration failed; successful state unavailable" in source
    assert '"; origin=" + std::to_string(' in source
    assert '"; code=" + std::to_string(' in source
    assert "iteration_stop_" in source
    assert " = 1; break;" in source
    assert p.to_graph().to_data()


def test_match_does_not_expose_success_in_error_arm_or_after_region():
    p, a, _ = _fixture("match")
    result = p.iterate(a.n, lambda t, x: t.value("next", 2*x),
                       lambda t, x: t.boolean(False), max_steps=0)
    escaped = []

    def ok(t, value):
        escaped.append(value["state"])
        return value["state"]

    def err(t, failure):
        with pytest.raises(KeyError):
            failure["state"]
        return a.n

    output = p.match(result, ok=ok, err=err)
    with pytest.raises(ValueError, match="region|escape|sub-block"):
        p.product(illegal=escaped[0])
    _commit(p, a, output)
    source = emit_cpp_program(p)
    assert "branch_choice_" in source
    assert "matched_" in source
    assert p.to_graph().to_data()


def test_dynamic_int64_count_is_exact_and_requires_a_capacity():
    p, a, _ = _fixture("dynamic-count")
    count = p.int64((1 << 53) + 1)
    with pytest.raises(ValueError, match="max_steps"):
        p.fold(a.n, count, lambda t, x, i: x)
    result = p.fold(a.n, count, lambda t, x, i: x, max_steps=(1 << 53) + 1)
    _commit(p, a, result)
    source = emit_cpp_program(p)
    assert "std::int64_t(9007199254740993)" in source
    assert "9007199254740992" not in source
    assert "fold count outside declared capacity" in source
    assert "std::numeric_limits<long>::digits >= std::numeric_limits<std::int64_t>::digits" in source
    assert "all_reduce_min(static_cast<long>(" in source
    assert p.to_graph().to_data()


def test_exact_points_are_preserved_by_product_projection():
    p, a, _ = _fixture("points")
    later = p.value("later", a.n, at=TimePoint(p.clock, step=1))
    product = p.product(old=a.n, new=later)
    assert product["old"].point == a.n.point
    assert product["new"].point == later.point
    assert product["new"].state_ref == later.state_ref
    _commit(p, a, product["new"])
    detached = detach_compiled_program(p)
    assert detached._ir_hash() == p._ir_hash()


def test_uncertified_failure_effect_is_refused_before_codegen():
    p, a, _ = _fixture("effect-profile")
    result = p.iterate(a.n, lambda t, x: t.fill_boundary(x),
                       lambda t, x: t.boolean(False), max_steps=1)
    _commit(p, a, p.require_success(result)["state"])
    with pytest.raises(NotImplementedError, match="lane-uniform failure contract"):
        emit_cpp_program(p)


def test_integer_range_overflow_division_and_branch_type_are_explicit():
    p, a, _ = _fixture("integers")
    with pytest.raises(ValueError, match="out of range"):
        p.int64(1 << 63)
    x = p.int64(-(1 << 63))
    q = p.branch(p.boolean(True), lambda t: x // -1, lambda t: t.int64(0))
    assert q.vtype == "int64"
    _commit(p, a, a.n)
    source = emit_cpp_program(p)
    assert "Int64 arithmetic overflow or invalid division" in source
    assert "std::numeric_limits<std::int64_t>::min()" in source


def test_general_result_capture_branch_has_closed_graph_and_exact_failure_code():
    p, a, _ = _fixture("general-result")
    calls = []
    escaped = []

    def body(t):
        calls.append("once")
        trial = t.branch(t.boolean(False), lambda q: q.success(a.n),
                         lambda q: q.failure(a.n, (1 << 40) + 17))
        return t.require_success(trial)

    result = p.capture(body)
    assert calls == ["once"]
    with pytest.raises(TypeError, match="opaque"):
        result["value"]
    with pytest.raises(ValueError, match="nonzero"):
        p.failure(a.n, 0)

    def ok(t, x):
        escaped.append(x)
        return x

    def err(t, error):
        with pytest.raises(KeyError):
            error["value"]
        assert error["origin"].vtype == "int64"
        return t.branch(error["code"] == (1 << 40) + 17,
                        lambda q: a.n, lambda q: q.value("wrong_code", 2*a.n))

    recovered = p.match(result, ok=ok, err=err)
    with pytest.raises(ValueError, match="region|escape|sub-block"):
        p.product(escaped=escaped[0])
    _commit(p, a, recovered)
    source = emit_cpp_program(p)
    assert "std::int64_t(1099511627793)" in source
    assert "failure.code;" in source
    assert "throw structured_failure_" in source
    graph = p.to_graph().to_data()
    capture = next(row for row in graph["nodes"] if row["kind"] == "capture")
    assert capture["body"]["captures"] and capture["body"]["nodes"]
    assert graph == detach_compiled_program(p).to_graph().to_data()


def test_iteration_step_result_and_nested_capture_propagate_without_failed_update():
    p, a, _ = _fixture("step-result")
    initial = p.product(u=a.n, k=p.int64(0))

    def step(t, x):
        return t.branch(x["k"] < 1,
                        lambda q: q.success(q.product(u=q.value("next", 2*x["u"]), k=x["k"]+1)),
                        lambda q: q.failure(x, 71))

    outer = p.capture(lambda t: t.require_success(t.capture(
        lambda q: q.require_success(q.iterate(initial, step, lambda s, x: x["k"] >= 3,
                                             max_steps=4))["state"])))
    recovered = p.match(outer, ok=lambda t, x: x["u"], err=lambda t, e: a.n)
    _commit(p, a, recovered)
    source = emit_cpp_program(p)
    import re
    status = re.search(r"std::int64_t (matched_\w+_status)", source)
    assert status is not None
    failure = source.index("break; }", status.start())
    copy = source.index("copy_with_provenance(next_", failure)
    assert failure < copy
    assert source.count("= failure.code;") == 3
    assert source.count("= failure.origin;") == 3
    assert p.to_graph().to_data()


def test_result_branch_bad_signature_is_atomic_and_capture_effect_is_checked():
    p, a, b = _fixture("result-types")
    flag = p.boolean(True)
    old = p._ir_hash()
    with pytest.raises(ValueError, match="structural"):
        p.branch(flag, lambda t: t.success(a.n), lambda t: t.failure(b.n, 5))
    assert p._ir_hash() == old
    result = p.capture(lambda t: t.fill_boundary(a.n))
    _commit(p, a, p.require_success(result))
    with pytest.raises(NotImplementedError, match="lane-uniform failure contract"):
        emit_cpp_program(p)


def test_empty_product_is_unit_and_capture_can_return_an_outer_value():
    p, a, _ = _fixture("unit-result")
    unit = p.product()
    result = p.capture(lambda t: unit)
    p.require_success(result)
    _commit(p, a, a.n)
    assert emit_cpp_program(p)
    assert p.to_graph().to_data()


def test_user_failure_one_and_runtime_budget_one_have_distinct_discriminants():
    p, a, _ = _fixture("failure-discriminant")
    user = p.capture(lambda t: t.require_success(t.failure(a.n, 1)))
    budget = p.capture(lambda t: t.require_success(t.iterate(
        a.n, lambda q, x: x, lambda q, x: q.boolean(False), max_steps=0))["state"])
    user_origin = p.match(user, ok=lambda t, x: t.int64(0), err=lambda t, e: e["origin"])
    budget_origin = p.match(budget, ok=lambda t, x: t.int64(0), err=lambda t, e: e["origin"])
    _commit(p, a, p.branch(user_origin == budget_origin,
                         lambda t: t.value("wrong_origin", 2*a.n), lambda t: a.n))
    source = emit_cpp_program(p)
    assert "std::int64_t(1)" in source
    assert "iteration_origin_" in source
    assert "failure.origin;" in source
    assert p.to_graph().to_data()
