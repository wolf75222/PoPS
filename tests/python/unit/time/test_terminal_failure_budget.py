"""Typed terminal errors have explicit resources and preserve structural depth."""
from __future__ import annotations

import pytest

from pops.codegen.program_emit_typed_failure import emit_terminal_failure
from pops.identity.semantic import program_semantic_data
from pops.time import Program
from pops.time._program.detach import detach_compiled_program


@pytest.mark.parametrize("field", ("terminal_failure_byte_capacity",
                                   "terminal_failure_aggregate_byte_capacity"))
@pytest.mark.parametrize("bad", (0, -1, True, 0.5, 1 << 63))
def test_terminal_resource_limits_are_positive_exact_int64(field, bad):
    with pytest.raises(ValueError, match=field):
        Program("failure", **{field: bad})


def test_terminal_limits_are_part_of_identity_and_survive_detach():
    p = Program("failure", terminal_failure_byte_capacity=8192,
                terminal_failure_aggregate_byte_capacity=32768)
    q = detach_compiled_program(p)
    for data in (p._serialize(), q._serialize(), program_semantic_data(p)):
        assert data["terminal_failure_byte_capacity"] == 8192
        assert data["terminal_failure_aggregate_byte_capacity"] == 32768
    assert q._ir_hash() == p._ir_hash()
    p.terminal_failure_aggregate_byte_capacity += 1
    assert p._ir_hash() != q._ir_hash()
    p.terminal_failure_byte_capacity = True
    with pytest.raises(ValueError, match="terminal_failure_byte_capacity"):
        p._serialize()


def test_terminal_codegen_uses_the_same_explicit_capacity_for_pack_and_transport():
    p = Program("failure", terminal_failure_byte_capacity=104857600,
                terminal_failure_aggregate_byte_capacity=536870912)
    lines = []
    emit_terminal_failure(p, {"vtype": "int64"}, "INT64_C(42)", lines, {}, "system")
    source = "\n".join(lines)
    assert ("Limits{" + ",".join(["UINT64_C(104857600)"] * 3) + ",&pops_finite_residency}"
            in source)
    assert "TypedFailureCapacity{UINT64_C(104857600),UINT64_C(536870912)}" in source
    assert "67108864" not in source and "268435456" not in source


def test_deep_terminal_type_is_not_limited_by_python_or_cpp_recursion():
    signature, tokens = {"vtype": "int64"}, "INT64_C(42)"
    for _ in range(1500):
        signature = {"vtype": "product", "members": (("detail", signature),)}
        tokens = {"detail": tokens}
    lines = []
    emit_terminal_failure(Program("deep"), signature, tokens, lines, {}, "system")
    source = "\n".join(lines)
    assert "pops.typed-failure@1" in source
    assert source.count("ContinuationKind::Product") == 3000
    assert "finite_structural::pack<" in source
    assert "std::tuple<" not in source


def test_terminal_failure_without_finite_operations_declares_the_finite_helpers():
    from typed_program_support import typed_state
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_emit_distributed_arrays import OPS as DISTRIBUTED_OPS
    from pops.codegen.program_emit_finite_arrays import FINITE_OPS
    from pops.codegen.program_emit_versioned import VERSIONED_OPS
    from pops.codegen.program_lowerability import all_ops
    from pops.time._program.field_representation import FIELD_REPRESENTATION_OPS
    from pops.time.variant_types import Failure, Int64, Product, Result

    p = Program("integer_failure")
    q = typed_state(p, "left", state_name="U")
    errors = Failure(retry=Product(code=Int64))
    p.require_success(p.variant(Result(Int64, errors), "err",
                                p.variant(errors, "retry", p.product(code=p.int64(7)))))
    p.commit(q.next, p.value("accepted", q.n, at=q.next.point))
    finite = FINITE_OPS | VERSIONED_OPS | FIELD_REPRESENTATION_OPS | DISTRIBUTED_OPS
    assert not finite & {value.op for value in all_ops(p)}
    source = emit_cpp_program(p)
    # The payload packs through finite structural storage, declared by the finite helpers.
    pack = source.index("finite_structural::pack<")
    assert source.index("namespace pops_algebra_finite {") < pack
    assert source.index("finite_device::Residency pops_finite_residency;") < pack


def test_every_finite_helper_declaration_selects_the_helpers():
    import pathlib
    import re
    from pops.codegen.program_emit_finite_arrays import finite_cpp_helpers, uses_finite_helpers

    helpers = finite_cpp_helpers()
    names = re.findall(r"^(?:std::atomic<\S+>|\S+::Residency) (\w+)", helpers, re.M)
    names += ["pops_algebra_finite::" + alias for alias in re.findall(r"^using (?:\S+::)?(\w+)", helpers, re.M)]
    # The namespaces of the finite headers that the helpers include, directly or not.
    include = pathlib.Path(__file__).resolve().parents[4] / "include"
    headers, seen = [helpers], set()
    while headers:
        for header in re.findall(r"#include <(pops/runtime/program/finite_\w+\.hpp)>", headers.pop()):
            if header not in seen:
                seen.add(header)
                headers.append((include / header).read_text())
                names += ["pops::runtime::program::%s::name" % namespace for namespace in
                          re.findall(r"^namespace pops::runtime::program::(\w+) \{", headers[-1], re.M)]
    assert {"pops_finite_residency", "pops_algebra_finite::make_values",
            "pops::runtime::program::finite_device::name"} <= set(names)
    assert [name for name in names if not uses_finite_helpers(name)] == []
    assert not uses_finite_helpers("pops::runtime::program::reject_typed_failure<2>(")
