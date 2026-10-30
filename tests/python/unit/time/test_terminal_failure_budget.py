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
