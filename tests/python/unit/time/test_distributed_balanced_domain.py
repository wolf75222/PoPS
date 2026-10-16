"""Placement-independent authoring and emission; native tests cover MPI execution."""

import pytest

from pops.codegen.program_emit_distributed_arrays import emit_distributed_op
from pops.time import Program
from pops.time._program.detach import detach_compiled_program
from pops.time._program.distributed_arrays import Distributed
from pops.time.variant_types import Int64, Result


def test_balanced_domain_keeps_size_ssa_and_defers_rank_count():
    program = Program("balanced")
    distributed = Distributed(program)
    size = program.int64(17)
    domain = distributed.balanced_domain(size, capacity=32)
    assert domain.inputs == (size,)
    assert "rank_count" not in domain.attrs
    lines = []
    emit_distributed_op(
        program, domain, {size.id: "runtime_size"}, None, lines, [], {}, None, "system"
    )
    source = "\n".join(lines)
    assert ".issue_balanced(0,32,runtime_size)" in source
    assert "MPI_COMM_WORLD" not in source
    assert "_offsets={" not in source
    assert program._ir_hash() == detach_compiled_program(program)._ir_hash()


def test_balanced_family_preserves_type_but_issues_fresh_domain():
    program = Program("balanced_family")
    distributed = Distributed(program)
    first = distributed.balanced_domain(0, capacity=(1 << 63) - 1)
    second = distributed.balanced_domain(3, family=first)
    assert first is not second
    for name in ("capacity", "family", "ownership"):
        assert first.attrs["signature"][name] == second.attrs["signature"][name]
    assert second.inputs[-1] is first
    before = program._ir_hash()
    with pytest.raises(TypeError, match="exact capacity"):
        distributed.balanced_domain(1, capacity=4, family=first)
    assert program._ir_hash() == before


@pytest.mark.parametrize("invalid", [True, 1.25, 1 << 63])
def test_invalid_size_is_rejected_without_partial_graph(invalid):
    program = Program("bad_size")
    distributed = Distributed(program)
    before = program._ir_hash()
    with pytest.raises((TypeError, ValueError, OverflowError)):
        distributed.balanced_domain(invalid, capacity=4)
    assert program._ir_hash() == before


def test_balanced_size_requires_owned_int64_and_collective_region():
    program = Program("collective")
    distributed = Distributed(program)
    domain = distributed.balanced_domain(2, capacity=2)
    before = program._ir_hash()
    with pytest.raises((TypeError, ValueError)):
        distributed.balanced_domain(Program("foreign").int64(1), capacity=2)
    assert program._ir_hash() == before

    def body(builder, index):
        distributed.balanced_domain(1, capacity=2)
        return builder.variant(Result(Int64, Int64), "ok", distributed.gid(index))

    with pytest.raises(TypeError, match="collective or unsupported"):
        distributed.local_map(domain, body, byte_capacity=4096)
    assert program._ir_hash() == before


def test_rank_environment_does_not_enter_equations_or_identity(monkeypatch):
    def build():
        program = Program("same_program")
        Distributed(program).balanced_domain(17, capacity=32)
        return program._ir_hash()

    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
    first = build()
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "3")
    assert build() == first


def test_balanced_domain_survives_full_program_codegen():
    from typed_program_support import typed_state
    from pops.codegen.program_codegen import emit_cpp_program

    program = Program("balanced_full_program")
    state = typed_state(program, "material", state_name="U")
    distributed = Distributed(program)
    domain = distributed.balanced_domain(3, capacity=10)
    result = distributed.local_map(
        domain,
        lambda builder, index: builder.variant(
            Result(Int64, Int64), "ok", distributed.gid(index) + 17
        ),
        byte_capacity=4096,
    )
    program.require_success(result)
    program.commit(state.next, program.value("accepted", state.n, at=state.next.point))
    assert program.validate()
    source = emit_cpp_program(program)
    assert ".issue_balanced(0,10," in source
    assert "pops_program_hash" in source
