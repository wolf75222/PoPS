"""ADC-683 fences for execution-owned System layout-transfer collectives."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/runtime/system/system_layout_transfer.cpp"


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    opening_brace = source.index("{", start)
    depth = 0
    for offset in range(opening_brace, len(source)):
        token = source[offset]
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"unterminated C++ function {signature}")


def test_layout_transfer_accepts_a_live_world_congruent_execution_context():
    source = SOURCE.read_text(encoding="utf-8")
    resolver = _function(source, "CommunicatorView resolve_execution_communicator(")

    assert "MPI_COMM_WORLD" not in source
    assert "MPI_Comm_f2c" in resolver
    assert "MPI_Comm_compare(communicator, field_rank_space.native_handle()" in resolver
    assert "relation != MPI_IDENT && relation != MPI_CONGRUENT" in resolver
    assert "POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1" in resolver
    assert "return CommunicatorView{communicator};" in resolver


def test_layout_transfer_retains_the_resolved_context_for_every_hot_collective():
    source = SOURCE.read_text(encoding="utf-8")
    implementation = source.split("struct PreparedSystemLayoutTransfer::Impl", maxsplit=1)[1]
    hot_path = source.split("void PreparedSystemLayoutTransfer::begin_transaction", maxsplit=1)[1]

    assert source.count("world_communicator_view()") == 1
    assert "CommunicatorView communicator;" in implementation
    assert "CommunicatorView world;" not in implementation
    assert "p_->world" not in hot_path
    assert "world_communicator_view()" not in hot_path
    assert "parallel_copy(p_->source_snapshot, p_->source_state(), p_->communicator)" in hot_path
    assert "collective_elements(local_source_elements, p_->communicator)" in hot_path
    assert "collective_elements(local_target_elements, p_->communicator)" in hot_path
