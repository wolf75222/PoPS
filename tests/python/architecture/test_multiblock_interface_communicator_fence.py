"""ADC-683 fences for execution-lane-owned multi-block interface collectives."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = ROOT / "include/pops/runtime/multiblock/interface_flux_scheduler.hpp"


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


def test_interface_scheduler_hot_path_never_falls_back_to_mpi_world():
    source = SCHEDULER.read_text(encoding="utf-8")
    consensus = _function(source, "static void require_distributed_flux_consensus_(")
    apply_one = _function(source, "static void apply_one_(")

    assert "MPI_COMM_WORLD" not in consensus
    assert "MPI_COMM_WORLD" not in apply_one
    assert "const CommunicatorView& communicator" in consensus
    assert "prepared.communicator" in apply_one
