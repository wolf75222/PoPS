"""Structural portability contract for GCC/NVCC exact-pair consensus calls.

The three affected one-pair calls use a named pair plus a one-element span.  This avoids the
ambiguous nested initializer-list syntax rejected by the ARM CUDA host compiler without altering
the pair ordering or collective route.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _region(source: str, start: str, end: str) -> str:
    start_offset = source.index(start)
    end_offset = source.index(end, start_offset)
    return source[start_offset:end_offset]


def _assert_one_pair_span(
    source: str,
    *,
    start: str,
    end: str,
    pair: str,
    pairs: str,
    label: str,
    payload: str,
    communicator: str,
) -> None:
    region = _region(source, start, end)
    normalized = " ".join(region.split())

    assert "{{std::string_view" not in region
    assert f"const ExactOrderedBytePair {pair}" in region
    assert label in region
    assert payload in region
    assert f"const std::span<const ExactOrderedBytePair> {pairs}" in region
    assert f"&{pair}, 1" in region
    assert f"all_ranks_agree_exact_ordered_byte_pairs({pairs}, {communicator})" in normalized


def _assert_fixed_pair_span(
    source: str,
    *,
    start: str,
    end: str,
    storage: str,
    size: int,
    span: str,
    ordered_values: tuple[str, ...],
    communicator: str,
) -> None:
    region = _region(source, start, end)
    normalized = " ".join(region.split())

    assert "{{std::string_view" not in region
    assert f"const std::array<ExactOrderedBytePair, {size}> {storage}" in region
    offsets = [normalized.index(value) for value in ordered_values]
    assert offsets == sorted(offsets)
    assert f"const std::span<const ExactOrderedBytePair> {span}" in region
    assert f"all_ranks_agree_exact_ordered_byte_pairs({span}, {communicator})" in normalized


def test_cuda_host_parser_portable_exact_one_pair_consensus_calls() -> None:
    ghost_fill = (ROOT / "include/pops/runtime/amr/prepared_amr_ghost_fill.hpp").read_text()
    tagging = (ROOT / "include/pops/runtime/amr/prepared_tagging_execution.hpp").read_text()
    interface = (ROOT / "include/pops/runtime/multiblock/interface_flux_scheduler.hpp").read_text()
    partitioned_fac = (ROOT / "include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp").read_text()

    _assert_one_pair_span(
        ghost_fill,
        start="PreparedAmrGhostFill<Dim, MemorySpace> prepare_amr_ghost_fill(",
        end="  state->remote_parent_collective =",
        pair="ghost_fill_contract_pair",
        pairs="ghost_fill_contract_pairs",
        label='std::string_view("pops-prepared-amr-ghost-fill")',
        payload="std::string_view(state->exact_contract)",
        communicator="lane.communicator()",
    )
    _assert_one_pair_span(
        tagging,
        start="  static PreparedTaggingExecutionPlan prepare(",
        end="    candidate->prepared_ = true;",
        pair="tagging_collective_pair",
        pairs="tagging_collective_pairs",
        label='std::string_view("prepared-tagging")',
        payload="std::string_view(candidate->collective_contract_)",
        communicator="communicator",
    )
    _assert_one_pair_span(
        interface,
        start="    const std::string collective_identity = collective_plan_identity_(",
        end="    PreparedInterface prepared;",
        pair="route_collective_pair",
        pairs="route_collective_pairs",
        label="std::string_view(route.identity)",
        payload="std::string_view(collective_identity)",
        communicator="execution_communicator",
    )
    _assert_fixed_pair_span(
        partitioned_fac,
        start="  CompositeFacPoisson(request_type request, CompositeFacOptions options = {},",
        end="    lane_.emplace(ExecutionLane::duplicate_world_collectively(lane_identity_));",
        storage="fac_contract_pairs",
        size=1,
        span="fac_contract_pair_span",
        ordered_values=(
            'std::string_view("pops-partitioned-composite-fac")',
            "std::string_view(exact_contract_)",
        ),
        communicator="world_communicator_view()",
    )
    _assert_fixed_pair_span(
        partitioned_fac,
        start="    for (std::size_t connection = 0; connection < connections_.size(); ++connection) {",
        end="    }\n\n    lane_.emplace(ExecutionLane::duplicate_world_collectively(lane_identity_));",
        storage="connection_contract_pairs",
        size=3,
        span="connection_contract_pair_span",
        ordered_values=(
            'std::string_view("pops-fac-parent-gather")',
            "std::string_view(connections_[connection]->gather_contract)",
            'std::string_view("pops-fac-fine-restriction")',
            "std::string_view(connections_[connection]->restriction_contract)",
            'std::string_view("pops-fac-flux-mismatch")',
            "std::string_view(connections_[connection]->flux_contract)",
        ),
        communicator="world_communicator_view()",
    )
