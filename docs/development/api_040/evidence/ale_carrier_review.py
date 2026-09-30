"""Independent structural reception of a pinned native ALE carrier revision.

This reads source only. It never claims System, MPI, or public descriptor execution.
"""
import argparse
from pathlib import Path
import subprocess


def receive(revision):
    root = Path(__file__).resolve().parents[4]

    def source(path):
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"], cwd=root, text=True)

    provider = source("include/pops/runtime/program/program_context_moving_interval.inc")
    prepared = provider.index("validate_program_state_publication_candidate_")
    ledger = provider.index("stage_exchange_batch")
    geometry = provider.index("std::swap(runtime_state().moving_interval_geometry_")
    state = provider.index("std::swap(state(program_block)")
    assert prepared < ledger < geometry < state
    assert provider.index("pointwise_active_mask") < provider.index("Real invalid = 0")
    assert "embedded-boundary moving measures are not prepared" in provider
    assert "moving numeric preparation failed collectively" in provider
    assert provider.index("Kokkos::fence();", provider.index("Real invalid = 0")) < prepared
    contract = provider[provider.index("pops.moving-interval-publication.v2"):
                        provider.index("moving publication contract failed collectively")]
    for authority in ("lane.identity()", "rank_space().origin()", "rank_space().extent()",
                      "geometry().domain().lo", "geometry().domain().hi", "is_periodic",
                      "input.layout()[global].lo", "input.layout()[global].hi",
                      "input.distribution().owner(global)", "owner >="):
        assert authority in contract, authority
    runtime = source("include/pops/runtime/program/program_runtime_state.hpp")
    assert "moving_interval_geometry_(accepted.moving_interval_geometry_)" in runtime
    assert "moving_interval_geometry_.swap(prepared.moving_interval_geometry_)" in runtime
    fab = source("include/pops/mesh/storage/fab.hpp")
    assert "Kokkos::deep_copy(data_, other.data_)" in fab
    snapshot = source("src/runtime/system/system_impl.hpp")
    assert "program(owner.program_)" in snapshot
    assert "owner.program_.prepare_accepted_restore(program)" in snapshot
    print(f"{revision}: structural carrier/recovery/collective authority checks pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="ed8b4e0")
    receive(parser.parse_args().revision)
