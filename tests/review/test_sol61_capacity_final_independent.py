"""Independent exact consensus lifetime/source-contract check, no Native."""
from pathlib import Path


def test_named_consensus_views_are_live_and_allocation_is_in_voted_phase():
    root=Path(__file__).resolve().parents[2]
    comm=(root/'include/pops/parallel/comm.hpp').read_text()
    assert 'using ExactOrderedBytePair = std::pair<std::string_view, std::string_view>;' in comm
    source=(root/'src/runtime/amr/amr_system.cpp').read_text()
    start=source.index('std::uint64_t AmrSystem<Dim>::checkpoint_state_carriers_byte_capacity() const')
    end=source.index('void AmrSystem<Dim>::validate_checkpoint_state_carriers(',start)
    body=source[start:end]
    assert body.index('std::string contract;') < body.index('std::vector<ExactOrderedBytePair> named_pairs;')
    assert body.index('contract = std::move(exact).release();') < body.index('named_pairs.emplace_back("state-carriers-capacity", contract);') < body.index('catch (...)') < body.index('rethrow_collective_failure') < body.index('all_ranks_agree_exact_ordered_byte_pairs(named_pairs, lane)')
    assert body.count('contract =')==1
