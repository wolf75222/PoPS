// Independent bounded host probe; build against the reviewed wire9 header.
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <cassert>
#include <iostream>
#include <limits>

namespace p = pops::runtime::program;

p::AmrProgramAcceptedState<2> state(std::size_t extra) {
  p::AmrProgramAcceptedState<2> value;
  value.spatial_contract = "capacity-review";
  value.topology_epoch = 9;
  value.materialization_generation = 11;
  value.logical_clock_ticks = {{"macro", 3}};
  value.flux_budget_contract = "no-flux";
  value.coupling_contract = "no-coupling";
  const p::HistorySampleIdentity sample{std::bit_cast<std::uint64_t>(.5),
      std::bit_cast<std::uint64_t>(.25), 7, p::HistorySampleKind::Publication};
  for (int level = 0; level < 5; ++level)
    value.level_clocks.push_back({level, 3, {0, 1}, .75});
  for (int ring = 0; ring < 3; ++ring) {
    std::string name = "ring" + std::to_string(ring) + std::string(13 * ring, 'n');
    value.histories.push_back({name, ring, "state" + std::string(31 * ring, 's'),
        "space" + std::string(17 * ring, 'p'), "macro",
        "interpolation" + std::string(29 * ring + (ring == 2 ? extra : 0), 'i'), 2, ring + 1});
    for (int level = 0; level < 5; ++level)
      for (int slot = 0; slot < 2; ++slot)
        value.history_slots.push_back({name, level, slot, .25, true, 2, sample});
  }
  for (const auto& ring : value.histories)
    for (int child = 1; child < 5; ++child) {
      const auto key = "pops.amr.level-history.v1/" + std::to_string(child) + "/" +
          std::to_string(ring.name.size()) + ":" + ring.name;
      p::AmrProgramPendingHistoryRemap pending{key, child - 1, child, 8, 10,
          9, 11, 3, 2, 1, .25, .125, false};
      pending.qualified_topology_epoch = 9;
      pending.qualified_materialization_generation = 11;
      pending.source_sample = sample;
      pending.retained_ring_contract = p::checkpoint_detail::pending_ring_contract(
          value.histories, value.history_slots, ring.name, child, value.level_clocks[child]);
      value.pending_history_remaps.push_back(pending);
    }
  std::sort(value.pending_history_remaps.begin(), value.pending_history_remaps.end(),
            [](const auto& a, const auto& b) { return a.key < b.key; });
  return value;
}

p::AmrProgramAcceptedStateCapacity<2> shape(const p::AmrProgramAcceptedState<2>& state) {
  p::AmrProgramAcceptedStateCapacity<2> result;
  result.spatial_contract_characters = state.spatial_contract.size();
  result.level_count = state.level_clocks.size();
  result.logical_clock_identities = {"macro"};
  result.histories = state.histories;  // Same frozen-descriptor source used by bind.
  result.temporal_provider_identity = state.temporal_partition.provider_identity;
  result.flux_budget_contract_characters = state.flux_budget_contract.size();
  result.coupling_contract_characters = state.coupling_contract.size();
  result.pending_history_remap_count = result.histories.size() * (result.level_count - 1);
  for (const auto& marker : state.pending_history_remaps)
    result.pending_history_remap_key_characters =
        std::max(result.pending_history_remap_key_characters, marker.key.size());
  return result;
}

int main() {
  std::size_t previous_capacity = 0, previous_extra = 0;
  for (std::size_t extra : {0u, 1u, 8192u, 131071u}) {
    const auto value = state(extra);
    const auto capacity_shape = shape(value);
    const auto bytes = p::serialize_amr_program_accepted_state(value);
    const auto capacity = p::serialized_amr_program_accepted_state_capacity(capacity_shape);
    assert(p::serialized_amr_program_accepted_state_size(value) == bytes.size());
    assert(bytes.size() <= capacity);
    assert(p::deserialize_amr_program_accepted_state<2>(bytes).pending_history_remaps ==
           value.pending_history_remaps);
    if (previous_capacity)
      // One descriptor plus max-contract bound for each of 12 pending records.
      assert(capacity - previous_capacity == 13 * (extra - previous_extra));
    previous_capacity = capacity;
    previous_extra = extra;
  }
  auto enormous = shape(state(0));
  enormous.pending_history_remap_count = std::numeric_limits<std::size_t>::max();
  bool refused = false;
  try { (void)p::serialized_amr_program_accepted_state_capacity(enormous); }
  catch (const std::length_error&) { refused = true; }
  assert(refused);
  std::cout << "wire9 capacity: four unequal long-contract cases and overflow passed\n";
}
