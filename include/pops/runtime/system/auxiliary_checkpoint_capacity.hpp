#pragma once
#include <pops/runtime/system/auxiliary_checkpoint.hpp>
#include <pops/runtime/program/module_metadata.hpp>
#include <algorithm>
#include <limits>
#include <map>

namespace pops::runtime::system {
inline std::size_t checked_auxiliary_capacity_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("auxiliary checkpoint capacity exceeds size_t");
  return a + b;
}

/// Pending inputs belong to the full attempt journal, which is separately snapshotted for rollback.
/// Durable accepted-image capture rejects that work rather than publishing it or inventing points.
template<int Dim>
void require_no_pending_auxiliary_input_checkpoint(const ExactAuxiliaryRegistry<Dim>& registry,
    const std::vector<std::string>& dirty) {
  for (const auto& identity : dirty) {
    const PreparedAuxiliaryProvider<Dim>* found = nullptr;
    for (std::size_t index = 0; index < registry.provider_count(); ++index)
      if (registry.provider(index).identity() == identity) { found = &registry.provider(index); break; }
    if (!found) throw std::invalid_argument("auxiliary task contains a foreign dirty provider");
    if (found->kind() == AuxiliaryProviderKind::input)
      throw std::logic_error("durable auxiliary checkpoint refuses unpublished staged input work");
  }
}

/// Observe accepted provenance and accepted stale images only. The complete transient dirty
/// set remains owned by the task snapshot; input and unpublished work cannot be durable invalidations.
template<int Dim>
AuxiliaryCheckpointAcceptedState<Dim> current_accepted_auxiliary_checkpoint_observation(
    const ExactAuxiliaryRegistry<Dim>& registry, const std::vector<std::string>& dirty) {
  std::set<std::string> unique;
  std::vector<std::string> accepted_invalidations;
  for (const auto& identity : dirty) {
    if (!unique.insert(identity).second)
      throw std::invalid_argument("auxiliary task contains duplicate dirty providers");
    const PreparedAuxiliaryProvider<Dim>* found = nullptr;
    for (std::size_t index = 0; index < registry.provider_count(); ++index)
      if (registry.provider(index).identity() == identity) { found = &registry.provider(index); break; }
    if (!found) throw std::invalid_argument("auxiliary task contains a foreign dirty provider");
    if (found->kind() != AuxiliaryProviderKind::input && registry.last_accepted_point(identity))
      accepted_invalidations.push_back(identity);
  }
  return capture_auxiliary_checkpoint_state(registry, accepted_invalidations);
}

/// Reserve derives from installed ownership data, independently of transient task work.
/// Arbitrary raw System points remain outside this future certificate and use current-size queries.
template<int Dim>
std::size_t program_auxiliary_metadata_capacity(const ExactAuxiliaryRegistry<Dim>& registry,
    const program::ProgramOwnedClockManifest& manifest) {
  manifest.validate();
  const auto actual = capture_auxiliary_checkpoint_state(registry);
  std::size_t bound = serialize_auxiliary_checkpoint_state(actual).size();
  std::size_t maximum_clock = 0;
  for (const auto& clock : manifest.logical_clock_identities)
    maximum_clock = std::max(maximum_clock, clock.size());
  for (const auto& provider : actual.providers)
    if (provider.accepted_point)
      maximum_clock = std::max(maximum_clock, provider.accepted_point->clock.size());
  // POPSAUX2/3: clock length prefix plus seven u64/i32 slots (i32 uses a u64 wire slot).
  constexpr std::size_t point_fixed = 8 * sizeof(std::uint64_t);
  const auto future_point = checked_auxiliary_capacity_add(point_fixed, maximum_clock);
  std::size_t extension = 0;
  bool noninput = false;
  for (const auto& provider : actual.providers) {
    const auto present = provider.accepted_point
        ? checked_auxiliary_capacity_add(point_fixed, provider.accepted_point->clock.size()) : 0;
    bound = checked_auxiliary_capacity_add(bound, future_point - present);
    if (provider.kind != AuxiliaryProviderKind::input) {
      noninput = true;
      extension = checked_auxiliary_capacity_add(extension,
          checked_auxiliary_capacity_add(sizeof(std::uint64_t), provider.identity.size()));
    }
  }
  if (noninput) bound = checked_auxiliary_capacity_add(bound,
      checked_auxiliary_capacity_add(sizeof(std::uint64_t), extension));
  return bound;
}

/// Accepted-only restore replaces newer staged values with the actual accepted input payload.
/// A provider lacking accepted provenance remains absent and uninitialized, hence due on first use.
template<int Dim>
std::map<std::string, std::vector<double>> restored_accepted_auxiliary_inputs(
    const AuxiliaryCheckpointAcceptedState<Dim>& state, std::size_t cells) {
  auxiliary_checkpoint_detail::validate_state(state);
  std::map<std::string, const AuxiliaryCheckpointStorageGroup<Dim>*> groups;
  for (const auto& group : state.groups) {
    if (cells != 0 && group.component_count > std::numeric_limits<std::size_t>::max() / cells)
      throw std::overflow_error("restored auxiliary input payload shape exceeds size_t");
    if (group.payload.size() != group.component_count * cells)
      throw std::invalid_argument("restored auxiliary input payload differs from exact domain shape");
    groups.emplace(group.identity, &group);
  }
  std::map<std::string, std::vector<double>> result;
  for (const auto& component : state.components) {
    if (component.provider_kind != AuxiliaryProviderKind::input) continue;
    const auto provider = std::find_if(state.providers.begin(), state.providers.end(),
        [&](const auto& value) { return value.identity == component.provider_identity; });
    if (provider == state.providers.end())
      throw std::invalid_argument("restored input component has no provider provenance");
    if (!provider->accepted_point) continue;
    const auto* group = groups.at(component.address.group);
    const auto offset = component.address.component * cells;
    const auto begin = group->payload.begin() + offset;
    if (!result.emplace(component.key.exact_key(), std::vector<double>(begin, begin + cells)).second)
      throw std::invalid_argument("restored input cache has duplicate component keys");
  }
  return result;
}
} // namespace pops::runtime::system
