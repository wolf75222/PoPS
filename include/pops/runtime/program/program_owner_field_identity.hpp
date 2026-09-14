/// @file
/// @brief Stateless identity lookup over an existing Program's owned field registries.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>

namespace pops::runtime::program {
// No registry, lifetime, clock or sampling authority is retained here. The caller supplies
// its current accepted/attempt states and exact existing scratch/history ownership maps.
struct ProgramOwnerFieldIdentity {
  enum class Family : std::uint8_t { Unset, State, Scratch, History, Direct };
  Family family = Family::Unset;
  int scratch_kind = -1;
  int scratch_owner = -1;
  std::int64_t scratch_value_id = -1;
  int scratch_subslot = -1;
  std::string history_name;
  int history_lag = -1;
  int direct_level = -1;
};

template <class Field>
bool same_program_owner_layout(const Field& a, const Field& b) {
  return a.layout() == b.layout() && a.distribution() == b.distribution() &&
         a.local_rank() == b.local_rank() && a.local_size() == b.local_size();
}

template <class Field, class StateAt, class AttemptAt, class Scratches, class History, class Decode>
ProgramOwnerFieldIdentity classify_program_owner_field(
    int runtime_block, const Field& seed, int levels, StateAt&& accepted_at, AttemptAt&& attempt_at,
    const Scratches& scratches, const History& manager, Decode&& decode_history, const char* role) {
  using Identity = ProgramOwnerFieldIdentity;
  using Family = Identity::Family;
  Identity identity;
  for (int level = 0; level < levels; ++level) {
    if (&seed == &accepted_at(level)) {
      identity.family = Family::State;
      identity.direct_level = level;
    }
    if (const Field* attempt = attempt_at(level); attempt != nullptr && &seed == attempt) {
      identity.family = Family::State;
      identity.direct_level = level;
    }
  }
  for (const auto& [key, scratch] : scratches) {
    if (&seed != &scratch)
      continue;
    if (std::get<2>(key) != runtime_block)
      throw std::invalid_argument(std::string(role) +
                                  " received a foreign AMR Program owner field");
    if (identity.family != Family::Unset && identity.family != Family::Scratch)
      throw std::logic_error(std::string(role) + " aliases multiple AMR Program field identities");
    identity.family = Family::Scratch;
    identity.scratch_kind = static_cast<int>(std::get<0>(key));
    identity.direct_level = std::get<1>(key);
    identity.scratch_owner = std::get<2>(key);
    identity.scratch_value_id = std::get<3>(key);
    identity.scratch_subslot = std::get<4>(key);
  }
  for (const auto& [key, ring] : manager.histories) {
    for (std::size_t lag = 0; lag < ring.size(); ++lag) {
      if (&seed != &ring[lag])
        continue;
      const auto decoded = decode_history(key);
      const auto owner = manager.owner.find(key);
      if (!decoded || owner == manager.owner.end() || owner->second != runtime_block)
        throw std::invalid_argument(std::string(role) +
                                    " received a foreign AMR Program history field");
      if (identity.family != Family::Unset && identity.family != Family::History)
        throw std::logic_error(std::string(role) +
                               " aliases multiple AMR Program field identities");
      identity.family = Family::History;
      identity.direct_level = decoded->first;
      identity.history_name = decoded->second;
      identity.history_lag = static_cast<int>(lag);
    }
  }
  if (identity.family == Family::Unset) {
    for (int level = 0; level < levels; ++level) {
      if (!same_program_owner_layout(seed, accepted_at(level)))
        continue;
      if (identity.direct_level >= 0)
        throw std::logic_error(std::string(role) + " matches multiple AMR Program owner levels");
      identity.direct_level = level;
    }
    if (identity.direct_level < 0)
      throw std::invalid_argument(std::string(role) +
                                  " has no authenticated AMR Program owner layout");
    identity.family = Family::Direct;
  }
  return identity;
}

template <class Field, class StateAt, class AttemptAt, class Scratches, class History,
          class HistoryKey>
const Field& resolve_program_owner_field(int runtime_block, const Field& seed,
                                         const ProgramOwnerFieldIdentity& identity, int level,
                                         StateAt&& accepted_at, AttemptAt&& attempt_at,
                                         const Scratches& scratches, const History& manager,
                                         HistoryKey&& history_key, const char* role) {
  using Family = ProgramOwnerFieldIdentity::Family;
  if (identity.family == Family::Scratch && identity.scratch_owner != runtime_block)
    throw std::invalid_argument(std::string(role) + " has a foreign scratch owner identity");
  const Field& accepted = accepted_at(level);
  const Field* current = nullptr;
  if (identity.family == Family::State) {
    current = attempt_at(level);
    if (current == nullptr)
      current = &accepted;
  } else if (identity.family == Family::Scratch) {
    for (const auto& [key, scratch] : scratches) {
      if (static_cast<int>(std::get<0>(key)) == identity.scratch_kind &&
          std::get<1>(key) == level && std::get<2>(key) == identity.scratch_owner &&
          std::get<3>(key) == identity.scratch_value_id &&
          std::get<4>(key) == identity.scratch_subslot) {
        current = &scratch;
        break;
      }
    }
    if (current == nullptr)
      throw std::runtime_error(std::string(role) +
                               " scratch is missing a live AMR Program owner level");
  } else if (identity.family == Family::History) {
    const auto found = manager.histories.find(history_key(identity.history_name, level));
    if (found == manager.histories.end() || identity.history_lag < 0 ||
        static_cast<std::size_t>(identity.history_lag) >= found->second.size())
      throw std::runtime_error(std::string(role) +
                               " history is missing a live AMR Program owner level");
    current = &found->second[static_cast<std::size_t>(identity.history_lag)];
  } else {
    if (level != identity.direct_level)
      throw std::runtime_error(std::string(role) +
                               " cannot cover every live AMR Program owner level");
    current = &seed;
  }
  if (!same_program_owner_layout(*current, accepted))
    throw std::invalid_argument(std::string(role) + " fields have different exact layouts");
  return *current;
}

}  // namespace pops::runtime::program
