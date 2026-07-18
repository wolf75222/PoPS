#pragma once

#include <cstdint>

namespace pops {

/// Physical distribution of a field over the active communicator. Distributed fields own disjoint
/// pieces whose scientific reductions use SUM. Replicated fields own one complete copy per rank;
/// their copies must agree and contribute exactly once.
enum class FieldDistribution : std::uint8_t { Distributed, Replicated };

constexpr bool field_distribution_is_valid(FieldDistribution distribution) noexcept {
  return distribution == FieldDistribution::Distributed ||
         distribution == FieldDistribution::Replicated;
}

}  // namespace pops
