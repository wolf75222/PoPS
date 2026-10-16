#pragma once

#include <pops/mesh/geometry/geometry.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <bit>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pops::runtime::program {

// A first production has an exact native point. This descriptor is persisted
// in the copied Field, not regenerated from the consuming clock after History
// or restart. It describes a scientific origin; live storage permission remains
// the responsibility of the context's exact allocation/attempt registry.
inline std::string field_transform_production_origin(
    std::string_view scientific_contract, std::int64_t producer,
    const multiblock::BoundaryEvaluationPoint& point, std::uint64_t capacity) {
  if (scientific_contract.empty() || producer < 0 || point.clock.empty() || point.tick < 0 ||
      point.level < 0 || point.substep < 0 || point.stage < 0 ||
      point.stage_fraction.denominator <= 0 || !std::isfinite(point.dt) || point.dt < 0 ||
      !std::isfinite(point.physical_time))
    throw std::invalid_argument("Field transformation production lacks an exact native point");
  constexpr std::string_view magic = "POPSFTP1";
  // Five length-prefixed strings and ten scalar words. No allocation precedes
  // the complete source/destination preflight, including the borrowed point.
  std::uint64_t bytes = magic.size() + 16 * 15;
  const auto add = [&](std::uint64_t size) {
    if (size > std::numeric_limits<std::uint64_t>::max() - bytes)
      throw std::length_error("Field transformation origin is not representable");
    bytes += size;
  };
  for (const auto text :
       {scientific_contract, std::string_view(point.clock), std::string_view(point.graph_identity),
        std::string_view(point.rate_identity), std::string_view(point.application_identity)})
    add(text.size());
  const auto charge = [&](std::uint64_t size) {
    if (size > capacity)
      throw std::length_error("Field transformation origin exceeds its workspace");
    capacity -= size;
  };
  charge(sizeof(point) + sizeof(std::string));
  charge(scientific_contract.size());
  charge(point.clock.size());
  charge(point.graph_identity.size());
  charge(point.rate_identity.size());
  charge(point.application_identity.size());
  charge(bytes);
  if (bytes > std::string{}.max_size())
    throw std::length_error("Field transformation origin exceeds native string storage");
  std::string out;
  out.reserve(static_cast<std::size_t>(bytes));
  out.append(magic);
  const auto word = [&](std::uint64_t value) {
    constexpr char digits[] = "0123456789abcdef";
    for (int shift = 60; shift >= 0; shift -= 4)
      out.push_back(digits[(value >> shift) & 15]);
  };
  const auto text = [&](std::string_view value) {
    word(value.size());
    out.append(value);
  };
  text(scientific_contract);
  text(point.clock);
  text(point.graph_identity);
  text(point.rate_identity);
  text(point.application_identity);
  word(std::bit_cast<std::uint64_t>(producer));
  word(std::bit_cast<std::uint64_t>(point.tick));
  word(std::uint64_t(point.level));
  word(std::uint64_t(point.substep));
  word(std::uint64_t(point.stage));
  word(std::bit_cast<std::uint64_t>(point.stage_fraction.numerator));
  word(std::bit_cast<std::uint64_t>(point.stage_fraction.denominator));
  word(std::bit_cast<std::uint64_t>(point.dt));
  word(std::bit_cast<std::uint64_t>(point.physical_time));
  word(1);  // Exact framing revision of this production point.
  if (out.size() != bytes)
    throw std::logic_error("Field production origin framing mismatch");
  return out;
}

// This value borrows an origin already authenticated by ProgramContext. Encoding
// it grants no storage permission. In particular, a scientific point is not
// inferred from the current clock: a retained old Field keeps its old origin.
struct FieldTransformSupport {
  std::string_view program, origin, provider;
  int owner = -1, level = -1;
  std::uint64_t epoch = 0, generation = 0;

  template <int Dim>
  std::string encode(const Geometry<Dim>& geometry, int components,
                     std::uint64_t workspace_byte_capacity) const {
    static_assert(sizeof(Real) == sizeof(std::uint64_t));
    if (program.empty() || origin.empty() || level < 0 || components < 1 ||
        !((owner >= 0 && provider.empty()) || (owner == -1 && !provider.empty())))
      throw std::invalid_argument("transform support has no exact scientific source");
    constexpr std::string_view magic = "POPSFTS1";
    std::uint64_t output = magic.size() + 16 * (9 + 4 * Dim);
    const auto add = [&](std::uint64_t size) {
      if (size > std::numeric_limits<std::uint64_t>::max() - output)
        throw std::length_error("transform support size is not representable");
      output += size;
    };
    add(program.size());
    add(origin.size());
    add(provider.size());
    // Borrowed source strings and the destination coexist. No temporary string
    // or partially constructed contract is allocated before this preflight.
    std::uint64_t available = workspace_byte_capacity;
    const auto charge = [&](std::uint64_t size) {
      if (size > available)
        throw std::length_error("transform support exceeds its explicit workspace");
      available -= size;
    };
    charge(sizeof(FieldTransformSupport) + sizeof(std::string));
    charge(program.size());
    charge(origin.size());
    charge(provider.size());
    charge(output);
    if (output > std::string{}.max_size())
      throw std::length_error("transform support exceeds native string storage");
    std::string result;
    result.reserve(static_cast<std::size_t>(output));
    const auto word = [&](std::uint64_t value) {
      constexpr char digits[] = "0123456789abcdef";
      for (int shift = 60; shift >= 0; shift -= 4)
        result.push_back(digits[(value >> shift) & 15]);
    };
    const auto text = [&](std::string_view value) {
      word(value.size());
      result.append(value);
    };
    result.append(magic);
    text(program);
    text(origin);
    text(provider);
    word(std::bit_cast<std::uint64_t>(std::int64_t(owner)));
    word(std::bit_cast<std::uint64_t>(std::int64_t(level)));
    word(epoch);
    word(generation);
    word(Dim);
    word(components);
    for (int axis = 0; axis < Dim; ++axis) {
      word(std::bit_cast<std::uint64_t>(std::int64_t(geometry.domain().lo[axis])));
      word(std::bit_cast<std::uint64_t>(std::int64_t(geometry.domain().hi[axis])));
      word(std::bit_cast<std::uint64_t>(geometry.lower()[axis]));
      word(std::bit_cast<std::uint64_t>(geometry.upper()[axis]));
    }
    if (result.size() != output)
      throw std::logic_error("transform support framing differs from its preflight");
    return result;
  }
};

}  // namespace pops::runtime::program
