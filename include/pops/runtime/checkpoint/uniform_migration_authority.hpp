#pragma once
#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <pops/core/foundation/types.hpp>

namespace pops::runtime::checkpoint {
/// Read-only Uniform valid projection for migration authority@1. Grown storage is
/// decoded and authenticated, never used to manufacture storage for a legacy image.
template <int Dim>
std::vector<std::vector<std::uint64_t>> uniform_migration_valid_projection(
    std::span<const std::uint8_t> bytes, const std::array<std::uint64_t, Dim>& shape,
    const std::vector<std::string>& blocks, const std::vector<std::uint64_t>& components) {
  const auto archive = decode_state_carriers<Dim>(bytes);
  validate_complete_state_carriers(archive);
  if (archive.levels != 1 || archive.blocks != blocks || components.size() != blocks.size() ||
      archive.real_bits != sizeof(RealBits) * 8)
    throw std::invalid_argument("Uniform migration carrier block/level/Real authority differs");
  std::uint64_t cells = 1;
  for (auto extent : shape) {
    if (!extent || extent > std::uint64_t(INT64_MAX))
      throw std::invalid_argument("Uniform migration carrier domain extent is invalid");
    cells = state_carrier_detail::product(cells, extent);
  }
  std::vector<std::vector<std::uint64_t>> result;
  for (std::size_t b = 0; b < blocks.size(); ++b) {
    if (!components[b]) throw std::invalid_argument("Uniform migration component authority is empty");
    const auto count = state_carrier_detail::product(cells, components[b]);
    // Complete in-domain disjoint coverage must precede allocating the projection.
    std::uint64_t covered = 0;
    for (const auto& row : archive.patches) if (row.block == b) {
      if (row.components != components[b])
        throw std::invalid_argument("Uniform migration component authority differs");
      std::uint64_t valid = 1;
      for (int d = 0; d < Dim; ++d) {
        if (row.lo[d] < 0 || row.hi[d] >= std::int64_t(shape[d]))
          throw std::invalid_argument("Uniform migration valid geometry exceeds domain");
        valid = state_carrier_detail::product(valid, std::uint64_t(row.hi[d] - row.lo[d]) + 1);
      }
      if (valid > cells - covered)
        throw std::invalid_argument("Uniform migration valid geometry overcovers domain");
      covered += valid;
    }
    if (covered != cells || count > SIZE_MAX / sizeof(std::uint64_t))
      throw std::invalid_argument("Uniform migration valid geometry has holes/size overflow");
    auto& projection = result.emplace_back(static_cast<std::size_t>(count));
    for (const auto& row : archive.patches) if (row.block == b) {
      std::uint64_t valid = 1, grown = 1;
      for (int d = 0; d < Dim; ++d) {
        valid = state_carrier_detail::product(valid, std::uint64_t(row.hi[d] - row.lo[d]) + 1);
        grown = state_carrier_detail::product(grown,
            std::uint64_t(row.grown_hi[d]) - std::uint64_t(row.grown_lo[d]) + 1);
      }
      for (std::uint64_t c = 0; c < row.components; ++c)
        for (std::uint64_t i = 0; i < valid; ++i) {
          auto remaining = i;
          std::uint64_t src = c * grown, dst = c * cells, gs = 1, ds = 1;
          for (int d = 0; d < Dim; ++d) {
            const auto extent = std::uint64_t(row.hi[d] - row.lo[d]) + 1;
            const auto coordinate = row.lo[d] + std::int64_t(remaining % extent);
            remaining /= extent;
            src += (std::uint64_t(coordinate) - std::uint64_t(row.grown_lo[d])) * gs;
            dst += std::uint64_t(coordinate) * ds;
            gs *= std::uint64_t(row.grown_hi[d]) - std::uint64_t(row.grown_lo[d]) + 1;
            ds *= shape[d];
          }
          const auto bits = static_cast<RealBits>(row.bits.at(static_cast<std::size_t>(src)));
          // Same native Real→double projection as the public state_global binding.
          projection.at(static_cast<std::size_t>(dst)) =
              std::bit_cast<std::uint64_t>(static_cast<double>(std::bit_cast<Real>(bits)));
        }
    }
  }
  return result;
}
} // namespace pops::runtime::checkpoint
