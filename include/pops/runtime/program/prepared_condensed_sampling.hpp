/// @file
/// @brief Stateless footprint authentication for generated condensed pointwise laws.
#pragma once

#include <pops/mesh/storage/multifab.hpp>
#include <cstdint>
#include <stdexcept>

namespace pops::runtime::program {
// The law evaluates storage indices, including authenticated coarse/fine state ghosts.
// A compact provider may not silently contribute a view with a smaller sampling footprint.
template <int Dim, class Providers>
void require_condensed_sampling_footprint(const MultiFab<Dim>& output, const MultiFab<Dim>& state,
                                          Providers&& providers) {
  if (output.layout() != state.layout() || output.distribution() != state.distribution() ||
      output.local_rank() != state.local_rank() || output.local_size() != state.local_size())
    throw std::invalid_argument("AMR condensed sampling state differs from the output ownership");
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    const auto& box = output.fab(local).grown_box();
    const auto require = [&](const auto& view) {
      if (view.data == nullptr)
        throw std::invalid_argument("AMR condensed sampling has an absent state/provider view");
      for (int axis = 0; axis < Dim; ++axis) {
        const auto low = static_cast<std::int64_t>(box.lo[axis]) - view.origin[axis];
        const auto high = static_cast<std::int64_t>(box.hi[axis]) - view.origin[axis];
        if (low < 0 || high >= static_cast<std::int64_t>(view.extents[axis]))
          throw std::invalid_argument(
              "AMR condensed sampling lacks its exact grown state/provider footprint");
      }
    };
    require(state.fab(local).view());
    const auto views = providers(local);
    for (const auto& view : views.storage)
      require(view);
  }
}
}  // namespace pops::runtime::program
