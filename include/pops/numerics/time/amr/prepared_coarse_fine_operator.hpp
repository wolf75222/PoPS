/// @file
/// @brief Exact-ranked type-erased launch authority for prepared coarse/fine algorithms.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

namespace pops {

/// Immutable affine relation between one coarse and fine index space.
///
/// Origins and ratios retain their exact ranked types.  A launcher never decodes a dimension tag
/// or substitutes an isotropic scalar ratio for a directional relation.
template <int Dim>
struct PreparedCoarseFineTransform {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedCoarseFineTransform only supports dimensions 1, 2, and 3");

  amr::transfer::IndexMapping<Dim> mapping{};
  amr::RefinementRatio<Dim> refinement_ratio{};

  constexpr bool operator==(const PreparedCoarseFineTransform&) const = default;
};

/// Type-erased host launch authority for one conservative cell-average coarse/fine algorithm.
///
/// Registry providers prepare this value once.  Carrier and FillPatch workspaces invoke one
/// dimension-specialized host launcher per local patch.  The launcher submits a named Kokkos
/// functor; `std::function` never crosses the device boundary.
template <int Dim>
struct PreparedCoarseFineOperator {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedCoarseFineOperator only supports dimensions 1, 2, and 3");

  using transform_type = PreparedCoarseFineTransform<Dim>;
  using box_type = Box<Dim>;
  using write_view_type = FieldView<Real, Dim>;
  using read_view_type = FieldView<const Real, Dim>;
  using topology_type = BoundaryTopology<Dim>;

  using SpatialLauncher = std::function<void(
      write_view_type, read_view_type, const box_type&, const box_type&, const box_type&,
      const box_type&, const transform_type&, int, bool, bool, const topology_type&)>;
  using SpaceTimeLauncher =
      std::function<void(write_view_type, read_view_type, read_view_type, const box_type&,
                         const box_type&, const box_type&, const box_type&, const transform_type&,
                         int, Real, Real, int, const topology_type&)>;

  Extent<Dim> parent_reach{};
  Extent<Dim> minimum_axis_cells{};
  std::function<void(const box_type&)> validate_extra;
  SpatialLauncher launch_spatial;
  SpaceTimeLauncher launch_space_time;

  void validate() const {
    for (int axis = 0; axis < Dim; ++axis) {
      if (parent_reach[axis] < 0 || minimum_axis_cells[axis] < 1)
        throw std::invalid_argument("incomplete prepared coarse/fine operator authority");
    }
    if (!launch_spatial || !launch_space_time)
      throw std::invalid_argument("incomplete prepared coarse/fine operator authority");
  }

  void validate_domain(const box_type& coarse_domain) const {
    validate();
    if (coarse_domain.empty())
      throw std::invalid_argument(
          "prepared coarse/fine operator requires a non-empty parent domain");
    for (int axis = 0; axis < Dim; ++axis) {
      if (coarse_domain.length(axis) < minimum_axis_cells[axis])
        throw std::invalid_argument(
            "prepared coarse/fine operator requires a larger parent domain on every axis; no "
            "lower-order fallback is permitted");
    }
    if (validate_extra)
      validate_extra(coarse_domain);
  }
};

namespace detail {

template <int Dim>
Extent<Dim> checked_coarse_fine_carrier_growth(const Extent<Dim>& fine_ghost_depth,
                                               const amr::RefinementRatio<Dim>& refinement_ratio,
                                               const Extent<Dim>& parent_reach) {
  Extent<Dim> growth{};
  for (int axis = 0; axis < Dim; ++axis) {
    if (fine_ghost_depth[axis] < 0 || parent_reach[axis] < 0)
      throw std::invalid_argument("prepared coarse/fine carrier has invalid growth metadata");
    if (parent_reach[axis] > (std::numeric_limits<std::int64_t>::max() - fine_ghost_depth[axis]) /
                                 refinement_ratio[axis])
      throw std::overflow_error("prepared coarse/fine carrier growth exceeds int64_t");
    growth[axis] = fine_ghost_depth[axis] + parent_reach[axis] * refinement_ratio[axis];
    if (growth[axis] > std::numeric_limits<int>::max())
      throw std::invalid_argument(
          "prepared coarse/fine carrier growth exceeds the native integer index range");
  }
  return growth;
}

}  // namespace detail

}  // namespace pops
