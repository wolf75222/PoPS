#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/storage/fab2d.hpp>

#include <functional>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pops {

struct PreparedCoarseFineTransform2D {
  int coarse_origin_x = 0;
  int coarse_origin_y = 0;
  int fine_origin_x = 0;
  int fine_origin_y = 0;
  int refinement_ratio_x = 0;
  int refinement_ratio_y = 0;
};

/// Type-erased host launch authority for one conservative cell-average coarse/fine algorithm.
///
/// Registry providers prepare this value once.  Generic carrier and FillPatch workspaces use its
/// reach/validation metadata when allocating, then invoke the host launchers per patch.  Each
/// launcher submits its own named Kokkos functor; no `std::function` crosses the device boundary and
/// adding another provider requires no branch in either workspace.
struct PreparedCoarseFineOperator {
  using SpatialLauncher = std::function<void(
      Array4, ConstArray4, const Box2D&, const Box2D&, const Box2D&, const Box2D&,
      const PreparedCoarseFineTransform2D&, int, bool, bool, Periodicity)>;
  using SpaceTimeLauncher = std::function<void(
      Array4, ConstArray4, ConstArray4, const Box2D&, const Box2D&, const Box2D&, const Box2D&,
      const PreparedCoarseFineTransform2D&, int, Real, Real, int, Periodicity)>;

  // Directional 2D requirements of the current backend.  Builtins are isotropic, but the seam does
  // not present one scalar as a universal multi-dimensional contract.
  int parent_reach_x = -1;
  int parent_reach_y = -1;
  int minimum_axis_cells_x = 0;
  int minimum_axis_cells_y = 0;
  std::function<void(const Box2D&)> validate_extra;
  SpatialLauncher launch_spatial;
  SpaceTimeLauncher launch_space_time;

  void validate() const {
    if (parent_reach_x < 0 || parent_reach_y < 0 || minimum_axis_cells_x < 1 ||
        minimum_axis_cells_y < 1 || !launch_spatial || !launch_space_time)
      throw std::invalid_argument("incomplete prepared coarse/fine operator authority");
  }

  void validate_domain(const Box2D& coarse_domain) const {
    validate();
    if (coarse_domain.empty() || coarse_domain.nx() < minimum_axis_cells_x ||
        coarse_domain.ny() < minimum_axis_cells_y)
      throw std::invalid_argument(
          "prepared coarse/fine operator requires a larger parent domain on every axis; no "
          "lower-order fallback is permitted");
    if (validate_extra)
      validate_extra(coarse_domain);
  }
};

namespace detail {

inline int checked_coarse_fine_carrier_growth(int fine_ghost_depth, int refinement_ratio,
                                               int parent_reach) {
  if (fine_ghost_depth < 0 || refinement_ratio <= 0 || parent_reach < 0)
    throw std::invalid_argument("prepared coarse/fine carrier has invalid growth metadata");
  const std::int64_t growth = static_cast<std::int64_t>(fine_ghost_depth) +
                              static_cast<std::int64_t>(refinement_ratio) * parent_reach;
  if (growth > std::numeric_limits<int>::max())
    throw std::invalid_argument(
        "prepared coarse/fine carrier growth exceeds the native integer index range");
  return static_cast<int>(growth);
}

}  // namespace detail

}  // namespace pops
