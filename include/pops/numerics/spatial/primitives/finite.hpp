#pragma once

/// @file
/// @brief Collective fail-closed preflight for finite-volume fields.
///
/// A device flux/source kernel cannot throw coherently under MPI.  A fail-closed pointwise
/// evaluator therefore materializes a non-finite value; host-side operator/publication boundaries
/// use this preflight before that value can update state or enter a conservation ledger.  The
/// reduction is collective: all ranks either continue or throw at the same program point.

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <stdexcept>
#include <string>

namespace pops::detail {

struct NonFiniteFiniteVolumeKernel {
  ConstArray4 values;
  int ncomp;

  POPS_HD Real operator()(int i, int j) const {
    for (int component = 0; component < ncomp; ++component)
      if (!Kokkos::isfinite(values(i, j, component)))
        return Real(1);
    return Real(0);
  }
};

inline bool local_finite_volume_data_has_nonfinite(const MultiFab& values) {
  for (int local = 0; local < values.local_size(); ++local) {
    const Real found = for_each_cell_reduce_max(
        values.box(local),
        NonFiniteFiniteVolumeKernel{values.fab(local).const_array(), values.ncomp()});
    if (found != Real(0))
      return true;
  }
  return false;
}

/// Reject one or more fields with exactly one MPI collective.  The variadic form keeps the paired
/// x/y face-flux preflight atomic: no rank can throw after checking Fx while a peer enters Fy.
template <class... Remaining>
inline void reject_nonfinite_finite_volume_data(const char* where, const MultiFab& first,
                                                const Remaining&... remaining) {
  const bool local_failed = local_finite_volume_data_has_nonfinite(first) ||
                            (local_finite_volume_data_has_nonfinite(remaining) || ...);
  const double global_failed = all_reduce_max(local_failed ? 1.0 : 0.0);
  if (global_failed != 0.0)
    throw std::runtime_error(
        std::string(where) +
        " produced non-finite finite-volume data; a physical flux, numerical flux, source, or "
        "metric evaluation failed");
}

/// Validate a complete AMR hierarchy with one collective.  The level type is intentionally a
/// protocol (it only needs a ``U`` MultiFab), keeping this primitive independent of AMR ownership.
template <class LevelRange>
inline void reject_nonfinite_finite_volume_hierarchy(const char* where,
                                                     const LevelRange& levels) {
  bool local_failed = false;
  for (const auto& level : levels)
    local_failed = local_finite_volume_data_has_nonfinite(level.U) || local_failed;
  const double global_failed = all_reduce_max(local_failed ? 1.0 : 0.0);
  if (global_failed != 0.0)
    throw std::runtime_error(
        std::string(where) +
        " produced a non-finite AMR state during reflux, average-down, or synchronization");
}

}  // namespace pops::detail
