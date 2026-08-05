/// @file
/// @brief Multi-patch publication seam for the canonical compile-time-ranked face operator.

#pragma once

#include <pops/numerics/spatial/operators/cartesian_operator.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace pops::nd {

/// Materialize the integrated face fluxes of one prepared Cartesian specialization.
///
/// The numerical algorithm remains owned by `PreparedCartesianOperator`: reconstruction uses
/// `FieldView<const Real, Dim>` and `Index<Dim>`, while axis traversal is instantiated recursively
/// through `for_each_face<Axis>`.  This function only lifts that patch operation over the local
/// patches of one ranked `MultiFab`; it does not provide a two-dimensional storage adapter.
template <int Dim, class Model, class Metric, class Reconstruction, class NumericalFlux,
          ReconstructionVariables Variables, class MemorySpace>
void compute_face_fluxes(const PreparedCartesianOperator<Dim, Model, Metric, Reconstruction,
                                                         NumericalFlux, Variables>& prepared,
                         const MultiFab<Dim, MemorySpace>& state,
                         std::vector<FaceField<Dim, MemorySpace>>& integrated_fluxes) {
  if (state.ncomp() != Model::n_vars || integrated_fluxes.size() != state.local_size())
    throw std::invalid_argument("prepared ND face workspace does not match the state MultiFab");

  for (std::size_t local = 0; local < state.local_size(); ++local) {
    if (!(integrated_fluxes[local].cell_box() == state.box(local)) ||
        integrated_fluxes[local].ncomp() != Model::n_vars)
      throw std::invalid_argument(
          "prepared ND face workspace patch does not match the state patch");
  }

  auto candidate = make_face_flux_workspace(state);
  for (std::size_t local = 0; local < state.local_size(); ++local)
    prepared.materialize_face_fluxes(state.fab(local), candidate[local]);
  for (std::size_t local = 0; local < state.local_size(); ++local)
    cartesian_operator_detail::copy_face_axes<0>(candidate[local], integrated_fluxes[local],
                                                 Model::n_vars);
  device_fence();
}

}  // namespace pops::nd
