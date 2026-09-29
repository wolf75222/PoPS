/// Frequency of a conservative update from the numerical speeds of its incident faces.
#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pops::nd {

template <int Dim>
POPS_HD Real incident_face_frequency_at(
    const FaceFieldView<const Real, Dim>& speeds, const Index<Dim>& cell,
    const std::array<Real, static_cast<std::size_t>(Dim)>& inverse_spacing) {
  Real result = Real(0);
  for (int axis = 0; axis < Dim; ++axis) {
    Index<Dim> upper = cell;
    ++upper[axis];
    const Real lower_speed = speeds.axes[axis](cell, 0);
    const Real upper_speed = speeds.axes[axis](upper, 0);
    if (!Kokkos::isfinite(lower_speed) || !Kokkos::isfinite(upper_speed) ||
        lower_speed < Real(0) || upper_speed < Real(0))
      return std::numeric_limits<Real>::infinity();
    result += Kokkos::max(lower_speed, upper_speed) * inverse_spacing[axis];
  }
  return Kokkos::isfinite(result) ? result : std::numeric_limits<Real>::infinity();
}

/// Return the local maximum of sum_axis(max(speed_lower, speed_upper)/dx_axis).
/// The caller performs the communicator reduction using its prepared execution lane.
/// Speeds must come from the exact FluxEvaluation used to publish the shared face flux.
template <int Dim, class MemorySpace>
Real maximum_incident_face_frequency(
    const MultiFab<Dim, MemorySpace>& state,
    const std::vector<FaceField<Dim, MemorySpace>>& speeds,
    const Geometry<Dim>& geometry,
    MultiFab<Dim, MemorySpace>& frequency_scratch) {
  if (state.layout() != frequency_scratch.layout() ||
      state.distribution() != frequency_scratch.distribution() ||
      state.local_rank() != frequency_scratch.local_rank() ||
      state.local_size() != frequency_scratch.local_size() ||
      frequency_scratch.ncomp() != 1 || speeds.size() != state.local_size())
    throw std::invalid_argument("face frequency scratch differs from the prepared state layout");
  std::array<Real, Dim> inverse_spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    const Real spacing = geometry.spacing(axis);
    if (!std::isfinite(spacing) || !(spacing > Real(0)))
      throw std::invalid_argument("face frequency requires positive finite axis spacing");
    inverse_spacing[axis] = Real(1) / spacing;
  }
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    if (speeds[local].cell_box() != state.box(local) || speeds[local].ncomp() != 1 ||
        frequency_scratch.box(local) != state.box(local))
      throw std::invalid_argument("face frequency patch or component authority differs");
    const auto values = speeds[local].view();
    const auto frequency = frequency_scratch.fab(local).view();
    const auto inv = inverse_spacing;
    for_each_cell(state.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      frequency(cell, 0) = incident_face_frequency_at(values, cell, inv);
    });
  }
  device_fence();
  return reduce_max_local(frequency_scratch);
}

}  // namespace pops::nd
