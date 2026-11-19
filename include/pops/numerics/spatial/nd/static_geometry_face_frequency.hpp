/// Frequency of a conservative update on a layout served by static geometry arrays (N17, C1).
///
/// The same rules as face_frequency.hpp, with each face speed weighed by the measure of its face
/// over the measure of its cell in place of the inverse spacing of its axis. A separate header so
/// that the box contract keeps its own bound and the adaptive consumers do not read the arrays.
#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/static_geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/numerics/spatial/nd/face_frequency_rule.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pops::nd {

/// The face frequency of ``rule`` on one cell: the incident sum is
/// sum_axis max(speed_lower |S_lower|, speed_upper |S_upper|) / |K|, and the largest-axis rule
/// takes the largest speed times the largest |S| / |K|. On the uniform box |S| / |K| is
/// 1 / dx_axis and both are the rules of face_frequency.hpp; on node positions it is the inverse
/// width of the cell along the axis. Infinity on a nonfinite or negative speed or reading.
template <int Dim>
POPS_HD Real incident_face_frequency_at(const FaceFieldView<const Real, Dim>& speeds,
                                        const Index<Dim>& cell,
                                        const StaticGeometryView<Dim>& arrays,
                                        FaceFrequencyRule rule = FaceFrequencyRule::incident_sum) {
  const Real measure = arrays.measure(cell);
  Real sum = Real(0), largest_speed = Real(0), largest_ratio = Real(0);
  for (int axis = 0; axis < Dim; ++axis) {
    Index<Dim> upper = cell;
    ++upper[axis];
    const Real lower_speed = speeds.axes[axis](cell, 0);
    const Real upper_speed = speeds.axes[axis](upper, 0);
    if (!Kokkos::isfinite(lower_speed) || !Kokkos::isfinite(upper_speed) || lower_speed < Real(0) ||
        upper_speed < Real(0))
      return std::numeric_limits<Real>::infinity();
    const Real lower_ratio = arrays.face_measure_at(axis, cell) / measure;
    const Real upper_ratio = arrays.face_measure_at(axis, upper) / measure;
    if (!Kokkos::isfinite(lower_ratio) || !Kokkos::isfinite(upper_ratio) || lower_ratio < Real(0) ||
        upper_ratio < Real(0))
      return std::numeric_limits<Real>::infinity();
    sum += Kokkos::max(lower_speed * lower_ratio, upper_speed * upper_ratio);
    largest_speed = Kokkos::max(largest_speed, Kokkos::max(lower_speed, upper_speed));
    largest_ratio = Kokkos::max(largest_ratio, Kokkos::max(lower_ratio, upper_ratio));
  }
  const Real result = rule == FaceFrequencyRule::largest_axis ? largest_speed * largest_ratio : sum;
  return Kokkos::isfinite(result) ? result : std::numeric_limits<Real>::infinity();
}

/// The local maximum of the face frequency of ``rule``, each cell reading the measures of its
/// faces and its own measure, reduced in one pass over the recorded speeds. The caller performs
/// the communicator reduction using its prepared execution lane.
template <int Dim, class MemorySpace>
Real maximum_incident_face_frequency(const MultiFab<Dim, MemorySpace>& state,
                                     const std::vector<FaceField<Dim, MemorySpace>>& speeds,
                                     const StaticGeometryView<Dim>& arrays,
                                     FaceFrequencyRule rule = FaceFrequencyRule::incident_sum) {
  if (speeds.size() != state.local_size())
    throw std::invalid_argument("face frequency speeds differ from the prepared state layout");
  if (arrays.cell_measure == nullptr)
    throw std::invalid_argument("face frequency on static geometry arrays requires the arrays");
  Real result = -std::numeric_limits<Real>::infinity();
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    if (speeds[local].cell_box() != state.box(local) || speeds[local].ncomp() != 1)
      throw std::invalid_argument("face frequency patch or component authority differs");
    const auto values = speeds[local].view();
    result = std::max(
        result, for_each_cell_reduce_max(state.box(local), [=] POPS_HD(const Index<Dim>& cell) {
          return incident_face_frequency_at(values, cell, arrays, rule);
        }));
  }
  return result;
}

}  // namespace pops::nd
