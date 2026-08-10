#pragma once

// Native geometry projection used only by the dimension-specialized System/AmrSystem Python
// Writer seams. Keep this out of bindings_detail.hpp: that header is included by every binding
// translation unit, while this implementation is relevant only to init_system.cpp and init_amr.cpp.

#include "../bindings_detail.hpp"

#include <pops/mesh/index/box.hpp>
#include <pops/mesh/layout/refinement.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::python::detail {

/// One materialized hierarchy patch in the native coordinate order (x[, y[, z]]).
template <int Dim>
struct OutputGeometryPatch {
  static_assert(Dim >= 1 && Dim <= 3, "output geometry only supports dimensions 1, 2, and 3");

  int level = 0;
  Box<Dim> box{};
};

template <int Dim>
constexpr std::string_view cartesian_cell_measure() {
  static_assert(Dim >= 1 && Dim <= 3, "output geometry only supports dimensions 1, 2, and 3");
  constexpr std::array<std::string_view, 3> measures{
      "pops://cell-measures/cartesian-length@1",
      "pops://cell-measures/cartesian-area@1",
      "pops://cell-measures/cartesian-volume@1",
  };
  return measures[static_cast<std::size_t>(Dim - 1)];
}

template <int Dim>
double cartesian_cell_volume(const std::array<double, Dim>& spacing) {
  double result = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    result *= spacing[static_cast<std::size_t>(axis)];
  return result;
}

template <int Dim>
double output_cell_volume(const std::string& cell_measure,
                          const std::array<double, Dim>& /*origin*/,
                          const std::array<double, Dim>& spacing, const Index<Dim>& /*index*/) {
  if (cell_measure != cartesian_cell_measure<Dim>())
    throw std::invalid_argument("native output geometry has no registered cell-measure kernel");
  return cartesian_cell_volume<Dim>(spacing);
}

template <>
inline double output_cell_volume<2>(const std::string& cell_measure,
                                    const std::array<double, 2>& origin,
                                    const std::array<double, 2>& spacing, const Index<2>& index) {
  if (cell_measure == "pops://cell-measures/polar-annulus-area@1") {
    const double inner = origin[0] + static_cast<double>(index[0]) * spacing[0];
    return 0.5 * ((inner + spacing[0]) * (inner + spacing[0]) - inner * inner) * spacing[1];
  }
  if (cell_measure != cartesian_cell_measure<2>())
    throw std::invalid_argument("native output geometry has no registered cell-measure kernel");
  return cartesian_cell_volume<2>(spacing);
}

template <int Dim, class Function>
void for_each_output_index(const Box<Dim>& box, Function&& function) {
  if (box.empty())
    return;
  Index<Dim> index = box.lo;
  while (true) {
    function(index);
    int axis = 0;
    for (; axis < Dim; ++axis) {
      if (index[axis] < box.hi[axis]) {
        ++index[axis];
        break;
      }
      index[axis] = box.lo[axis];
    }
    if (axis == Dim)
      return;
  }
}

template <int Dim>
std::size_t output_linear_offset(const Index<Dim>& index,
                                 const std::array<std::int64_t, Dim>& cell_shape) {
  std::size_t offset = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    offset += static_cast<std::size_t>(index[axis]) * stride;
    stride *= static_cast<std::size_t>(cell_shape[static_cast<std::size_t>(axis)]);
  }
  return offset;
}

/// Build one Writer geometry payload on the native side.
///
/// ``origin``, ``spacing``, ``cell_shape`` and every Box use the native coordinate order
/// (x[, y[, z]]). Dense NumPy arrays use the reverse order so x remains the contiguous axis. The
/// Writer v1 ABI consumes host memory, so the arrays deliberately live in NumPy-owned host storage.
/// RuntimeOutputSnapshot retains them by (layout, level, topology epoch), and the generated Writer
/// marshaller borrows their buffers without another allocation.
template <int Dim>
py::dict native_output_geometry_snapshot(int level, std::uint64_t topology_epoch,
                                         const std::array<double, Dim>& origin,
                                         const std::array<double, Dim>& spacing,
                                         const std::array<std::int64_t, Dim>& cell_shape,
                                         const std::string& cell_measure,
                                         const std::vector<OutputGeometryPatch<Dim>>& patches,
                                         const std::array<int, Dim>& next_refinement_ratio,
                                         bool adaptive) {
  static_assert(Dim >= 1 && Dim <= 3, "output geometry only supports dimensions 1, 2, and 3");
  if (level < 0)
    throw std::invalid_argument("native output geometry level must be nonnegative");
  bool has_next_refinement = false;
  bool inactive_refinement = true;
  Extent<Dim> next_ratio{};
  for (int axis = 0; axis < Dim; ++axis) {
    const int value = next_refinement_ratio[static_cast<std::size_t>(axis)];
    inactive_refinement = inactive_refinement && value == 0;
    if (value > 0) {
      has_next_refinement = has_next_refinement || value > 1;
      next_ratio[axis] = value;
    }
  }
  if (!inactive_refinement &&
      (std::any_of(next_refinement_ratio.begin(), next_refinement_ratio.end(),
                   [](int value) { return value < 1; }) ||
       !has_next_refinement))
    throw std::invalid_argument(
        "native output geometry next refinement ratio must be all-zero or exact-rank, "
        "positive, and refine at least one axis");

  Index<Dim> domain_lower{};
  Index<Dim> domain_upper{};
  std::size_t count = 1;
  std::vector<py::ssize_t> numpy_shape(static_cast<std::size_t>(Dim));
  for (int axis = 0; axis < Dim; ++axis) {
    const auto position = static_cast<std::size_t>(axis);
    const auto extent = cell_shape[position];
    if (extent < 1 || extent > std::numeric_limits<int>::max() ||
        extent > std::numeric_limits<py::ssize_t>::max() || !std::isfinite(origin[position]) ||
        !std::isfinite(spacing[position]) || spacing[position] <= 0.0)
      throw std::invalid_argument("native output geometry has invalid axes or shape");
    const auto size = static_cast<std::size_t>(extent);
    if (count > std::numeric_limits<std::size_t>::max() / size)
      throw std::overflow_error("native output geometry cell count exceeds size_t");
    count *= size;
    domain_upper[axis] = static_cast<int>(extent - 1);
    numpy_shape[static_cast<std::size_t>(Dim - 1 - axis)] = static_cast<py::ssize_t>(extent);
  }
  const Box<Dim> domain{domain_lower, domain_upper};
  const auto checked_box = [&domain](const Box<Dim>& box) {
    if (box.empty() || !domain.contains(box))
      throw std::invalid_argument("native output geometry patch lies outside its cell shape");
    return box;
  };

  std::vector<Box<Dim>> boxes;
  if (!adaptive) {
    boxes.push_back(domain);
  } else {
    for (const OutputGeometryPatch<Dim>& patch : patches)
      if (patch.level == level)
        boxes.push_back(checked_box(patch.box));
    if (boxes.empty())
      throw std::invalid_argument("native output geometry level has no materialized patch");
  }
  // Preserve the per-level BoxArray order: OutputPiece.global_box_index indexes this exact order.
  // Sorting geometry independently would silently point each piece at a different patch.

  py::array_t<bool> valid_cells(numpy_shape);
  py::array_t<bool> coverage(numpy_shape);
  py::array_t<double> cell_volumes(numpy_shape);
  auto* valid = valid_cells.mutable_data();
  auto* covered = coverage.mutable_data();
  auto* volumes = cell_volumes.mutable_data();
  std::fill_n(valid, count, false);
  std::fill_n(covered, count, false);

  for (const Box<Dim>& box : boxes)
    for_each_output_index(box, [&](const Index<Dim>& index) {
      valid[output_linear_offset<Dim>(index, cell_shape)] = true;
    });

  if (adaptive && has_next_refinement) {
    for (const OutputGeometryPatch<Dim>& patch : patches) {
      if (patch.level != level + 1)
        continue;
      const Box<Dim> parent = checked_box(pops::coarsen(patch.box, next_ratio));
      for_each_output_index(parent, [&](const Index<Dim>& index) {
        covered[output_linear_offset<Dim>(index, cell_shape)] = true;
      });
    }
  }

  // This is the full coordinate-cell measure.  Embedded-boundary volume fractions are emitted
  // independently as the reserved ``pops_kappa`` sidecar; folding them into this geometry array
  // would make the mesh metric ambiguous and would double-apply kappa in downstream consumers.
  for_each_output_index(domain, [&](const Index<Dim>& index) {
    volumes[output_linear_offset<Dim>(index, cell_shape)] =
        output_cell_volume<Dim>(cell_measure, origin, spacing, index);
  });

  py::list box_rows;
  for (const Box<Dim>& box : boxes) {
    py::tuple row(static_cast<py::ssize_t>(2 * Dim));
    for (int array_axis = 0; array_axis < Dim; ++array_axis) {
      const int native_axis = Dim - 1 - array_axis;
      row[static_cast<py::ssize_t>(array_axis)] = box.lo[native_axis];
      row[static_cast<py::ssize_t>(Dim + array_axis)] = box.hi[native_axis] + 1;
    }
    box_rows.append(std::move(row));
  }
  py::tuple output_shape(static_cast<py::ssize_t>(Dim));
  for (int axis = 0; axis < Dim; ++axis)
    output_shape[static_cast<py::ssize_t>(axis)] = numpy_shape[static_cast<std::size_t>(axis)];

  // Prevent mutation while the native Writer borrows these exact buffers.
  valid_cells.attr("setflags")(false);
  coverage.attr("setflags")(false);
  cell_volumes.attr("setflags")(false);
  py::dict result;
  result["dimension"] = Dim;
  result["topology_epoch"] = topology_epoch;
  result["cell_shape"] = std::move(output_shape);
  result["boxes"] = std::move(box_rows);
  result["valid_cells"] = std::move(valid_cells);
  result["coverage"] = std::move(coverage);
  result["cell_volumes"] = std::move(cell_volumes);
  return result;
}

}  // namespace pops::python::detail
