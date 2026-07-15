#pragma once

// Native geometry projection used only by the System/AmrSystem Python Writer seams.  Keep this out
// of bindings_detail.hpp: that header is included by every binding translation unit, while this
// implementation is relevant only to init_system.cpp and init_amr.cpp.

#include "../bindings_detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops::python::detail {

/// Build one Writer geometry payload on the native side.
///
/// The Writer v1 ABI consumes host memory, so these three arrays deliberately live in NumPy-owned
/// host storage. Filling them here keeps the per-cell geometry work out of the Python scheduler;
/// RuntimeOutputSnapshot retains the returned arrays by (layout, level, topology epoch), and the
/// generated Writer marshaller borrows their buffers without another allocation.
inline py::dict native_output_geometry_snapshot(int level, std::uint64_t topology_epoch,
                                                const std::array<double, 2>& origin,
                                                const std::array<double, 2>& spacing,
                                                const std::array<std::int64_t, 2>& cell_shape,
                                                const std::string& cell_measure,
                                                const std::vector<PatchBox>& patch_boxes,
                                                int next_refinement_ratio, bool adaptive) {
  if (level < 0 || cell_shape[0] < 1 || cell_shape[1] < 1 || !std::isfinite(origin[0]) ||
      !std::isfinite(origin[1]) || !std::isfinite(spacing[0]) || !std::isfinite(spacing[1]) ||
      spacing[0] <= 0.0 || spacing[1] <= 0.0)
    throw std::invalid_argument("native output geometry has invalid axes or shape");
  if (next_refinement_ratio < 0 || next_refinement_ratio == 1)
    throw std::invalid_argument(
        "native output geometry next refinement ratio must be zero or greater than one");

  const auto ny = cell_shape[0];
  const auto nx = cell_shape[1];
  const auto checked_box = [ny, nx](std::int64_t jlo, std::int64_t ilo, std::int64_t jhi,
                                    std::int64_t ihi) {
    if (jlo < 0 || ilo < 0 || jhi <= jlo || ihi <= ilo || jhi > ny || ihi > nx)
      throw std::invalid_argument("native output geometry patch lies outside its cell shape");
    return std::array<std::int64_t, 4>{jlo, ilo, jhi, ihi};
  };

  std::vector<std::array<std::int64_t, 4>> boxes;
  if (!adaptive || level == 0) {
    boxes.push_back({0, 0, ny, nx});
  } else {
    for (const PatchBox& box : patch_boxes)
      if (box.level == level)
        boxes.push_back(checked_box(box.jlo, box.ilo, static_cast<std::int64_t>(box.jhi) + 1,
                                    static_cast<std::int64_t>(box.ihi) + 1));
    if (boxes.empty())
      throw std::invalid_argument("native output geometry level has no materialized patch");
  }
  std::sort(boxes.begin(), boxes.end());

  const std::array<py::ssize_t, 2> shape{static_cast<py::ssize_t>(ny),
                                         static_cast<py::ssize_t>(nx)};
  py::array_t<bool> valid_cells(shape);
  py::array_t<bool> coverage(shape);
  py::array_t<double> cell_volumes(shape);
  auto* valid = valid_cells.mutable_data();
  auto* covered = coverage.mutable_data();
  auto* volumes = cell_volumes.mutable_data();
  const auto count = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
  std::fill_n(valid, count, false);
  std::fill_n(covered, count, false);

  // Patch rectangles are disjoint by the native hierarchy contract. Rectangular fills are O(the
  // represented cells), rather than scanning every patch for every cell or selected quantity.
  for (const auto& box : boxes)
    for (std::int64_t j = box[0]; j < box[2]; ++j)
      std::fill(valid + j * nx + box[1], valid + j * nx + box[3], true);

  if (adaptive && next_refinement_ratio > 1) {
    const auto ratio = static_cast<std::int64_t>(next_refinement_ratio);
    for (const PatchBox& fine : patch_boxes) {
      if (fine.level != level + 1)
        continue;
      const auto parent = checked_box(fine.jlo / ratio, fine.ilo / ratio,
                                      (static_cast<std::int64_t>(fine.jhi) + ratio) / ratio,
                                      (static_cast<std::int64_t>(fine.ihi) + ratio) / ratio);
      for (std::int64_t j = parent[0]; j < parent[2]; ++j)
        std::fill(covered + j * nx + parent[1], covered + j * nx + parent[3], true);
    }
  }

  if (cell_measure == "pops://cell-measures/cartesian-area@1") {
    std::fill_n(volumes, count, spacing[0] * spacing[1]);
  } else if (cell_measure == "pops://cell-measures/polar-annulus-area@1") {
    for (std::int64_t j = 0; j < ny; ++j)
      for (std::int64_t i = 0; i < nx; ++i) {
        const double inner = origin[0] + static_cast<double>(i) * spacing[0];
        volumes[j * nx + i] =
            0.5 * ((inner + spacing[0]) * (inner + spacing[0]) - inner * inner) * spacing[1];
      }
  } else {
    throw std::invalid_argument("native output geometry has no registered cell-measure kernel");
  }

  py::list box_rows;
  for (const auto& box : boxes)
    box_rows.append(py::make_tuple(box[0], box[1], box[2], box[3]));
  // Prevent mutation while the native Writer borrows these exact buffers.
  valid_cells.attr("setflags")(false);
  coverage.attr("setflags")(false);
  cell_volumes.attr("setflags")(false);
  py::dict result;
  result["topology_epoch"] = topology_epoch;
  result["boxes"] = std::move(box_rows);
  result["valid_cells"] = std::move(valid_cells);
  result["coverage"] = std::move(coverage);
  result["cell_volumes"] = std::move(cell_volumes);
  return result;
}

}  // namespace pops::python::detail
