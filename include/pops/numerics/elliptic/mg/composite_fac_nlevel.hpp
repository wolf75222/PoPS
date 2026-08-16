/// @file
/// @brief Prepared exact-ranked transfer and masking primitives used by composite FAC.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/field_view.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::elliptic::mg::fac_detail {

template <int Dim>
struct SetScalarKernel {
  FieldView<Real, Dim> values{};
  Real value = Real(0);

  POPS_HD void operator()(const Index<Dim>& index) const { values(index, 0) = value; }
};

template <int Dim>
struct MaskResidualKernel {
  FieldView<Real, Dim> residual{};
  FieldView<const Real, Dim> covered{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (covered(index, 0) >= Real(0.5))
      residual(index, 0) = Real(0);
  }
};

template <int Dim>
struct MaskedAddKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> increment{};
  FieldView<const Real, Dim> covered{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (covered(index, 0) < Real(0.5))
      destination(index, 0) += increment(index, 0);
  }
};

template <int Dim>
struct MaskedJacobiKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> iterate{};
  FieldView<const Real, Dim> right_hand_side{};
  FieldView<const Real, Dim> covered{};
  Real inverse_spacing_squared[Dim]{};
  Real inverse_diagonal = Real(0);
  Real relaxation = Real(2) / Real(3);
  Real reaction = Real(0);

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (covered(index, 0) >= Real(0.5)) {
      destination(index, 0) = iterate(index, 0);
      return;
    }
    Real image = reaction * iterate(index, 0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = index;
      Index<Dim> upper = index;
      --lower[axis];
      ++upper[axis];
      image += (Real(2) * iterate(index, 0) - iterate(lower, 0) - iterate(upper, 0)) *
               inverse_spacing_squared[axis];
    }
    destination(index, 0) =
        iterate(index, 0) + relaxation * inverse_diagonal * (right_hand_side(index, 0) - image);
  }
};

template <int Dim>
std::vector<Box<Dim>> subtract_box(const Box<Dim>& subject, const Box<Dim>& cut) {
  const Box<Dim> overlap = subject.intersect(cut);
  if (overlap.empty())
    return {subject};
  if (overlap == subject)
    return {};
  std::vector<Box<Dim>> result;
  result.reserve(static_cast<std::size_t>(2 * Dim));
  Box<Dim> remainder = subject;
  for (int axis = 0; axis < Dim; ++axis) {
    if (remainder.lo[axis] < overlap.lo[axis]) {
      Box<Dim> lower = remainder;
      lower.hi[axis] = overlap.lo[axis] - 1;
      result.push_back(lower);
      remainder.lo[axis] = overlap.lo[axis];
    }
    if (overlap.hi[axis] < remainder.hi[axis]) {
      Box<Dim> upper = remainder;
      upper.lo[axis] = overlap.hi[axis] + 1;
      result.push_back(upper);
      remainder.hi[axis] = overlap.hi[axis];
    }
  }
  return result;
}

template <int Dim>
void subtract_from_regions(std::vector<Box<Dim>>& regions, const Box<Dim>& cut) {
  std::vector<Box<Dim>> next;
  for (const Box<Dim>& region : regions) {
    std::vector<Box<Dim>> pieces = subtract_box(region, cut);
    next.insert(next.end(), pieces.begin(), pieces.end());
  }
  regions = std::move(next);
}

template <int Dim>
using CellTransfer = ::pops::amr::transfer::PreparedTransfer<Dim>;

/// Tensor-product quadratic interpolation of a cell-centered point field at a fine cell center.
/// Each axis uses its parent cell and its two immediate neighbors, retaining all mixed terms.
template <int Dim>
struct QuadraticInterpolationTransfer {
  FieldView<const Real, Dim> coarse{};
  FieldView<Real, Dim> fine{};
  Box<Dim> destination{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  ::pops::amr::transfer::IndexMapping<Dim> mapping{};
  Box<Dim> sample_domain{};

  POPS_HD void operator()(const Index<Dim>& fine_index) const {
    Index<Dim> parent{};
    Index<Dim> child{};
    Index<Dim> sample = fine_index;
    if (!sample_domain.empty()) {
      for (int axis = 0; axis < Dim; ++axis) {
        const int lo = sample_domain.lo[axis];
        const int hi = sample_domain.hi[axis];
        const int length = hi - lo + 1;
        if (length <= 0)
          continue;
        while (sample[axis] < lo)
          sample[axis] += length;
        while (sample[axis] > hi)
          sample[axis] -= length;
      }
    }
    ::pops::amr::transfer::detail::fine_parent_and_child(sample, ratio, mapping, parent, child);
    Real weights[Dim][3]{};
    for (int axis = 0; axis < Dim; ++axis) {
      const Real s =
          static_cast<Real>(2 * child[axis] + 1 - ratio[axis]) / static_cast<Real>(2 * ratio[axis]);
      const Real d = s * s;
      weights[axis][0] = (d - s) / Real(2);
      weights[axis][1] = Real(1) - d;
      weights[axis][2] = (d + s) / Real(2);
    }

    Real value = Real(0);
    constexpr int points = Dim == 1 ? 3 : Dim == 2 ? 9 : 27;
    for (int ordinal = 0; ordinal < points; ++ordinal) {
      int remainder = ordinal;
      Index<Dim> source_index = parent;
      Real weight = Real(1);
      for (int axis = 0; axis < Dim; ++axis) {
        const int stencil = remainder % 3;
        remainder /= 3;
        source_index[axis] += stencil - 1;
        weight *= weights[axis][stencil];
      }
      if (weight != Real(0))
        value += weight * coarse_sample_(source_index);
    }
    fine(fine_index, 0) = value;
  }

  // Quadratic C/F interpolation at a domain face needs the parent cell two
  // interiors away.  A one-ghost parent allocation does not contain that
  // sample; wrap through the valid interior so a periodic coarse level supplies
  // the same value its halo would have carried with two ghosts.
  POPS_HD Real coarse_sample_(Index<Dim> index) const {
    for (int axis = 0; axis < Dim; ++axis) {
      const int allocated_lo = coarse.origin[axis];
      const int allocated_extent = static_cast<int>(coarse.extents[axis]);
      if (index[axis] >= allocated_lo && index[axis] < allocated_lo + allocated_extent)
        continue;
      const int valid_lo = allocated_lo + 1;
      const int valid_len = allocated_extent - 2;
      if (valid_len <= 0)
        continue;
      int shifted = (index[axis] - valid_lo) % valid_len;
      if (shifted < 0)
        shifted += valid_len;
      index[axis] = valid_lo + shifted;
    }
    return coarse(index, 0);
  }
};

template <int Dim>
void execute_quadratic_interpolations(
    const std::vector<QuadraticInterpolationTransfer<Dim>>& transfers) {
  for (const QuadraticInterpolationTransfer<Dim>& transfer : transfers)
    for_each_cell(transfer.destination, transfer);
  Kokkos::fence();
}

/// Add the conservative coarse/fine flux replacement to one uncovered parent cell.  The parent
/// operator contributes one coarse face difference; the composite operator replaces it with the
/// corresponding sum of fine face differences.  The factor converts a fine face contribution to
/// the parent control volume for arbitrary anisotropic refinement ratios.
template <int Dim>
struct FluxMismatchTransfer {
  FieldView<const Real, Dim> parent{};
  FieldView<const Real, Dim> fine{};
  FieldView<Real, Dim> residual{};
  FieldView<const Real, Dim> covered{};
  Box<Dim> destination{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  int normal_axis = 0;
  int child_side = 0;
  Real inverse_spacing_squared = Real(0);
  Real fine_face_weight = Real(0);
  Real sign = Real(1);
  Index<Dim> geometry_shift{};

  POPS_HD void operator()(const Index<Dim>& coarse_index) const {
    if (covered(coarse_index, 0) >= Real(0.5))
      return;

    Index<Dim> parent_neighbor = coarse_index;
    parent_neighbor[normal_axis] -= child_side;
    const Real coarse_face = parent(coarse_index, 0) - parent(parent_neighbor, 0);

    Index<Dim> geometry = coarse_index;
    for (int axis = 0; axis < Dim; ++axis)
      geometry[axis] += geometry_shift[axis];
    Index<Dim> fine_inner{};
    for (int axis = 0; axis < Dim; ++axis)
      fine_inner[axis] = geometry[axis] * ratio[axis];
    fine_inner[normal_axis] += child_side < 0 ? ratio[normal_axis] : -1;
    Index<Dim> fine_ghost = fine_inner;
    fine_ghost[normal_axis] += child_side;

    Real fine_faces = Real(0);
    std::int64_t tangential_count = 1;
    for (int axis = 0; axis < Dim; ++axis)
      if (axis != normal_axis)
        tangential_count *= ratio[axis];
    for (std::int64_t ordinal = 0; ordinal < tangential_count; ++ordinal) {
      std::int64_t remainder = ordinal;
      Index<Dim> fine_face_inner = fine_inner;
      Index<Dim> fine_face_ghost = fine_ghost;
      for (int axis = 0; axis < Dim; ++axis) {
        if (axis == normal_axis)
          continue;
        const int child = static_cast<int>(remainder % ratio[axis]);
        remainder /= ratio[axis];
        fine_face_inner[axis] += child;
        fine_face_ghost[axis] += child;
      }
      fine_faces += fine(fine_face_ghost, 0) - fine(fine_face_inner, 0);
    }
    residual(coarse_index, 0) +=
        sign * inverse_spacing_squared * (coarse_face - fine_face_weight * fine_faces);
  }
};

template <int Dim>
void execute_flux_mismatches(const std::vector<FluxMismatchTransfer<Dim>>& transfers) {
  for (const FluxMismatchTransfer<Dim>& transfer : transfers)
    for_each_cell(transfer.destination, transfer);
  Kokkos::fence();
}

template <int Dim>
void execute_transfers(const std::vector<CellTransfer<Dim>>& transfers) {
  for (const CellTransfer<Dim>& transfer : transfers)
    for_each_cell(transfer.destination_region(), transfer);
  Kokkos::fence();
}

template <int Dim>
void require_ratio(const ::pops::amr::RefinementRatio<Dim>& ratio) {
  if (ratio.is_identity())
    throw std::invalid_argument("composite FAC transition must refine at least one axis");
}

}  // namespace pops::elliptic::mg::fac_detail
