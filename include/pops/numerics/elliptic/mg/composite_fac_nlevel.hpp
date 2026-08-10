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

template <int Dim>
struct InjectionTransfer {
  FieldView<const Real, Dim> coarse{};
  FieldView<Real, Dim> fine{};
  Box<Dim> destination{};
  ::pops::amr::RefinementRatio<Dim> ratio{};
  ::pops::amr::transfer::IndexMapping<Dim> mapping{};

  POPS_HD void operator()(const Index<Dim>& fine_index) const {
    Index<Dim> parent{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t relative =
          static_cast<std::int64_t>(fine_index[axis]) - mapping.fine_origin[axis];
      std::int64_t quotient = relative / ratio[axis];
      if (relative % ratio[axis] < 0)
        --quotient;
      parent[axis] =
          static_cast<int>(static_cast<std::int64_t>(mapping.coarse_origin[axis]) + quotient);
    }
    fine(fine_index, 0) = coarse(parent, 0);
  }
};

template <int Dim>
void execute_injections(const std::vector<InjectionTransfer<Dim>>& transfers) {
  for (const InjectionTransfer<Dim>& transfer : transfers)
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
