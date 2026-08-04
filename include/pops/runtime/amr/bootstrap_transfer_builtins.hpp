/// @file
/// @brief Named bootstrap routes over the canonical compile-time-ranked transfer authority.

#pragma once

#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>

#include <cstddef>

namespace pops::runtime::amr {

/// Prepare parent-to-child conservative linear prolongation.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_conservative_linear(
    const AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& child_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  return ::pops::numerics::time::amr::prepare_linear_prolongation(
      runtime, parent_level, parent, child, child_region, mapping, components);
}

/// Prepare fine-to-parent conservative volume restriction.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_volume_average(
    const AmrRuntime<Dim, MemorySpace>& runtime, std::size_t fine_level,
    FieldView<const Real, Dim> fine, FieldView<Real, Dim> parent, const Box<Dim>& parent_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  return ::pops::numerics::time::amr::prepare_average_down(runtime, fine_level, fine, parent,
                                                           parent_region, mapping, components);
}

/// Prepare parent-to-child coarse/fine ghost interpolation.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_conservative_coarse_fine(
    const AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& ghost_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  return ::pops::numerics::time::amr::prepare_fill_patch(runtime, parent_level, parent, child,
                                                         ghost_region, mapping, components);
}

}  // namespace pops::runtime::amr
