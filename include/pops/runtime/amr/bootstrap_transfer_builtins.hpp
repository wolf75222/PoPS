/// @file
/// @brief Named bootstrap routes over the canonical compile-time-ranked transfer authority.

#pragma once

#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace pops::runtime::amr {

template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedDivergencePreservingFaceTransfer<Dim>
prepare_divergence_preserving_face(
    const AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    std::type_identity_t<std::array<FieldView<const Real, Dim>, Dim>> parent_faces,
    std::type_identity_t<std::array<FieldView<Real, Dim>, Dim>> child_faces,
    const Box<Dim>& child_cell_region, ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  return ::pops::numerics::time::amr::prepare_divergence_preserving_faces(
      runtime, parent_level, parent_faces, child_faces, child_cell_region, mapping, components);
}

template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_node_multilinear(
    const AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent_nodes, FieldView<Real, Dim> child_nodes,
    const Box<Dim>& child_node_region, ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  return ::pops::numerics::time::amr::prepare_node_multilinear(
      runtime, parent_level, parent_nodes, child_nodes, child_node_region, mapping, components);
}

template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedLinearTemporalInterpolation<Dim> prepare_linear_time_interpolation(
    const AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> older, FieldView<const Real, Dim> newer,
    FieldView<Real, Dim> candidate, const Box<Dim>& destination_region,
    const ::pops::amr::transfer::QualifiedTemporalState& older_state,
    const ::pops::amr::transfer::QualifiedTemporalState& newer_state,
    const ::pops::amr::transfer::QualifiedTemporalState& target_state,
    ::pops::amr::transfer::TemporalComponentRange components = {}) {
  return ::pops::numerics::time::amr::prepare_linear_time_interpolation(
      runtime, parent_level, older, newer, candidate, destination_region, older_state, newer_state,
      target_state, components);
}

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

/// Prepare explicitly requested first-order conservative parent injection.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_conservative_injection(
    const AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& child_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  return ::pops::numerics::time::amr::prepare_constant_injection(
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

/// Prepare the authenticated fifth-order coarse/fine cell-average interpolation route.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_conservative_polynomial5_coarse_fine(
    const AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& ghost_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  return ::pops::numerics::time::amr::prepare_fifth_order_fill_patch(
      runtime, parent_level, parent, child, ghost_region, mapping, components);
}

}  // namespace pops::runtime::amr
