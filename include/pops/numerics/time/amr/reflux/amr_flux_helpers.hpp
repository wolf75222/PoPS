/// @file
/// @brief Ranked AMR transfer preparation and execution helpers.

#pragma once

#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <cstddef>
#include <stdexcept>
namespace pops::numerics::time::amr {

namespace detail {

template <int Dim>
struct PreparedTransferKernel {
  ::pops::amr::transfer::PreparedTransfer<Dim> transfer;

  POPS_HD void operator()(const Index<Dim>& index) const { transfer(index); }
};

}  // namespace detail

/// Execute an already authenticated transfer on an explicit execution-space instance.
template <class ExecutionSpace, int Dim>
void execute_prepared_transfer(const ExecutionSpace& execution,
                               const ::pops::amr::transfer::PreparedTransfer<Dim>& prepared) {
  for_each_cell(execution, prepared.destination_region(),
                detail::PreparedTransferKernel<Dim>{prepared});
}

/// Execute an already authenticated transfer on the default execution-space instance.
template <int Dim>
void execute_prepared_transfer(const ::pops::amr::transfer::PreparedTransfer<Dim>& prepared) {
  for_each_cell(prepared.destination_region(), detail::PreparedTransferKernel<Dim>{prepared});
}

/// Prepare conservative fine-to-parent restriction through the live AMR runtime authority.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_average_down(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t fine_level,
    FieldView<const Real, Dim> fine, FieldView<Real, Dim> parent, const Box<Dim>& parent_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (fine_level == 0)
    throw std::invalid_argument("AMR average-down requires a fine level");
  const std::size_t parent_level = fine_level - 1;
  return runtime.template prepare_transfer<::pops::amr::transfer::Centering::Cell>(
      fine_level, parent_level, runtime.hierarchy().level(fine_level).spatial_contract(),
      runtime.hierarchy().level(parent_level).spatial_contract(),
      ::pops::amr::transfer::TransferKind::ConservativeRestriction, fine, parent, parent_region,
      mapping, components);
}

/// Prepare parent-to-child linear prolongation through the live AMR runtime authority.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_linear_prolongation(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& child_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (parent_level >= runtime.hierarchy().num_levels() ||
      runtime.hierarchy().num_levels() - parent_level < 2)
    throw std::invalid_argument("AMR prolongation requires adjacent live levels");
  const std::size_t child_level = parent_level + 1;
  return runtime.template prepare_transfer<::pops::amr::transfer::Centering::Cell>(
      parent_level, child_level, runtime.hierarchy().level(parent_level).spatial_contract(),
      runtime.hierarchy().level(child_level).spatial_contract(),
      ::pops::amr::transfer::TransferKind::LinearProlongation, parent, child, child_region, mapping,
      components);
}

/// Prepare parent-to-child coarse/fine ghost interpolation through the live runtime authority.
template <int Dim, class MemorySpace>
::pops::amr::transfer::PreparedTransfer<Dim> prepare_fill_patch(
    const ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>& runtime, std::size_t parent_level,
    FieldView<const Real, Dim> parent, FieldView<Real, Dim> child, const Box<Dim>& ghost_region,
    ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
    ::pops::amr::transfer::ComponentRange components = {}) {
  if (parent_level >= runtime.hierarchy().num_levels() ||
      runtime.hierarchy().num_levels() - parent_level < 2)
    throw std::invalid_argument("AMR fill-patch preparation requires adjacent live levels");
  const std::size_t child_level = parent_level + 1;
  return runtime.template prepare_transfer<::pops::amr::transfer::Centering::Cell>(
      parent_level, child_level, runtime.hierarchy().level(parent_level).spatial_contract(),
      runtime.hierarchy().level(child_level).spatial_contract(),
      ::pops::amr::transfer::TransferKind::CoarseFineGhostInterpolation, parent, child,
      ghost_region, mapping, components);
}

}  // namespace pops::numerics::time::amr
