/// @file
/// @brief Ranked AMR transfer preparation and execution helpers.

#pragma once

#include <pops/amr/transfer/nd/transfer_provider.hpp>
#include <pops/mesh/execution/for_each.hpp>

namespace pops::numerics::time::amr {

namespace detail {

template <int Dim>
struct PreparedTransferKernel {
  ::pops::amr::transfer::nd::PreparedTransfer<Dim> transfer;

  POPS_HD void operator()(const Index<Dim>& index) const { transfer(index); }
};

}  // namespace detail

/// Execute one authenticated transfer on an explicit execution-space instance.
template <class ExecutionSpace, int Dim>
void execute_prepared_transfer(const ExecutionSpace& execution,
                               const ::pops::amr::transfer::nd::PreparedTransfer<Dim>& prepared) {
  for_each_cell(execution, prepared.destination_region(),
                detail::PreparedTransferKernel<Dim>{prepared});
}

/// Execute one authenticated transfer on the default execution-space instance.
template <int Dim>
void execute_prepared_transfer(const ::pops::amr::transfer::nd::PreparedTransfer<Dim>& prepared) {
  for_each_cell(prepared.destination_region(), detail::PreparedTransferKernel<Dim>{prepared});
}

template <int Dim>
auto prepare_average_down(FieldView<const Real, Dim> fine, FieldView<Real, Dim> parent,
                          const Box<Dim>& parent_region,
                          ::pops::amr::transfer::nd::RefinementRatio<Dim> ratio,
                          ::pops::amr::transfer::nd::IndexMapping<Dim> mapping = {},
                          ::pops::amr::transfer::nd::ComponentRange components = {}) {
  const auto provider = ::pops::amr::transfer::nd::TransferProvider<
      Dim, ::pops::amr::transfer::nd::Centering::Cell>::conservative_restriction();
  return provider.prepare(fine, parent, parent_region, ratio, mapping, components);
}

template <int Dim>
auto prepare_linear_prolongation(FieldView<const Real, Dim> parent, FieldView<Real, Dim> child,
                                 const Box<Dim>& child_region,
                                 ::pops::amr::transfer::nd::RefinementRatio<Dim> ratio,
                                 ::pops::amr::transfer::nd::IndexMapping<Dim> mapping = {},
                                 ::pops::amr::transfer::nd::ComponentRange components = {}) {
  const auto provider = ::pops::amr::transfer::nd::TransferProvider<
      Dim, ::pops::amr::transfer::nd::Centering::Cell>::linear_prolongation();
  return provider.prepare(parent, child, child_region, ratio, mapping, components);
}

template <int Dim>
auto prepare_fill_patch(FieldView<const Real, Dim> parent, FieldView<Real, Dim> child,
                        const Box<Dim>& ghost_region,
                        ::pops::amr::transfer::nd::RefinementRatio<Dim> ratio,
                        ::pops::amr::transfer::nd::IndexMapping<Dim> mapping = {},
                        ::pops::amr::transfer::nd::ComponentRange components = {}) {
  const auto provider = ::pops::amr::transfer::nd::TransferProvider<
      Dim, ::pops::amr::transfer::nd::Centering::Cell>::coarse_fine_ghost_interpolation();
  return provider.prepare(parent, child, ghost_region, ratio, mapping, components);
}

}  // namespace pops::numerics::time::amr
