/// @file
/// @brief Prepared facade for authenticated AMR reflux corrections.

#pragma once

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::amr {

/// Input to a local Reflux correction kernel after the AMR runtime authenticated and reconciled
/// the complete metric-time face product.
template <int Dim, class Payload>
struct PreparedRefluxRequest {
  const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>* key = nullptr;
  const ::pops::amr::reflux::MetricFaceReflux<Payload>* reconciliation = nullptr;
  double coarse_cell_measure = 0.0;
  ::pops::amr::reflux::CoarseCellFaceSide side = ::pops::amr::reflux::CoarseCellFaceSide::Lower;
};

template <int Dim, class Payload>
using PreparedRefluxKernel = PreparedProvider<Payload(const PreparedRefluxRequest<Dim, Payload>&)>;

/// Reflux adapter that delegates every topology, clock, and metric decision to AmrRuntime first.
template <int Dim, class Payload>
class PreparedRefluxComponent {
 public:
  explicit PreparedRefluxComponent(PreparedRefluxKernel<Dim, Payload> kernel)
      : kernel_(std::move(kernel)) {
    if (!kernel_)
      throw std::invalid_argument("prepared ND Reflux component requires a kernel");
    ExactContractBuilder contract;
    contract.text("pops.amr.prepared-reflux")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(kernel_.collective_contract());
    collective_contract_ = std::move(contract).release();
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(kernel_); }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  template <class MemorySpace, class Axpy>
  Payload correction(const AmrRuntime<Dim, MemorySpace>& runtime,
                     const ::pops::amr::reflux::TransactionalFaceFluxLedger<Dim, Payload>& ledger,
                     const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>& key,
                     std::string_view state_identity,
                     const ::pops::amr::reflux::FaceRefinementMapping<Dim>& mapping,
                     const ::pops::amr::reflux::MetricRefluxBudget& budget,
                     double coarse_cell_measure, ::pops::amr::reflux::CoarseCellFaceSide side,
                     Axpy&& axpy) const {
    if (!(coarse_cell_measure > 0.0) || !std::isfinite(coarse_cell_measure))
      throw std::invalid_argument(
          "prepared ND Reflux correction requires a finite positive cell measure");
    const auto reconciliation = runtime.reconcile_reflux(ledger, key, state_identity, mapping,
                                                         budget, std::forward<Axpy>(axpy));
    return kernel_(
        PreparedRefluxRequest<Dim, Payload>{&key, &reconciliation, coarse_cell_measure, side});
  }

 private:
  PreparedRefluxKernel<Dim, Payload> kernel_{};
  std::string collective_contract_{};
};

}  // namespace pops::runtime::amr
