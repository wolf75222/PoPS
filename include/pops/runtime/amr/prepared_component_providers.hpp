/// @file
/// @brief Compile-time-ranked adapters for prepared AMR Tagger, Reflux, and Clustering providers.

#pragma once

#include <pops/amr/tagging/clustering_provider.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::amr {

/// Authenticated, non-owning input presented to a local Tagger kernel.
template <int Dim, class MemorySpace>
struct PreparedTaggingRequest {
  const MultiFab<Dim, MemorySpace>* state = nullptr;
  ::pops::amr::hierarchy::LevelStateSpatialContract<Dim> source_level{};
  ::pops::amr::tagging::TagMaskBudget budget{};
  std::string_view runtime_spatial_contract{};
};

template <int Dim, class MemorySpace>
using PreparedTaggerKernel = PreparedProvider<::pops::amr::tagging::TagMask<Dim>(
    const PreparedTaggingRequest<Dim, MemorySpace>&)>;

/// Local Tagger adapter.  It owns neither topology nor communicator collectives.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedTaggerComponent {
 public:
  using runtime_type = AmrRuntime<Dim, MemorySpace>;
  using mask_type = ::pops::amr::tagging::TagMask<Dim>;

  explicit PreparedTaggerComponent(PreparedTaggerKernel<Dim, MemorySpace> kernel)
      : kernel_(std::move(kernel)) {
    if (!kernel_)
      throw std::invalid_argument("prepared ND Tagger component requires a kernel");
    collective_contract_ = ranked_contract_("pops.amr.prepared-tagger", kernel_);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(kernel_); }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  mask_type apply(const runtime_type& runtime, std::size_t level,
                  ::pops::amr::tagging::TagMaskBudget budget) const {
    if (level >= runtime.hierarchy().num_levels())
      throw std::out_of_range("prepared ND Tagger level is outside the live hierarchy");
    const auto& source = runtime.hierarchy().level(level);
    PreparedTaggingRequest<Dim, MemorySpace> request{&source.field(), source.spatial_contract(),
                                                     budget, runtime.spatial_contract()};
    mask_type result = kernel_(request);
    if (result.level_identity() != source.layout().exact_identity() ||
        result.local_rank() != source.field().local_rank())
      throw std::runtime_error(
          "prepared ND Tagger returned tags for another layout or rank coordinate");
    return result;
  }

 private:
  template <class Kernel>
  static std::string ranked_contract_(std::string_view role, const Kernel& kernel) {
    ExactContractBuilder contract;
    contract.text(role)
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(kernel.collective_contract());
    return std::move(contract).release();
  }

  PreparedTaggerKernel<Dim, MemorySpace> kernel_{};
  std::string collective_contract_{};
};

/// Ranked wrapper around the canonical clustering authority.
template <int Dim>
class PreparedClusteringComponent {
 public:
  using provider_type = ::pops::amr::tagging::ClusterProvider<Dim>;
  using mask_type = ::pops::amr::tagging::TagMask<Dim>;
  using options_type = ::pops::amr::tagging::ClusterOptions<Dim>;
  using result_type = ::pops::amr::tagging::ClusterResult<Dim>;

  explicit PreparedClusteringComponent(std::shared_ptr<const provider_type> provider)
      : provider_(std::move(provider)) {
    if (!provider_ || provider_->provider_identity().empty())
      throw std::invalid_argument("prepared ND Clustering component requires an identity");
    ExactContractBuilder contract;
    contract.text("pops.amr.prepared-clustering")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(provider_->provider_identity());
    collective_contract_ = std::move(contract).release();
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(provider_); }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  result_type cluster(std::span<const mask_type> shards, const options_type& options) const {
    result_type result = provider_->cluster(shards, options);
    if (std::string_view(result.identity.provider) != provider_->provider_identity() ||
        result.identity.boxes != result.boxes.boxes())
      throw std::runtime_error(
          "prepared ND Clustering provider returned an unauthenticated result");
    return result;
  }

 private:
  std::shared_ptr<const provider_type> provider_{};
  std::string collective_contract_{};
};

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
