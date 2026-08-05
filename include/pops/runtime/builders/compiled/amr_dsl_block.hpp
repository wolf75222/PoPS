/// @file
/// @brief Immutable binding of one ranked spatial block provider to a live AMR level.

#pragma once

#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/builders/block/block_builder.hpp>
#include <pops/runtime/builders/compiled/generated_amr_system_block.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::builders {

template <int Dim>
struct AmrDslBlockSpec {
  std::size_t level = 0;
  std::string state_identity{};
};

/// Prepared AMR spatial block with one immutable dimension, provider, and topology generation.
///
/// This object exposes spatial residual assembly only.  Program owns time methods and cadence;
/// AmrRuntime owns hierarchy mutation, transfer, load balance, and reflux.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedAmrDslBlock {
 public:
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using field_type = typename runtime_type::field_type;
  using operator_type = PreparedBlockOperator<Dim, MemorySpace>;

  PreparedAmrDslBlock(runtime_type& runtime, AmrDslBlockSpec<Dim> spec,
                      operator_type spatial_operator)
      : runtime_(&runtime),
        spec_(std::move(spec)),
        spatial_operator_(std::move(spatial_operator)),
        bound_spatial_contract_(runtime.spatial_contract()),
        topology_epoch_(runtime.topology_epoch()),
        materialization_generation_(runtime.materialization_generation()) {
    validate_prepared_();
    ExactContractBuilder contract;
    contract.text("pops.prepared-amr-dsl-block")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(spec_.level))
        .text(spec_.state_identity)
        .bytes(bound_spatial_contract_)
        .scalar(topology_epoch_)
        .scalar(materialization_generation_)
        .bytes(spatial_operator_.collective_contract());
    collective_contract_ = std::move(contract).release();
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return runtime_ != nullptr && static_cast<bool>(spatial_operator_);
  }
  [[nodiscard]] static constexpr int dimension() noexcept { return Dim; }
  [[nodiscard]] std::size_t level() const noexcept { return spec_.level; }
  [[nodiscard]] std::string_view state_identity() const noexcept { return spec_.state_identity; }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  const operator_type& spatial_operator() const noexcept { return spatial_operator_; }

  /// Assemble one complete residual candidate against the exact topology generation captured at
  /// preparation.  A regrid or rebalance makes this object stale and requires re-preparation.
  field_type assemble_residual() const {
    require_live_();
    return spatial_operator_.assemble_residual(runtime_->hierarchy().state(spec_.level),
                                               runtime_->spatial_contract(), spec_.state_identity);
  }

 private:
  void validate_prepared_() const {
    if (spec_.level >= runtime_->hierarchy().num_levels())
      throw std::out_of_range("prepared AMR DSL block level is outside the hierarchy");
    if (spec_.state_identity.empty() || spec_.state_identity != spatial_operator_.state_identity())
      throw std::invalid_argument("prepared AMR DSL block state identity is inconsistent");
    if (bound_spatial_contract_ != spatial_operator_.spatial_contract())
      throw std::invalid_argument("prepared AMR DSL block spatial contract is inconsistent");
    const auto& state = runtime_->hierarchy().state(spec_.level);
    const auto& capabilities = spatial_operator_.capabilities();
    const PreparedProviderSupport support = spatial_operator_.support(
        BlockProviderRequest<Dim>{Dim, capabilities.geometry, state.ncomp(), state.ghosts(), true});
    if (!support.accepted())
      throw std::invalid_argument(std::string(support.reason));
  }

  void require_live_() const {
    if (runtime_ == nullptr || bound_spatial_contract_ != runtime_->spatial_contract() ||
        topology_epoch_ != runtime_->topology_epoch() ||
        materialization_generation_ != runtime_->materialization_generation())
      throw std::invalid_argument("prepared AMR DSL block is stale after a topology mutation");
    if (spec_.level >= runtime_->hierarchy().num_levels())
      throw std::out_of_range("prepared AMR DSL block level is no longer live");
  }

  runtime_type* runtime_ = nullptr;
  AmrDslBlockSpec<Dim> spec_{};
  operator_type spatial_operator_;
  std::string bound_spatial_contract_{};
  std::uint64_t topology_epoch_ = 0;
  std::uint64_t materialization_generation_ = 0;
  std::string collective_contract_{};
};

}  // namespace pops::runtime::builders
