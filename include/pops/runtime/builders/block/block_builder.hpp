/// @file
/// @brief Prepared compile-time-ranked spatial block provider boundary.

#pragma once

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::builders {

/// Geometry is a provider capability, not a dimension-selection mechanism.
enum class BlockGeometryCapability : std::uint8_t {
  Cartesian = 0,
  Polar = 1,
  EmbeddedBoundary = 2,
};

/// Dynamic plan facts checked once against a rank-specialized provider.
template <int Dim>
struct BlockProviderRequest {
  int dimension = 0;
  BlockGeometryCapability geometry = BlockGeometryCapability::Cartesian;
  int state_components = 0;
  Extent<Dim> available_ghosts{};
  bool amr = false;
};

/// Immutable capabilities declared by the concrete spatial provider.
template <int Dim>
struct BlockProviderCapabilities {
  static_assert(Dim >= 1 && Dim <= 3,
                "BlockProviderCapabilities only supports dimensions 1, 2, and 3");

  BlockGeometryCapability geometry = BlockGeometryCapability::Cartesian;
  int state_components = 0;
  Extent<Dim> required_ghosts{};
  bool supports_amr = false;

  [[nodiscard]] PreparedProviderSupport support(const BlockProviderRequest<Dim>& request) const {
    if (request.dimension != Dim)
      return PreparedProviderSupport::reject(1, "block provider dimension differs from the plan");
    if (request.geometry != geometry)
      return PreparedProviderSupport::reject(2, "block provider geometry differs from the plan");
    if (request.state_components != state_components)
      return PreparedProviderSupport::reject(
          3, "block provider component count differs from the plan");
    if (request.amr && !supports_amr)
      return PreparedProviderSupport::reject(4, "block provider does not support AMR execution");
    for (int axis = 0; axis < Dim; ++axis)
      if (request.available_ghosts[axis] < required_ghosts[axis])
        return PreparedProviderSupport::reject(5, "block provider requires a wider ghost region");
    return PreparedProviderSupport::accept();
  }
};

/// Non-owning input to one already prepared spatial residual evaluation.
template <int Dim, class MemorySpace>
struct BlockResidualRequest {
  const MultiFab<Dim, MemorySpace>* state = nullptr;
  std::string_view spatial_contract{};
  std::string_view state_identity{};
};

template <int Dim, class MemorySpace>
using PreparedBlockResidualKernel =
    PreparedProvider<MultiFab<Dim, MemorySpace>(const BlockResidualRequest<Dim, MemorySpace>&)>;

/// One immutable spatial specialization selected after Python resolved the dimension and geometry.
///
/// Cartesian, polar, and embedded-boundary implementations may all satisfy this protocol, but they
/// remain different prepared sources with different identities and capability contracts.  The
/// generic builder never switches algorithms on `Dim` and never silently routes an unsupported
/// geometry through the Cartesian implementation.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedBlockOperator {
 public:
  using field_type = MultiFab<Dim, MemorySpace>;
  using capabilities_type = BlockProviderCapabilities<Dim>;
  using kernel_type = PreparedBlockResidualKernel<Dim, MemorySpace>;

  PreparedBlockOperator(capabilities_type capabilities, kernel_type residual,
                        std::string spatial_contract, std::string state_identity)
      : capabilities_(capabilities),
        residual_(std::move(residual)),
        spatial_contract_(std::move(spatial_contract)),
        state_identity_(std::move(state_identity)) {
    validate_prepared_();
    ExactContractBuilder contract;
    contract.text("pops.prepared-block-operator")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(capabilities_.geometry)
        .scalar(std::int32_t{capabilities_.state_components})
        .scalar(capabilities_.supports_amr)
        .text(spatial_contract_)
        .text(state_identity_);
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(std::int64_t{capabilities_.required_ghosts[axis]});
    contract.bytes(residual_.collective_contract());
    collective_contract_ = std::move(contract).release();
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(residual_); }
  [[nodiscard]] static constexpr int dimension() noexcept { return Dim; }
  [[nodiscard]] const capabilities_type& capabilities() const noexcept { return capabilities_; }
  [[nodiscard]] std::string_view spatial_contract() const noexcept { return spatial_contract_; }
  [[nodiscard]] std::string_view state_identity() const noexcept { return state_identity_; }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  [[nodiscard]] PreparedProviderSupport support(const BlockProviderRequest<Dim>& request) const {
    return capabilities_.support(request);
  }

  /// Materialize a complete candidate residual.  The provider cannot partially mutate live state;
  /// publication remains the caller's transaction.
  field_type assemble_residual(const field_type& state, std::string_view live_spatial_contract,
                               std::string_view live_state_identity) const {
    if (live_spatial_contract != spatial_contract_ || live_state_identity != state_identity_)
      throw std::invalid_argument("prepared block operator is stale for the live spatial state");
    if (state.ncomp() != capabilities_.state_components)
      throw std::invalid_argument("prepared block state component count differs from its provider");
    for (int axis = 0; axis < Dim; ++axis)
      if (state.ghosts()[axis] < capabilities_.required_ghosts[axis])
        throw std::invalid_argument("prepared block state has insufficient ghost storage");

    field_type candidate = residual_(
        BlockResidualRequest<Dim, MemorySpace>{&state, live_spatial_contract, live_state_identity});
    if (candidate.layout() != state.layout() || candidate.distribution() != state.distribution() ||
        candidate.local_rank() != state.local_rank() || candidate.ncomp() != state.ncomp() ||
        candidate.ghosts() != state.ghosts() || candidate.local_size() != state.local_size() ||
        candidate.shares_storage_with(state))
      throw std::runtime_error(
          "prepared block provider returned a residual for another layout or ownership");
    return candidate;
  }

 private:
  void validate_prepared_() const {
    if (!residual_)
      throw std::invalid_argument("prepared block operator requires a residual provider");
    if (capabilities_.state_components < 1)
      throw std::invalid_argument("prepared block provider requires a positive component count");
    for (int axis = 0; axis < Dim; ++axis)
      if (capabilities_.required_ghosts[axis] < 0)
        throw std::invalid_argument("prepared block provider ghost widths must be non-negative");
    if (spatial_contract_.empty() || state_identity_.empty())
      throw std::invalid_argument(
          "prepared block operator requires exact spatial and state identities");
  }

  capabilities_type capabilities_{};
  kernel_type residual_{};
  std::string spatial_contract_{};
  std::string state_identity_{};
  std::string collective_contract_{};
};

}  // namespace pops::runtime::builders
