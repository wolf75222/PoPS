/// @file
/// @brief Prepared facade for the canonical ranked clustering provider.

#pragma once

#include <pops/amr/tagging/clustering_provider.hpp>
#include <pops/core/identity/prepared_provider.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::amr {

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

}  // namespace pops::runtime::amr
