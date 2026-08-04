/// @file
/// @brief Thin ranked coupling facade for canonical clustering and regrid transactions.

#pragma once

#include <pops/amr/tagging/clustering_provider.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace pops::coupling::amr {

/// Couples a prepared clustering provider to one immutable-rank AMR runtime.
///
/// This facade does not grow tags, reconstruct ownership, or install hierarchy state itself. The
/// provider authenticates its shards and the runtime owns both ownership preparation and atomic
/// topology publication.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrRegridCoupler {
 public:
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using field_type = typename runtime_type::field_type;
  using prepared_type = ::pops::amr::regridding::PreparedRegrid<Dim>;
  using provider_type = ::pops::amr::tagging::ClusterProvider<Dim>;

  AmrRegridCoupler(runtime_type& runtime, const provider_type& provider)
      : runtime_(&runtime), provider_(&provider) {
    if (provider.provider_identity().empty())
      throw std::invalid_argument("AMR regrid coupler requires an identified cluster provider");
  }

  runtime_type& runtime() const noexcept { return *runtime_; }
  std::string_view provider_identity() const noexcept { return provider_->provider_identity(); }

  prepared_type prepare(std::size_t parent_level, ::pops::amr::RefinementRatio<Dim> ratio,
                        std::span<const ::pops::amr::tagging::TagMask<Dim>> tag_shards,
                        const ::pops::amr::tagging::ClusterOptions<Dim>& cluster_options,
                        ::pops::amr::regridding::RegridPreparationBudget preparation_budget,
                        const ExecutionLane& lane = ExecutionLane::world()) const {
    ::pops::amr::tagging::ClusterResult<Dim> clustered =
        provider_->cluster(tag_shards, cluster_options);
    if (clustered.identity.provider != provider_->provider_identity())
      throw std::invalid_argument("AMR cluster result changed its prepared provider identity");
    return runtime_->prepare_regrid(parent_level, ratio, std::move(clustered), preparation_budget,
                                    lane);
  }

  void publish(prepared_type prepared, std::optional<field_type> child_state) const {
    const int parent_level = prepared.source_level().level;
    if (parent_level < 0)
      throw std::invalid_argument("prepared AMR regrid has no source level");
    runtime_->publish_regrid(static_cast<std::size_t>(parent_level), std::move(prepared),
                             std::move(child_state));
  }

 private:
  runtime_type* runtime_ = nullptr;
  const provider_type* provider_ = nullptr;
};

}  // namespace pops::coupling::amr
