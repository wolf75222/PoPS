/// @file
/// @brief Program-facing spatial facade for one compile-time-ranked AMR runtime.

#pragma once

#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops::runtime::program {

template <int Dim>
struct ProgramSpatialSnapshot {
  std::string spatial_contract;
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;

  bool operator==(const ProgramSpatialSnapshot&) const = default;
};

/// Spatial services available to a Program after Python selected and bound one native rank.
///
/// Time methods, equation assembly, provider resolution, and Python binding do not live here. The
/// context preserves the AMR runtime as the sole topology authority and exposes only prepared
/// transactions or authenticated state views.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrProgramContext {
 public:
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using hierarchy_type = typename runtime_type::hierarchy_type;
  using field_type = typename runtime_type::field_type;
  using subcycle_plan_type = ::pops::numerics::time::amr::PreparedAmrSubcyclePlan<Dim, MemorySpace>;

  explicit AmrProgramContext(runtime_type& runtime) : runtime_(&runtime) {}

  static constexpr int dimension = Dim;

  runtime_type& runtime() const noexcept { return *runtime_; }
  hierarchy_type& hierarchy() const noexcept { return runtime_->hierarchy(); }
  const ::pops::amr::hierarchy::LevelLayout<Dim>& layout(std::size_t level) const {
    return runtime_->hierarchy().layout(level);
  }
  field_type& state(std::size_t level) const { return runtime_->hierarchy().state(level); }

  ProgramSpatialSnapshot<Dim> spatial_snapshot() const {
    return {std::string(runtime_->spatial_contract()), runtime_->topology_epoch(),
            runtime_->materialization_generation()};
  }

  void require_live(const ProgramSpatialSnapshot<Dim>& snapshot) const {
    if (snapshot.spatial_contract != runtime_->spatial_contract() ||
        snapshot.topology_epoch != runtime_->topology_epoch() ||
        snapshot.materialization_generation != runtime_->materialization_generation())
      throw std::invalid_argument("AMR Program spatial snapshot is stale");
  }

  subcycle_plan_type prepare_subcycling(
      std::span<const int> temporal_substeps,
      ::pops::numerics::time::amr::AmrSubcyclePreparationBudget budget) const {
    return subcycle_plan_type::prepare(*runtime_, temporal_substeps, budget);
  }

  ::pops::amr::regridding::PreparedRegrid<Dim> prepare_regrid(
      std::size_t parent_level, ::pops::amr::RefinementRatio<Dim> ratio,
      ::pops::amr::tagging::ClusterResult<Dim> clustered,
      ::pops::amr::regridding::RegridPreparationBudget preparation_budget,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    return runtime_->prepare_regrid(parent_level, ratio, std::move(clustered), preparation_budget,
                                    lane);
  }

  void publish_regrid(::pops::amr::regridding::PreparedRegrid<Dim> prepared,
                      std::optional<field_type> child_state) const {
    const int parent_level = prepared.source_level().level;
    if (parent_level < 0)
      throw std::invalid_argument("AMR Program regrid has no source level");
    runtime_->publish_regrid(static_cast<std::size_t>(parent_level), std::move(prepared),
                             std::move(child_state));
  }

  PreparedRebalanceDecision<Dim> prepare_rebalance(
      std::size_t level, ResourceEstimates estimates,
      parallel::LoadBalancePreparationBudget preparation_budget, const RebalancePolicy& policy,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    return runtime_->prepare_rebalance(level, estimates, preparation_budget, policy, lane);
  }

  PreparedRebalanceDecision<Dim> prepare_rebalance(
      std::size_t level, ResourceEstimates estimates,
      parallel::LoadBalancePreparationBudget preparation_budget,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    return runtime_->prepare_rebalance(level, estimates, preparation_budget, lane);
  }

  void apply_rebalance(std::size_t level, PreparedRebalanceDecision<Dim> decision,
                       field_type remapped_state) const {
    runtime_->apply_rebalance(level, std::move(decision), std::move(remapped_state));
  }

  template <class Payload, class Axpy>
  ::pops::amr::reflux::MetricFaceReflux<Payload> reconcile_reflux(
      const ::pops::amr::reflux::TransactionalFaceFluxLedger<Dim, Payload>& ledger,
      const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>& key, std::string_view state_identity,
      const ::pops::amr::reflux::FaceRefinementMapping<Dim>& mapping,
      const ::pops::amr::reflux::MetricRefluxBudget& budget, Axpy&& axpy) const {
    return runtime_->reconcile_reflux(ledger, key, state_identity, mapping, budget,
                                      std::forward<Axpy>(axpy));
  }

 private:
  runtime_type* runtime_ = nullptr;
};

}  // namespace pops::runtime::program
