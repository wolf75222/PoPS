/// @file
/// @brief Compile-time-ranked multipatch AMR coupling facade.

#pragma once

#include <pops/coupling/amr/amr_regrid_coupler.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace pops::coupling::amr {

template <int Dim>
struct CoarseHierarchyRequest {
  Box<Dim> domain{};
  Extent<Dim> max_grid_size{};
  mesh::RankSpace<Dim> rank_space{};
  Index<Dim> local_rank{};
  int components = 0;
  Extent<Dim> ghosts{};
  mesh::BoxArrayValidationBudget layout_budget{};
  ::pops::amr::hierarchy::HierarchyValidationBudget hierarchy_budget{};
  parallel::LoadBalancePreparationBudget load_balance_budget{};
};

/// Materialize one coarse hierarchy from the same prepared ownership authority retained by the
/// runtime. No integer rank mapping is reconstructed at this boundary.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
::pops::amr::hierarchy::AmrHierarchy<Dim, MemorySpace> make_coarse_hierarchy(
    CoarseHierarchyRequest<Dim> request, const PreparedLoadBalanceAuthority<Dim>& load_balance,
    const ExecutionLane& lane = ExecutionLane::world()) {
  if (request.domain.empty() || !request.rank_space.contains(request.local_rank) ||
      request.components < 1)
    throw std::invalid_argument("coarse AMR hierarchy request is incomplete");
  mesh::BoxArray<Dim> patches =
      mesh::BoxArray<Dim>::from_domain(request.domain, request.max_grid_size);
  PreparedLoadBalanceResult<Dim> ownership =
      load_balance.prepare(patches, request.rank_space, request.load_balance_budget, {}, lane);
  ::pops::amr::hierarchy::LevelLayout<Dim> layout(
      0, request.domain, patches, ownership.plan().distribution(),
      ::pops::amr::RefinementRatio<Dim>{}, request.layout_budget);
  MultiFab<Dim, MemorySpace> field(std::move(patches), ownership.plan().distribution(),
                                   request.local_rank, request.components, request.ghosts);
  return ::pops::amr::hierarchy::AmrHierarchy<Dim, MemorySpace>::from_coarse(
      std::move(layout), std::move(field), request.hierarchy_budget);
}

/// Construct the spatial runtime and its coarse hierarchy from one shared prepared authority.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
::pops::runtime::amr::AmrRuntime<Dim, MemorySpace> make_amr_runtime(
    CoarseHierarchyRequest<Dim> request,
    std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>> load_balance,
    std::string spatial_identity, const ExecutionLane& lane = ExecutionLane::world()) {
  if (!load_balance)
    throw std::invalid_argument("AMR runtime construction requires a load-balance authority");
  auto hierarchy = make_coarse_hierarchy<Dim, MemorySpace>(request, *load_balance, lane);
  return ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>(
      std::move(hierarchy), std::move(load_balance), std::move(spatial_identity));
}

/// Spatial-only multipatch facade used by coupling code after Python selected `Dim` once.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrCouplerMP {
 public:
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using field_type = typename runtime_type::field_type;
  using subcycle_plan_type = ::pops::numerics::time::amr::PreparedAmrSubcyclePlan<Dim, MemorySpace>;

  explicit AmrCouplerMP(runtime_type& runtime) : runtime_(&runtime) {}

  static constexpr int dimension = Dim;

  runtime_type& runtime() const noexcept { return *runtime_; }
  const ::pops::amr::hierarchy::LevelLayout<Dim>& layout(std::size_t level) const {
    return runtime_->hierarchy().layout(level);
  }
  field_type& state(std::size_t level) const { return runtime_->hierarchy().state(level); }

  subcycle_plan_type prepare_subcycling(
      std::span<const int> temporal_substeps,
      ::pops::numerics::time::amr::AmrSubcyclePreparationBudget budget) const {
    return subcycle_plan_type::prepare(*runtime_, temporal_substeps, budget);
  }

  AmrRegridCoupler<Dim, MemorySpace> regrid_coupler(
      const ::pops::amr::tagging::ClusterProvider<Dim>& provider) const {
    return AmrRegridCoupler<Dim, MemorySpace>(*runtime_, provider);
  }

 private:
  runtime_type* runtime_ = nullptr;
};

}  // namespace pops::coupling::amr
