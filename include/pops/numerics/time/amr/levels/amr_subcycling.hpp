/// @file
/// @brief Prepared ranked AMR subcycle transitions over one live spatial runtime.

#pragma once

#include <pops/numerics/time/amr/levels/amr_patch_range.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::numerics::time::amr {

struct AmrSubcyclePreparationBudget {
  std::size_t transitions = 0;
  mesh::BoxArrayValidationBudget interface{};

  bool operator==(const AmrSubcyclePreparationBudget&) const = default;
};

/// Refine a base index domain through an explicit sequence of spatial ratios.
template <int Dim>
Box<Dim> amr_level_index_domain(Box<Dim> base_domain,
                                std::span<const ::pops::amr::RefinementRatio<Dim>> ratios,
                                std::size_t level) {
  if (base_domain.empty() || level > ratios.size())
    throw std::invalid_argument("AMR level domain request is outside its prepared ratio sequence");
  for (std::size_t transition = 0; transition < level; ++transition)
    base_domain = ::pops::amr::hierarchy::refine_box(base_domain, ratios[transition]);
  return base_domain;
}

/// One adjacent level pair qualified by both spatial identity and an explicit temporal ratio.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedAmrSubcycleTransition {
 public:
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using spatial_contract_type = ::pops::amr::hierarchy::LevelStateSpatialContract<Dim>;

  static PreparedAmrSubcycleTransition prepare(const runtime_type& runtime,
                                               std::size_t parent_level, int temporal_substeps,
                                               mesh::BoxArrayValidationBudget interface_budget) {
    if (parent_level >= runtime.hierarchy().num_levels() ||
        runtime.hierarchy().num_levels() - parent_level < 2 || temporal_substeps < 1)
      throw std::invalid_argument(
          "AMR subcycle transition requires adjacent live levels and positive temporal substeps");
    const std::size_t child_level = parent_level + 1;
    CoarseFineInterface<Dim> interface(runtime.hierarchy().layout(parent_level),
                                       runtime.hierarchy().layout(child_level), interface_budget);
    return PreparedAmrSubcycleTransition(parent_level,
                                         runtime.hierarchy().level(parent_level).spatial_contract(),
                                         runtime.hierarchy().level(child_level).spatial_contract(),
                                         interface.exact_identity(), temporal_substeps);
  }

  std::size_t parent_level() const noexcept { return parent_level_; }
  std::size_t child_level() const noexcept { return parent_level_ + 1; }
  int temporal_substeps() const noexcept { return temporal_substeps_; }
  const spatial_contract_type& parent_contract() const noexcept { return parent_; }
  const spatial_contract_type& child_contract() const noexcept { return child_; }
  const CoarseFineInterfaceIdentity<Dim>& interface_identity() const noexcept { return interface_; }

  void require_live(const runtime_type& runtime) const {
    if (parent_level_ >= runtime.hierarchy().num_levels() ||
        runtime.hierarchy().num_levels() - parent_level_ < 2 ||
        runtime.hierarchy().level(parent_level_).spatial_contract() != parent_ ||
        runtime.hierarchy().level(parent_level_ + 1).spatial_contract() != child_)
      throw std::invalid_argument("prepared AMR subcycle transition is stale");
  }

  ::pops::amr::transfer::PreparedTransfer<Dim> prepare_fill_patch(
      const runtime_type& runtime, FieldView<const Real, Dim> parent, FieldView<Real, Dim> child,
      const Box<Dim>& ghost_region, ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
      ::pops::amr::transfer::ComponentRange components = {}) const {
    require_live(runtime);
    return ::pops::numerics::time::amr::prepare_fill_patch(runtime, parent_level_, parent, child,
                                                           ghost_region, mapping, components);
  }

  ::pops::amr::transfer::PreparedTransfer<Dim> prepare_average_down(
      const runtime_type& runtime, FieldView<const Real, Dim> child, FieldView<Real, Dim> parent,
      const Box<Dim>& parent_region, ::pops::amr::transfer::IndexMapping<Dim> mapping = {},
      ::pops::amr::transfer::ComponentRange components = {}) const {
    require_live(runtime);
    return ::pops::numerics::time::amr::prepare_average_down(runtime, child_level(), child, parent,
                                                             parent_region, mapping, components);
  }

  template <class Payload, class Axpy>
  ::pops::amr::reflux::MetricFaceReflux<Payload> reconcile_reflux(
      const runtime_type& runtime,
      const ::pops::amr::reflux::TransactionalFaceFluxLedger<Dim, Payload>& ledger,
      const ::pops::amr::reflux::CoarseFaceRefluxKey<Dim>& key, std::string_view state_identity,
      const ::pops::amr::reflux::MetricRefluxBudget& budget, Axpy&& axpy) const {
    require_live(runtime);
    if (key.levels.coarse != static_cast<int>(parent_level_) ||
        key.levels.fine != static_cast<int>(child_level()))
      throw std::invalid_argument("prepared AMR subcycle reflux key names another transition");
    const ::pops::amr::reflux::FaceRefinementMapping<Dim> mapping{interface_.parent.domain.lo,
                                                                  interface_.child.domain.lo};
    return runtime.reconcile_reflux(ledger, key, state_identity, mapping, budget,
                                    std::forward<Axpy>(axpy));
  }

 private:
  PreparedAmrSubcycleTransition(std::size_t parent_level, spatial_contract_type parent,
                                spatial_contract_type child,
                                CoarseFineInterfaceIdentity<Dim> interface, int temporal_substeps)
      : parent_level_(parent_level),
        parent_(std::move(parent)),
        child_(std::move(child)),
        interface_(std::move(interface)),
        temporal_substeps_(temporal_substeps) {}

  std::size_t parent_level_ = 0;
  spatial_contract_type parent_{};
  spatial_contract_type child_{};
  CoarseFineInterfaceIdentity<Dim> interface_{};
  int temporal_substeps_ = 1;
};

/// Complete prepared transition set for the current runtime topology.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedAmrSubcyclePlan {
 public:
  using runtime_type = ::pops::runtime::amr::AmrRuntime<Dim, MemorySpace>;
  using transition_type = PreparedAmrSubcycleTransition<Dim, MemorySpace>;

  static PreparedAmrSubcyclePlan prepare(const runtime_type& runtime,
                                         std::span<const int> temporal_substeps,
                                         AmrSubcyclePreparationBudget budget) {
    const std::size_t transition_count = runtime.hierarchy().num_levels() - 1;
    if (temporal_substeps.size() != transition_count || transition_count > budget.transitions)
      throw std::invalid_argument(
          "AMR subcycle preparation requires one bounded temporal ratio per transition");

    std::vector<transition_type> transitions;
    transitions.reserve(transition_count);
    for (std::size_t parent = 0; parent < transition_count; ++parent)
      transitions.push_back(
          transition_type::prepare(runtime, parent, temporal_substeps[parent], budget.interface));
    return PreparedAmrSubcyclePlan(runtime, std::move(transitions), budget);
  }

  std::size_t size() const noexcept { return transitions_.size(); }
  std::string_view spatial_contract() const noexcept { return spatial_contract_; }
  std::uint64_t topology_epoch() const noexcept { return topology_epoch_; }
  std::uint64_t materialization_generation() const noexcept { return materialization_generation_; }
  const AmrSubcyclePreparationBudget& preparation_budget() const noexcept { return budget_; }

  const transition_type& transition(std::size_t parent_level) const {
    if (parent_level >= transitions_.size())
      throw std::out_of_range("AMR subcycle transition lies outside the prepared plan");
    return transitions_[parent_level];
  }

  void require_live(const runtime_type& runtime) const {
    if (runtime.topology_epoch() != topology_epoch_ ||
        runtime.materialization_generation() != materialization_generation_ ||
        runtime.spatial_contract() != spatial_contract_)
      throw std::invalid_argument("prepared AMR subcycle plan is stale");
    for (const transition_type& current : transitions_)
      current.require_live(runtime);
  }

 private:
  PreparedAmrSubcyclePlan(const runtime_type& runtime, std::vector<transition_type> transitions,
                          AmrSubcyclePreparationBudget budget)
      : spatial_contract_(runtime.spatial_contract()),
        topology_epoch_(runtime.topology_epoch()),
        materialization_generation_(runtime.materialization_generation()),
        transitions_(std::move(transitions)),
        budget_(budget) {}

  std::string spatial_contract_;
  std::uint64_t topology_epoch_ = 0;
  std::uint64_t materialization_generation_ = 0;
  std::vector<transition_type> transitions_;
  AmrSubcyclePreparationBudget budget_{};
};

}  // namespace pops::numerics::time::amr
