/// @file
/// @brief Transactional compile-time-ranked AMR hierarchy and field-layout authority.

#pragma once

#include <pops/amr/hierarchy/hierarchy_plan.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::amr::hierarchy {

/// Exact spatial metadata retained beside one materialized level field.
///
/// This contract deliberately excludes field values.  It proves that storage was allocated for the
/// same ordered patches, spatial owner coordinates, process space, component count, ghost extent,
/// and local rank as the hierarchy level that publishes it.
template <int Dim>
struct LevelStateSpatialContract {
  static_assert(Dim >= 1 && Dim <= 3,
                "LevelStateSpatialContract only supports dimensions 1, 2, and 3");

  LevelLayoutIdentity<Dim> layout{};
  Index<Dim> local_rank{};
  int components = 0;
  Extent<Dim> ghosts{};

  bool operator==(const LevelStateSpatialContract&) const = default;
};

/// One hierarchy level whose geometry, ownership, and local field allocation are inseparable.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrLevelState {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "AmrLevelState only supports dimensions 1, 2, and 3");

  using field_type = MultiFab<Dim, MemorySpace>;

  AmrLevelState(LevelLayout<Dim> layout, field_type field)
      : layout_(std::move(layout)), field_(std::move(field)) {
    validate_();
  }

  const LevelLayout<Dim>& layout() const noexcept { return layout_; }
  field_type& field() noexcept { return field_; }
  const field_type& field() const noexcept { return field_; }

  LevelStateSpatialContract<Dim> spatial_contract() const {
    return {layout_.exact_identity(), field_.local_rank(), field_.ncomp(), field_.ghosts()};
  }

 private:
  void validate_() const {
    if (field_.layout() != layout_.patches() || field_.distribution() != layout_.distribution())
      throw std::invalid_argument(
          "AMR level field does not authenticate the level layout and ownership");
    if (!layout_.distribution().rank_space().contains(field_.local_rank()))
      throw std::invalid_argument("AMR level local rank lies outside the level process space");
  }

  LevelLayout<Dim> layout_;
  field_type field_;
};

/// A complete hierarchy value.  Topology replacement returns a validated candidate and leaves the
/// source untouched on validation or allocation failure.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class AmrHierarchy {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "AmrHierarchy only supports dimensions 1, 2, and 3");

  using level_type = AmrLevelState<Dim, MemorySpace>;
  using field_type = typename level_type::field_type;

  AmrHierarchy(std::vector<level_type> levels, HierarchyValidationBudget validation_budget)
      : levels_(std::move(levels)), validation_budget_(validation_budget) {
    validate_();
  }

  static AmrHierarchy from_coarse(LevelLayout<Dim> layout, field_type field,
                                  HierarchyValidationBudget validation_budget) {
    if (layout.level() != 0)
      throw std::invalid_argument("AMR coarse level must have level index zero");
    std::vector<level_type> levels;
    levels.emplace_back(std::move(layout), std::move(field));
    return AmrHierarchy(std::move(levels), validation_budget);
  }

  std::size_t num_levels() const noexcept { return levels_.size(); }

  const level_type& level(std::size_t index) const {
    if (index >= levels_.size())
      throw std::out_of_range("AMR hierarchy level lies outside [0, num_levels)");
    return levels_[index];
  }

  level_type& level(std::size_t index) {
    if (index >= levels_.size())
      throw std::out_of_range("AMR hierarchy level lies outside [0, num_levels)");
    return levels_[index];
  }

  const LevelLayout<Dim>& layout(std::size_t index) const { return level(index).layout(); }
  field_type& state(std::size_t index) { return level(index).field(); }
  const field_type& state(std::size_t index) const { return level(index).field(); }
  const HierarchyValidationBudget& validation_budget() const noexcept { return validation_budget_; }

  HierarchyPlan<Dim> plan() const {
    std::vector<LevelLayout<Dim>> layouts;
    layouts.reserve(levels_.size());
    for (const level_type& current : levels_)
      layouts.push_back(current.layout());
    return HierarchyPlan<Dim>(std::move(layouts), validation_budget_);
  }

  std::vector<LevelStateSpatialContract<Dim>> spatial_contract() const {
    std::vector<LevelStateSpatialContract<Dim>> contract;
    contract.reserve(levels_.size());
    for (const level_type& current : levels_)
      contract.push_back(current.spatial_contract());
    return contract;
  }

  /// Append or replace one level, truncating all finer state in the returned candidate.
  AmrHierarchy with_level(level_type candidate) const {
    const int candidate_index = candidate.layout().level();
    if (candidate_index < 0 || static_cast<std::size_t>(candidate_index) > levels_.size())
      throw std::out_of_range("AMR hierarchy replacement level is not contiguous");
    std::vector<level_type> next;
    next.reserve(static_cast<std::size_t>(candidate_index) + 1);
    for (int level_index = 0; level_index < candidate_index; ++level_index)
      next.push_back(levels_[static_cast<std::size_t>(level_index)]);
    next.push_back(std::move(candidate));
    return AmrHierarchy(std::move(next), validation_budget_);
  }

  /// Return a candidate containing levels zero through `finest`, inclusive.
  AmrHierarchy truncated(std::size_t finest) const {
    if (finest >= levels_.size())
      throw std::out_of_range("AMR hierarchy truncation level lies outside the hierarchy");
    std::vector<level_type> next;
    next.reserve(finest + 1);
    for (std::size_t level_index = 0; level_index <= finest; ++level_index)
      next.push_back(levels_[level_index]);
    return AmrHierarchy(std::move(next), validation_budget_);
  }

 private:
  void validate_() const {
    if (levels_.empty())
      throw std::invalid_argument("AMR hierarchy requires a materialized coarse level");
    (void)plan();

    const int components = levels_.front().field().ncomp();
    const Extent<Dim> ghosts = levels_.front().field().ghosts();
    const Index<Dim> local_rank = levels_.front().field().local_rank();
    for (const level_type& current : levels_) {
      if (current.field().ncomp() != components || current.field().ghosts() != ghosts)
        throw std::invalid_argument(
            "AMR hierarchy levels must share one component and ghost-width contract");
      if (current.field().local_rank() != local_rank)
        throw std::invalid_argument("AMR hierarchy levels must share one local rank coordinate");
    }
  }

  std::vector<level_type> levels_;
  HierarchyValidationBudget validation_budget_{};
};

}  // namespace pops::amr::hierarchy
