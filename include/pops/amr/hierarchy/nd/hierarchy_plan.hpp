/// @file
/// @brief Exact ND AMR hierarchy plan with anisotropic parent/child validation.

#pragma once

#include <pops/amr/hierarchy/nd/level_layout.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pops::amr::hierarchy::nd {

struct HierarchyValidationBudget {
  std::size_t levels = 0;
  std::size_t parent_child_patch_pairs = 0;

  bool operator==(const HierarchyValidationBudget&) const = default;
};

template <int Dim>
struct HierarchyPlanIdentity {
  std::vector<LevelLayoutIdentity<Dim>> levels{};
  HierarchyValidationBudget validation_budget{};

  bool operator==(const HierarchyPlanIdentity&) const = default;
};

/// Pure geometry/ownership plan. Publication into fields or an MPI runtime is a later cutover.
template <int Dim>
class HierarchyPlan {
  static_assert(Dim >= 1 && Dim <= 3, "HierarchyPlan only supports dimensions 1, 2, and 3");

 public:
  HierarchyPlan(std::vector<LevelLayout<Dim>> levels, HierarchyValidationBudget budget)
      : levels_(std::move(levels)), budget_(budget) {
    validate_();
  }

  std::size_t num_levels() const noexcept { return levels_.size(); }

  const LevelLayout<Dim>& level(std::size_t index) const {
    if (index >= levels_.size())
      throw std::out_of_range("HierarchyPlan level is outside [0, num_levels)");
    return levels_[index];
  }

  const HierarchyValidationBudget& validation_budget() const noexcept { return budget_; }

  HierarchyPlanIdentity<Dim> exact_identity() const {
    HierarchyPlanIdentity<Dim> identity;
    identity.levels.reserve(levels_.size());
    for (const LevelLayout<Dim>& level_layout : levels_)
      identity.levels.push_back(level_layout.exact_identity());
    identity.validation_budget = budget_;
    return identity;
  }

  /// Return a validated append or replacement, truncating levels finer than the candidate.
  HierarchyPlan with_level(LevelLayout<Dim> candidate) const {
    if (candidate.level() < 0 || static_cast<std::size_t>(candidate.level()) > levels_.size())
      throw std::out_of_range("HierarchyPlan replacement level is not contiguous");
    std::vector<LevelLayout<Dim>> next;
    next.reserve(static_cast<std::size_t>(candidate.level()) + 1);
    for (int level_index = 0; level_index < candidate.level(); ++level_index)
      next.push_back(levels_[static_cast<std::size_t>(level_index)]);
    next.push_back(std::move(candidate));
    return HierarchyPlan(std::move(next), budget_);
  }

  bool operator==(const HierarchyPlan& other) const {
    return exact_identity() == other.exact_identity();
  }

 private:
  static std::size_t checked_pair_count_(std::size_t children, std::size_t parents) {
    if (parents != 0 && children > std::numeric_limits<std::size_t>::max() / parents)
      throw std::length_error("HierarchyPlan parent/child patch pair count exceeds size_t");
    return children * parents;
  }

  void validate_() const {
    if (levels_.empty())
      throw std::invalid_argument("HierarchyPlan requires level zero");
    if (levels_.size() > budget_.levels)
      throw std::length_error("HierarchyPlan exceeds its explicit level budget");
    if (levels_.front().level() != 0)
      throw std::invalid_argument("HierarchyPlan first level must be level zero");

    std::size_t pair_count = 0;
    for (std::size_t level_index = 1; level_index < levels_.size(); ++level_index) {
      const LevelLayout<Dim>& parent = levels_[level_index - 1];
      const LevelLayout<Dim>& child = levels_[level_index];
      if (child.level() != static_cast<int>(level_index))
        throw std::invalid_argument("HierarchyPlan levels must be consecutive and ordered");
      if (child.distribution().rank_space() != parent.distribution().rank_space())
        throw std::invalid_argument("HierarchyPlan levels must share one exact process space");
      if (child.domain() != refine_box(parent.domain(), child.ratio_from_parent()))
        throw std::invalid_argument(
            "HierarchyPlan child domain is not the anisotropic refinement of its parent");

      const std::size_t current_pairs =
          checked_pair_count_(child.patches().size(), parent.patches().size());
      if (pair_count > budget_.parent_child_patch_pairs ||
          current_pairs > budget_.parent_child_patch_pairs - pair_count)
        throw std::length_error("HierarchyPlan exceeds its explicit parent/child pair budget");
      pair_count += current_pairs;

      for (const Box<Dim>& fine_patch : child.patches().boxes()) {
        const Box<Dim> footprint = coarsen_box(fine_patch, child.ratio_from_parent());
        if (refine_box(footprint, child.ratio_from_parent()) != fine_patch)
          throw std::invalid_argument(
              "HierarchyPlan fine patches must contain complete anisotropic parent cells");
        mesh::ExactCellCount covered;
        for (const Box<Dim>& parent_patch : parent.patches().boxes())
          if (!covered.add(mesh::ExactCellCount::from_box(footprint.intersect(parent_patch))))
            throw std::overflow_error("HierarchyPlan parent coverage exceeds exact count capacity");
        if (covered != mesh::ExactCellCount::from_box(footprint))
          throw std::invalid_argument(
              "HierarchyPlan fine patch footprint is not covered by the parent level");
      }
    }
  }

  std::vector<LevelLayout<Dim>> levels_{};
  HierarchyValidationBudget budget_{};
};

}  // namespace pops::amr::hierarchy::nd
