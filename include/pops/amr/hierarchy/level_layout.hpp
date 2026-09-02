/// @file
/// @brief Exact, immutable-by-value ND AMR level layout contract.

#pragma once

#include <pops/amr/refinement_ratio.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::amr::hierarchy {

namespace detail {

inline int checked_index(std::int64_t value, const char* operation) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error(operation);
  return static_cast<int>(value);
}

inline int floor_div(int numerator, int denominator) {
  if (denominator <= 0)
    throw std::invalid_argument("ND refinement ratios must be strictly positive");
  const int quotient = numerator / denominator;
  const int remainder = numerator % denominator;
  return remainder < 0 ? quotient - 1 : quotient;
}

template <int Dim>
void validate_ratio(const std::type_identity_t<RefinementRatio<Dim>>& ratio) {
  for (int axis = 0; axis < Dim; ++axis)
    if (ratio[axis] <= 0)
      throw std::invalid_argument("ND refinement ratios must be strictly positive");
}

}  // namespace detail

/// Refine an inclusive box independently along every axis.
template <int Dim>
Box<Dim> refine_box(const Box<Dim>& box, const std::type_identity_t<RefinementRatio<Dim>>& ratio) {
  detail::validate_ratio<Dim>(ratio);
  if (box.empty())
    return box;
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    result.lo[axis] = detail::checked_index(static_cast<std::int64_t>(box.lo[axis]) * ratio[axis],
                                            "refine_box lower bound exceeds signed index range");
    result.hi[axis] = detail::checked_index(
        static_cast<std::int64_t>(box.hi[axis]) * ratio[axis] + ratio[axis] - 1,
        "refine_box upper bound exceeds signed index range");
  }
  return result;
}

/// Coarsen an inclusive box with mathematical floor division on negative origins.
template <int Dim>
Box<Dim> coarsen_box(const Box<Dim>& box, const std::type_identity_t<RefinementRatio<Dim>>& ratio) {
  detail::validate_ratio<Dim>(ratio);
  if (box.empty())
    return box;
  Box<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    result.lo[axis] = detail::floor_div(box.lo[axis], ratio[axis]);
    result.hi[axis] = detail::floor_div(box.hi[axis], ratio[axis]);
  }
  return result;
}

template <int Dim>
struct LevelLayoutIdentity {
  int level = -1;
  Box<Dim> domain{};
  RefinementRatio<Dim> ratio_from_parent{};
  std::vector<Box<Dim>> patches{};
  mesh::RankSpace<Dim> rank_space{};
  mesh::DistributionMode distribution_mode = mesh::DistributionMode::replicated;
  std::vector<Index<Dim>> owners{};
  mesh::BoxArrayValidationBudget validation_budget{};

  bool operator==(const LevelLayoutIdentity&) const = default;
};

/// A geometric level and its exact patch ownership. No field storage or execution state is owned.
template <int Dim>
class LevelLayout {
  static_assert(Dim >= 1 && Dim <= 3, "LevelLayout only supports dimensions 1, 2, and 3");

 public:
  LevelLayout(int level, Box<Dim> domain, mesh::BoxArray<Dim> patches,
              mesh::Distribution<Dim> distribution, RefinementRatio<Dim> ratio_from_parent,
              mesh::BoxArrayValidationBudget validation_budget)
      : level_(level),
        domain_(domain),
        patches_(std::move(patches)),
        distribution_(std::move(distribution)),
        ratio_from_parent_(ratio_from_parent),
        validation_budget_(validation_budget) {
    validate_(validation_budget);
  }

  int level() const noexcept { return level_; }
  const Box<Dim>& domain() const noexcept { return domain_; }
  const mesh::BoxArray<Dim>& patches() const noexcept { return patches_; }
  const mesh::Distribution<Dim>& distribution() const noexcept { return distribution_; }
  const RefinementRatio<Dim>& ratio_from_parent() const noexcept { return ratio_from_parent_; }
  const mesh::BoxArrayValidationBudget& validation_budget() const noexcept {
    return validation_budget_;
  }

  LevelLayoutIdentity<Dim> exact_identity() const {
    return LevelLayoutIdentity<Dim>{level_,
                                    domain_,
                                    ratio_from_parent_,
                                    patches_.boxes(),
                                    distribution_.rank_space(),
                                    distribution_.mode(),
                                    distribution_.owners(),
                                    validation_budget_};
  }

  /// Compare against a previously materialized exact identity without rebuilding its owning
  /// vectors.  Accepted-step liveness checks use this path after the identity has been frozen at
  /// preparation; constructing another identity would allocate once per level on every step.
  bool matches_exact_identity(const LevelLayoutIdentity<Dim>& identity) const {
    return level_ == identity.level && domain_ == identity.domain &&
           ratio_from_parent_ == identity.ratio_from_parent &&
           patches_.boxes() == identity.patches &&
           distribution_.rank_space() == identity.rank_space &&
           distribution_.mode() == identity.distribution_mode &&
           distribution_.owners() == identity.owners &&
           validation_budget_ == identity.validation_budget;
  }

  bool operator==(const LevelLayout& other) const {
    return exact_identity() == other.exact_identity();
  }

 private:
  void validate_(mesh::BoxArrayValidationBudget budget) const {
    if (level_ < 0)
      throw std::invalid_argument("LevelLayout level must be non-negative");
    if (domain_.empty())
      throw std::invalid_argument("LevelLayout domain must be non-empty");
    if (patches_.empty())
      throw std::invalid_argument("LevelLayout must contain at least one patch");
    if (!distribution_.matches_layout(patches_))
      throw std::invalid_argument(
          "LevelLayout distribution does not authenticate its patch layout");
    detail::validate_ratio<Dim>(ratio_from_parent_);
    if (level_ == 0) {
      if (!ratio_from_parent_.is_identity())
        throw std::invalid_argument("LevelLayout level zero must use the identity ratio");
      if (!patches_.tiles_exactly(domain_, budget))
        throw std::invalid_argument("LevelLayout level zero patches must exactly tile the domain");
    } else {
      if (!ratio_from_parent_.refines_any_axis())
        throw std::invalid_argument("a fine LevelLayout must refine at least one axis");
      if (!patches_.is_disjoint_within(domain_, budget))
        throw std::invalid_argument(
            "a fine LevelLayout requires non-empty disjoint patches inside its domain");
    }
  }

  int level_ = -1;
  Box<Dim> domain_{};
  mesh::BoxArray<Dim> patches_{};
  mesh::Distribution<Dim> distribution_{};
  RefinementRatio<Dim> ratio_from_parent_{};
  mesh::BoxArrayValidationBudget validation_budget_{};
};

}  // namespace pops::amr::hierarchy
