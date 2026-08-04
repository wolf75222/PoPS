/// @file
/// @brief Immutable compile-time-ranked patch ownership plan.

#pragma once

#include <pops/mesh/layout/nd/distribution.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace pops::parallel::nd {

enum class LoadBalanceStrategy : std::uint8_t {
  space_filling_curve = 0,
  weighted_lpt = 1,
  round_robin = 2,
};

/// Explicit upper bounds authenticated before load-balance work allocates temporary metadata.
struct LoadBalancePreparationBudget {
  std::size_t patches = 0;
  std::size_t ranks = 0;
  std::int64_t total_weight = 0;

  bool operator==(const LoadBalancePreparationBudget&) const = default;
};

template <int Dim>
class LoadBalanceProvider;

/// Exact result of one prepared ownership decision. Patch ordinals and linear rank ordinals are
/// retained alongside the spatial Distribution so reports never have to reconstruct the decision.
template <int Dim>
class OwnershipPlan {
  static_assert(Dim >= 1 && Dim <= 3, "OwnershipPlan only supports dimensions 1, 2, and 3");

 public:
  LoadBalanceStrategy strategy() const noexcept { return strategy_; }
  const mesh::Distribution<Dim>& distribution() const noexcept { return distribution_; }
  const std::vector<std::int64_t>& weights() const noexcept { return weights_; }
  const std::vector<std::size_t>& linear_owners() const noexcept { return linear_owners_; }
  const std::vector<std::int64_t>& linear_rank_loads() const noexcept { return rank_loads_; }
  const std::vector<std::size_t>& traversal() const noexcept { return traversal_; }
  std::int64_t total_weight() const noexcept { return total_weight_; }

  std::int64_t max_load() const noexcept {
    return rank_loads_.empty() ? 0 : *std::max_element(rank_loads_.begin(), rank_loads_.end());
  }

  double imbalance() const noexcept {
    if (total_weight_ == 0 || rank_loads_.empty())
      return 1.0;
    const double average =
        static_cast<double>(total_weight_) / static_cast<double>(rank_loads_.size());
    return static_cast<double>(max_load()) / average;
  }

 private:
  friend class LoadBalanceProvider<Dim>;

  OwnershipPlan(LoadBalanceStrategy strategy, mesh::Distribution<Dim> distribution,
                std::vector<std::int64_t> weights, std::vector<std::size_t> linear_owners,
                std::vector<std::int64_t> rank_loads, std::vector<std::size_t> traversal,
                std::int64_t total_weight)
      : strategy_(strategy),
        distribution_(std::move(distribution)),
        weights_(std::move(weights)),
        linear_owners_(std::move(linear_owners)),
        rank_loads_(std::move(rank_loads)),
        traversal_(std::move(traversal)),
        total_weight_(total_weight) {}

  LoadBalanceStrategy strategy_ = LoadBalanceStrategy::space_filling_curve;
  mesh::Distribution<Dim> distribution_{};
  std::vector<std::int64_t> weights_;
  std::vector<std::size_t> linear_owners_;
  std::vector<std::int64_t> rank_loads_;
  std::vector<std::size_t> traversal_;
  std::int64_t total_weight_ = 0;
};

}  // namespace pops::parallel::nd
