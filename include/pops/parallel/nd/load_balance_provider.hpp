/// @file
/// @brief One bounded deterministic load-balance core for spatial dimensions 1..3.

#pragma once

#include <pops/parallel/nd/ownership_plan.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::parallel::nd {
namespace detail {

inline void validate_budget(const LoadBalancePreparationBudget& budget) {
  if (budget.patches == 0 || budget.ranks == 0 || budget.total_weight <= 0)
    throw std::invalid_argument("ND load-balance preparation budgets must be strictly positive");
}

template <int Dim>
std::pair<std::vector<std::int64_t>, std::int64_t> prepare_weights(
    const mesh::BoxArray<Dim>& patches, std::span<const std::int64_t> supplied,
    const LoadBalancePreparationBudget& budget) {
  if (patches.size() > budget.patches)
    throw std::length_error("ND load-balance patch count exceeds its prepared budget");
  if (!supplied.empty() && supplied.size() != patches.size())
    throw std::invalid_argument("ND load-balance weight count must equal its patch count");

  std::int64_t total = 0;
  for (std::size_t patch = 0; patch < patches.size(); ++patch) {
    if (patches[patch].empty())
      throw std::invalid_argument("ND load-balance layouts cannot contain empty patches");
    const std::int64_t weight = supplied.empty() ? patches[patch].numPts() : supplied[patch];
    if (weight <= 0)
      throw std::invalid_argument("ND load-balance weights must be strictly positive");
    if (weight > std::numeric_limits<std::int64_t>::max() - total)
      throw std::overflow_error("ND load-balance total weight exceeds int64_t");
    if (weight > budget.total_weight - total)
      throw std::length_error("ND load-balance total weight exceeds its prepared budget");
    total += weight;
  }

  std::vector<std::int64_t> weights;
  weights.reserve(patches.size());
  for (std::size_t patch = 0; patch < patches.size(); ++patch)
    weights.push_back(supplied.empty() ? patches[patch].numPts() : supplied[patch]);
  return {std::move(weights), total};
}

inline std::int64_t cumulative_target(std::int64_t total, std::size_t partitions,
                                      std::size_t rank_count) {
  const auto ranks = static_cast<std::int64_t>(rank_count);
  const auto cut = static_cast<std::int64_t>(partitions);
  const std::int64_t quotient = total / ranks;
  const std::int64_t remainder = total % ranks;
  const std::int64_t remainder_product = remainder * cut;
  const std::int64_t remainder_share = (remainder_product + ranks - 1) / ranks;
  return quotient * cut + remainder_share;
}

inline void checked_add_load(std::int64_t& destination, std::int64_t weight) {
  if (weight > std::numeric_limits<std::int64_t>::max() - destination)
    throw std::overflow_error("ND load-balance rank load exceeds int64_t");
  destination += weight;
}

template <int Dim>
std::vector<std::size_t> morton_order(const mesh::BoxArray<Dim>& patches) {
  std::vector<std::size_t> order(patches.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  if (patches.empty())
    return order;

  std::array<int, Dim> lower{};
  for (int axis = 0; axis < Dim; ++axis)
    lower[static_cast<std::size_t>(axis)] = patches[0].lo[axis];
  for (std::size_t patch = 1; patch < patches.size(); ++patch)
    for (int axis = 0; axis < Dim; ++axis)
      lower[static_cast<std::size_t>(axis)] =
          std::min(lower[static_cast<std::size_t>(axis)], patches[patch].lo[axis]);

  std::vector<std::array<std::uint32_t, Dim>> coordinate(patches.size());
  for (std::size_t patch = 0; patch < patches.size(); ++patch) {
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t offset = static_cast<std::int64_t>(patches[patch].lo[axis]) -
                                  lower[static_cast<std::size_t>(axis)];
      if (offset < 0 ||
          static_cast<std::uint64_t>(offset) > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("ND Morton coordinate is outside its exact 32-bit range");
      coordinate[patch][static_cast<std::size_t>(axis)] = static_cast<std::uint32_t>(offset);
    }
  }

  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    for (int bit = 31; bit >= 0; --bit) {
      for (int axis = Dim - 1; axis >= 0; --axis) {
        const auto lhs =
            (coordinate[left][static_cast<std::size_t>(axis)] >> bit) & std::uint32_t{1};
        const auto rhs =
            (coordinate[right][static_cast<std::size_t>(axis)] >> bit) & std::uint32_t{1};
        if (lhs != rhs)
          return lhs < rhs;
      }
    }
    return left < right;
  });
  return order;
}

template <int Dim>
std::vector<Index<Dim>> spatial_owners(const mesh::RankSpace<Dim>& ranks,
                                       const std::vector<std::size_t>& linear_owners) {
  std::vector<Index<Dim>> owners;
  owners.reserve(linear_owners.size());
  for (const std::size_t owner : linear_owners)
    owners.push_back(ranks.coordinate(owner));
  return owners;
}

}  // namespace detail

/// Prepared immutable strategy. Route selection remains owned by the existing public registry;
/// this class is the single dimension-generic algorithm authority consumed by its adapters.
template <int Dim>
class LoadBalanceProvider {
  static_assert(Dim >= 1 && Dim <= 3, "LoadBalanceProvider only supports dimensions 1, 2, and 3");

 public:
  explicit LoadBalanceProvider(LoadBalanceStrategy strategy) : strategy_(strategy) {
    if (strategy_ != LoadBalanceStrategy::space_filling_curve &&
        strategy_ != LoadBalanceStrategy::weighted_lpt &&
        strategy_ != LoadBalanceStrategy::round_robin)
      throw std::invalid_argument("ND load-balance strategy is invalid");
  }

  static LoadBalanceProvider space_filling_curve() {
    return LoadBalanceProvider{LoadBalanceStrategy::space_filling_curve};
  }
  static LoadBalanceProvider weighted_lpt() {
    return LoadBalanceProvider{LoadBalanceStrategy::weighted_lpt};
  }
  static LoadBalanceProvider round_robin() {
    return LoadBalanceProvider{LoadBalanceStrategy::round_robin};
  }

  LoadBalanceStrategy strategy() const noexcept { return strategy_; }

  std::string_view provider_identity() const noexcept {
    switch (strategy_) {
      case LoadBalanceStrategy::space_filling_curve:
        return "pops.load_balance.nd.morton_sfc@2";
      case LoadBalanceStrategy::weighted_lpt:
        return "pops.load_balance.nd.weighted_lpt@1";
      case LoadBalanceStrategy::round_robin:
        return "pops.load_balance.nd.round_robin@1";
    }
    return "pops.load_balance.nd.invalid@1";
  }

  OwnershipPlan<Dim> prepare(const mesh::BoxArray<Dim>& patches,
                             const mesh::RankSpace<Dim>& rank_space,
                             LoadBalancePreparationBudget budget,
                             std::span<const std::int64_t> supplied_weights = {}) const {
    detail::validate_budget(budget);
    if (rank_space.empty())
      throw std::invalid_argument("ND load balance requires a non-empty rank space");
    if (rank_space.size() > budget.ranks)
      throw std::length_error("ND load-balance rank count exceeds its prepared budget");
    if (rank_space.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::length_error("ND load-balance rank count exceeds its bounded algorithm range");

    auto [weights, total_weight] = detail::prepare_weights(patches, supplied_weights, budget);
    const std::size_t patch_count = patches.size();
    const std::size_t rank_count = rank_space.size();
    std::vector<std::size_t> traversal(patch_count);
    std::iota(traversal.begin(), traversal.end(), std::size_t{0});
    std::vector<std::size_t> linear_owners(patch_count, 0);
    std::vector<std::int64_t> rank_loads(rank_count, 0);

    switch (strategy_) {
      case LoadBalanceStrategy::space_filling_curve: {
        traversal = detail::morton_order(patches);
        std::int64_t cumulative = 0;
        std::size_t owner = 0;
        const std::size_t active_rank_count = std::min(patch_count, rank_count);
        for (std::size_t position = 0; position < traversal.size(); ++position) {
          const std::size_t patch = traversal[position];
          linear_owners[patch] = owner;
          detail::checked_add_load(rank_loads[owner], weights[patch]);
          detail::checked_add_load(cumulative, weights[patch]);
          const std::size_t patches_left = patch_count - position - 1;
          const std::size_t ranks_left = active_rank_count - owner - 1;
          const bool reached_weight_target =
              cumulative >= detail::cumulative_target(total_weight, owner + 1, active_rank_count);
          const bool must_seed_remaining_ranks = patches_left == ranks_left;
          if (owner + 1 < active_rank_count && patches_left >= ranks_left &&
              (reached_weight_target || must_seed_remaining_ranks))
            ++owner;
        }
        break;
      }
      case LoadBalanceStrategy::weighted_lpt:
        std::sort(traversal.begin(), traversal.end(), [&](std::size_t left, std::size_t right) {
          return weights[left] > weights[right] ||
                 (weights[left] == weights[right] && left < right);
        });
        for (const std::size_t patch : traversal) {
          std::size_t owner = 0;
          for (std::size_t rank = 1; rank < rank_count; ++rank)
            if (rank_loads[rank] < rank_loads[owner])
              owner = rank;
          linear_owners[patch] = owner;
          detail::checked_add_load(rank_loads[owner], weights[patch]);
        }
        break;
      case LoadBalanceStrategy::round_robin:
        for (std::size_t patch = 0; patch < patch_count; ++patch) {
          const std::size_t owner = patch % rank_count;
          linear_owners[patch] = owner;
          detail::checked_add_load(rank_loads[owner], weights[patch]);
        }
        break;
      default:
        throw std::logic_error("ND load-balance provider retained an invalid strategy");
    }

    auto distribution = mesh::Distribution<Dim>::partitioned(
        patches, rank_space, detail::spatial_owners(rank_space, linear_owners));
    return OwnershipPlan<Dim>{strategy_,
                              std::move(distribution),
                              std::move(weights),
                              std::move(linear_owners),
                              std::move(rank_loads),
                              std::move(traversal),
                              total_weight};
  }

  mesh::Distribution<Dim> distribute(const mesh::BoxArray<Dim>& patches,
                                     const mesh::RankSpace<Dim>& rank_space,
                                     LoadBalancePreparationBudget budget,
                                     std::span<const std::int64_t> supplied_weights = {}) const {
    const OwnershipPlan<Dim> plan = prepare(patches, rank_space, budget, supplied_weights);
    return plan.distribution();
  }

 private:
  LoadBalanceStrategy strategy_;
};

}  // namespace pops::parallel::nd
