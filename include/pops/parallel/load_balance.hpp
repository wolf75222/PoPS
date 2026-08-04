#pragma once

#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/parallel/nd/load_balance_provider.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

/// @file
/// @brief AMR load balancing: builds a DistributionMapping (box -> rank) from a
///        BoxArray, using a Z-order space-filling curve (SFC) or LPT knapsack.
///
/// Layer: `include/pops/parallel`.
/// Role: distribute boxes over ranks with replicated metadata (AMReX style). Two
/// strategies: make_sfc_distribution (contiguous segments of equal load along the
/// Morton curve -> spatial locality) and make_knapsack_distribution (heaviest box
/// to the least loaded rank -> minimizes the maximum imbalance). load_imbalance
/// measures the max load / average load ratio. Morton tooling exposed: part1by1,
/// morton_key, morton_order.
/// Contract: a box weight defaults to its cell count (proxy for compute cost), or is supplied by a
/// prepared ownership authority as one positive integer per box.
///
/// Invariants:
/// - PURE functions, no MPI: testable in serial; they will feed the comm seam once
///   an MPI backend is wired in;
/// - SFC guarantees that with nboxes >= nranks each rank receives at least one box;
/// - an empty BoxArray yields an empty mapping; nranks must otherwise be positive.

namespace pops {

// Spread the bits of x (16 useful bits) onto the even positions of a 64-bit word.
inline std::uint64_t part1by1(std::uint64_t x) {
  x &= 0xffffffffULL;
  x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
  x = (x | (x << 8)) & 0x00ff00ff00ff00ffULL;
  x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fULL;
  x = (x | (x << 2)) & 0x3333333333333333ULL;
  x = (x | (x << 1)) & 0x5555555555555555ULL;
  return x;
}

// Morton key (Z-order) interleaving (x, y): x on the even bits, y on the odd bits.
inline std::uint64_t morton_key(std::uint32_t x, std::uint32_t y) {
  return part1by1(x) | (part1by1(y) << 1);
}

namespace detail {

inline mesh::BoxArray<2> legacy_load_balance_boxes(const BoxArray& boxes) {
  std::vector<Box<2>> converted;
  converted.reserve(static_cast<std::size_t>(boxes.size()));
  for (const Box2D& box : boxes.boxes())
    converted.emplace_back(Index<2>{box.lo[0], box.lo[1]}, Index<2>{box.hi[0], box.hi[1]});
  return mesh::BoxArray<2>(std::move(converted));
}

inline mesh::RankSpace<2> legacy_load_balance_ranks(int nranks) {
  return mesh::RankSpace<2>(Index<2>{0, 0}, Extent<2>{nranks, 1});
}

inline parallel::nd::LoadBalancePreparationBudget legacy_load_balance_budget(std::size_t patches,
                                                                             std::size_t ranks) {
  return {
      .patches = std::max(std::size_t{1}, patches),
      .ranks = ranks,
      .total_weight = std::numeric_limits<std::int64_t>::max(),
  };
}

inline DistributionMapping legacy_distribution_mapping(const mesh::Distribution<2>& distribution) {
  std::vector<int> ranks;
  ranks.reserve(distribution.box_count());
  for (const Index<2>& owner : distribution.owners()) {
    const std::size_t linear = distribution.rank_space().linear_rank(owner);
    if (linear > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("load-balance owner rank exceeds the legacy signed-int contract");
    ranks.push_back(static_cast<int>(linear));
  }
  return DistributionMapping(std::move(ranks));
}

inline DistributionMapping legacy_provider_distribution(
    const BoxArray& boxes, int nranks, std::span<const std::int64_t> supplied_weights,
    const parallel::nd::LoadBalanceProvider<2>& provider) {
  if (!supplied_weights.empty() &&
      supplied_weights.size() != static_cast<std::size_t>(boxes.size()))
    throw std::invalid_argument("load-balance weight count must equal the BoxArray size");
  std::int64_t total_weight = 0;
  for (int index = 0; index < boxes.size(); ++index) {
    if (boxes[index].empty())
      throw std::invalid_argument("load-balance layouts cannot contain empty boxes");
    const std::int64_t weight =
        supplied_weights.empty() ? boxes[index].num_cells() : supplied_weights[index];
    if (weight <= 0)
      throw std::invalid_argument("load-balance weights must be strictly positive");
    if (weight > std::numeric_limits<std::int64_t>::max() - total_weight)
      throw std::overflow_error("load-balance total weight exceeds int64_t");
    total_weight += weight;
  }
  const mesh::BoxArray<2> converted = legacy_load_balance_boxes(boxes);
  const mesh::RankSpace<2> rank_space = legacy_load_balance_ranks(nranks);
  const auto budget = legacy_load_balance_budget(converted.size(), rank_space.size());
  return legacy_distribution_mapping(
      provider.distribute(converted, rank_space, budget, supplied_weights));
}

}  // namespace detail

// Box indices sorted along the Morton curve (low corner, shifted by the bounding
// box to stay positive).
inline std::vector<int> morton_order(const BoxArray& ba) {
  const int n = ba.size();
  if (n == 0)
    return {};

  const mesh::BoxArray<2> boxes = detail::legacy_load_balance_boxes(ba);
  const mesh::RankSpace<2> rank_space = detail::legacy_load_balance_ranks(1);
  const std::vector<std::int64_t> unit_weights(static_cast<std::size_t>(n), 1);
  const auto plan = parallel::nd::LoadBalanceProvider<2>::space_filling_curve().prepare(
      boxes, rank_space, detail::legacy_load_balance_budget(boxes.size(), rank_space.size()),
      unit_weights);

  std::vector<int> order;
  order.reserve(plan.traversal().size());
  for (const std::size_t patch : plan.traversal())
    order.push_back(static_cast<int>(patch));
  return order;
}

namespace detail {

inline std::vector<std::int64_t> load_balance_weights(const BoxArray& boxes,
                                                      std::span<const std::int64_t> supplied) {
  if (!supplied.empty() && supplied.size() != static_cast<std::size_t>(boxes.size()))
    throw std::invalid_argument("load-balance weight count must equal the BoxArray size");
  std::vector<std::int64_t> weights(static_cast<std::size_t>(boxes.size()));
  std::int64_t total = 0;
  for (int index = 0; index < boxes.size(); ++index) {
    const std::int64_t value = supplied.empty() ? boxes[index].num_cells() : supplied[index];
    if (value <= 0)
      throw std::invalid_argument("load-balance weights must be strictly positive");
    if (value > std::numeric_limits<std::int64_t>::max() - total)
      throw std::overflow_error("load-balance total weight exceeds int64_t");
    total += value;
    weights[static_cast<std::size_t>(index)] = value;
  }
  return weights;
}

}  // namespace detail

// Z-order distribution: contiguous segments of ~equal load along the SFC.
// Guarantees that with nboxes >= nranks each rank receives at least one box.
inline DistributionMapping make_sfc_distribution(
    const BoxArray& ba, int nranks, std::span<const std::int64_t> supplied_weights = {}) {
  if (nranks <= 0)
    throw std::invalid_argument("SFC load balance requires a positive rank count");
  return detail::legacy_provider_distribution(
      ba, nranks, supplied_weights, parallel::nd::LoadBalanceProvider<2>::space_filling_curve());
}

// Knapsack distribution (LPT): heaviest box -> least loaded rank.
inline DistributionMapping make_knapsack_distribution(
    const BoxArray& ba, int nranks, std::span<const std::int64_t> supplied_weights = {}) {
  if (nranks <= 0)
    throw std::invalid_argument("knapsack load balance requires a positive rank count");
  return detail::legacy_provider_distribution(ba, nranks, supplied_weights,
                                              parallel::nd::LoadBalanceProvider<2>::weighted_lpt());
}

inline DistributionMapping make_round_robin_distribution(
    const BoxArray& ba, int nranks, std::span<const std::int64_t> supplied_weights = {}) {
  if (nranks <= 0)
    throw std::invalid_argument("round-robin load balance requires a positive rank count");
  return detail::legacy_provider_distribution(ba, nranks, supplied_weights,
                                              parallel::nd::LoadBalanceProvider<2>::round_robin());
}

// Imbalance = max load / average load (1.0 = perfect).
inline double load_imbalance(const BoxArray& ba, const DistributionMapping& dm, int nranks,
                             std::span<const std::int64_t> supplied_weights = {}) {
  if (nranks <= 0)
    throw std::invalid_argument("load_imbalance requires a positive rank count");
  if (dm.size() != ba.size())
    throw std::invalid_argument("load_imbalance mapping size differs from the BoxArray");
  const auto weights = detail::load_balance_weights(ba, supplied_weights);
  if (ba.size() == 0)
    return 1.0;
  std::vector<std::int64_t> load(nranks, 0);
  for (int i = 0; i < ba.size(); ++i) {
    if (dm[i] < 0 || dm[i] >= nranks)
      throw std::invalid_argument("load_imbalance mapping contains an invalid owner rank");
    load[dm[i]] += weights[static_cast<std::size_t>(i)];
  }
  std::int64_t mx = 0, sum = 0;
  for (std::int64_t l : load) {
    mx = std::max(mx, l);
    sum += l;
  }
  const double avg = double(sum) / nranks;
  return avg > 0 ? mx / avg : 1.0;
}

}  // namespace pops
