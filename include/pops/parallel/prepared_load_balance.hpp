/// @file
/// @brief Prepared, extensible ownership authority for AMR BoxArray layouts.

#pragma once

#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/load_balance.hpp>
#include <pops/parallel/ownership_plan.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

using LoadBalanceWeights = std::span<const std::int64_t>;

/// One compile-time-ranked provider result.  The provider returns the complete OwnershipPlan so
/// neither prepared authority nor a later consumer reconstructs strategy metadata from owners.
template <int Dim>
using PreparedLoadBalanceProvider = PreparedProvider<parallel::OwnershipPlan<Dim>(
    const mesh::BoxArray<Dim>&, const mesh::RankSpace<Dim>&, parallel::LoadBalancePreparationBudget,
    LoadBalanceWeights)>;

template <int Dim>
class PreparedLoadBalanceAuthority;

/// Authenticated result retained by hierarchy planning and later installation transactions.
template <int Dim>
class PreparedLoadBalanceResult {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedLoadBalanceResult only supports dimensions 1, 2, and 3");

 public:
  PreparedLoadBalanceResult(const PreparedLoadBalanceResult&) = default;
  PreparedLoadBalanceResult(PreparedLoadBalanceResult&&) noexcept = default;
  PreparedLoadBalanceResult& operator=(const PreparedLoadBalanceResult&) = delete;
  PreparedLoadBalanceResult& operator=(PreparedLoadBalanceResult&&) = delete;

  [[nodiscard]] const parallel::OwnershipPlan<Dim>& plan() const noexcept { return plan_; }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }
  [[nodiscard]] std::string_view collective_context_contract() const noexcept {
    return collective_context_contract_;
  }
  [[nodiscard]] std::string_view source_contract() const noexcept { return source_contract_; }
  [[nodiscard]] std::string_view exact_contract() const noexcept { return exact_contract_; }

 private:
  friend class PreparedLoadBalanceAuthority<Dim>;

  PreparedLoadBalanceResult(parallel::OwnershipPlan<Dim> plan, std::string collective_contract,
                            std::string collective_context_contract, std::string source_contract,
                            std::string exact_contract)
      : plan_(std::move(plan)),
        collective_contract_(std::move(collective_contract)),
        collective_context_contract_(std::move(collective_context_contract)),
        source_contract_(std::move(source_contract)),
        exact_contract_(std::move(exact_contract)) {}

  parallel::OwnershipPlan<Dim> plan_;
  std::string collective_contract_;
  std::string collective_context_contract_;
  std::string source_contract_;
  std::string exact_contract_;
};

/// Measured, topology-qualified resource cost for one AMR patch.
///
/// Integer counters keep the collective contract bit-exact across ranks.  Compute and
/// communication time are accumulated over ``samples`` observations; resident bytes are the
/// amount that must move if ownership changes.  No field is an optional hint: a rebalance request
/// with stale or incomplete evidence is rejected before invoking a policy.
struct ResourceEstimate {
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;
  std::int64_t samples = 0;
  std::int64_t cell_updates = 0;
  std::int64_t compute_nanoseconds = 0;
  std::int64_t memory_bytes = 0;
  std::int64_t communication_bytes = 0;
  std::int64_t communication_nanoseconds = 0;
  std::int64_t resident_bytes = 0;
};

using ResourceEstimates = std::span<const ResourceEstimate>;

/// Policy used to decide whether a measured candidate repays its migration cost.
struct RebalancePolicy {
  /// Required net reduction over the complete amortization horizon, in parts per million.
  std::int64_t minimum_improvement_ppm = 50'000;
  std::int64_t amortization_steps = 20;
  std::int64_t migration_bandwidth_bytes_per_second = 1'000'000'000;
  std::int64_t per_patch_migration_latency_nanoseconds = 0;
};

enum class RebalanceReason : std::uint8_t {
  EmptyHierarchy = 0,
  MappingUnchanged = 1,
  InsufficientNetBenefit = 2,
  NetBenefit = 3,
};

/// Topology-qualified ownership proposal with the complete prepared result retained as evidence.
///
/// A hierarchy may consume `proposed.plan().distribution()` only when `accepted` is true and the
/// source contract still matches its live level identity.  Retaining the full prepared result keeps
/// spatial owners, linear owners, weights, traversal, rank loads, provider identity, and collective
/// context inseparable throughout migration.
template <int Dim>
struct PreparedRebalanceDecision {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedRebalanceDecision only supports dimensions 1, 2, and 3");

  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;
  std::string source_contract;
  PreparedLoadBalanceResult<Dim> proposed;
  RebalanceReason reason = RebalanceReason::EmptyHierarchy;
  bool accepted = false;
  std::int64_t moved_patches = 0;
  std::int64_t migration_bytes = 0;
  std::int64_t migration_nanoseconds = 0;
  std::int64_t current_max_nanoseconds_per_step = 0;
  std::int64_t proposed_max_nanoseconds_per_step = 0;
  double current_imbalance = 1.0;
  double proposed_imbalance = 1.0;
  double predicted_net_speedup = 1.0;
  std::string exact_contract;
};

namespace detail {

template <int Dim>
inline std::string exact_load_balance_collective(
    std::string_view semantic_identity, std::string_view provider_collective_contract,
    const std::optional<RebalancePolicy>& default_rebalance_policy) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-load-balance-collective")
      .scalar(std::uint32_t{2})
      .scalar(static_cast<std::uint32_t>(Dim))
      .text(semantic_identity)
      .bytes(provider_collective_contract)
      .scalar(static_cast<std::uint8_t>(default_rebalance_policy.has_value() ? 1 : 0));
  if (default_rebalance_policy) {
    contract.scalar(default_rebalance_policy->minimum_improvement_ppm)
        .scalar(default_rebalance_policy->amortization_steps)
        .scalar(default_rebalance_policy->migration_bandwidth_bytes_per_second)
        .scalar(default_rebalance_policy->per_patch_migration_latency_nanoseconds);
  }
  return std::move(contract).release();
}

inline std::string exact_load_balance_collective_context(const ExecutionLane& lane) {
  if (lane.identity().empty())
    throw std::invalid_argument("prepared load-balance execution lane identity is empty");
#ifdef POPS_HAS_MPI
  if (!lane.active())
    throw std::invalid_argument("prepared load-balance execution lane is inactive");
#endif
  const int communicator_size = lane.size();
  if (communicator_size <= 0)
    throw std::invalid_argument("prepared load-balance execution lane has no ranks");

  ExactContractBuilder contract;
  contract.text("pops.prepared-load-balance-collective-context")
      .scalar(std::uint32_t{1})
      .text(lane.identity())
      .scalar(static_cast<std::uint64_t>(communicator_size))
      .scalar(static_cast<std::uint8_t>(lane.owns_communicator() ? 1 : 0))
      .text("process-world-rank-order");
  return std::move(contract).release();
}

template <int Dim>
inline std::string exact_load_balance_source(std::string_view collective_contract,
                                             std::string_view collective_context_contract,
                                             const mesh::BoxArray<Dim>& patches,
                                             const mesh::RankSpace<Dim>& rank_space,
                                             parallel::LoadBalancePreparationBudget budget,
                                             LoadBalanceWeights weights) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-load-balance-source")
      .scalar(std::uint32_t{2})
      .scalar(static_cast<std::uint32_t>(Dim))
      .bytes(collective_contract)
      .bytes(collective_context_contract)
      .scalar(static_cast<std::uint64_t>(budget.patches))
      .scalar(static_cast<std::uint64_t>(budget.ranks))
      .scalar(budget.total_weight)
      .sequence(patches.boxes(), [](ExactContractBuilder& item, const Box<Dim>& patch) {
        for (int axis = 0; axis < Dim; ++axis)
          item.scalar(static_cast<std::int32_t>(patch.lo[axis]))
              .scalar(static_cast<std::int32_t>(patch.hi[axis]));
      });
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(static_cast<std::int32_t>(rank_space.origin()[axis]))
        .scalar(static_cast<std::int64_t>(rank_space.extent()[axis]));
  contract.scalar(static_cast<std::uint64_t>(rank_space.size()))
      .scalar(static_cast<std::uint8_t>(weights.empty() ? 0 : 1))
      .sequence(weights);
  return std::move(contract).release();
}

inline std::int64_t checked_add_cost(std::int64_t lhs, std::int64_t rhs, std::string_view context) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<std::int64_t>::max() - lhs)
    throw std::overflow_error(std::string(context) + " exceeds int64_t");
  return lhs + rhs;
}

inline std::int64_t estimate_weight(const ResourceEstimate& estimate, std::uint64_t topology_epoch,
                                    std::uint64_t materialization_generation) {
  if (estimate.topology_epoch != topology_epoch ||
      estimate.materialization_generation != materialization_generation)
    throw std::invalid_argument("load-balance resource estimate is stale for the live topology");
  if (estimate.samples <= 0 || estimate.cell_updates <= 0 || estimate.compute_nanoseconds < 0 ||
      estimate.memory_bytes < 0 || estimate.communication_bytes < 0 ||
      estimate.communication_nanoseconds < 0 || estimate.resident_bytes <= 0)
    throw std::invalid_argument("load-balance resource estimate is incomplete or negative");
  const std::int64_t total =
      checked_add_cost(estimate.compute_nanoseconds, estimate.communication_nanoseconds,
                       "load-balance measured time");
  if (total <= 0)
    throw std::invalid_argument("load-balance resource estimate has no measured time");
  const std::int64_t quotient = total / estimate.samples;
  return checked_add_cost(quotient, total % estimate.samples == 0 ? 0 : 1,
                          "load-balance per-sample weight");
}

template <int Dim>
inline void append_rank_space(ExactContractBuilder& contract,
                              const mesh::RankSpace<Dim>& rank_space) {
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(static_cast<std::int32_t>(rank_space.origin()[axis]))
        .scalar(static_cast<std::int64_t>(rank_space.extent()[axis]));
  contract.scalar(static_cast<std::uint64_t>(rank_space.size()));
}

template <int Dim>
inline void append_distribution(ExactContractBuilder& contract,
                                const mesh::Distribution<Dim>& distribution) {
  contract.scalar(static_cast<std::uint8_t>(distribution.mode()));
  append_rank_space(contract, distribution.rank_space());
  contract
      .sequence(distribution.layout().boxes(),
                [](ExactContractBuilder& item, const Box<Dim>& patch) {
                  for (int axis = 0; axis < Dim; ++axis)
                    item.scalar(static_cast<std::int32_t>(patch.lo[axis]))
                        .scalar(static_cast<std::int32_t>(patch.hi[axis]));
                })
      .sequence(distribution.owners(), [](ExactContractBuilder& item, const Index<Dim>& owner) {
        for (int axis = 0; axis < Dim; ++axis)
          item.scalar(static_cast<std::int32_t>(owner[axis]));
      });
}

template <int Dim>
inline std::string exact_rebalance_request(const mesh::Distribution<Dim>& current,
                                           std::uint64_t topology_epoch,
                                           std::uint64_t materialization_generation,
                                           ResourceEstimates estimates,
                                           const RebalancePolicy& policy) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-rebalance-request")
      .scalar(std::uint32_t{3})
      .scalar(static_cast<std::uint32_t>(Dim))
      .scalar(topology_epoch)
      .scalar(materialization_generation)
      .scalar(policy.minimum_improvement_ppm)
      .scalar(policy.amortization_steps)
      .scalar(policy.migration_bandwidth_bytes_per_second)
      .scalar(policy.per_patch_migration_latency_nanoseconds);
  append_distribution(contract, current);
  contract.scalar(static_cast<std::uint64_t>(estimates.size()));
  for (const ResourceEstimate& estimate : estimates) {
    contract.scalar(estimate.topology_epoch)
        .scalar(estimate.materialization_generation)
        .scalar(estimate.samples)
        .scalar(estimate.cell_updates)
        .scalar(estimate.compute_nanoseconds)
        .scalar(estimate.memory_bytes)
        .scalar(estimate.communication_bytes)
        .scalar(estimate.communication_nanoseconds)
        .scalar(estimate.resident_bytes);
  }
  return std::move(contract).release();
}

template <int Dim>
inline std::string exact_rebalance_source(std::string_view authority_identity,
                                          std::string_view authority_collective_contract,
                                          int source_level, std::uint64_t topology_epoch,
                                          std::uint64_t materialization_generation,
                                          const mesh::Distribution<Dim>& current) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-rebalance-source")
      .scalar(std::uint32_t{3})
      .scalar(static_cast<std::uint32_t>(Dim))
      .text(authority_identity)
      .bytes(authority_collective_contract)
      .scalar(source_level)
      .scalar(topology_epoch)
      .scalar(materialization_generation);
  append_distribution(contract, current);
  return std::move(contract).release();
}

template <int Dim>
inline std::vector<std::int64_t> rank_costs(const mesh::Distribution<Dim>& distribution,
                                            LoadBalanceWeights weights) {
  if (distribution.mode() != mesh::DistributionMode::partitioned ||
      distribution.owners().size() != weights.size())
    throw std::invalid_argument(
        "rebalance requires a partitioned distribution and one weight per patch");
  std::vector<std::int64_t> costs(distribution.rank_space().size(), 0);
  for (std::size_t patch = 0; patch < weights.size(); ++patch) {
    const std::size_t owner = distribution.rank_space().linear_rank(distribution.owners()[patch]);
    costs[owner] =
        checked_add_cost(costs[owner], weights[patch], "rebalance per-rank measured cost");
  }
  return costs;
}

template <int Dim>
inline std::int64_t maximum_rank_cost(const mesh::Distribution<Dim>& distribution,
                                      LoadBalanceWeights weights) {
  const std::vector<std::int64_t> costs = rank_costs(distribution, weights);
  return costs.empty() ? 0 : *std::max_element(costs.begin(), costs.end());
}

template <int Dim>
inline double measured_imbalance(const mesh::Distribution<Dim>& distribution,
                                 LoadBalanceWeights weights) {
  if (weights.empty())
    return 1.0;
  std::int64_t total = 0;
  for (const std::int64_t weight : weights)
    total = checked_add_cost(total, weight, "rebalance measured total");
  if (total == 0 || distribution.rank_space().empty())
    return 1.0;
  const double average =
      static_cast<double>(total) / static_cast<double>(distribution.rank_space().size());
  return static_cast<double>(maximum_rank_cost(distribution, weights)) / average;
}

inline std::int64_t migration_time_nanoseconds(std::int64_t bytes, std::int64_t moved_patches,
                                               const RebalancePolicy& policy) {
  if (bytes < 0 || moved_patches < 0 || policy.migration_bandwidth_bytes_per_second <= 0 ||
      policy.per_patch_migration_latency_nanoseconds < 0)
    throw std::invalid_argument("rebalance migration cost model is invalid");
  const long double transfer =
      std::ceil(static_cast<long double>(bytes) * 1.0e9L /
                static_cast<long double>(policy.migration_bandwidth_bytes_per_second));
  const long double latency =
      static_cast<long double>(moved_patches) * policy.per_patch_migration_latency_nanoseconds;
  const long double total = transfer + latency;
  if (total > static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("rebalance migration time exceeds int64_t");
  return static_cast<std::int64_t>(total);
}

template <int Dim>
inline std::string exact_rebalance_decision(const PreparedRebalanceDecision<Dim>& decision) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-rebalance-decision")
      .scalar(std::uint32_t{3})
      .scalar(static_cast<std::uint32_t>(Dim))
      .scalar(decision.topology_epoch)
      .scalar(decision.materialization_generation)
      .bytes(decision.source_contract)
      .bytes(decision.proposed.exact_contract())
      .scalar(static_cast<std::uint8_t>(decision.reason))
      .scalar(static_cast<std::uint8_t>(decision.accepted ? 1 : 0))
      .scalar(decision.moved_patches)
      .scalar(decision.migration_bytes)
      .scalar(decision.migration_nanoseconds)
      .scalar(decision.current_max_nanoseconds_per_step)
      .scalar(decision.proposed_max_nanoseconds_per_step)
      .scalar(decision.current_imbalance)
      .scalar(decision.proposed_imbalance)
      .scalar(decision.predicted_net_speedup);
  return std::move(contract).release();
}

template <int Dim>
inline void validate_prepared_load_balance_request(const mesh::BoxArray<Dim>& patches,
                                                   const mesh::RankSpace<Dim>& rank_space,
                                                   parallel::LoadBalancePreparationBudget budget,
                                                   LoadBalanceWeights weights,
                                                   int communicator_size) {
  if (communicator_size <= 0 || rank_space.empty() ||
      rank_space.size() != static_cast<std::size_t>(communicator_size))
    throw std::invalid_argument(
        "prepared load-balance rank space must equal the execution communicator");
  if (budget.patches == 0 || budget.ranks == 0 || budget.total_weight <= 0)
    throw std::invalid_argument("prepared load-balance budgets must be strictly positive");
  if (patches.size() > budget.patches)
    throw std::length_error("prepared load-balance patch count exceeds its budget");
  if (rank_space.size() > budget.ranks)
    throw std::length_error("prepared load-balance rank count exceeds its budget");
  if (!weights.empty() && weights.size() != patches.size())
    throw std::invalid_argument("prepared load-balance weight count must equal its patch count");

  std::int64_t total_weight = 0;
  for (std::size_t patch = 0; patch < patches.size(); ++patch) {
    if (patches[patch].empty())
      throw std::invalid_argument("prepared load-balance layouts cannot contain empty patches");
    const std::int64_t weight = weights.empty() ? patches[patch].numPts() : weights[patch];
    if (weight <= 0)
      throw std::invalid_argument("prepared load-balance weights must be strictly positive");
    if (weight > std::numeric_limits<std::int64_t>::max() - total_weight)
      throw std::overflow_error("prepared load-balance total weight exceeds int64_t");
    if (weight > budget.total_weight - total_weight)
      throw std::length_error("prepared load-balance total weight exceeds its budget");
    total_weight += weight;
  }
}

template <int Dim>
inline void validate_prepared_load_balance_plan(const parallel::OwnershipPlan<Dim>& plan,
                                                const mesh::BoxArray<Dim>& patches,
                                                const mesh::RankSpace<Dim>& rank_space,
                                                LoadBalanceWeights supplied_weights) {
  const auto& distribution = plan.distribution();
  if (!distribution.matches_layout(patches) || distribution.rank_space() != rank_space ||
      distribution.mode() != mesh::DistributionMode::partitioned)
    throw std::invalid_argument(
        "prepared load-balance provider returned a distribution for another source");

  const std::size_t patch_count = patches.size();
  const std::size_t rank_count = rank_space.size();
  if (distribution.owners().size() != patch_count || plan.weights().size() != patch_count ||
      plan.linear_owners().size() != patch_count || plan.traversal().size() != patch_count ||
      plan.linear_rank_loads().size() != rank_count)
    throw std::invalid_argument(
        "prepared load-balance provider returned inconsistent plan cardinalities");
  if (plan.strategy() != parallel::LoadBalanceStrategy::space_filling_curve &&
      plan.strategy() != parallel::LoadBalanceStrategy::weighted_lpt &&
      plan.strategy() != parallel::LoadBalanceStrategy::round_robin)
    throw std::invalid_argument("prepared load-balance provider returned an invalid strategy");

  std::vector<char> traversed(patch_count, 0);
  for (const std::size_t patch : plan.traversal()) {
    if (patch >= patch_count || traversed[patch] != 0)
      throw std::invalid_argument(
          "prepared load-balance traversal must be one exact patch permutation");
    traversed[patch] = 1;
  }

  std::vector<std::int64_t> rank_loads(rank_count, 0);
  std::int64_t total_weight = 0;
  for (std::size_t patch = 0; patch < patch_count; ++patch) {
    const std::int64_t expected_weight =
        supplied_weights.empty() ? patches[patch].numPts() : supplied_weights[patch];
    if (plan.weights()[patch] != expected_weight)
      throw std::invalid_argument(
          "prepared load-balance provider changed an authenticated patch weight");
    const std::size_t linear_owner = plan.linear_owners()[patch];
    if (linear_owner >= rank_count ||
        distribution.owners()[patch] != rank_space.coordinate(linear_owner))
      throw std::invalid_argument(
          "prepared load-balance spatial and linear owners are inconsistent");
    rank_loads[linear_owner] = checked_add_cost(rank_loads[linear_owner], expected_weight,
                                                "prepared load-balance rank load");
    total_weight =
        checked_add_cost(total_weight, expected_weight, "prepared load-balance total weight");
  }
  if (plan.linear_rank_loads() != rank_loads || plan.total_weight() != total_weight)
    throw std::invalid_argument(
        "prepared load-balance provider returned inconsistent retained loads");
}

template <int Dim>
inline std::string exact_load_balance_result(std::string_view source_contract,
                                             const parallel::OwnershipPlan<Dim>& plan) {
  const auto& distribution = plan.distribution();
  ExactContractBuilder contract;
  contract.text("pops.prepared-load-balance-result")
      .scalar(std::uint32_t{2})
      .scalar(static_cast<std::uint32_t>(Dim))
      .bytes(source_contract)
      .scalar(static_cast<std::uint8_t>(plan.strategy()))
      .scalar(static_cast<std::uint8_t>(distribution.mode()))
      .sequence(distribution.layout().boxes(),
                [](ExactContractBuilder& item, const Box<Dim>& patch) {
                  for (int axis = 0; axis < Dim; ++axis)
                    item.scalar(static_cast<std::int32_t>(patch.lo[axis]))
                        .scalar(static_cast<std::int32_t>(patch.hi[axis]));
                });
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(static_cast<std::int32_t>(distribution.rank_space().origin()[axis]))
        .scalar(static_cast<std::int64_t>(distribution.rank_space().extent()[axis]));
  contract.scalar(static_cast<std::uint64_t>(distribution.rank_space().size()))
      .sequence(distribution.owners(),
                [](ExactContractBuilder& item, const Index<Dim>& owner) {
                  for (int axis = 0; axis < Dim; ++axis)
                    item.scalar(static_cast<std::int32_t>(owner[axis]));
                })
      .sequence(plan.weights())
      .sequence(plan.linear_owners(),
                [](ExactContractBuilder& item, std::size_t owner) {
                  item.scalar(static_cast<std::uint64_t>(owner));
                })
      .sequence(plan.linear_rank_loads())
      .sequence(plan.traversal(),
                [](ExactContractBuilder& item, std::size_t patch) {
                  item.scalar(static_cast<std::uint64_t>(patch));
                })
      .scalar(plan.total_weight());
  return std::move(contract).release();
}

template <class Operation>
inline void collective_load_balance_preflight(std::string_view context,
                                              const CommunicatorView& communicator,
                                              Operation&& operation) {
  std::string local_error;
  try {
    std::invoke(std::forward<Operation>(operation));
  } catch (const std::exception& error) {
    local_error = error.what();
  } catch (...) {
    local_error = "unknown native exception";
  }
  if (all_reduce_max(local_error.empty() ? 0L : 1L, communicator) != 0) {
    std::string message(context);
    message += " failed on at least one rank";
    if (!local_error.empty())
      message += ": " + local_error;
    throw std::invalid_argument(message);
  }
}

inline void require_empty_load_balance_options(const PreparedProviderOptions& options,
                                               std::string_view expected_schema) {
  if (options.schema_identity != expected_schema || !options.values.empty())
    throw std::invalid_argument("builtin load-balance provider options are not canonical");
}

inline std::int64_t require_signed_option(const PreparedProviderOptions& options,
                                          std::string_view key) {
  const auto found = options.values.find(std::string(key));
  if (found == options.values.end() || !std::holds_alternative<std::int64_t>(found->second))
    throw std::invalid_argument("measured load-balance option '" + std::string(key) +
                                "' must be one exact int64");
  return std::get<std::int64_t>(found->second);
}

inline RebalancePolicy measured_rebalance_policy(const PreparedProviderOptions& options) {
  static const std::string schema = "pops.amr.load-balance.measured-knapsack@1";
  static const std::array<std::string_view, 4> keys{
      "minimum_improvement_ppm",
      "amortization_steps",
      "migration_bandwidth_bytes_per_second",
      "per_patch_migration_latency_nanoseconds",
  };
  if (options.schema_identity != schema || options.values.size() != keys.size())
    throw std::invalid_argument("measured knapsack options are not canonical");
  for (const std::string_view key : keys)
    if (!options.values.contains(std::string(key)))
      throw std::invalid_argument("measured knapsack options are not canonical");
  RebalancePolicy policy{
      .minimum_improvement_ppm = require_signed_option(options, keys[0]),
      .amortization_steps = require_signed_option(options, keys[1]),
      .migration_bandwidth_bytes_per_second = require_signed_option(options, keys[2]),
      .per_patch_migration_latency_nanoseconds = require_signed_option(options, keys[3]),
  };
  if (policy.minimum_improvement_ppm < 0 || policy.minimum_improvement_ppm >= 1'000'000 ||
      policy.amortization_steps <= 0 || policy.migration_bandwidth_bytes_per_second <= 0 ||
      policy.per_patch_migration_latency_nanoseconds < 0)
    throw std::invalid_argument("measured knapsack policy is outside its bounded envelope");
  return policy;
}

struct SpaceFillingCurveLoadBalance {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.load_balance.space_filling_curve", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("space-filling-curve")
        .scalar(std::uint32_t{2})
        .text("compile-time-ranked-ownership-plan");
  }
  template <int Dim>
  parallel::OwnershipPlan<Dim> operator()(const mesh::BoxArray<Dim>& patches,
                                          const mesh::RankSpace<Dim>& rank_space,
                                          parallel::LoadBalancePreparationBudget budget,
                                          LoadBalanceWeights weights) const {
    return parallel::LoadBalanceProvider<Dim>::space_filling_curve().prepare(patches, rank_space,
                                                                             budget, weights);
  }
};

struct KnapsackLoadBalance {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.load_balance.knapsack", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("knapsack").scalar(std::uint32_t{2}).text("compile-time-ranked-ownership-plan");
  }
  template <int Dim>
  parallel::OwnershipPlan<Dim> operator()(const mesh::BoxArray<Dim>& patches,
                                          const mesh::RankSpace<Dim>& rank_space,
                                          parallel::LoadBalancePreparationBudget budget,
                                          LoadBalanceWeights weights) const {
    return parallel::LoadBalanceProvider<Dim>::weighted_lpt().prepare(patches, rank_space, budget,
                                                                      weights);
  }
};

struct MeasuredKnapsackLoadBalance {
  RebalancePolicy policy;

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.load_balance.measured_knapsack", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("measured-knapsack")
        .scalar(std::uint32_t{2})
        .text("compile-time-ranked-ownership-plan")
        .scalar(policy.minimum_improvement_ppm)
        .scalar(policy.amortization_steps)
        .scalar(policy.migration_bandwidth_bytes_per_second)
        .scalar(policy.per_patch_migration_latency_nanoseconds);
  }
  template <int Dim>
  parallel::OwnershipPlan<Dim> operator()(const mesh::BoxArray<Dim>& patches,
                                          const mesh::RankSpace<Dim>& rank_space,
                                          parallel::LoadBalancePreparationBudget budget,
                                          LoadBalanceWeights weights) const {
    return parallel::LoadBalanceProvider<Dim>::weighted_lpt().prepare(patches, rank_space, budget,
                                                                      weights);
  }
};

struct RoundRobinLoadBalance {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.load_balance.round_robin", 2};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    // Round-robin is deliberately index based.  Supplied weights are still validated and enter the
    // collective request contract, but this provider intentionally does not consume them when it
    // chooses owners.  Keep that capability in the prepared identity so the behavior can neither be
    // mistaken for a weighted policy nor changed silently in a later implementation.
    contract.text("round-robin")
        .scalar(std::uint32_t{2})
        .text("compile-time-ranked-ownership-plan")
        .text("weights-authenticated-index-policy");
  }
  template <int Dim>
  parallel::OwnershipPlan<Dim> operator()(const mesh::BoxArray<Dim>& patches,
                                          const mesh::RankSpace<Dim>& rank_space,
                                          parallel::LoadBalancePreparationBudget budget,
                                          LoadBalanceWeights weights) const {
    return parallel::LoadBalanceProvider<Dim>::round_robin().prepare(patches, rank_space, budget,
                                                                     weights);
  }
};

}  // namespace detail

/// Evaluate an authenticated candidate ownership plan without mutating hierarchy state.
template <int Dim>
inline PreparedRebalanceDecision<Dim> make_rebalance_decision(
    const mesh::BoxArray<Dim>& patches, const mesh::Distribution<Dim>& current,
    PreparedLoadBalanceResult<Dim> proposed, std::uint64_t topology_epoch,
    std::uint64_t materialization_generation, ResourceEstimates estimates,
    const RebalancePolicy& policy, std::string source_contract) {
  const mesh::Distribution<Dim>& candidate = proposed.plan().distribution();
  if (!current.matches_layout(patches) || !candidate.matches_layout(patches) ||
      current.mode() != mesh::DistributionMode::partitioned ||
      candidate.mode() != mesh::DistributionMode::partitioned ||
      current.rank_space() != candidate.rank_space() || estimates.size() != patches.size() ||
      source_contract.empty())
    throw std::invalid_argument(
        "prepared rebalance distributions, estimates, and source must describe one layout");
  if (policy.minimum_improvement_ppm < 0 || policy.minimum_improvement_ppm >= 1'000'000 ||
      policy.amortization_steps <= 0 || policy.migration_bandwidth_bytes_per_second <= 0 ||
      policy.per_patch_migration_latency_nanoseconds < 0)
    throw std::invalid_argument("rebalance policy is outside its exact bounded envelope");

  std::vector<std::int64_t> weights;
  weights.reserve(estimates.size());
  for (const ResourceEstimate& estimate : estimates)
    weights.push_back(
        detail::estimate_weight(estimate, topology_epoch, materialization_generation));
  if (proposed.plan().weights() != weights)
    throw std::invalid_argument(
        "prepared rebalance proposal does not retain the authenticated measured weights");

  PreparedRebalanceDecision<Dim> decision{
      .topology_epoch = topology_epoch,
      .materialization_generation = materialization_generation,
      .source_contract = std::move(source_contract),
      .proposed = std::move(proposed),
  };
  if (patches.empty()) {
    decision.reason = RebalanceReason::EmptyHierarchy;
    decision.exact_contract = detail::exact_rebalance_decision(decision);
    return decision;
  }

  const mesh::Distribution<Dim>& proposed_distribution = decision.proposed.plan().distribution();
  decision.current_max_nanoseconds_per_step = detail::maximum_rank_cost(current, weights);
  decision.proposed_max_nanoseconds_per_step =
      detail::maximum_rank_cost(proposed_distribution, weights);
  decision.current_imbalance = detail::measured_imbalance(current, weights);
  decision.proposed_imbalance = detail::measured_imbalance(proposed_distribution, weights);
  for (std::size_t patch = 0; patch < patches.size(); ++patch) {
    if (current.owners()[patch] == proposed_distribution.owners()[patch])
      continue;
    ++decision.moved_patches;
    decision.migration_bytes = detail::checked_add_cost(
        decision.migration_bytes, estimates[patch].resident_bytes, "rebalance migration bytes");
  }
  decision.migration_nanoseconds =
      detail::migration_time_nanoseconds(decision.migration_bytes, decision.moved_patches, policy);

  const long double current_horizon =
      static_cast<long double>(decision.current_max_nanoseconds_per_step) *
      policy.amortization_steps;
  const long double proposed_horizon =
      static_cast<long double>(decision.proposed_max_nanoseconds_per_step) *
          policy.amortization_steps +
      decision.migration_nanoseconds;
  if (!(current_horizon > 0.0L) || !(proposed_horizon > 0.0L))
    throw std::invalid_argument("rebalance measured horizon must be strictly positive");
  decision.predicted_net_speedup = static_cast<double>(current_horizon / proposed_horizon);
  const long double required_fraction =
      1.0L - static_cast<long double>(policy.minimum_improvement_ppm) / 1.0e6L;
  decision.accepted =
      decision.moved_patches > 0 && proposed_horizon <= current_horizon * required_fraction;
  if (decision.moved_patches == 0)
    decision.reason = RebalanceReason::MappingUnchanged;
  else if (decision.accepted)
    decision.reason = RebalanceReason::NetBenefit;
  else
    decision.reason = RebalanceReason::InsufficientNetBenefit;
  decision.exact_contract = detail::exact_rebalance_decision(decision);
  return decision;
}

/// Immutable compile-time-ranked authority prepared before hierarchy materialization.
///
/// The provider is invoked exactly once.  Request provenance and every retained OwnershipPlan
/// component are then authenticated collectively before the result can escape this method.
template <int Dim>
class PreparedLoadBalanceAuthority {
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedLoadBalanceAuthority only supports dimensions 1, 2, and 3");

 public:
  PreparedLoadBalanceAuthority(
      std::string semantic_identity, PreparedLoadBalanceProvider<Dim> provider,
      std::optional<RebalancePolicy> default_rebalance_policy = std::nullopt)
      : semantic_identity_(std::move(semantic_identity)),
        provider_(std::move(provider)),
        default_rebalance_policy_(std::move(default_rebalance_policy)) {
    if (semantic_identity_.empty() || !provider_)
      throw std::invalid_argument("prepared load-balance authority is incomplete");
    if (default_rebalance_policy_) {
      const RebalancePolicy& policy = *default_rebalance_policy_;
      if (policy.minimum_improvement_ppm < 0 || policy.minimum_improvement_ppm >= 1'000'000 ||
          policy.amortization_steps <= 0 || policy.migration_bandwidth_bytes_per_second <= 0 ||
          policy.per_patch_migration_latency_nanoseconds < 0)
        throw std::invalid_argument("prepared load-balance default rebalance policy is invalid");
    }
    collective_contract_ = detail::exact_load_balance_collective<Dim>(
        semantic_identity_, provider_.collective_contract(), default_rebalance_policy_);
  }

  [[nodiscard]] const std::string& semantic_identity() const noexcept { return semantic_identity_; }
  [[nodiscard]] const std::string& implementation() const noexcept {
    return provider_.implementation();
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }
  [[nodiscard]] std::string_view provider_collective_contract() const noexcept {
    return provider_.collective_contract();
  }
  [[nodiscard]] bool has_default_rebalance_policy() const noexcept {
    return default_rebalance_policy_.has_value();
  }
  [[nodiscard]] const RebalancePolicy& default_rebalance_policy() const {
    if (!default_rebalance_policy_)
      throw std::logic_error("load-balance authority has no measured rebalance policy");
    return *default_rebalance_policy_;
  }

  [[nodiscard]] std::string rebalance_source_contract(
      int source_level, const mesh::Distribution<Dim>& current, std::uint64_t topology_epoch,
      std::uint64_t materialization_generation) const {
    if (source_level < 0 || current.mode() != mesh::DistributionMode::partitioned ||
        current.rank_space().empty())
      throw std::invalid_argument("prepared rebalance source is not a partitioned hierarchy level");
    return detail::exact_rebalance_source(semantic_identity_, collective_contract_, source_level,
                                          topology_epoch, materialization_generation, current);
  }

  [[nodiscard]] PreparedLoadBalanceResult<Dim> prepare(
      const mesh::BoxArray<Dim>& patches, const mesh::RankSpace<Dim>& rank_space,
      parallel::LoadBalancePreparationBudget budget, LoadBalanceWeights weights = {},
      const ExecutionLane& lane = ExecutionLane::world()) const {
    const CommunicatorView communicator = lane.communicator();
    std::string collective_context_contract;
    std::string source_contract;
    detail::collective_load_balance_preflight("prepared load-balance source", communicator, [&] {
      detail::validate_prepared_load_balance_request(patches, rank_space, budget, weights,
                                                     communicator.size());
      collective_context_contract = detail::exact_load_balance_collective_context(lane);
      source_contract = detail::exact_load_balance_source<Dim>(
          collective_contract_, collective_context_contract, patches, rank_space, budget, weights);
    });

    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"collective", collective_contract_},
             {"collective-context", collective_context_contract},
             {"source", source_contract}},
            communicator))
      throw std::invalid_argument(
          "prepared load-balance authority or source differs across MPI ranks");

    std::optional<parallel::OwnershipPlan<Dim>> plan;
    detail::collective_load_balance_preflight("prepared load-balance provider", communicator, [&] {
      plan.emplace(provider_(patches, rank_space, budget, weights));
    });

    std::string exact_contract;
    detail::collective_load_balance_preflight("prepared load-balance result", communicator, [&] {
      if (!plan)
        throw std::logic_error("prepared load-balance provider produced no ownership plan");
      detail::validate_prepared_load_balance_plan(*plan, patches, rank_space, weights);
      exact_contract = detail::exact_load_balance_result<Dim>(source_contract, *plan);
    });
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"collective", collective_contract_},
             {"collective-context", collective_context_contract},
             {"result", exact_contract}},
            communicator))
      throw std::invalid_argument(
          "prepared load-balance provider returned different plans across MPI ranks");
    return PreparedLoadBalanceResult<Dim>(std::move(*plan), collective_contract_,
                                          std::move(collective_context_contract),
                                          std::move(source_contract), std::move(exact_contract));
  }

  /// Produce one collective, topology-qualified migration decision from measured patch costs.
  /// The hierarchy remains unchanged until it validates and consumes the returned source contract.
  [[nodiscard]] PreparedRebalanceDecision<Dim> decide_rebalance(
      int source_level, const mesh::BoxArray<Dim>& patches, const mesh::Distribution<Dim>& current,
      std::uint64_t topology_epoch, std::uint64_t materialization_generation,
      ResourceEstimates estimates, parallel::LoadBalancePreparationBudget preparation_budget,
      const RebalancePolicy& policy, const ExecutionLane& lane = ExecutionLane::world()) const {
    const CommunicatorView communicator = lane.communicator();
    std::vector<std::int64_t> weights;
    std::string request_contract;
    std::string source_contract;
    detail::collective_load_balance_preflight("prepared rebalance request", communicator, [&] {
      if (source_level < 0 || current.mode() != mesh::DistributionMode::partitioned ||
          !current.matches_layout(patches) ||
          current.rank_space().size() != static_cast<std::size_t>(communicator.size()) ||
          estimates.size() != patches.size())
        throw std::invalid_argument(
            "prepared rebalance source must be one partitioned level on the execution rank space");
      if (policy.minimum_improvement_ppm < 0 || policy.minimum_improvement_ppm >= 1'000'000 ||
          policy.amortization_steps <= 0 || policy.migration_bandwidth_bytes_per_second <= 0 ||
          policy.per_patch_migration_latency_nanoseconds < 0)
        throw std::invalid_argument("rebalance policy is outside its exact bounded envelope");
      weights.reserve(estimates.size());
      for (const ResourceEstimate& estimate : estimates)
        weights.push_back(
            detail::estimate_weight(estimate, topology_epoch, materialization_generation));
      request_contract = detail::exact_rebalance_request(
          current, topology_epoch, materialization_generation, estimates, policy);
      source_contract = rebalance_source_contract(source_level, current, topology_epoch,
                                                  materialization_generation);
    });

    if (!all_ranks_agree_exact_ordered_byte_pairs({{"collective", collective_contract_},
                                                   {"rebalance-source", source_contract},
                                                   {"rebalance-request", request_contract}},
                                                  communicator))
      throw std::invalid_argument(
          "prepared rebalance authority or measured request differs across MPI ranks");

    PreparedLoadBalanceResult<Dim> proposal =
        prepare(patches, current.rank_space(), preparation_budget, weights, lane);
    std::optional<PreparedRebalanceDecision<Dim>> result;
    detail::collective_load_balance_preflight("prepared rebalance decision", communicator, [&] {
      result.emplace(make_rebalance_decision(patches, current, std::move(proposal), topology_epoch,
                                             materialization_generation, estimates, policy,
                                             source_contract));
    });
    if (!result)
      throw std::logic_error("prepared rebalance decision was not materialized");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"collective", collective_contract_}, {"decision", result->exact_contract}},
            communicator))
      throw std::invalid_argument("prepared rebalance decision differs across MPI ranks");
    return std::move(*result);
  }

  [[nodiscard]] PreparedRebalanceDecision<Dim> decide_rebalance(
      int source_level, const mesh::BoxArray<Dim>& patches, const mesh::Distribution<Dim>& current,
      std::uint64_t topology_epoch, std::uint64_t materialization_generation,
      ResourceEstimates estimates, parallel::LoadBalancePreparationBudget preparation_budget,
      const ExecutionLane& lane = ExecutionLane::world()) const {
    return decide_rebalance(source_level, patches, current, topology_epoch,
                            materialization_generation, estimates, preparation_budget,
                            default_rebalance_policy(), lane);
  }

 private:
  std::string semantic_identity_;
  PreparedLoadBalanceProvider<Dim> provider_;
  std::optional<RebalancePolicy> default_rebalance_policy_;
  std::string collective_contract_;
};

template <int Dim>
using LoadBalanceAuthorityFactory = std::function<PreparedLoadBalanceAuthority<Dim>(
    std::string semantic_identity, const PreparedProviderOptions& options)>;

/// Open preparation registry.  Route lookup occurs once during bind; hierarchy/regrid code keeps
/// only one PreparedLoadBalanceAuthority<Dim> and cannot branch on route or concrete provider type.
template <int Dim>
class LoadBalanceProviderRegistry {
 public:
  void add(std::string route, LoadBalanceAuthorityFactory<Dim> factory) {
    if (route.empty() || !factory)
      throw std::invalid_argument("load-balance provider registration is incomplete");
    std::lock_guard<std::mutex> guard(mutex_);
    if (!factories_.emplace(std::move(route), std::move(factory)).second)
      throw std::invalid_argument("load-balance provider route is already registered");
  }

  [[nodiscard]] PreparedLoadBalanceAuthority<Dim> prepare(
      std::string_view route, std::string semantic_identity,
      const PreparedProviderOptions& options) const {
    LoadBalanceAuthorityFactory<Dim> factory;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      const auto found = factories_.find(std::string(route));
      if (found == factories_.end())
        throw std::invalid_argument("load-balance provider route is not registered");
      factory = found->second;
    }
    return factory(std::move(semantic_identity), options);
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::string, LoadBalanceAuthorityFactory<Dim>, std::less<>> factories_;
};

template <int Dim>
inline LoadBalanceProviderRegistry<Dim>& load_balance_provider_registry() {
  static LoadBalanceProviderRegistry<Dim> registry;
  static std::once_flag builtins;
  std::call_once(builtins, [&] {
    registry.add("space_filling_curve",
                 [](std::string identity, const PreparedProviderOptions& options) {
                   detail::require_empty_load_balance_options(
                       options, "pops.amr.load-balance.space-filling-curve@1");
                   return PreparedLoadBalanceAuthority<Dim>(
                       std::move(identity),
                       PreparedLoadBalanceProvider<Dim>(detail::SpaceFillingCurveLoadBalance{}));
                 });
    registry.add("knapsack", [](std::string identity, const PreparedProviderOptions& options) {
      detail::require_empty_load_balance_options(options, "pops.amr.load-balance.knapsack@1");
      return PreparedLoadBalanceAuthority<Dim>(
          std::move(identity), PreparedLoadBalanceProvider<Dim>(detail::KnapsackLoadBalance{}));
    });
    registry.add("measured_knapsack", [](std::string identity,
                                         const PreparedProviderOptions& options) {
      const RebalancePolicy policy = detail::measured_rebalance_policy(options);
      return PreparedLoadBalanceAuthority<Dim>(
          std::move(identity),
          PreparedLoadBalanceProvider<Dim>(detail::MeasuredKnapsackLoadBalance{policy}), policy);
    });
    registry.add("round_robin", [](std::string identity, const PreparedProviderOptions& options) {
      detail::require_empty_load_balance_options(options, "pops.amr.load-balance.round-robin@1");
      return PreparedLoadBalanceAuthority<Dim>(
          std::move(identity), PreparedLoadBalanceProvider<Dim>(detail::RoundRobinLoadBalance{}));
    });
  });
  return registry;
}

template <int Dim>
inline void register_load_balance_provider(std::string route,
                                           LoadBalanceAuthorityFactory<Dim> factory) {
  load_balance_provider_registry<Dim>().add(std::move(route), std::move(factory));
}

template <int Dim>
inline PreparedLoadBalanceAuthority<Dim> prepare_load_balance_authority(
    std::string_view route, std::string semantic_identity, const PreparedProviderOptions& options) {
  return load_balance_provider_registry<Dim>().prepare(route, std::move(semantic_identity),
                                                       options);
}

}  // namespace pops
