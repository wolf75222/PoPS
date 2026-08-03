/// @file
/// @brief Prepared, extensible ownership authority for AMR BoxArray layouts.

#pragma once

#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/load_balance.hpp>

#include <cstdint>
#include <cmath>
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
using PreparedLoadBalanceProvider =
    PreparedProvider<DistributionMapping(const BoxArray&, int, LoadBalanceWeights)>;

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

/// Immutable measured decision.  ``proposed_mapping`` is never silently applied; the hierarchy
/// must consume it through its migration transaction when ``accepted`` is true.
struct RebalanceDecision {
  std::uint64_t topology_epoch = 0;
  std::uint64_t materialization_generation = 0;
  /// Exact prepared-authority, level, BoxArray and current-owner identity consumed by migration.
  std::string source_contract;
  DistributionMapping proposed_mapping;
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

inline std::string exact_load_balance_request(const BoxArray& boxes, int rank_count,
                                              LoadBalanceWeights weights) {
  ExactContractBuilder contract;
  contract.text("pops.load-balance-request")
      .scalar(std::uint32_t{1})
      .scalar(static_cast<std::int32_t>(rank_count))
      .scalar(static_cast<std::int32_t>(boxes.size()))
      .scalar(static_cast<std::uint8_t>(weights.empty() ? 0 : 1));
  for (int index = 0; index < boxes.size(); ++index) {
    const Box2D& box = boxes[index];
    contract.scalar(static_cast<std::int32_t>(box.lo[0]))
        .scalar(static_cast<std::int32_t>(box.lo[1]))
        .scalar(static_cast<std::int32_t>(box.hi[0]))
        .scalar(static_cast<std::int32_t>(box.hi[1]));
  }
  contract.sequence(weights);
  return std::move(contract).release();
}

inline std::string exact_load_balance_mapping(const DistributionMapping& mapping) {
  ExactContractBuilder contract;
  contract.text("pops.load-balance-mapping").scalar(std::uint32_t{1});
  contract.scalar(static_cast<std::int32_t>(mapping.size()));
  for (const int owner : mapping.ranks())
    contract.scalar(static_cast<std::int32_t>(owner));
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
  const std::int64_t remainder = total % estimate.samples;
  return checked_add_cost(quotient, remainder == 0 ? 0 : 1, "load-balance per-sample weight");
}

inline std::string exact_rebalance_request(const DistributionMapping& current,
                                           std::uint64_t topology_epoch,
                                           std::uint64_t materialization_generation,
                                           ResourceEstimates estimates,
                                           const RebalancePolicy& policy) {
  ExactContractBuilder contract;
  contract.text("pops.rebalance-request")
      .scalar(std::uint32_t{1})
      .scalar(topology_epoch)
      .scalar(materialization_generation)
      .scalar(policy.minimum_improvement_ppm)
      .scalar(policy.amortization_steps)
      .scalar(policy.migration_bandwidth_bytes_per_second)
      .scalar(policy.per_patch_migration_latency_nanoseconds)
      .sequence(current.ranks());
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

inline std::string exact_rebalance_source(std::string_view authority_identity,
                                          std::string_view authority_collective_contract,
                                          int source_level, int source_rank_count,
                                          std::uint64_t topology_epoch,
                                          std::uint64_t materialization_generation,
                                          const BoxArray& source_boxes,
                                          const DistributionMapping& source_mapping) {
  ExactContractBuilder contract;
  contract.text("pops.rebalance-source")
      .scalar(std::uint32_t{1})
      .text(authority_identity)
      .text(authority_collective_contract)
      .scalar(source_level)
      .scalar(source_rank_count)
      .scalar(topology_epoch)
      .scalar(materialization_generation)
      .scalar(static_cast<std::uint64_t>(source_boxes.size()));
  for (const Box2D& box : source_boxes.boxes())
    contract.scalar(box.lo[0]).scalar(box.lo[1]).scalar(box.hi[0]).scalar(box.hi[1]);
  contract.sequence(source_mapping.ranks());
  return std::move(contract).release();
}

inline std::int64_t maximum_rank_cost(const DistributionMapping& mapping, int rank_count,
                                      LoadBalanceWeights weights) {
  std::vector<std::int64_t> costs(static_cast<std::size_t>(rank_count), 0);
  for (int index = 0; index < mapping.size(); ++index) {
    const int owner = mapping[index];
    if (owner < 0 || owner >= rank_count)
      throw std::invalid_argument("rebalance mapping contains an invalid owner rank");
    costs[static_cast<std::size_t>(owner)] = checked_add_cost(
        costs[static_cast<std::size_t>(owner)], weights[static_cast<std::size_t>(index)],
        "rebalance per-rank measured cost");
  }
  return costs.empty() ? 0 : *std::max_element(costs.begin(), costs.end());
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

inline std::string exact_rebalance_decision(const RebalanceDecision& decision) {
  ExactContractBuilder contract;
  contract.text("pops.rebalance-decision")
      .scalar(std::uint32_t{2})
      .scalar(decision.topology_epoch)
      .scalar(decision.materialization_generation)
      .text(decision.source_contract)
      .scalar(static_cast<std::uint8_t>(decision.reason))
      .scalar(static_cast<std::uint8_t>(decision.accepted ? 1 : 0))
      .scalar(decision.moved_patches)
      .scalar(decision.migration_bytes)
      .scalar(decision.migration_nanoseconds)
      .scalar(decision.current_max_nanoseconds_per_step)
      .scalar(decision.proposed_max_nanoseconds_per_step)
      .scalar(decision.current_imbalance)
      .scalar(decision.proposed_imbalance)
      .scalar(decision.predicted_net_speedup)
      .sequence(decision.proposed_mapping.ranks());
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

struct SpaceFillingCurveLoadBalance {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.load_balance.space_filling_curve", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("space-filling-curve").scalar(std::uint32_t{1});
  }
  DistributionMapping operator()(const BoxArray& boxes, int ranks,
                                 LoadBalanceWeights weights) const {
    return make_sfc_distribution(boxes, ranks, weights);
  }
};

struct KnapsackLoadBalance {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.load_balance.knapsack", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("knapsack").scalar(std::uint32_t{1});
  }
  DistributionMapping operator()(const BoxArray& boxes, int ranks,
                                 LoadBalanceWeights weights) const {
    return make_knapsack_distribution(boxes, ranks, weights);
  }
};

struct RoundRobinLoadBalance {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.load_balance.round_robin", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    // Round-robin is deliberately index based.  Supplied weights are still validated and enter the
    // collective request contract, but this provider intentionally does not consume them when it
    // chooses owners.  Keep that capability in the prepared identity so the behavior can neither be
    // mistaken for a weighted policy nor changed silently in a later implementation.
    contract.text("round-robin")
        .scalar(std::uint32_t{1})
        .text("weights-authenticated-index-policy");
  }
  DistributionMapping operator()(const BoxArray& boxes, int ranks,
                                 LoadBalanceWeights weights) const {
    return make_round_robin_distribution(boxes, ranks, weights);
  }
};

}  // namespace detail

/// Evaluate one proposed ownership map without mutating hierarchy state.
///
/// This pure host routine is also the executable specification used by the collective authority and
/// by migration transactions: every cost is measured, every estimate is tied to the live topology,
/// and migration must be repaid over the declared horizon before adoption is allowed.
inline RebalanceDecision make_rebalance_decision(
    const BoxArray& boxes, const DistributionMapping& current, const DistributionMapping& proposed,
    int rank_count, std::uint64_t topology_epoch, std::uint64_t materialization_generation,
    ResourceEstimates estimates, const RebalancePolicy& policy, std::string source_contract) {
  if (rank_count <= 0 || current.size() != boxes.size() || proposed.size() != boxes.size() ||
      estimates.size() != static_cast<std::size_t>(boxes.size()) || source_contract.empty())
    throw std::invalid_argument(
        "rebalance mappings, estimates and source contract must match a positive-rank BoxArray");
  if (policy.minimum_improvement_ppm < 0 || policy.minimum_improvement_ppm >= 1'000'000 ||
      policy.amortization_steps <= 0 || policy.migration_bandwidth_bytes_per_second <= 0 ||
      policy.per_patch_migration_latency_nanoseconds < 0)
    throw std::invalid_argument("rebalance policy is outside its exact bounded envelope");

  std::vector<std::int64_t> weights;
  weights.reserve(estimates.size());
  for (const ResourceEstimate& estimate : estimates)
    weights.push_back(
        detail::estimate_weight(estimate, topology_epoch, materialization_generation));

  RebalanceDecision decision;
  decision.topology_epoch = topology_epoch;
  decision.materialization_generation = materialization_generation;
  decision.source_contract = std::move(source_contract);
  decision.proposed_mapping = proposed;
  if (boxes.size() == 0) {
    decision.reason = RebalanceReason::EmptyHierarchy;
    decision.exact_contract = detail::exact_rebalance_decision(decision);
    return decision;
  }

  decision.current_max_nanoseconds_per_step =
      detail::maximum_rank_cost(current, rank_count, weights);
  decision.proposed_max_nanoseconds_per_step =
      detail::maximum_rank_cost(proposed, rank_count, weights);
  decision.current_imbalance = load_imbalance(boxes, current, rank_count, weights);
  decision.proposed_imbalance = load_imbalance(boxes, proposed, rank_count, weights);
  for (int index = 0; index < boxes.size(); ++index) {
    if (current[index] == proposed[index])
      continue;
    ++decision.moved_patches;
    decision.migration_bytes = detail::checked_add_cost(
        decision.migration_bytes, estimates[static_cast<std::size_t>(index)].resident_bytes,
        "rebalance migration bytes");
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

/// Immutable authority prepared before hierarchy materialization.  Every invocation validates the
/// same provider/request/result contract collectively; regrid consumers call this object directly
/// and never inspect an implementation name.
class PreparedLoadBalanceAuthority {
 public:
  PreparedLoadBalanceAuthority(std::string semantic_identity, PreparedLoadBalanceProvider provider)
      : semantic_identity_(std::move(semantic_identity)), provider_(std::move(provider)) {
    if (semantic_identity_.empty() || !provider_)
      throw std::invalid_argument("prepared load-balance authority is incomplete");
  }

  [[nodiscard]] const std::string& semantic_identity() const noexcept { return semantic_identity_; }
  [[nodiscard]] const std::string& implementation() const noexcept {
    return provider_.implementation();
  }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return provider_.collective_contract();
  }

  [[nodiscard]] DistributionMapping distribute(
      const BoxArray& boxes, int rank_count, LoadBalanceWeights weights = {},
      const CommunicatorView& communicator = world_communicator_view()) const {
    std::string request_contract;
    detail::collective_load_balance_preflight("load-balance request", communicator, [&] {
      if (rank_count <= 0 || rank_count != communicator.size())
        throw std::invalid_argument(
            "load-balance rank count must equal the execution communicator size");
      if (!weights.empty() && weights.size() != static_cast<std::size_t>(boxes.size()))
        throw std::invalid_argument("load-balance weight count must equal the BoxArray size");
      for (int index = 0; index < boxes.size(); ++index) {
        if (boxes[index].empty())
          throw std::invalid_argument("load-balance BoxArray contains an empty box");
        if (!weights.empty() && weights[static_cast<std::size_t>(index)] <= 0)
          throw std::invalid_argument("load-balance weights must be strictly positive");
      }
      request_contract = detail::exact_load_balance_request(boxes, rank_count, weights);
    });

    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{semantic_identity_, provider_.collective_contract()}, {"request", request_contract}},
            communicator))
      throw std::invalid_argument(
          "load-balance provider identity or request differs across MPI ranks");

    std::optional<DistributionMapping> mapping;
    detail::collective_load_balance_preflight("load-balance provider", communicator, [&] {
      mapping.emplace(provider_(boxes, rank_count, weights));
    });

    std::string mapping_contract;
    detail::collective_load_balance_preflight("load-balance mapping", communicator, [&] {
      if (!mapping || mapping->size() != boxes.size())
        throw std::invalid_argument("load-balance provider returned a mapping of the wrong size");
      for (const int owner : mapping->ranks())
        if (owner < 0 || owner >= rank_count)
          throw std::invalid_argument("load-balance provider returned an invalid owner rank");
      mapping_contract = detail::exact_load_balance_mapping(*mapping);
    });
    if (!all_ranks_agree_exact_ordered_byte_pairs({{semantic_identity_, mapping_contract}},
                                                  communicator))
      throw std::invalid_argument("load-balance provider returned different mappings across ranks");
    return std::move(*mapping);
  }

  /// Produce one collective, topology-qualified migration decision from measured patch costs.
  ///
  /// The method does not mutate hierarchy ownership.  It authenticates the observations, prepares
  /// a policy candidate through the same immutable authority, accounts for migration over the
  /// configured horizon, and returns a decision that a hierarchy migration transaction may consume.
  [[nodiscard]] RebalanceDecision decide_rebalance(
      int source_level, const BoxArray& boxes, const DistributionMapping& current, int rank_count,
      std::uint64_t topology_epoch, std::uint64_t materialization_generation,
      ResourceEstimates estimates, const RebalancePolicy& policy,
      const CommunicatorView& communicator = world_communicator_view()) const {
    std::vector<std::int64_t> weights;
    std::string request_contract;
    std::string source_contract;
    detail::collective_load_balance_preflight("rebalance request", communicator, [&] {
      if (source_level < 0 || rank_count <= 0 || rank_count != communicator.size())
        throw std::invalid_argument(
            "rebalance source level must be nonnegative and rank count must equal the execution "
            "communicator size");
      if (current.size() != boxes.size() ||
          estimates.size() != static_cast<std::size_t>(boxes.size()))
        throw std::invalid_argument(
            "rebalance current mapping and resource estimates must match the BoxArray");
      if (policy.minimum_improvement_ppm < 0 || policy.minimum_improvement_ppm >= 1'000'000 ||
          policy.amortization_steps <= 0 || policy.migration_bandwidth_bytes_per_second <= 0 ||
          policy.per_patch_migration_latency_nanoseconds < 0)
        throw std::invalid_argument("rebalance policy is outside its exact bounded envelope");
      for (const int owner : current.ranks())
        if (owner < 0 || owner >= rank_count)
          throw std::invalid_argument("rebalance current mapping contains an invalid owner rank");
      weights.reserve(estimates.size());
      for (const ResourceEstimate& estimate : estimates)
        weights.push_back(
            detail::estimate_weight(estimate, topology_epoch, materialization_generation));
      request_contract = detail::exact_rebalance_request(
          current, topology_epoch, materialization_generation, estimates, policy);
      source_contract = detail::exact_rebalance_source(
          semantic_identity_, provider_.collective_contract(), source_level, rank_count,
          topology_epoch, materialization_generation, boxes, current);
    });

    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{semantic_identity_, provider_.collective_contract()},
             {"rebalance-source", source_contract},
             {"rebalance-request", request_contract}},
            communicator))
      throw std::invalid_argument(
          "rebalance provider identity or measured request differs across MPI ranks");

    DistributionMapping proposed = distribute(boxes, rank_count, weights, communicator);
    std::optional<RebalanceDecision> result;
    detail::collective_load_balance_preflight("rebalance decision", communicator, [&] {
      result.emplace(make_rebalance_decision(boxes, current, proposed, rank_count, topology_epoch,
                                             materialization_generation, estimates, policy,
                                             source_contract));
    });
    if (!result)
      throw std::logic_error("rebalance decision was not materialized");
    if (!all_ranks_agree_exact_ordered_byte_pairs({{semantic_identity_, result->exact_contract}},
                                                  communicator))
      throw std::invalid_argument("rebalance decision differs across MPI ranks");
    return std::move(*result);
  }

 private:
  std::string semantic_identity_;
  PreparedLoadBalanceProvider provider_;
};

using LoadBalanceAuthorityFactory = std::function<PreparedLoadBalanceAuthority(
    std::string semantic_identity, const PreparedProviderOptions& options)>;

/// Open preparation registry.  Route lookup occurs once during bind; hierarchy/regrid code keeps
/// only PreparedLoadBalanceAuthority and cannot branch on route or concrete provider type.
class LoadBalanceProviderRegistry {
 public:
  void add(std::string route, LoadBalanceAuthorityFactory factory) {
    if (route.empty() || !factory)
      throw std::invalid_argument("load-balance provider registration is incomplete");
    std::lock_guard<std::mutex> guard(mutex_);
    if (!factories_.emplace(std::move(route), std::move(factory)).second)
      throw std::invalid_argument("load-balance provider route is already registered");
  }

  [[nodiscard]] PreparedLoadBalanceAuthority prepare(std::string_view route,
                                                     std::string semantic_identity,
                                                     const PreparedProviderOptions& options) const {
    LoadBalanceAuthorityFactory factory;
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
  std::map<std::string, LoadBalanceAuthorityFactory, std::less<>> factories_;
};

inline LoadBalanceProviderRegistry& load_balance_provider_registry() {
  static LoadBalanceProviderRegistry registry;
  static std::once_flag builtins;
  std::call_once(builtins, [&] {
    registry.add("space_filling_curve", [](std::string identity,
                                           const PreparedProviderOptions& options) {
      detail::require_empty_load_balance_options(options,
                                                 "pops.amr.load-balance.space-filling-curve@1");
      return PreparedLoadBalanceAuthority(
          std::move(identity), PreparedLoadBalanceProvider(detail::SpaceFillingCurveLoadBalance{}));
    });
    registry.add("knapsack", [](std::string identity, const PreparedProviderOptions& options) {
      detail::require_empty_load_balance_options(options, "pops.amr.load-balance.knapsack@1");
      return PreparedLoadBalanceAuthority(
          std::move(identity), PreparedLoadBalanceProvider(detail::KnapsackLoadBalance{}));
    });
    registry.add("round_robin", [](std::string identity, const PreparedProviderOptions& options) {
      detail::require_empty_load_balance_options(options, "pops.amr.load-balance.round-robin@1");
      return PreparedLoadBalanceAuthority(
          std::move(identity), PreparedLoadBalanceProvider(detail::RoundRobinLoadBalance{}));
    });
  });
  return registry;
}

inline void register_load_balance_provider(std::string route, LoadBalanceAuthorityFactory factory) {
  load_balance_provider_registry().add(std::move(route), std::move(factory));
}

inline PreparedLoadBalanceAuthority prepare_load_balance_authority(
    std::string_view route, std::string semantic_identity, const PreparedProviderOptions& options) {
  return load_balance_provider_registry().prepare(route, std::move(semantic_identity), options);
}

}  // namespace pops
