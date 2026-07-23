/// @file
/// @brief Prepared, extensible ownership authority for AMR BoxArray layouts.

#pragma once

#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/load_balance.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops {

using LoadBalanceWeights = std::span<const std::int64_t>;
using PreparedLoadBalanceProvider =
    PreparedProvider<DistributionMapping(const BoxArray&, int, LoadBalanceWeights)>;

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

  [[nodiscard]] const std::string& semantic_identity() const noexcept {
    return semantic_identity_;
  }
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
    detail::collective_load_balance_preflight(
        "load-balance request", communicator, [&] {
          if (rank_count <= 0 || rank_count != communicator.size())
            throw std::invalid_argument(
                "load-balance rank count must equal the execution communicator size");
          if (!weights.empty() && weights.size() != static_cast<std::size_t>(boxes.size()))
            throw std::invalid_argument(
                "load-balance weight count must equal the BoxArray size");
          for (int index = 0; index < boxes.size(); ++index) {
            if (boxes[index].empty())
              throw std::invalid_argument("load-balance BoxArray contains an empty box");
            if (!weights.empty() && weights[static_cast<std::size_t>(index)] <= 0)
              throw std::invalid_argument("load-balance weights must be strictly positive");
          }
          request_contract = detail::exact_load_balance_request(boxes, rank_count, weights);
        });

    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{semantic_identity_, provider_.collective_contract()},
             {"request", request_contract}},
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
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{semantic_identity_, mapping_contract}}, communicator))
      throw std::invalid_argument("load-balance provider returned different mappings across ranks");
    return std::move(*mapping);
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

  [[nodiscard]] PreparedLoadBalanceAuthority prepare(
      std::string_view route, std::string semantic_identity,
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
      detail::require_empty_load_balance_options(
          options, "pops.amr.load-balance.space-filling-curve@1");
      return PreparedLoadBalanceAuthority(
          std::move(identity), PreparedLoadBalanceProvider(detail::SpaceFillingCurveLoadBalance{}));
    });
    registry.add("knapsack", [](std::string identity, const PreparedProviderOptions& options) {
      detail::require_empty_load_balance_options(options, "pops.amr.load-balance.knapsack@1");
      return PreparedLoadBalanceAuthority(
          std::move(identity), PreparedLoadBalanceProvider(detail::KnapsackLoadBalance{}));
    });
    registry.add("round_robin", [](std::string identity,
                                   const PreparedProviderOptions& options) {
      detail::require_empty_load_balance_options(options, "pops.amr.load-balance.round-robin@1");
      return PreparedLoadBalanceAuthority(
          std::move(identity), PreparedLoadBalanceProvider(detail::RoundRobinLoadBalance{}));
    });
  });
  return registry;
}

inline void register_load_balance_provider(std::string route,
                                           LoadBalanceAuthorityFactory factory) {
  load_balance_provider_registry().add(std::move(route), std::move(factory));
}

inline PreparedLoadBalanceAuthority prepare_load_balance_authority(
    std::string_view route, std::string semantic_identity,
    const PreparedProviderOptions& options) {
  return load_balance_provider_registry().prepare(route, std::move(semantic_identity), options);
}

}  // namespace pops
