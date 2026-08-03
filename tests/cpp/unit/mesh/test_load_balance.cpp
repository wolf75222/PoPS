// Equilibrage de charge : Z-order (SFC) vs knapsack. On verifie la validite
// (toutes les boxes assignees, tous les rangs utilises), la propriete de
// localite de la SFC (segments contigus le long de la courbe de Morton -> il y
// a exactement nranks-1 transitions de rang dans l'ordre de Morton), et que le
// knapsack equilibre au moins aussi bien que la SFC (il optimise le desequilibre
// max, la SFC le troque contre la localite).

#include <gtest/gtest.h>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/parallel/load_balance.hpp>
#include <pops/parallel/prepared_load_balance.hpp>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace pops;

static_assert(std::is_copy_constructible_v<PreparedLoadBalanceProvider>);
static_assert(std::is_nothrow_move_constructible_v<PreparedLoadBalanceProvider>);
static_assert(std::is_copy_assignable_v<PreparedLoadBalanceProvider>);
static_assert(std::is_nothrow_move_assignable_v<PreparedLoadBalanceProvider>);
static_assert(std::is_copy_constructible_v<PreparedLoadBalanceAuthority>);
static_assert(std::is_nothrow_move_constructible_v<PreparedLoadBalanceAuthority>);

namespace {

// nombre de changements de rang le long de l'ordre de Morton.
int rank_transitions(const BoxArray& ba, const DistributionMapping& dm) {
  const std::vector<int> order = morton_order(ba);
  int t = 0;
  for (std::size_t k = 1; k < order.size(); ++k)
    if (dm[order[k]] != dm[order[k - 1]])
      ++t;
  return t;
}

bool all_in_range(const DistributionMapping& dm, int nranks) {
  for (int r : dm.ranks())
    if (r < 0 || r >= nranks)
      return false;
  return true;
}

int n_ranks_used(const DistributionMapping& dm, int nranks) {
  std::vector<char> seen(nranks, 0);
  for (int r : dm.ranks())
    seen[r] = 1;
  int u = 0;
  for (char c : seen)
    u += c;
  return u;
}

struct ExternalIndexLoadBalance {
  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.test.load_balance.external_index", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    contract.text("external-index-policy").scalar(std::uint32_t{1});
  }

  DistributionMapping operator()(const BoxArray& boxes, int ranks,
                                 LoadBalanceWeights weights) const {
    if (!weights.empty() && weights.size() != static_cast<std::size_t>(boxes.size()))
      throw std::invalid_argument("external load-balance weight count mismatch");
    return DistributionMapping(boxes.size(), ranks);
  }
};

ResourceEstimate measured_patch_cost(std::int64_t nanoseconds, std::int64_t resident_bytes = 1024) {
  return ResourceEstimate{
      .topology_epoch = 7,
      .materialization_generation = 3,
      .samples = 1,
      .cell_updates = 1,
      .compute_nanoseconds = nanoseconds,
      .memory_bytes = 64,
      .communication_bytes = 0,
      .communication_nanoseconds = 0,
      .resident_bytes = resident_bytes,
  };
}

}  // namespace

TEST(test_load_balance, morton_key_reference_values) {
  EXPECT_EQ(morton_key(0, 0), 0) << "morton_00";
  EXPECT_EQ(morton_key(1, 0), 1) << "morton_10";
  EXPECT_EQ(morton_key(0, 1), 2) << "morton_01";
  EXPECT_EQ(morton_key(1, 1), 3) << "morton_11";
  EXPECT_EQ(morton_key(2, 0), 4) << "morton_20";
  EXPECT_EQ(morton_key(0, 2), 8) << "morton_02";
}

TEST(test_load_balance, uniform_case_balances_and_sfc_is_local) {
  const int nranks = 4;

  // --- cas uniforme : 8x8 = 64 boxes de 16x16 (charge egale) ---
  BoxArray ba = BoxArray::from_domain(Box2D::from_extents(128, 128), 16);
  ASSERT_EQ(ba.size(), 64) << "uniform_64_boxes";

  DistributionMapping sfc = make_sfc_distribution(ba, nranks);
  DistributionMapping knap = make_knapsack_distribution(ba, nranks);

  EXPECT_TRUE(all_in_range(sfc, nranks)) << "sfc_ranks_in_range";
  EXPECT_TRUE(all_in_range(knap, nranks)) << "knap_ranks_in_range";
  EXPECT_EQ(n_ranks_used(sfc, nranks), nranks) << "sfc_all_ranks_used";
  EXPECT_EQ(n_ranks_used(knap, nranks), nranks) << "knap_all_ranks_used";

  const double sfc_imb = load_imbalance(ba, sfc, nranks);
  const double knap_imb = load_imbalance(ba, knap, nranks);
  std::printf("uniforme : sfc_imb=%.4f knap_imb=%.4f\n", sfc_imb, knap_imb);
  EXPECT_LE(sfc_imb, 1.001) << "sfc_uniform_balanced";
  EXPECT_LE(knap_imb, 1.001) << "knap_uniform_balanced";

  // localite : la SFC fait des segments contigus (nranks-1 transitions), le
  // knapsack disperse (beaucoup plus de transitions).
  const int sfc_t = rank_transitions(ba, sfc);
  const int knap_t = rank_transitions(ba, knap);
  std::printf("localite : sfc_transitions=%d knap_transitions=%d\n", sfc_t, knap_t);
  EXPECT_EQ(sfc_t, nranks - 1) << "sfc_contiguous_segments";
  EXPECT_GT(knap_t, sfc_t) << "knap_less_local_than_sfc";
}

TEST(test_load_balance, nonuniform_case_knapsack_beats_sfc) {
  // cas non-uniforme concu : poids [5,4,3,2,2,2], 3 rangs, places le long
  // de l'axe x pour que l'ordre de Morton = l'ordre d'insertion. Le knapsack
  // (LPT) doit equilibrer strictement mieux que la coupe contigue SFC.
  const int w[] = {5, 4, 3, 2, 2, 2};
  std::vector<Box2D> bx;
  for (int k = 0; k < 6; ++k)
    bx.push_back(Box2D{{k * 100, 0}, {k * 100, w[k] - 1}});  // 1 x w_k cellules
  BoxArray ban(std::move(bx));
  for (int k = 0; k < 6; ++k)
    EXPECT_EQ(ban[k].num_cells(), w[k]) << "nonuniform_weights";

  DistributionMapping sfc3 = make_sfc_distribution(ban, 3);
  DistributionMapping knap3 = make_knapsack_distribution(ban, 3);
  EXPECT_TRUE(all_in_range(sfc3, 3) && all_in_range(knap3, 3)) << "nonuniform_in_range";
  EXPECT_TRUE(n_ranks_used(sfc3, 3) == 3 && n_ranks_used(knap3, 3) == 3)
      << "nonuniform_all_ranks_used";

  const double sfc3_imb = load_imbalance(ban, sfc3, 3);
  const double knap3_imb = load_imbalance(ban, knap3, 3);
  std::printf("non-uniforme : sfc_imb=%.4f knap_imb=%.4f\n", sfc3_imb, knap3_imb);
  EXPECT_LE(knap3_imb, sfc3_imb + 1e-9) << "knap_balances_at_least_as_well";
  EXPECT_LT(knap3_imb, sfc3_imb) << "knap_strictly_better_here";
}

TEST(test_load_balance, supplied_weights_drive_weighted_policies) {
  const BoxArray boxes = BoxArray::from_domain(Box2D::from_extents(4, 1), 1);
  ASSERT_EQ(boxes.size(), 4);
  const std::vector<std::int64_t> weights{100, 1, 1, 1};

  const DistributionMapping unweighted_sfc = make_sfc_distribution(boxes, 2);
  const DistributionMapping weighted_sfc = make_sfc_distribution(boxes, 2, weights);
  EXPECT_EQ(unweighted_sfc.ranks(), (std::vector<int>{0, 0, 1, 1}));
  EXPECT_EQ(weighted_sfc.ranks(), (std::vector<int>{0, 1, 1, 1}));

  const DistributionMapping unweighted_knapsack = make_knapsack_distribution(boxes, 2);
  const DistributionMapping weighted_knapsack = make_knapsack_distribution(boxes, 2, weights);
  EXPECT_EQ(unweighted_knapsack.ranks(), (std::vector<int>{0, 1, 0, 1}));
  EXPECT_EQ(weighted_knapsack.ranks(), (std::vector<int>{0, 1, 1, 1}));

  EXPECT_LT(load_imbalance(boxes, weighted_knapsack, 2, weights),
            load_imbalance(boxes, unweighted_knapsack, 2, weights));
}

TEST(test_load_balance, round_robin_authenticates_but_does_not_consume_weights) {
  const BoxArray boxes = BoxArray::from_domain(Box2D::from_extents(4, 1), 1);
  const std::vector<std::int64_t> weights{100, 1, 1, 1};

  EXPECT_EQ(make_round_robin_distribution(boxes, 2).ranks(), (std::vector<int>{0, 1, 0, 1}));
  EXPECT_EQ(make_round_robin_distribution(boxes, 2, weights).ranks(),
            (std::vector<int>{0, 1, 0, 1}));

  EXPECT_THROW(make_round_robin_distribution(boxes, 2, std::vector<std::int64_t>{1, 2}),
               std::invalid_argument);
  EXPECT_THROW(make_round_robin_distribution(boxes, 2, std::vector<std::int64_t>{1, 0, 1, 1}),
               std::invalid_argument);

  const auto authority = prepare_load_balance_authority(
      "round_robin", "test.round-robin.identity",
      PreparedProviderOptions{"pops.amr.load-balance.round-robin@1", {}});
  EXPECT_EQ(authority.implementation(), "pops.load_balance.round_robin");
  EXPECT_FALSE(authority.collective_contract().empty());
}

TEST(test_load_balance, third_party_provider_registers_without_core_changes) {
  register_load_balance_provider("test_external_index", [](std::string semantic_identity,
                                                           const PreparedProviderOptions& options) {
    if (options.schema_identity != "pops.test.load-balance.external-index@1" ||
        !options.values.empty())
      throw std::invalid_argument("external load-balance options are not canonical");
    return PreparedLoadBalanceAuthority(std::move(semantic_identity),
                                        PreparedLoadBalanceProvider(ExternalIndexLoadBalance{}));
  });

  const auto authority = prepare_load_balance_authority(
      "test_external_index", "test.external-index.semantic-identity",
      PreparedProviderOptions{"pops.test.load-balance.external-index@1", {}});
  EXPECT_EQ(authority.semantic_identity(), "test.external-index.semantic-identity");
  EXPECT_EQ(authority.implementation(), "pops.test.load_balance.external_index");

  const BoxArray boxes = BoxArray::from_domain(Box2D::from_extents(4, 1), 1);
  const DistributionMapping mapping = authority.distribute(boxes, 1);
  EXPECT_EQ(mapping.ranks(), (std::vector<int>{0, 0, 0, 0}));

  EXPECT_THROW(prepare_load_balance_authority(
                   "test_external_index", "test.external-index.semantic-identity",
                   PreparedProviderOptions{"pops.test.load-balance.wrong-schema@1", {}}),
               std::invalid_argument);
}

TEST(test_load_balance, measured_rebalance_accepts_only_net_benefit_after_migration) {
  const BoxArray boxes = BoxArray::from_domain(Box2D::from_extents(4, 1), 1);
  const DistributionMapping current(std::vector<int>{0, 0, 1, 1});
  const DistributionMapping proposed(std::vector<int>{0, 1, 0, 1});
  const std::vector<ResourceEstimate> estimates{measured_patch_cost(100), measured_patch_cost(100),
                                                measured_patch_cost(1), measured_patch_cost(1)};
  const RebalancePolicy profitable{
      .minimum_improvement_ppm = 50'000,
      .amortization_steps = 100,
      .migration_bandwidth_bytes_per_second = 1'000'000'000'000,
      .per_patch_migration_latency_nanoseconds = 0,
  };
  const std::string source_contract = detail::exact_rebalance_source(
      "test.load-balance", "test.load-balance@1", 1, 2, 7, 3, boxes, current);

  const RebalanceDecision accepted = make_rebalance_decision(
      boxes, current, proposed, 2, 7, 3, estimates, profitable, source_contract);
  EXPECT_TRUE(accepted.accepted);
  EXPECT_EQ(accepted.reason, RebalanceReason::NetBenefit);
  EXPECT_EQ(accepted.moved_patches, 2);
  EXPECT_EQ(accepted.migration_bytes, 2048);
  EXPECT_LT(accepted.proposed_imbalance, accepted.current_imbalance);
  EXPECT_GT(accepted.predicted_net_speedup, 1.05);
  EXPECT_FALSE(accepted.exact_contract.empty());

  RebalancePolicy expensive = profitable;
  expensive.amortization_steps = 1;
  expensive.migration_bandwidth_bytes_per_second = 1;
  const RebalanceDecision refused = make_rebalance_decision(boxes, current, proposed, 2, 7, 3,
                                                            estimates, expensive, source_contract);
  EXPECT_FALSE(refused.accepted);
  EXPECT_EQ(refused.reason, RebalanceReason::InsufficientNetBenefit);
  EXPECT_LT(refused.predicted_net_speedup, 1.0);
}

TEST(test_load_balance, measured_rebalance_refuses_stale_or_incomplete_evidence) {
  const BoxArray boxes = BoxArray::from_domain(Box2D::from_extents(2, 1), 1);
  const DistributionMapping current(std::vector<int>{0, 0});
  const DistributionMapping proposed(std::vector<int>{0, 1});
  std::vector<ResourceEstimate> estimates{measured_patch_cost(100), measured_patch_cost(1)};
  const RebalancePolicy policy{};
  const std::string source_contract = detail::exact_rebalance_source(
      "test.load-balance", "test.load-balance@1", 1, 2, 7, 3, boxes, current);

  estimates[1].topology_epoch = 6;
  EXPECT_THROW(make_rebalance_decision(boxes, current, proposed, 2, 7, 3, estimates, policy,
                                       source_contract),
               std::invalid_argument);
  estimates[1] = measured_patch_cost(1);
  estimates[1].samples = 0;
  EXPECT_THROW(make_rebalance_decision(boxes, current, proposed, 2, 7, 3, estimates, policy,
                                       source_contract),
               std::invalid_argument);
}

TEST(test_load_balance, measured_rebalance_keeps_an_unchanged_mapping) {
  const BoxArray boxes = BoxArray::from_domain(Box2D::from_extents(2, 1), 1);
  const DistributionMapping current(std::vector<int>{0, 1});
  const std::vector<ResourceEstimate> estimates{measured_patch_cost(1), measured_patch_cost(1)};

  const RebalanceDecision decision = make_rebalance_decision(
      boxes, current, current, 2, 7, 3, estimates, RebalancePolicy{},
      detail::exact_rebalance_source("test.load-balance", "test.load-balance@1", 1, 2, 7, 3, boxes,
                                     current));
  EXPECT_FALSE(decision.accepted);
  EXPECT_EQ(decision.reason, RebalanceReason::MappingUnchanged);
  EXPECT_EQ(decision.moved_patches, 0);
  EXPECT_EQ(decision.migration_bytes, 0);
  EXPECT_DOUBLE_EQ(decision.predicted_net_speedup, 1.0);
}

TEST(test_load_balance, measured_knapsack_provider_owns_exact_default_decision_policy) {
  const PreparedProviderOptions options{
      "pops.amr.load-balance.measured-knapsack@1",
      {
          {"minimum_improvement_ppm", std::int64_t{125'000}},
          {"amortization_steps", std::int64_t{40}},
          {"migration_bandwidth_bytes_per_second", std::int64_t{25'000'000'000}},
          {"per_patch_migration_latency_nanoseconds", std::int64_t{2'500}},
      },
  };
  const PreparedLoadBalanceAuthority authority = prepare_load_balance_authority(
      "measured_knapsack", "test.measured-knapsack.identity", options);
  ASSERT_TRUE(authority.has_default_rebalance_policy());
  EXPECT_EQ(authority.implementation(), "pops.load_balance.measured_knapsack");
  const RebalancePolicy& policy = authority.default_rebalance_policy();
  EXPECT_EQ(policy.minimum_improvement_ppm, 125'000);
  EXPECT_EQ(policy.amortization_steps, 40);
  EXPECT_EQ(policy.migration_bandwidth_bytes_per_second, 25'000'000'000);
  EXPECT_EQ(policy.per_patch_migration_latency_nanoseconds, 2'500);

  const BoxArray boxes = BoxArray::from_domain(Box2D::from_extents(2, 1), 1);
  const DistributionMapping current(std::vector<int>{0, 0});
  const RebalanceDecision defaulted = authority.decide_rebalance(
      1, boxes, current, 1, 7, 3,
      std::vector<ResourceEstimate>{measured_patch_cost(100), measured_patch_cost(1)});
  EXPECT_FALSE(defaulted.accepted);
  EXPECT_EQ(defaulted.reason, RebalanceReason::MappingUnchanged);
  EXPECT_FALSE(defaulted.source_contract.empty());

  PreparedProviderOptions incomplete = options;
  incomplete.values.erase("amortization_steps");
  EXPECT_THROW(prepare_load_balance_authority("measured_knapsack", "test.invalid", incomplete),
               std::invalid_argument);
  PreparedProviderOptions wrong_type = options;
  wrong_type.values["amortization_steps"] = std::uint64_t{40};
  EXPECT_THROW(prepare_load_balance_authority("measured_knapsack", "test.invalid", wrong_type),
               std::invalid_argument);
}
