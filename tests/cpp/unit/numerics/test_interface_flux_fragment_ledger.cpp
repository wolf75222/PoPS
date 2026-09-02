#include <gtest/gtest.h>

#include <set>
#include <vector>

#include <pops/numerics/time/amr/reflux/amr_interface_flux_ledger.hpp>

namespace pops {
namespace {

amr::InterfaceFluxLedgerBudget test_budget() {
  return {16, 16, 2, 2048, "test-interface-budget"};
}

amr::InterfaceFluxFragmentKey fragment_key(
    amr::InterfaceFluxOrientation orientation = amr::InterfaceFluxOrientation::FineOutward) {
  return {"fluid-plasma.x-high",
          9,
          0,
          1,
          {1, 7, amr::Rational(1, 2), 0.125},
          "ssprk2.stage.0",
          "test.program.graph@1",
          "test.program.rate@1",
          "test.program.application@1",
          {{1, 7, amr::Rational(0, 1), 0.0}, {1, 7, amr::Rational(1, 1), 0.25}},
          orientation};
}

void scalar_axpy(double& destination, double scale, const double& payload) {
  destination += scale * payload;
}

TEST(test_interface_flux_fragment_ledger, full_fragment_identity_keeps_every_qualifier) {
  const amr::InterfaceFluxFragmentKey base = fragment_key();
  std::set<amr::InterfaceFluxFragmentKey> identities;
  identities.insert(base);

  auto distinct = base;
  distinct.interface_identity = "fluid-plasma.y-high";
  identities.insert(distinct);
  distinct = base;
  distinct.topology_epoch = 10;
  identities.insert(distinct);
  distinct = base;
  distinct.coarse_level = 1;
  distinct.fine_level = 2;
  identities.insert(distinct);
  distinct = base;
  distinct.clock.phase = amr::Rational(3, 4);
  identities.insert(distinct);
  distinct = base;
  distinct.stage_identity = "ssprk2.stage.1";
  identities.insert(distinct);
  distinct = base;
  distinct.graph_identity = "test.program.graph@2";
  identities.insert(distinct);
  distinct = base;
  distinct.rate_identity = "test.program.rate@2";
  identities.insert(distinct);
  distinct = base;
  distinct.application_identity = "test.program.application@2";
  identities.insert(distinct);
  distinct = base;
  distinct.interval.end.phase = amr::Rational(3, 4);
  identities.insert(distinct);
  distinct = base;
  distinct.orientation = amr::InterfaceFluxOrientation::CoarseOutward;
  identities.insert(distinct);
  distinct = base;
  distinct.left_block = 2;
  identities.insert(distinct);
  distinct = base;
  distinct.right_block = 3;
  identities.insert(distinct);

  EXPECT_EQ(identities.size(), 13u);
}

TEST(test_interface_flux_fragment_ledger,
     opposite_interface_consumers_receive_equal_and_opposite_fluxes) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  const auto coarse = fragment_key(amr::InterfaceFluxOrientation::CoarseOutward);
  const auto fine = fragment_key(amr::InterfaceFluxOrientation::FineOutward);

  ledger.begin();
  ledger.accumulate(coarse, {amr::Rational(1, 1), 0.5, 0.25}, 4.0);
  ledger.accumulate(fine, {amr::Rational(1, 1), 0.5, 0.25}, 4.0);
  ledger.commit();

  const auto accepted = ledger.aggregate(scalar_axpy);
  ASSERT_EQ(accepted.size(), 2u);
  const double coarse_flux = accepted.at(amr::interface_flux_accumulation_key(coarse));
  const double fine_flux = accepted.at(amr::interface_flux_accumulation_key(fine));
  EXPECT_DOUBLE_EQ(coarse_flux, -1.0);
  EXPECT_DOUBLE_EQ(fine_flux, 1.0);
  EXPECT_DOUBLE_EQ(coarse_flux + fine_flux, 0.0);
}

TEST(test_interface_flux_fragment_ledger, ssprk2_stages_accumulate_with_exact_weights) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  auto stage_0 = fragment_key();
  stage_0.clock = {1, 7, amr::Rational(0, 1), 0.0};
  auto stage_1 = fragment_key();
  stage_1.clock = {1, 7, amr::Rational(1, 1), 0.25};
  stage_1.stage_identity = "ssprk2.stage.1";

  ledger.begin();
  ledger.accumulate(stage_0, {amr::Rational(1, 2), 0.5, 0.25}, 2.0);
  ledger.accumulate(stage_1, {amr::Rational(1, 2), 0.5, 0.25}, 6.0);
  ledger.commit();

  const auto accepted = ledger.aggregate(scalar_axpy);
  ASSERT_EQ(accepted.size(), 1u) << "stage identity is integrated, not part of the final balance";
  EXPECT_DOUBLE_EQ(accepted.begin()->second, 1.0);
}

TEST(test_interface_flux_fragment_ledger, rollback_never_publishes_pending_fragments) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(1, 2), 0.5, 0.25}, 2.0);
  EXPECT_TRUE(ledger.aggregate(scalar_axpy).empty());

  ledger.begin();
  auto nested = fragment_key();
  nested.stage_identity = "ssprk2.stage.1";
  ledger.accumulate(nested, {amr::Rational(1, 2), 0.5, 0.25}, 6.0);
  ledger.commit();
  EXPECT_EQ(ledger.pending_size(), 2u);
  EXPECT_EQ(ledger.published_size(), 0u);
  EXPECT_TRUE(ledger.aggregate(scalar_axpy).empty());

  ledger.rollback();
  EXPECT_TRUE(ledger.empty());
  EXPECT_TRUE(ledger.aggregate(scalar_axpy).empty());
}

TEST(test_interface_flux_fragment_ledger, stale_topology_epoch_is_rejected_before_storage) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  auto stale = fragment_key();
  stale.topology_epoch = 8;

  ledger.begin();
  EXPECT_THROW(ledger.accumulate(stale, {amr::Rational(1, 1), 0.5, 0.25}, 4.0),
               std::invalid_argument);
  EXPECT_EQ(ledger.pending_size(), 0u);
  ledger.rollback();

  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 4.0);
  ledger.commit();
  ASSERT_EQ(ledger.published_size(), 1u);

  ledger.advance_topology_epoch(10);
  EXPECT_TRUE(ledger.empty());
  ledger.begin();
  EXPECT_THROW(ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 4.0),
               std::invalid_argument);
  ledger.rollback();
}

TEST(test_interface_flux_fragment_ledger,
     unresolved_program_weight_cannot_escape_the_attempt_transaction) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(0, 1), 0.5, 0.25, false}, 4.0);

  EXPECT_THROW(ledger.commit(), std::runtime_error);
  ASSERT_TRUE(ledger.in_transaction());
  ASSERT_EQ(ledger.pending_size(), 1u);
  ledger.resolve_pending_stage_weight(0, amr::Rational(3, 4));
  ledger.commit();

  const auto accepted = ledger.aggregate(scalar_axpy);
  ASSERT_EQ(accepted.size(), 1u);
  EXPECT_DOUBLE_EQ(accepted.begin()->second, 0.75);
}

TEST(test_interface_flux_fragment_ledger, nested_savepoint_cannot_resolve_an_outer_fragment) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(0, 1), 0.5, 0.25, false}, 4.0);
  ledger.begin();
  EXPECT_THROW(ledger.resolve_pending_stage_weight(0, amr::Rational(1, 2)), std::runtime_error);
  ledger.rollback();
  ASSERT_EQ(ledger.pending_size(), 1u);
  EXPECT_FALSE(ledger.cold_pending_fragments()[0].measure.stage_weight_resolved);
  ledger.resolve_pending_stage_weight(0, amr::Rational(1, 2));
  ledger.commit();
  EXPECT_EQ(ledger.published_size(), 1u);
}

TEST(test_interface_flux_fragment_ledger,
     duplicate_stage_clock_identity_is_rejected_before_publication) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 4.0);
  EXPECT_THROW(ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 4.0),
               std::runtime_error);
  EXPECT_EQ(ledger.pending_size(), 1u);
  ledger.rollback();
}

TEST(test_interface_flux_fragment_ledger,
     authoritative_substep_duration_is_not_reconstructed_from_physical_timestamps) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  auto key = fragment_key();
  key.interval.begin.physical_time = 0.1;
  key.interval.end.physical_time = 0.3;
  key.clock.physical_time = 0.2;
  ledger.begin();
  ledger.accumulate(key, {amr::Rational(1, 1), 0.5, 0.2}, 5.0);
  ledger.commit();

  const auto accepted = ledger.aggregate(scalar_axpy);
  ASSERT_EQ(accepted.size(), 1u);
  EXPECT_DOUBLE_EQ(accepted.begin()->second, 1.0);
}

TEST(test_interface_flux_fragment_ledger,
     cold_prime_allows_zero_to_n_to_m_accepted_windows_without_hot_growth) {
  amr::TransactionalInterfaceFluxLedger<std::vector<double>> owner(9, test_budget());
  amr::TransactionalInterfaceFluxLedger<std::vector<double>> rollback_image(9, test_budget());
  const auto hot_carriers_before = owner.hot_carrier_capacities();
  const auto rollback_carriers_before = rollback_image.hot_carrier_capacities();

  // The artifact primes all finite dense slots before the first candidate.  The later 0 -> N ->
  // M copies use only this storage; no post-seal re-prime is permitted.
  rollback_image.prime_snapshot_slots_at_bind();
  rollback_image.prime_snapshot_arenas_at_bind();
  EXPECT_EQ(rollback_image.snapshot_identity_arena_capacity(),
            test_budget().max_identity_characters);
  EXPECT_EQ(rollback_image.snapshot_payload_arena_capacity(),
            test_budget().max_payload_terms_per_window);
  EXPECT_EQ(rollback_image.published_size(), 0u);

  auto first = fragment_key();
  auto second = fragment_key();
  second.stage_identity = "ssprk2.stage.1";
  owner.begin();
  owner.accumulate(first, {amr::Rational(1, 1), 0.5, 0.25}, {1.0, 2.0});
  owner.accumulate(second, {amr::Rational(1, 1), 0.5, 0.25}, {3.0});
  owner.commit();

  rollback_image.copy_from_preallocated(owner);
  ASSERT_EQ(rollback_image.published_size(), 2u);
  EXPECT_EQ(rollback_image.cold_published_fragments()[0].payload.size(), 2u);

  auto third = fragment_key();
  third.stage_identity = "ssprk2.stage.2";
  owner.begin();
  owner.accumulate(first, {amr::Rational(1, 1), 0.5, 0.25}, {4.0, 5.0});
  owner.accumulate(second, {amr::Rational(1, 1), 0.5, 0.25}, {6.0});
  owner.accumulate(third, {amr::Rational(1, 1), 0.5, 0.25}, {7.0, 8.0, 9.0});
  owner.commit();

  rollback_image.copy_from_preallocated(owner);
  ASSERT_EQ(rollback_image.published_size(), 3u);
  EXPECT_EQ(rollback_image.cold_published_fragments()[2].key.stage_identity, "ssprk2.stage.2");
  EXPECT_EQ(rollback_image.cold_published_fragments()[2].payload,
            (std::vector<double>{7.0, 8.0, 9.0}));
  EXPECT_EQ(owner.hot_carrier_capacities(), hot_carriers_before)
      << "0 -> N -> M windows may change sizes but cannot grow a prepared carrier";
  EXPECT_EQ(rollback_image.hot_carrier_capacities(), rollback_carriers_before)
      << "resident rollback copies use the same sealed dense slot/arena capacity";
}

TEST(test_interface_flux_fragment_ledger,
     inactive_budget_prepares_the_full_contract_image_before_noexcept_publication) {
  const amr::InterfaceFluxLedgerBudget inactive{0, 0, 1, 0,
                                                "pops.amr-program.interface-flux-budget/inactive"};
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, inactive);
  const auto inactive_capacities = ledger.hot_carrier_capacities();

  auto prepared = ledger.prepare_budget(test_budget());
  EXPECT_EQ(ledger.budget(), inactive) << "PreparedBudget must not mutate the inactive live ledger";
  EXPECT_EQ(ledger.hot_carrier_capacities(), inactive_capacities);
  ledger.publish_prepared_budget(prepared);
  EXPECT_EQ(ledger.budget(), test_budget());
  const auto bound_capacities = ledger.hot_carrier_capacities();
  EXPECT_GT(bound_capacities.begin_contract, inactive_capacities.begin_contract);

  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 2.0);
  ledger.commit();
  auto second = fragment_key();
  second.stage_identity = "ssprk2.stage.1";
  ledger.begin();
  ledger.accumulate(second, {amr::Rational(1, 1), 0.5, 0.25}, 3.0);
  ledger.commit();
  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 4.0);
  ledger.rollback();
  EXPECT_EQ(ledger.hot_carrier_capacities(), bound_capacities)
      << "two accepted windows and a rejected retry must not grow contract carriers";
}

TEST(test_interface_flux_fragment_ledger,
     moved_prepared_carriers_publish_once_and_restore_the_exact_prior_window) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  auto begin = ledger.prepare_begin();
  auto moved_begin = std::move(begin);
  ledger.publish_prepared_begin(moved_begin);
  ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 3.0);
  auto commit = ledger.prepare_commit();
  auto moved_commit = std::move(commit);
  ledger.publish_prepared_commit(moved_commit);
  ASSERT_EQ(ledger.published_size(), 1u);
  ledger.restore_prepared_commit(moved_commit);
  EXPECT_TRUE(ledger.in_transaction());
  EXPECT_EQ(ledger.pending_size(), 1u);
  EXPECT_EQ(ledger.published_size(), 0u);
  ledger.rollback();
  EXPECT_TRUE(ledger.empty());
}

TEST(test_interface_flux_fragment_ledger,
     payload_overrun_is_rejected_before_mutating_the_preallocated_candidate) {
  auto constrained_budget = test_budget();
  constrained_budget.max_payload_terms_per_window = 1;
  amr::TransactionalInterfaceFluxLedger<std::vector<double>> owner(9, constrained_budget);
  const auto key = fragment_key();
  owner.begin();
  owner.accumulate(key, {amr::Rational(1, 1), 0.5, 0.25}, {1.0});
  owner.commit();

  amr::TransactionalInterfaceFluxLedger<std::vector<double>> rollback_image(9, constrained_budget);
  rollback_image.prime_snapshot_slots_at_bind();
  rollback_image.prime_snapshot_arenas_at_bind();
  rollback_image.copy_from_preallocated(owner);
  ASSERT_EQ(rollback_image.cold_published_fragments().size(), 1u);
  const auto prior_payload = rollback_image.cold_published_fragments()[0].payload;

  owner.begin();
  EXPECT_THROW(owner.accumulate(key, {amr::Rational(1, 1), 0.5, 0.25}, {2.0, 3.0}),
               std::length_error);
  owner.rollback();
  ASSERT_EQ(rollback_image.cold_published_fragments().size(), 1u);
  EXPECT_EQ(rollback_image.cold_published_fragments()[0].payload, prior_payload);
}

TEST(test_interface_flux_fragment_ledger,
     bind_sized_snapshot_identity_arena_rejects_overrun_without_clobbering_the_image) {
  auto constrained_budget = test_budget();
  constrained_budget.max_identity_characters = 1;
  amr::TransactionalInterfaceFluxLedger<std::vector<double>> ledger(9, constrained_budget);
  ledger.begin();
  EXPECT_THROW(ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, {1.0}),
               std::length_error);
  EXPECT_TRUE(ledger.empty());
  ledger.rollback();
}

}  // namespace
}  // namespace pops
