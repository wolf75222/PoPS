#include <gtest/gtest.h>

#include <set>

#include <pops/numerics/time/amr/reflux/amr_interface_flux_ledger.hpp>

namespace pops {
namespace {

amr::InterfaceFluxFragmentKey fragment_key(
    amr::InterfaceFluxOrientation orientation = amr::InterfaceFluxOrientation::FineOutward) {
  return {"fluid-plasma.x-high",
          9,
          0,
          1,
          {1, 7, amr::Rational(1, 2), 0.125},
          "ssprk2.stage.0",
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

  EXPECT_EQ(identities.size(), 10u);
}

TEST(test_interface_flux_fragment_ledger,
     opposite_interface_consumers_receive_equal_and_opposite_fluxes) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9);
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
  amr::TransactionalInterfaceFluxLedger<double> ledger(9);
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
  amr::TransactionalInterfaceFluxLedger<double> ledger(9);
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
  amr::TransactionalInterfaceFluxLedger<double> ledger(9);
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
  amr::TransactionalInterfaceFluxLedger<double> ledger(9);
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
  amr::TransactionalInterfaceFluxLedger<double> ledger(9);
  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(0, 1), 0.5, 0.25, false}, 4.0);
  ledger.begin();
  EXPECT_THROW(ledger.resolve_pending_stage_weight(0, amr::Rational(1, 2)), std::runtime_error);
  ledger.rollback();
  ASSERT_EQ(ledger.pending_size(), 1u);
  EXPECT_FALSE(ledger.pending_entries()[0].measure.stage_weight_resolved);
  ledger.resolve_pending_stage_weight(0, amr::Rational(1, 2));
  ledger.commit();
  EXPECT_EQ(ledger.published_size(), 1u);
}

TEST(test_interface_flux_fragment_ledger,
     duplicate_stage_clock_identity_is_rejected_before_publication) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9);
  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 4.0);
  EXPECT_THROW(ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 4.0),
               std::runtime_error);
  EXPECT_EQ(ledger.pending_size(), 1u);
  ledger.rollback();
}

TEST(test_interface_flux_fragment_ledger,
     authoritative_substep_duration_is_not_reconstructed_from_physical_timestamps) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9);
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

}  // namespace
}  // namespace pops
