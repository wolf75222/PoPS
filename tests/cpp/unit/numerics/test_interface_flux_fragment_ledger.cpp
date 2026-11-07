#include <gtest/gtest.h>

#include <set>

#include <pops/numerics/time/amr/reflux/amr_interface_flux_ledger.hpp>

namespace pops {
namespace {

amr::InterfaceFluxLedgerBudget test_budget() {
  return {16, 16, 2, "test-interface-budget"};
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

TEST(test_interface_flux_fragment_ledger,
     rollback_only_evaluation_has_separate_capacity_and_survives_snapshot_copy) {
  auto budget = test_budget();
  budget.max_evaluation_fragments = 1;
  budget.max_evaluation_payload_terms = 1;
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, budget);
  ledger.begin();
  ledger.accumulate(fragment_key(), {amr::Rational(1, 1), 0.5, 0.25}, 2.0);
  auto evaluation = ledger.prepare_evaluation_begin();
  ledger.publish_prepared_begin(evaluation);
  auto probe = fragment_key();
  probe.graph_identity = "pops.amr-program.paired-evaluation/test-program";
  ledger.accumulate(probe, {amr::Rational(1, 1), 0.5, 0.25}, 4.0);
  auto extra = probe;
  extra.application_identity = "another-probe";
  // The outer accepted budget still has fourteen unused entries. A trial cannot borrow them.
  EXPECT_THROW(ledger.accumulate(extra, {amr::Rational(1, 1), 0.5, 0.25}, 6.0), std::length_error);
  EXPECT_EQ(ledger.pending_size(), 2u);
  auto snapshot = ledger;
  EXPECT_THROW(snapshot.commit(), std::runtime_error);
  EXPECT_THROW(snapshot.begin(), std::runtime_error);
  snapshot.rollback();
  EXPECT_EQ(snapshot.pending_size(), 1u);
  snapshot.commit();
  ASSERT_EQ(snapshot.published_size(), 1u);
  EXPECT_EQ(snapshot.published_entries().front().key.graph_identity, fragment_key().graph_identity);
  EXPECT_THROW(ledger.commit(), std::runtime_error);
  ledger.rollback();
  ledger.commit();
  EXPECT_EQ(ledger.published_entries().front().payload, 2.0);
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
  EXPECT_FALSE(ledger.pending_entries()[0].measure.stage_weight_resolved);
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
     affine_publication_keeps_evaluation_quadrature_and_opposite_consumers) {
  amr::TransactionalInterfaceFluxLedger<double> ledger(9, test_budget());
  const auto coarse = fragment_key(amr::InterfaceFluxOrientation::CoarseOutward);
  const auto fine = fragment_key(amr::InterfaceFluxOrientation::FineOutward);
  ledger.begin();
  ledger.accumulate(coarse, {amr::Rational(1, 2), .5, .25}, 4.);
  ledger.accumulate(fine, {amr::Rational(1, 2), .5, .25}, 4.);
  EXPECT_THROW(ledger.finalize_affine_publication({0., .25, .2, .8}), std::logic_error);
  ledger.commit();
  const auto original = ledger;
  EXPECT_THROW(ledger.finalize_affine_publication({0., .25, .21, .8}), std::invalid_argument);
  EXPECT_THROW(ledger.finalize_affine_publication({1., .25, .2, .8}), std::invalid_argument);
  EXPECT_FALSE(ledger.published_entries().front().measure.affine_publication);
  ledger.finalize_affine_publication({0., .25, .2, .8});
  for (std::size_t index = 0; index < 2; ++index) {
    const auto& a = original.published_entries()[index];
    const auto& b = ledger.published_entries()[index];
    EXPECT_EQ(a.key.clock, b.key.clock);
    EXPECT_EQ(a.key.interval.begin, b.key.interval.begin);
    EXPECT_EQ(a.key.interval.end, b.key.interval.end);
    EXPECT_EQ(a.measure.stage_weight, b.measure.stage_weight);
    EXPECT_EQ(a.measure.substep_duration, b.measure.substep_duration);
    EXPECT_EQ(a.payload, b.payload);
    ASSERT_TRUE(b.measure.affine_publication);
  }
  const auto accepted = ledger.aggregate(scalar_axpy);
  EXPECT_DOUBLE_EQ(accepted.at(amr::interface_flux_accumulation_key(coarse)), -.4);
  EXPECT_DOUBLE_EQ(accepted.at(amr::interface_flux_accumulation_key(fine)), .4);
  EXPECT_THROW(ledger.finalize_affine_publication({0., .25, .2, .8}), std::logic_error);
  ledger = original;  // The enclosing accepted snapshot restores this entire object.
  EXPECT_FALSE(ledger.published_entries().front().measure.affine_publication);
  EXPECT_DOUBLE_EQ(ledger.aggregate(scalar_axpy).at(amr::interface_flux_accumulation_key(fine)),
                   .5);
}

}  // namespace
}  // namespace pops

// Generic arithmetic/codec receivers share the real native ledger target.
TEST(test_interface_flux_fragment_ledger, exact_weight_ieee_extremes_and_non_dyadic_ratios) {
  using pops::ExactWeight;
  for (double x : {0.0, -0.0, 1e-6, -1e-6, 1.0, std::numeric_limits<double>::denorm_min(),
                   std::numeric_limits<double>::min(), std::numeric_limits<double>::max()}) {
    const auto weight = ExactWeight::from_double(x, 512);
    EXPECT_EQ(weight.value(), x);
    EXPECT_EQ(ExactWeight::decode(weight.encode(), 512), weight);
  }
  const auto tiny = ExactWeight::from_double(1e-6, 512);
  EXPECT_FALSE(tiny.fits_int64());
  const auto a = ExactWeight::from_double(.1, 512), b = ExactWeight::from_double(.3, 512);
  EXPECT_EQ((a / b) * b, a);
  EXPECT_EQ((tiny + -tiny), ExactWeight(0));
  EXPECT_EQ((a / b).value(), .1 / .3);
  const auto huge = ExactWeight::from_double(std::numeric_limits<double>::max(), 512);
  const auto small = ExactWeight::from_double(std::numeric_limits<double>::denorm_min(), 512);
  EXPECT_EQ((huge / small) * small, huge);
}

TEST(test_interface_flux_fragment_ledger, exact_weight_rounding_and_canonical_wire) {
  using pops::ExactWeight;
  EXPECT_EQ(ExactWeight(1, 3).value(), 1.0 / 3.0);
  EXPECT_EQ(ExactWeight::from_hex("-0x100000000000000000000", "0x3", 512),
            ExactWeight::from_decimal("-1208925819614629174706176", "3", 512));
  EXPECT_THROW(ExactWeight::from_hex("0x02", "0x1", 512), std::invalid_argument);
  EXPECT_EQ(ExactWeight(9007199254740993LL).value(), 9007199254740992.0);
  EXPECT_EQ(ExactWeight(9007199254740995LL).value(), 9007199254740996.0);
  auto negative_zero = ExactWeight(0).encode();
  negative_zero[3] = 1;
  EXPECT_THROW(ExactWeight::decode(negative_zero, 512), std::invalid_argument);
  auto nonreduced = ExactWeight(1).encode();
  nonreduced[20] = 2;
  nonreduced[24] = 2;
  EXPECT_THROW(ExactWeight::decode(nonreduced, 512), std::invalid_argument);
  auto truncated = ExactWeight(1).encode();
  truncated.pop_back();
  EXPECT_THROW(ExactWeight::decode(truncated, 512), std::invalid_argument);
  EXPECT_THROW(ExactWeight::from_decimal("02", "1", 512), std::invalid_argument);
  EXPECT_THROW(ExactWeight::from_decimal("2", "2", 512), std::invalid_argument);
  EXPECT_EQ(ExactWeight::from_decimal("10000000000000000000000000000000000000000", "3", 512).text(),
            "10000000000000000000000000000000000000000/3");
}

TEST(test_interface_flux_fragment_ledger, exact_weight_multiword_division_and_gcd_are_exact) {
  using pops::ExactWeight;
  // The multiword common factor 2^127 - 1 cancels through the general long division.
  const auto product =
      ExactWeight::from_decimal(
          "170141183460469231731687303725582153184246746208706176311805394026439", "1", 512) *
      ExactWeight::from_decimal(
          "1", "425889840005138625341827496619373129859156554350279099253808298414557223", 512);
  EXPECT_EQ(product.text(), "1000000000000000000000000000057/2503155504993241601315571986085849");
  EXPECT_EQ(product.value(), 0.0003994957556592953);
  // Multiword power-of-two divisors, then a one-limb divisor of a multiword numerator.
  EXPECT_EQ((ExactWeight::from_decimal("1267650600228229401496703205376", "1", 512) *
             ExactWeight::from_decimal("1", "1393796574908163946345982392040522594123776", 512))
                .text(),
            "1/1099511627776");
  const auto odd = ExactWeight::from_decimal("3802951800684688204490109616129",
                                             "696898287454081973172991196020261297061888", 512);
  EXPECT_EQ(odd.value(), 5.4569682106375694e-12);
  EXPECT_EQ((odd * ExactWeight(2)).text(),
            "3802951800684688204490109616129/348449143727040986586495598010130648530944");
  const auto third =
      ExactWeight::from_decimal("4820814132776970826625886277023487807566608981348378505941163",
                                "1", 512) *
      ExactWeight(1, 9);
  EXPECT_EQ(third.text(), "1606938044258990275541962092341162602522202993782792835313721/3");
  EXPECT_EQ(third.value(), 5.356460147529967e+59);
  // Power-of-two divisors on a limb boundary (2^64, 2^96): value() rounds with their remainder.
  EXPECT_EQ(ExactWeight::from_double(0x1.999999999999ap-13).value(), 0x1.999999999999ap-13);
  EXPECT_EQ(ExactWeight::from_double(0x1.0000000000001p-12).value(), 0x1.0000000000001p-12);
  EXPECT_EQ(ExactWeight::from_double(0x1.0000000000001p-44).value(), 0x1.0000000000001p-44);
  // Canonical limbs do not depend on the division path that normalized them.
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  const auto next_hex = [&state](int words) {
    std::string text = "0x";
    for (int word = 0; word < words; ++word) {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      const std::uint64_t value = word == 0 ? state | (std::uint64_t{1} << 63) : state;
      for (int nibble = 15; nibble >= 0; --nibble)
        text.push_back("0123456789abcdef"[(value >> (4 * nibble)) & 15]);
    }
    return text;
  };
  for (int words = 1; words <= 6; ++words)
    for (int trial = 0; trial < 8; ++trial) {
      const auto x = ExactWeight::from_hex(next_hex(words), "0x1", 512) /
                     ExactWeight::from_hex(next_hex(7 - words), "0x1", 512);
      const auto y = ExactWeight::from_hex(next_hex(words), "0x1", 512) * ExactWeight(1, 6);
      EXPECT_EQ((x * y) / y, x);
      EXPECT_EQ((x + y) - y, x);
      EXPECT_EQ(x / x, ExactWeight(1));
      EXPECT_EQ(ExactWeight::decode((x * y).encode(), 512), x * y);
    }
}

TEST(test_interface_flux_fragment_ledger, exact_weight_budget_failure_keeps_source_and_ledger) {
  using pops::ExactWeight;
  const auto small = ExactWeight::from_double(1e-6, 512);
  const auto before = small.encode();
  EXPECT_THROW(ExactWeight(0, 1, 4), std::invalid_argument);
  EXPECT_THROW(ExactWeight(1, 1, 4), std::invalid_argument);
  EXPECT_THROW(ExactWeight(0, 1, 8) + small, std::length_error);
  EXPECT_THROW(ExactWeight(0, 1, 8) / small, std::length_error);
  EXPECT_THROW(small.with_budget(8), std::length_error);
  EXPECT_THROW(ExactWeight::decode(before, 8), std::length_error);
  EXPECT_THROW(ExactWeight::from_double(1e-6, 8), std::length_error);
  EXPECT_EQ(small.encode(), before);
  const auto bounded = ExactWeight(2147483647, 1, 8);
  EXPECT_THROW(bounded * bounded, std::length_error);
  EXPECT_EQ(bounded, ExactWeight(2147483647));
  EXPECT_THROW(small / ExactWeight(0), std::domain_error);
  pops::amr::TransactionalInterfaceFluxLedger<double> ledger(9, pops::test_budget());
  auto key = pops::fragment_key();
  key.clock.physical_time = .035;
  key.interval.end.physical_time = .07;
  ledger.begin();
  ledger.accumulate(key, {ExactWeight(0), 1., .07, false}, 2.);
  EXPECT_THROW(ledger.resolve_pending_stage_weight(0, ExactWeight(1, 1, 8)), std::length_error);
  EXPECT_FALSE(ledger.pending_entries().front().measure.stage_weight_resolved);
  EXPECT_EQ(ledger.pending_entries().front().measure.stage_weight, ExactWeight(0));
  EXPECT_EQ(ledger.pending_entries().front().payload, 2.);
}

TEST(test_interface_flux_fragment_ledger, exact_ratio_is_rounded_once_with_its_duration) {
  const auto numerator = pops::ExactWeight::from_double(.3, 512);
  const auto duration = pops::ExactWeight::from_double(.07, 512);
  const auto ratio = numerator / duration;
  ASSERT_NE((.3 / .07) * .07, .3);
  pops::amr::InterfaceFluxFragmentKey key;
  key.orientation = pops::amr::InterfaceFluxOrientation::FineOutward;
  const pops::amr::InterfaceFluxFragmentMeasure measure{ratio, 1., .07, true};
  EXPECT_EQ(pops::amr::interface_flux_fragment_scale(key, measure), .3);
  const pops::amr::reflux::FaceFluxFragmentMeasure face{ratio, {0, 1}, {1, 1}, .07, 1.};
  EXPECT_EQ(pops::amr::reflux::weighted_face_flux_scale(face), .3);
  const double least = std::numeric_limits<double>::denorm_min();
  const auto wide_ratio = numerator / pops::ExactWeight::from_double(least, 512);
  ASSERT_FALSE(std::isfinite(wide_ratio.value()));
  const pops::amr::reflux::FaceFluxFragmentMeasure tiny_interval{
      wide_ratio, {0, 1}, {1, 1}, least, 1.};
  EXPECT_EQ(pops::amr::reflux::weighted_face_flux_scale(tiny_interval), .3);
}
