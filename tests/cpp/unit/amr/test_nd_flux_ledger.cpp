#include <gtest/gtest.h>

#include <pops/amr/reflux/nd/metric_reflux.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using pops::Index;
using pops::amr::ClockStamp;
using pops::amr::Rational;
using pops::amr::reflux::nd::CoarseCellFaceSide;
using pops::amr::reflux::nd::CoarseFaceRefluxKey;
using pops::amr::reflux::nd::FaceFluxFragmentKey;
using pops::amr::reflux::nd::FaceFluxFragmentMeasure;
using pops::amr::reflux::nd::FaceLedgerCentering;
using pops::amr::reflux::nd::FaceLedgerContribution;
using pops::amr::reflux::nd::FaceLedgerRole;
using pops::amr::reflux::nd::FaceRefinementMapping;
using pops::amr::reflux::nd::LevelTransition;
using pops::amr::reflux::nd::TransactionalFaceFluxLedger;
using pops::amr::reflux::nd::coarse_cell_reflux_correction;
using pops::amr::reflux::nd::fine_faces_for_coarse_face;
using pops::amr::reflux::nd::metric_reflux;
using pops::amr::transfer::nd::RefinementRatio;

void scalar_axpy(double& destination, double coefficient, const double& source) {
  destination += coefficient * source;
}

template <int Dim>
RefinementRatio<Dim> sample_ratio() {
  if constexpr (Dim == 1)
    return RefinementRatio<1>{2};
  else if constexpr (Dim == 2)
    return RefinementRatio<2>{2, 3};
  else
    return RefinementRatio<3>{2, 3, 4};
}

template <int Dim>
FaceRefinementMapping<Dim> sample_mapping() {
  FaceRefinementMapping<Dim> mapping;
  for (int axis = 0; axis < Dim; ++axis) {
    mapping.coarse_origin[axis] = -3 + axis;
    mapping.fine_origin[axis] = 5 - 2 * axis;
  }
  return mapping;
}

template <int Dim>
CoarseFaceRefluxKey<Dim> sample_query(int axis, std::uint64_t attempt) {
  CoarseFaceRefluxKey<Dim> query;
  query.owner = "transport";
  query.state = "U";
  query.levels = LevelTransition{2, 3};
  query.centering = FaceLedgerCentering::Face;
  query.axis = axis;
  query.attempt = attempt;
  for (int direction = 0; direction < Dim; ++direction)
    query.coarse_face[direction] = -1 + 2 * direction;
  return query;
}

ClockStamp clock_at(int level, std::int64_t macro_step, Rational phase, double physical_time) {
  return ClockStamp{level, macro_step, phase, physical_time};
}

template <int Dim>
FaceFluxFragmentKey<Dim> fragment_key(
    const CoarseFaceRefluxKey<Dim>& query, FaceLedgerRole role, Index<Dim> face, std::string stage,
    Rational phase, FaceLedgerContribution contribution = FaceLedgerContribution::NumericalFlux) {
  FaceFluxFragmentKey<Dim> key;
  key.owner = query.owner;
  key.state = query.state;
  key.levels = query.levels;
  key.centering = query.centering;
  key.axis = query.axis;
  key.face = face;
  key.coarse_face = query.coarse_face;
  key.clock = clock_at(role == FaceLedgerRole::Coarse ? query.levels.coarse : query.levels.fine, 9,
                       phase, 1.25 + phase.value());
  key.stage = std::move(stage);
  key.attempt = query.attempt;
  key.role = role;
  key.contribution = contribution;
  return key;
}

template <int Dim>
void accumulate_stage(TransactionalFaceFluxLedger<Dim, double>& ledger,
                      const CoarseFaceRefluxKey<Dim>& query, const RefinementRatio<Dim>& ratio,
                      const FaceRefinementMapping<Dim>& mapping, const std::string& stage,
                      Rational phase, Rational stage_weight, double duration,
                      double coarse_face_measure, double fine_face_measure, double coarse_flux,
                      double fine_flux) {
  ledger.accumulate(fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, stage, phase),
                    FaceFluxFragmentMeasure{stage_weight, duration, coarse_face_measure},
                    coarse_flux);
  for (const auto& fine_face : fine_faces_for_coarse_face(query, ratio, mapping))
    ledger.accumulate(fragment_key(query, FaceLedgerRole::Fine, fine_face, stage, phase),
                      FaceFluxFragmentMeasure{stage_weight, duration, fine_face_measure},
                      fine_flux);
}

template <int Dim>
void expect_composite_conservation() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  const auto query = sample_query<Dim>(0, 12);
  const auto fine_faces = fine_faces_for_coarse_face(query, ratio, mapping);
  const double fine_measure = 0.75;
  const double coarse_measure = fine_measure * static_cast<double>(fine_faces.size());
  const double duration = 0.4;
  TransactionalFaceFluxLedger<Dim, double> ledger;

  ledger.begin(query.attempt);
  accumulate_stage(ledger, query, ratio, mapping, "advance", Rational{1, 2}, Rational{1, 1},
                   duration, coarse_measure, fine_measure, 2.0, 3.0);
  ledger.commit();

  const auto result = metric_reflux(ledger, query, ratio, mapping, scalar_axpy);
  const double expected_mismatch = duration * coarse_measure;
  EXPECT_EQ(result.fine_face_count, fine_faces.size());
  EXPECT_NEAR(result.coarse_weighted_measure, duration * coarse_measure, 1e-14);
  EXPECT_NEAR(result.fine_weighted_measure, duration * coarse_measure, 1e-14);
  EXPECT_NEAR(result.mismatch, expected_mismatch, 1e-14);

  constexpr double coarse_cell_measure = 2.5;
  const double correction = coarse_cell_reflux_correction(result, coarse_cell_measure,
                                                          CoarseCellFaceSide::Upper, scalar_axpy);
  EXPECT_NEAR(correction * coarse_cell_measure + result.mismatch, 0.0, 1e-14);
}

template <int Dim>
std::size_t tangential_count(const RefinementRatio<Dim>& ratio, int normal_axis) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    if (axis != normal_axis)
      result *= static_cast<std::size_t>(ratio[axis]);
  return result;
}

}  // namespace

TEST(test_nd_flux_ledger, composite_reflux_conserves_accepted_transport_in_1d_2d_3d) {
  expect_composite_conservation<1>();
  expect_composite_conservation<2>();
  expect_composite_conservation<3>();
}

TEST(test_nd_flux_ledger, anisotropic_3d_faces_close_the_exact_tangential_surface_product) {
  const RefinementRatio<3> ratio{2, 3, 4};
  const auto mapping = sample_mapping<3>();
  constexpr std::array<std::size_t, 3> expected_counts{12, 8, 6};
  TransactionalFaceFluxLedger<3, double> ledger;

  for (int axis = 0; axis < 3; ++axis) {
    const auto query = sample_query<3>(axis, static_cast<std::uint64_t>(21 + axis));
    const auto fine_faces = fine_faces_for_coarse_face(query, ratio, mapping);
    ASSERT_EQ(fine_faces.size(), expected_counts[static_cast<std::size_t>(axis)]);
    ASSERT_EQ(fine_faces.size(), tangential_count(ratio, axis));
    const double fine_measure = 0.125 * static_cast<double>(axis + 1);
    const double coarse_measure = fine_measure * static_cast<double>(fine_faces.size());

    ledger.begin(query.attempt);
    accumulate_stage(ledger, query, ratio, mapping, "surface", Rational{1, 3}, Rational{1, 1}, 1.0,
                     coarse_measure, fine_measure, 1.75, 1.75);
    ledger.commit();

    const auto result = metric_reflux(ledger, query, ratio, mapping, scalar_axpy);
    EXPECT_NEAR(result.coarse_weighted_measure, coarse_measure, 1e-14);
    EXPECT_NEAR(result.fine_weighted_measure, coarse_measure, 1e-14);
    EXPECT_NEAR(result.mismatch, 0.0, 1e-14);
  }
}

TEST(test_nd_flux_ledger, axis_permutation_preserves_metric_reflux) {
  const RefinementRatio<3> original_ratio{2, 3, 4};
  const RefinementRatio<3> permuted_ratio{4, 2, 3};
  const auto mapping = sample_mapping<3>();
  TransactionalFaceFluxLedger<3, double> original;
  TransactionalFaceFluxLedger<3, double> permuted;
  const auto original_query = sample_query<3>(0, 31);
  const auto permuted_query = sample_query<3>(1, 31);

  ASSERT_EQ(fine_faces_for_coarse_face(original_query, original_ratio, mapping).size(), 12u);
  ASSERT_EQ(fine_faces_for_coarse_face(permuted_query, permuted_ratio, mapping).size(), 12u);
  original.begin(31);
  permuted.begin(31);
  accumulate_stage(original, original_query, original_ratio, mapping, "permuted", Rational{1, 4},
                   Rational{1, 1}, 0.5, 6.0, 0.5, 2.0, 2.5);
  accumulate_stage(permuted, permuted_query, permuted_ratio, mapping, "permuted", Rational{1, 4},
                   Rational{1, 1}, 0.5, 6.0, 0.5, 2.0, 2.5);
  original.commit();
  permuted.commit();

  const auto first = metric_reflux(original, original_query, original_ratio, mapping, scalar_axpy);
  const auto second = metric_reflux(permuted, permuted_query, permuted_ratio, mapping, scalar_axpy);
  EXPECT_NEAR(first.coarse_integrated, second.coarse_integrated, 1e-14);
  EXPECT_NEAR(first.fine_integrated, second.fine_integrated, 1e-14);
  EXPECT_NEAR(first.mismatch, second.mismatch, 1e-14);
}

TEST(test_nd_flux_ledger, exact_stage_weights_are_applied_before_metric_reflux) {
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto query = sample_query<2>(0, 42);
  TransactionalFaceFluxLedger<2, double> ledger;
  ledger.begin(query.attempt);
  accumulate_stage(ledger, query, ratio, mapping, "rk_a", Rational{1, 4}, Rational{1, 4}, 2.0, 2.0,
                   1.0, 2.0, 2.0);
  accumulate_stage(ledger, query, ratio, mapping, "rk_b", Rational{3, 4}, Rational{3, 4}, 2.0, 2.0,
                   1.0, 4.0, 4.0);
  ledger.commit();

  const auto result = metric_reflux(ledger, query, ratio, mapping, scalar_axpy);
  EXPECT_NEAR(result.coarse_integrated, 14.0, 1e-14);
  EXPECT_NEAR(result.fine_integrated, 14.0, 1e-14);
  EXPECT_NEAR(result.mismatch, 0.0, 1e-14);
  EXPECT_EQ(ledger.published_entries(0).size(), 6u);
  EXPECT_TRUE(ledger.published_entries(1).empty());
}

TEST(test_nd_flux_ledger, rejected_attempt_never_publishes_pending_faces) {
  const RefinementRatio<2> ratio{2, 3};
  const auto mapping = sample_mapping<2>();
  auto rejected_query = sample_query<2>(0, 0);
  TransactionalFaceFluxLedger<2, double> ledger;
  ledger.begin(rejected_query.attempt);
  accumulate_stage(ledger, rejected_query, ratio, mapping, "candidate", Rational{1, 2},
                   Rational{1, 1}, 0.25, 3.0, 1.0, 2.0, 2.0);
  EXPECT_EQ(ledger.pending_size(), 4u);
  EXPECT_EQ(ledger.published_size(), 0u);
  EXPECT_THROW((void)metric_reflux(ledger, rejected_query, ratio, mapping, scalar_axpy),
               std::runtime_error);
  ledger.rollback();
  EXPECT_EQ(ledger.pending_size(), 0u);
  EXPECT_EQ(ledger.published_size(), 0u);

  auto accepted_query = sample_query<2>(0, 1);
  ledger.begin(accepted_query.attempt);
  accumulate_stage(ledger, accepted_query, ratio, mapping, "retry", Rational{1, 2}, Rational{1, 1},
                   0.25, 3.0, 1.0, 2.0, 2.0);
  ledger.commit();
  EXPECT_EQ(ledger.published_size(), 4u);
  EXPECT_THROW(ledger.begin(1), std::invalid_argument);
}

TEST(test_nd_flux_ledger, sources_cell_centering_and_stale_attempts_fail_closed) {
  const auto query = sample_query<2>(1, 7);
  TransactionalFaceFluxLedger<2, double> ledger;
  ledger.begin(query.attempt);
  auto source = fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "source",
                             Rational{1, 2}, FaceLedgerContribution::Source);
  EXPECT_THROW(ledger.accumulate(source, FaceFluxFragmentMeasure{Rational{1, 1}, 0.1, 1.0}, 3.0),
               std::invalid_argument);
  auto cell =
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "cell", Rational{1, 2});
  cell.centering = FaceLedgerCentering::Cell;
  EXPECT_THROW(ledger.accumulate(cell, FaceFluxFragmentMeasure{Rational{1, 1}, 0.1, 1.0}, 3.0),
               std::invalid_argument);
  auto stale =
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "stale", Rational{1, 2});
  stale.attempt = 6;
  EXPECT_THROW(ledger.accumulate(stale, FaceFluxFragmentMeasure{Rational{1, 1}, 0.1, 1.0}, 3.0),
               std::invalid_argument);
  EXPECT_EQ(ledger.pending_size(), 0u);
  ledger.rollback();
  EXPECT_EQ(ledger.published_size(), 0u);
}
