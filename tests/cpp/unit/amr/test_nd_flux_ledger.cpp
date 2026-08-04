#include <gtest/gtest.h>

#include <pops/amr/reflux/nd/metric_reflux.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
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
using pops::amr::reflux::nd::FaceFluxLedgerBudget;
using pops::amr::reflux::nd::FaceLedgerCentering;
using pops::amr::reflux::nd::FaceLedgerContribution;
using pops::amr::reflux::nd::FaceLedgerRole;
using pops::amr::reflux::nd::FaceRefinementMapping;
using pops::amr::reflux::nd::LevelTransition;
using pops::amr::reflux::nd::MetricRefluxBudget;
using pops::amr::reflux::nd::TransactionalFaceFluxLedger;
using pops::amr::reflux::nd::coarse_cell_reflux_correction;
using pops::amr::reflux::nd::fine_faces_for_coarse_face;
using pops::amr::reflux::nd::metric_reflux;
using pops::amr::transfer::nd::RefinementRatio;

constexpr FaceFluxLedgerBudget ledger_budget() {
  return {512, 1024, 4};
}

constexpr MetricRefluxBudget reflux_budget() {
  return {256, 1024, 128};
}

void scalar_axpy(double& destination, double coefficient, const double& source) {
  destination += coefficient * source;
}

struct ThrowingPayload {
  double value = 0.0;
  inline static bool fail_copy = false;

  ThrowingPayload() = default;
  explicit ThrowingPayload(double input) : value(input) {}
  ThrowingPayload(const ThrowingPayload& other) : value(other.value) {
    if (fail_copy)
      throw std::runtime_error("injected payload copy failure");
  }
  ThrowingPayload& operator=(const ThrowingPayload&) = default;
  ThrowingPayload(ThrowingPayload&&) noexcept = default;
  ThrowingPayload& operator=(ThrowingPayload&&) noexcept = default;
};

struct CopyOnlyPayload {
  double value = 0.0;

  CopyOnlyPayload() = default;
  explicit CopyOnlyPayload(double input) : value(input) {}
  CopyOnlyPayload(const CopyOnlyPayload&) = default;
  CopyOnlyPayload(CopyOnlyPayload&&) noexcept = default;
  CopyOnlyPayload& operator=(const CopyOnlyPayload&) = delete;
  CopyOnlyPayload& operator=(CopyOnlyPayload&&) = delete;
};

static_assert(std::is_copy_constructible_v<CopyOnlyPayload>);
static_assert(!std::is_copy_assignable_v<CopyOnlyPayload>);

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
  query.macro_step = 9;
  query.window_begin = Rational{0, 1};
  query.window_end = Rational{1, 1};
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
  key.clock = clock_at(role == FaceLedgerRole::Coarse ? query.levels.coarse : query.levels.fine,
                       query.macro_step, phase, 1.25 + phase.value());
  key.stage = std::move(stage);
  key.attempt = query.attempt;
  key.role = role;
  key.contribution = contribution;
  return key;
}

template <int Dim>
void accumulate_stage(TransactionalFaceFluxLedger<Dim, double>& ledger,
                      const CoarseFaceRefluxKey<Dim>& query, const RefinementRatio<Dim>& ratio,
                      const FaceRefinementMapping<Dim>& mapping, const MetricRefluxBudget& budget,
                      const std::string& stage, Rational phase, Rational stage_weight,
                      Rational substep_begin, Rational substep_end, double duration,
                      double coarse_face_measure, double fine_face_measure, double coarse_flux,
                      double fine_flux) {
  ledger.accumulate(fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, stage, phase),
                    FaceFluxFragmentMeasure{stage_weight, substep_begin, substep_end, duration,
                                            coarse_face_measure},
                    coarse_flux);
  for (const auto& fine_face : fine_faces_for_coarse_face(query, ratio, mapping, budget))
    ledger.accumulate(fragment_key(query, FaceLedgerRole::Fine, fine_face, stage, phase),
                      FaceFluxFragmentMeasure{stage_weight, substep_begin, substep_end, duration,
                                              fine_face_measure},
                      fine_flux);
}

template <int Dim>
void expect_composite_conservation() {
  const auto ratio = sample_ratio<Dim>();
  const auto mapping = sample_mapping<Dim>();
  const auto query = sample_query<Dim>(0, 12);
  const auto budget = reflux_budget();
  const auto fine_faces = fine_faces_for_coarse_face(query, ratio, mapping, budget);
  const double fine_measure = 0.75;
  const double coarse_measure = fine_measure * static_cast<double>(fine_faces.size());
  const double duration = 0.4;
  TransactionalFaceFluxLedger<Dim, double> ledger{ledger_budget()};

  ledger.begin(query.attempt);
  accumulate_stage(ledger, query, ratio, mapping, budget, "advance", Rational{1, 2}, Rational{1, 1},
                   Rational{0, 1}, Rational{1, 1}, duration, coarse_measure, fine_measure, 2.0,
                   3.0);
  ledger.commit();

  const auto result = metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy);
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

template <int Dim>
std::vector<std::array<int, Dim>> coordinates(const std::vector<Index<Dim>>& faces) {
  std::vector<std::array<int, Dim>> result;
  result.reserve(faces.size());
  for (const auto& face : faces) {
    std::array<int, Dim> coordinate{};
    for (int axis = 0; axis < Dim; ++axis)
      coordinate[static_cast<std::size_t>(axis)] = face[axis];
    result.push_back(coordinate);
  }
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
  const auto budget = reflux_budget();
  constexpr std::array<std::size_t, 3> expected_counts{12, 8, 6};
  TransactionalFaceFluxLedger<3, double> ledger{ledger_budget()};

  for (int axis = 0; axis < 3; ++axis) {
    const auto query = sample_query<3>(axis, static_cast<std::uint64_t>(21 + axis));
    const auto fine_faces = fine_faces_for_coarse_face(query, ratio, mapping, budget);
    ASSERT_EQ(fine_faces.size(), expected_counts[static_cast<std::size_t>(axis)]);
    ASSERT_EQ(fine_faces.size(), tangential_count(ratio, axis));
    const double fine_measure = 0.125 * static_cast<double>(axis + 1);
    const double coarse_measure = fine_measure * static_cast<double>(fine_faces.size());

    ledger.begin(query.attempt);
    accumulate_stage(ledger, query, ratio, mapping, budget, "surface", Rational{1, 3},
                     Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0, coarse_measure,
                     fine_measure, 1.75, 1.75);
    ledger.commit();

    const auto result = metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy);
    EXPECT_NEAR(result.coarse_weighted_measure, coarse_measure, 1e-14);
    EXPECT_NEAR(result.fine_weighted_measure, coarse_measure, 1e-14);
    EXPECT_NEAR(result.mismatch, 0.0, 1e-14);
  }
}

TEST(test_nd_flux_ledger, axis_permutation_uses_explicit_coordinate_oracles) {
  const RefinementRatio<3> original_ratio{2, 3, 4};
  const RefinementRatio<3> permuted_ratio{4, 2, 3};
  const RefinementRatio<3> twice_permuted_ratio{3, 4, 2};
  FaceRefinementMapping<3> original_mapping;
  original_mapping.coarse_origin[0] = -3;
  original_mapping.coarse_origin[1] = -2;
  original_mapping.coarse_origin[2] = -1;
  original_mapping.fine_origin[0] = 5;
  original_mapping.fine_origin[1] = 3;
  original_mapping.fine_origin[2] = 1;
  FaceRefinementMapping<3> permuted_mapping;
  permuted_mapping.coarse_origin[0] = -1;
  permuted_mapping.coarse_origin[1] = -3;
  permuted_mapping.coarse_origin[2] = -2;
  permuted_mapping.fine_origin[0] = 1;
  permuted_mapping.fine_origin[1] = 5;
  permuted_mapping.fine_origin[2] = 3;
  FaceRefinementMapping<3> twice_permuted_mapping;
  twice_permuted_mapping.coarse_origin[0] = -2;
  twice_permuted_mapping.coarse_origin[1] = -1;
  twice_permuted_mapping.coarse_origin[2] = -3;
  twice_permuted_mapping.fine_origin[0] = 3;
  twice_permuted_mapping.fine_origin[1] = 1;
  twice_permuted_mapping.fine_origin[2] = 5;

  auto original_query = sample_query<3>(0, 31);
  auto permuted_query = sample_query<3>(1, 31);
  auto twice_permuted_query = sample_query<3>(2, 31);
  permuted_query.coarse_face[0] = original_query.coarse_face[2];
  permuted_query.coarse_face[1] = original_query.coarse_face[0];
  permuted_query.coarse_face[2] = original_query.coarse_face[1];
  twice_permuted_query.coarse_face[0] = original_query.coarse_face[1];
  twice_permuted_query.coarse_face[1] = original_query.coarse_face[2];
  twice_permuted_query.coarse_face[2] = original_query.coarse_face[0];
  const auto budget = reflux_budget();

  const std::vector<std::array<int, 3>> original_oracle{
      {9, 12, 17}, {9, 12, 18}, {9, 12, 19}, {9, 12, 20}, {9, 13, 17}, {9, 13, 18},
      {9, 13, 19}, {9, 13, 20}, {9, 14, 17}, {9, 14, 18}, {9, 14, 19}, {9, 14, 20}};
  const std::vector<std::array<int, 3>> permuted_oracle{
      {17, 9, 12}, {17, 9, 13}, {17, 9, 14}, {18, 9, 12}, {18, 9, 13}, {18, 9, 14},
      {19, 9, 12}, {19, 9, 13}, {19, 9, 14}, {20, 9, 12}, {20, 9, 13}, {20, 9, 14}};
  const std::vector<std::array<int, 3>> twice_permuted_oracle{
      {12, 17, 9}, {12, 18, 9}, {12, 19, 9}, {12, 20, 9}, {13, 17, 9}, {13, 18, 9},
      {13, 19, 9}, {13, 20, 9}, {14, 17, 9}, {14, 18, 9}, {14, 19, 9}, {14, 20, 9}};
  EXPECT_EQ(coordinates(fine_faces_for_coarse_face(original_query, original_ratio, original_mapping,
                                                   budget)),
            original_oracle);
  EXPECT_EQ(coordinates(fine_faces_for_coarse_face(permuted_query, permuted_ratio, permuted_mapping,
                                                   budget)),
            permuted_oracle);
  EXPECT_EQ(coordinates(fine_faces_for_coarse_face(twice_permuted_query, twice_permuted_ratio,
                                                   twice_permuted_mapping, budget)),
            twice_permuted_oracle);

  TransactionalFaceFluxLedger<3, double> original{ledger_budget()};
  TransactionalFaceFluxLedger<3, double> permuted{ledger_budget()};
  TransactionalFaceFluxLedger<3, double> twice_permuted{ledger_budget()};
  original.begin(31);
  permuted.begin(31);
  twice_permuted.begin(31);
  accumulate_stage(original, original_query, original_ratio, original_mapping, budget, "permuted",
                   Rational{1, 4}, Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.5, 6.0, 0.5,
                   2.0, 2.5);
  accumulate_stage(permuted, permuted_query, permuted_ratio, permuted_mapping, budget, "permuted",
                   Rational{1, 4}, Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.5, 6.0, 0.5,
                   2.0, 2.5);
  accumulate_stage(twice_permuted, twice_permuted_query, twice_permuted_ratio,
                   twice_permuted_mapping, budget, "permuted", Rational{1, 4}, Rational{1, 1},
                   Rational{0, 1}, Rational{1, 1}, 0.5, 6.0, 0.5, 2.0, 2.5);
  original.commit();
  permuted.commit();
  twice_permuted.commit();

  const auto first = metric_reflux(original, original_query, original_ratio, original_mapping,
                                   budget, scalar_axpy);
  const auto second = metric_reflux(permuted, permuted_query, permuted_ratio, permuted_mapping,
                                    budget, scalar_axpy);
  const auto third = metric_reflux(twice_permuted, twice_permuted_query, twice_permuted_ratio,
                                   twice_permuted_mapping, budget, scalar_axpy);
  EXPECT_NEAR(first.coarse_integrated, second.coarse_integrated, 1e-14);
  EXPECT_NEAR(first.fine_integrated, second.fine_integrated, 1e-14);
  EXPECT_NEAR(first.mismatch, second.mismatch, 1e-14);
  EXPECT_NEAR(first.coarse_integrated, third.coarse_integrated, 1e-14);
  EXPECT_NEAR(first.fine_integrated, third.fine_integrated, 1e-14);
  EXPECT_NEAR(first.mismatch, third.mismatch, 1e-14);
}

TEST(test_nd_flux_ledger, exact_stage_weights_are_applied_before_metric_reflux) {
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto query = sample_query<2>(0, 42);
  const auto budget = reflux_budget();
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  accumulate_stage(ledger, query, ratio, mapping, budget, "rk_a", Rational{1, 4}, Rational{1, 4},
                   Rational{0, 1}, Rational{1, 1}, 2.0, 2.0, 1.0, 2.0, 2.0);
  accumulate_stage(ledger, query, ratio, mapping, budget, "rk_b", Rational{3, 4}, Rational{3, 4},
                   Rational{0, 1}, Rational{1, 1}, 2.0, 2.0, 1.0, 4.0, 4.0);
  ledger.commit();

  const auto result = metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy);
  EXPECT_NEAR(result.coarse_integrated, 14.0, 1e-14);
  EXPECT_NEAR(result.fine_integrated, 14.0, 1e-14);
  EXPECT_NEAR(result.mismatch, 0.0, 1e-14);
  EXPECT_EQ(ledger.published_entries(0).size(), 6u);
  EXPECT_TRUE(ledger.published_entries(1).empty());
}

TEST(test_nd_flux_ledger, coarse_window_matches_two_exact_fine_substeps) {
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto query = sample_query<2>(0, 50);
  const auto budget = reflux_budget();
  const auto fine_faces = fine_faces_for_coarse_face(query, ratio, mapping, budget);
  ASSERT_EQ(fine_faces.size(), 2u);
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  ledger.accumulate(
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "advance", Rational{0, 1}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0, 2.0}, 3.0);
  for (const auto& fine_face : fine_faces) {
    ledger.accumulate(
        fragment_key(query, FaceLedgerRole::Fine, fine_face, "advance", Rational{0, 1}),
        FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 2}, 0.5, 1.0}, 3.0);
    ledger.accumulate(
        fragment_key(query, FaceLedgerRole::Fine, fine_face, "advance", Rational{1, 2}),
        FaceFluxFragmentMeasure{Rational{1, 1}, Rational{1, 2}, Rational{1, 1}, 0.5, 1.0}, 3.0);
  }
  ledger.commit();

  const auto result = metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy);
  EXPECT_NEAR(result.coarse_weighted_measure, 2.0, 1e-14);
  EXPECT_NEAR(result.fine_weighted_measure, 2.0, 1e-14);
  EXPECT_NEAR(result.mismatch, 0.0, 1e-14);
}

TEST(test_nd_flux_ledger, gaps_overlaps_duration_and_stage_weight_fail_closed) {
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto budget = reflux_budget();
  const auto populate = [&](TransactionalFaceFluxLedger<2, double>& ledger,
                            const CoarseFaceRefluxKey<2>& query, Rational second_begin,
                            Rational second_end, double first_duration, double second_duration,
                            double second_face_measure, Rational second_stage_weight) {
    const auto fine_faces = fine_faces_for_coarse_face(query, ratio, mapping, budget);
    ledger.begin(query.attempt);
    ledger.accumulate(
        fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "advance", Rational{0, 1}),
        FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0, 2.0}, 1.0);
    for (const auto& fine_face : fine_faces) {
      ledger.accumulate(
          fragment_key(query, FaceLedgerRole::Fine, fine_face, "advance", Rational{0, 1}),
          FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 2}, first_duration,
                                  1.0},
          1.0);
      ledger.accumulate(
          fragment_key(query, FaceLedgerRole::Fine, fine_face, "advance", second_begin),
          FaceFluxFragmentMeasure{second_stage_weight, second_begin, second_end, second_duration,
                                  second_face_measure},
          1.0);
    }
    ledger.commit();
  };

  const auto gap_query = sample_query<2>(0, 51);
  TransactionalFaceFluxLedger<2, double> gap{ledger_budget()};
  populate(gap, gap_query, Rational{3, 4}, Rational{1, 1}, 0.5, 0.5, 1.0, Rational{1, 1});
  EXPECT_THROW((void)metric_reflux(gap, gap_query, ratio, mapping, budget, scalar_axpy),
               std::runtime_error);

  const auto overlap_query = sample_query<2>(0, 52);
  TransactionalFaceFluxLedger<2, double> overlap{ledger_budget()};
  populate(overlap, overlap_query, Rational{1, 4}, Rational{1, 1}, 0.5, 0.5, 1.0, Rational{1, 1});
  EXPECT_THROW((void)metric_reflux(overlap, overlap_query, ratio, mapping, budget, scalar_axpy),
               std::runtime_error);

  const auto bad_duration_query = sample_query<2>(0, 53);
  TransactionalFaceFluxLedger<2, double> bad_duration{ledger_budget()};
  populate(bad_duration, bad_duration_query, Rational{1, 2}, Rational{1, 1}, 0.5, 0.6, 5.0 / 6.0,
           Rational{1, 1});
  EXPECT_THROW(
      (void)metric_reflux(bad_duration, bad_duration_query, ratio, mapping, budget, scalar_axpy),
      std::runtime_error);

  const auto bad_stage_weight_query = sample_query<2>(0, 54);
  TransactionalFaceFluxLedger<2, double> bad_stage_weight{ledger_budget()};
  populate(bad_stage_weight, bad_stage_weight_query, Rational{1, 2}, Rational{1, 1}, 0.5, 0.5, 2.0,
           Rational{1, 2});
  EXPECT_THROW((void)metric_reflux(bad_stage_weight, bad_stage_weight_query, ratio, mapping, budget,
                                   scalar_axpy),
               std::runtime_error);

  const auto distorted_clock_query = sample_query<2>(0, 55);
  TransactionalFaceFluxLedger<2, double> distorted_clock{ledger_budget()};
  populate(distorted_clock, distorted_clock_query, Rational{1, 2}, Rational{1, 1}, 0.75, 0.25, 1.0,
           Rational{1, 1});
  EXPECT_THROW((void)metric_reflux(distorted_clock, distorted_clock_query, ratio, mapping, budget,
                                   scalar_axpy),
               std::runtime_error);
}

TEST(test_nd_flux_ledger, tiny_physical_clock_mismatch_is_not_unit_scaled_roundoff) {
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto query = sample_query<2>(0, 56);
  const auto budget = reflux_budget();
  const auto fine_faces = fine_faces_for_coarse_face(query, ratio, mapping, budget);
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  ledger.accumulate(
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "advance", Rational{0, 1}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0e-16, 2.0}, 1.0);
  for (const auto& fine_face : fine_faces) {
    ledger.accumulate(
        fragment_key(query, FaceLedgerRole::Fine, fine_face, "first", Rational{0, 1}),
        FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 2}, 1.0e-16, 0.5}, 1.0);
    ledger.accumulate(
        fragment_key(query, FaceLedgerRole::Fine, fine_face, "second", Rational{1, 2}),
        FaceFluxFragmentMeasure{Rational{1, 1}, Rational{1, 2}, Rational{1, 1}, 1.0e-16, 0.5}, 1.0);
  }
  ledger.commit();
  EXPECT_THROW((void)metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy),
               std::runtime_error);
}

TEST(test_nd_flux_ledger, tangential_product_and_reflux_budgets_fail_closed) {
  auto query = sample_query<3>(0, 60);
  const auto mapping = sample_mapping<3>();
  const RefinementRatio<3> extreme_ratio{2, std::numeric_limits<int>::max(),
                                         std::numeric_limits<int>::max()};
  EXPECT_THROW(
      (void)fine_faces_for_coarse_face(query, extreme_ratio, mapping, MetricRefluxBudget{8, 8, 8}),
      std::length_error);
  EXPECT_THROW((void)fine_faces_for_coarse_face(query, RefinementRatio<3>{2, 2, 2}, mapping,
                                                MetricRefluxBudget{0, 8, 8}),
               std::invalid_argument);

  const RefinementRatio<3> ratio{2, 2, 2};
  const auto budget = reflux_budget();
  TransactionalFaceFluxLedger<3, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  accumulate_stage(ledger, query, ratio, mapping, budget, "advance", Rational{1, 2}, Rational{1, 1},
                   Rational{0, 1}, Rational{1, 1}, 1.0, 4.0, 1.0, 1.0, 1.0);
  ledger.commit();
  EXPECT_THROW(
      (void)metric_reflux(ledger, query, ratio, mapping,
                          MetricRefluxBudget{8, ledger.published_size() - 1, 8}, scalar_axpy),
      std::length_error);
  EXPECT_THROW((void)metric_reflux(ledger, query, ratio, mapping,
                                   MetricRefluxBudget{8, ledger.published_size(), 1}, scalar_axpy),
               std::length_error);
}

TEST(test_nd_flux_ledger, identity_ratios_and_incomplete_fine_surfaces_fail_closed) {
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto query = sample_query<2>(0, 41);
  const auto budget = reflux_budget();
  EXPECT_THROW((void)fine_faces_for_coarse_face(query, RefinementRatio<2>{1, 1}, mapping, budget),
               std::invalid_argument);

  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  ledger.accumulate(
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "coarse_stage",
                   Rational{1, 2}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.25, 2.0}, 3.0);
  const auto fine_faces = fine_faces_for_coarse_face(query, ratio, mapping, budget);
  ASSERT_EQ(fine_faces.size(), 2u);
  ledger.accumulate(
      fragment_key(query, FaceLedgerRole::Fine, fine_faces.front(), "fine_stage", Rational{1, 2}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.25, 1.0}, 3.0);
  ledger.commit();

  EXPECT_THROW((void)metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy),
               std::runtime_error);
}

TEST(test_nd_flux_ledger, rejected_attempt_never_publishes_pending_faces) {
  const RefinementRatio<2> ratio{2, 3};
  const auto mapping = sample_mapping<2>();
  const auto budget = reflux_budget();
  auto rejected_query = sample_query<2>(0, 0);
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(rejected_query.attempt);
  accumulate_stage(ledger, rejected_query, ratio, mapping, budget, "candidate", Rational{1, 2},
                   Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.25, 3.0, 1.0, 2.0, 2.0);
  EXPECT_EQ(ledger.pending_size(), 4u);
  EXPECT_EQ(ledger.published_size(), 0u);
  EXPECT_THROW((void)metric_reflux(ledger, rejected_query, ratio, mapping, budget, scalar_axpy),
               std::runtime_error);
  ledger.rollback();
  EXPECT_EQ(ledger.pending_size(), 0u);
  EXPECT_EQ(ledger.published_size(), 0u);

  auto accepted_query = sample_query<2>(0, 1);
  ledger.begin(accepted_query.attempt);
  accumulate_stage(ledger, accepted_query, ratio, mapping, budget, "retry", Rational{1, 2},
                   Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.25, 3.0, 1.0, 2.0, 2.0);
  ledger.commit();
  EXPECT_EQ(ledger.published_size(), 4u);
  EXPECT_THROW(ledger.begin(1), std::invalid_argument);
}

TEST(test_nd_flux_ledger, ledger_budgets_and_discard_published_attempt_fail_closed) {
  EXPECT_THROW(((void)TransactionalFaceFluxLedger<1, double>(FaceFluxLedgerBudget{0, 1, 1})),
               std::invalid_argument);

  const auto query0 = sample_query<1>(0, 0);
  TransactionalFaceFluxLedger<1, double> pending_limited{FaceFluxLedgerBudget{1, 4, 1}};
  pending_limited.begin(query0.attempt);
  pending_limited.accumulate(
      fragment_key(query0, FaceLedgerRole::Coarse, query0.coarse_face, "first", Rational{0, 1}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0, 1.0}, 1.0);
  EXPECT_THROW(
      pending_limited.accumulate(
          fragment_key(query0, FaceLedgerRole::Coarse, query0.coarse_face, "second",
                       Rational{1, 2}),
          FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0, 1.0}, 1.0),
      std::length_error);
  EXPECT_THROW(pending_limited.begin(query0.attempt), std::length_error);
  pending_limited.rollback();

  TransactionalFaceFluxLedger<1, double> publication_limited{FaceFluxLedgerBudget{2, 1, 1}};
  publication_limited.begin(query0.attempt);
  publication_limited.accumulate(
      fragment_key(query0, FaceLedgerRole::Coarse, query0.coarse_face, "accepted", Rational{0, 1}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0, 1.0}, 1.0);
  publication_limited.commit();
  const auto query1 = sample_query<1>(0, 1);
  publication_limited.begin(query1.attempt);
  publication_limited.accumulate(
      fragment_key(query1, FaceLedgerRole::Coarse, query1.coarse_face, "candidate", Rational{0, 1}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0, 1.0}, 1.0);
  EXPECT_THROW(publication_limited.commit(), std::length_error);
  EXPECT_TRUE(publication_limited.in_transaction());
  EXPECT_EQ(publication_limited.published_size(), 1u);
  EXPECT_EQ(publication_limited.pending_size(), 1u);
  publication_limited.rollback();

  TransactionalFaceFluxLedger<1, double> discardable{FaceFluxLedgerBudget{4, 4, 1}};
  for (std::uint64_t attempt = 0; attempt < 2; ++attempt) {
    const auto query = sample_query<1>(0, attempt);
    discardable.begin(attempt);
    discardable.accumulate(
        fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "published", Rational{0, 1}),
        FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0, 1.0}, 1.0);
    discardable.commit();
  }
  EXPECT_EQ(discardable.published_size(), 2u);
  EXPECT_EQ(discardable.discard_published_attempt(0), 1u);
  EXPECT_EQ(discardable.discard_published_attempt(0), 0u);
  EXPECT_EQ(discardable.published_size(), 1u);
  discardable.begin(2);
  EXPECT_THROW((void)discardable.discard_published_attempt(1), std::runtime_error);
  discardable.rollback();

  TransactionalFaceFluxLedger<1, CopyOnlyPayload> copy_only{FaceFluxLedgerBudget{2, 2, 1}};
  copy_only.begin(0);
  copy_only.accumulate(
      fragment_key(query0, FaceLedgerRole::Coarse, query0.coarse_face, "copy-only", Rational{0, 1}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 1.0, 1.0},
      CopyOnlyPayload{1.0});
  copy_only.commit();
  EXPECT_EQ(copy_only.discard_published_attempt(0), 1u);
  EXPECT_EQ(copy_only.published_size(), 0u);
}

TEST(test_nd_flux_ledger, failed_commit_preserves_accepted_and_pending_transactions) {
  TransactionalFaceFluxLedger<1, ThrowingPayload> ledger{FaceFluxLedgerBudget{4, 4, 1}};
  auto accepted = sample_query<1>(0, 0);
  ledger.begin(accepted.attempt);
  ledger.accumulate(
      fragment_key(accepted, FaceLedgerRole::Coarse, accepted.coarse_face, "accepted",
                   Rational{1, 2}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.1, 1.0},
      ThrowingPayload{2.0});
  ledger.commit();
  ASSERT_EQ(ledger.published_size(), 1u);

  auto candidate = sample_query<1>(0, 1);
  ledger.begin(candidate.attempt);
  ledger.accumulate(
      fragment_key(candidate, FaceLedgerRole::Coarse, candidate.coarse_face, "candidate",
                   Rational{1, 2}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.1, 1.0},
      ThrowingPayload{3.0});
  ThrowingPayload::fail_copy = true;
  EXPECT_THROW(ledger.commit(), std::runtime_error);
  ThrowingPayload::fail_copy = false;
  EXPECT_TRUE(ledger.in_transaction());
  EXPECT_EQ(ledger.published_size(), 1u);
  EXPECT_EQ(ledger.pending_size(), 1u);
  ledger.rollback();
  EXPECT_EQ(ledger.published_size(), 1u);
  EXPECT_EQ(ledger.pending_size(), 0u);

  auto survivor = sample_query<1>(0, 2);
  ledger.begin(survivor.attempt);
  ledger.accumulate(
      fragment_key(survivor, FaceLedgerRole::Coarse, survivor.coarse_face, "survivor",
                   Rational{1, 2}),
      FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.1, 1.0},
      ThrowingPayload{4.0});
  ledger.commit();
  ThrowingPayload::fail_copy = true;
  EXPECT_THROW((void)ledger.discard_published_attempt(0), std::runtime_error);
  ThrowingPayload::fail_copy = false;
  EXPECT_EQ(ledger.published_size(), 2u);
}

TEST(test_nd_flux_ledger, subnormal_cell_measure_fails_before_non_finite_axpy) {
  pops::amr::reflux::nd::MetricFaceReflux<double> reflux;
  reflux.mismatch = 1.0;
  EXPECT_THROW(
      (void)coarse_cell_reflux_correction(reflux, std::numeric_limits<double>::denorm_min(),
                                          CoarseCellFaceSide::Lower, scalar_axpy),
      std::overflow_error);
}

TEST(test_nd_flux_ledger, sources_cell_centering_and_stale_attempts_fail_closed) {
  const auto query = sample_query<2>(1, 7);
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  auto source = fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "source",
                             Rational{1, 2}, FaceLedgerContribution::Source);
  EXPECT_THROW(
      ledger.accumulate(
          source, FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.1, 1.0},
          3.0),
      std::invalid_argument);
  auto cell =
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "cell", Rational{1, 2});
  cell.centering = FaceLedgerCentering::Cell;
  EXPECT_THROW(
      ledger.accumulate(
          cell, FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.1, 1.0},
          3.0),
      std::invalid_argument);
  auto stale =
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "stale", Rational{1, 2});
  stale.attempt = 6;
  EXPECT_THROW(
      ledger.accumulate(
          stale, FaceFluxFragmentMeasure{Rational{1, 1}, Rational{0, 1}, Rational{1, 1}, 0.1, 1.0},
          3.0),
      std::invalid_argument);
  EXPECT_EQ(ledger.pending_size(), 0u);
  ledger.rollback();
  EXPECT_EQ(ledger.published_size(), 0u);
}
