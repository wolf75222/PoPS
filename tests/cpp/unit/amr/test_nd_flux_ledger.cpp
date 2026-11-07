#include <gtest/gtest.h>

#include <pops/amr/reflux/metric_reflux.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using pops::Index;
using pops::amr::ClockStamp;
using pops::amr::Rational;
using pops::amr::reflux::CoarseCellFaceSide;
using pops::amr::reflux::CoarseFaceRefluxKey;
using pops::amr::reflux::FaceFluxFragmentKey;
using pops::amr::reflux::FaceFluxFragmentMeasure;
using pops::amr::reflux::FaceFluxLedgerBudget;
using pops::amr::reflux::FaceLedgerCentering;
using pops::amr::reflux::FaceLedgerContribution;
using pops::amr::reflux::FaceLedgerRole;
using pops::amr::reflux::FaceRefinementMapping;
using pops::amr::reflux::LevelTransition;
using pops::amr::reflux::MetricRefluxBudget;
using pops::amr::reflux::PublishedCoarseFaceIndex;
using pops::amr::reflux::TransactionalFaceFluxLedger;
using pops::amr::reflux::coarse_cell_reflux_correction;
using pops::amr::reflux::fine_faces_for_coarse_face;
using pops::amr::reflux::metric_reflux;
using pops::amr::RefinementRatio;

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

struct ThrowingMovePayload {
  double value = 0.0;
  inline static int moves_before_failure = -1;

  explicit ThrowingMovePayload(double input = 0.0) : value(input) {}
  ThrowingMovePayload(const ThrowingMovePayload&) = default;
  ThrowingMovePayload(ThrowingMovePayload&& other) : value(other.value) {
    if (moves_before_failure == 0)
      throw std::runtime_error("injected payload move failure");
    if (moves_before_failure > 0)
      --moves_before_failure;
  }
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
    Rational phase, FaceLedgerContribution contribution = FaceLedgerContribution::NumericalFlux,
    std::string temporal_family = {}) {
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
  key.temporal_family = std::move(temporal_family);
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
                      double fine_flux, std::string temporal_family = {}) {
  ledger.accumulate(fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, stage, phase,
                                 FaceLedgerContribution::NumericalFlux, temporal_family),
                    FaceFluxFragmentMeasure{stage_weight, substep_begin, substep_end, duration,
                                            coarse_face_measure},
                    coarse_flux);
  for (const auto& fine_face : fine_faces_for_coarse_face(query, ratio, mapping, budget))
    ledger.accumulate(fragment_key(query, FaceLedgerRole::Fine, fine_face, stage, phase,
                                   FaceLedgerContribution::NumericalFlux, temporal_family),
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

TEST(test_nd_flux_ledger, legacy_aggregate_key_keeps_its_stage_and_empty_family) {
  FaceFluxFragmentKey<1> key{"owner",
                             "state",
                             {0, 1},
                             FaceLedgerCentering::Face,
                             0,
                             Index<1>{4},
                             Index<1>{2},
                             ClockStamp{0, 0, {0, 1}, 0.0},
                             "accepted-stage",
                             7,
                             FaceLedgerRole::Coarse,
                             FaceLedgerContribution::NumericalFlux};
  EXPECT_EQ(key.stage, "accepted-stage");
  EXPECT_EQ(key.attempt, 7u);
  EXPECT_TRUE(key.temporal_family.empty());
}

TEST(test_nd_flux_ledger, same_provider_occurrences_close_independent_temporal_quadratures) {
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto query = sample_query<2>(0, 43);
  const auto budget = reflux_budget();
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  accumulate_stage(ledger, query, ratio, mapping, budget, "provider4/a0", Rational{0, 1},
                   Rational{1, 2}, Rational{0, 1}, Rational{1, 1}, 1.0, 2.0, 1.0, -4.0, -4.0,
                   "occurrence-a");
  accumulate_stage(ledger, query, ratio, mapping, budget, "provider4/b0", Rational{0, 1},
                   Rational{1, 2}, Rational{0, 1}, Rational{1, 1}, 1.0, 2.0, 1.0, 3.0, 3.0,
                   "occurrence-b");
  accumulate_stage(ledger, query, ratio, mapping, budget, "provider4/a1", Rational{1, 1},
                   Rational{1, 2}, Rational{0, 1}, Rational{1, 1}, 1.0, 2.0, 1.0, -4.0, -4.0,
                   "occurrence-a");
  accumulate_stage(ledger, query, ratio, mapping, budget, "provider4/b1", Rational{1, 1},
                   Rational{1, 2}, Rational{0, 1}, Rational{1, 1}, 1.0, 2.0, 1.0, 3.0, 3.0,
                   "occurrence-b");
  ledger.commit();

  const auto result = metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy);
  EXPECT_NEAR(result.coarse_integrated, -2.0, 1e-14);
  EXPECT_NEAR(result.fine_integrated, -2.0, 1e-14);
  EXPECT_NEAR(result.mismatch, 0.0, 1e-14);
  EXPECT_NEAR(result.coarse_weighted_measure, 4.0, 1e-14);
  EXPECT_NEAR(result.fine_weighted_measure, 4.0, 1e-14);
}

TEST(test_nd_flux_ledger, missing_occurrence_stage_cannot_be_hidden_by_duplicate_family) {
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto query = sample_query<2>(0, 44);
  const auto budget = reflux_budget();
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  accumulate_stage(ledger, query, ratio, mapping, budget, "a0", Rational{0, 1}, Rational{1, 2},
                   Rational{0, 1}, Rational{1, 1}, 1.0, 2.0, 1.0, 1.0, 1.0, "occurrence-a");
  accumulate_stage(ledger, query, ratio, mapping, budget, "b0", Rational{0, 1}, Rational{1, 2},
                   Rational{0, 1}, Rational{1, 1}, 1.0, 2.0, 1.0, 1.0, 1.0, "occurrence-b");
  accumulate_stage(ledger, query, ratio, mapping, budget, "b1", Rational{1, 1}, Rational{1, 2},
                   Rational{0, 1}, Rational{1, 1}, 1.0, 2.0, 1.0, 1.0, 1.0, "occurrence-b");
  accumulate_stage(ledger, query, ratio, mapping, budget, "b1-duplicate", Rational{1, 1},
                   Rational{1, 2}, Rational{0, 1}, Rational{1, 1}, 1.0, 2.0, 1.0, 1.0, 1.0,
                   "occurrence-b");
  ledger.commit();
  EXPECT_THROW((void)metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy),
               std::runtime_error);
}

TEST(test_nd_flux_ledger, migrated_ab2_family_closes_across_two_fine_substeps) {
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto query = sample_query<2>(0, 45);
  const auto budget = reflux_budget();
  const auto fine_faces = fine_faces_for_coarse_face(query, ratio, mapping, budget);
  const std::string family = "resolved-operation/occurrence";
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  for (const auto& [stage, weight] : std::array<std::pair<std::string, Rational>, 2>{
           {{"fresh", {3, 2}}, {"retained-flx2", {-1, 2}}}}) {
    ledger.accumulate(fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, stage,
                                   Rational{0, 1}, FaceLedgerContribution::NumericalFlux, family),
                      FaceFluxFragmentMeasure{weight, Rational{0, 1}, Rational{1, 1}, 1.0, 2.0},
                      5.0);
    for (int substep = 0; substep < 2; ++substep)
      for (const auto& fine_face : fine_faces)
        ledger.accumulate(
            fragment_key(query, FaceLedgerRole::Fine, fine_face,
                         stage + "/fine-" + std::to_string(substep), Rational{substep, 2},
                         FaceLedgerContribution::NumericalFlux, family),
            FaceFluxFragmentMeasure{weight, Rational{substep, 2}, Rational{substep + 1, 2}, 0.5,
                                    1.0},
            5.0);
  }
  ledger.commit();

  const auto result = metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy);
  EXPECT_NEAR(result.coarse_integrated, 10.0, 1e-14);
  EXPECT_NEAR(result.fine_integrated, 10.0, 1e-14);
  EXPECT_NEAR(result.mismatch, 0.0, 1e-14);
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

TEST(test_nd_flux_ledger, declared_durations_preserve_clock_rate_after_timestamp_rounding) {
  namespace detail = pops::amr::reflux::detail;
  constexpr double macro_dt = 0.45 / 512.0;
  double time = 0.0;
  for (int step = 0; step < 71; ++step)
    time += macro_dt;
  const pops::amr::ClockWindow root{{0, 71, {0, 1}, time}, {0, 71, {1, 1}, time + macro_dt}};
  const pops::amr::ParentChildClockRelation first_relation(
      0, 1, {2, 1}, pops::amr::RemainderPolicy::IntegralOnly);
  const pops::amr::ParentChildClockRelation second_relation(
      1, 2, {2, 1}, pops::amr::RemainderPolicy::IntegralOnly);
  const auto parent = first_relation.partition(root).back().window;
  const auto children = second_relation.partition(parent);
  std::map<detail::StageSlice, detail::TemporalSliceMeasure> declared;
  auto rounded = declared;
  for (const auto& child : children) {
    const auto& window = child.window;
    const auto slice = detail::stage_slice(window.begin, "advance");
    const double duration = macro_dt * (window.end.phase - window.begin.phase).value();
    declared.emplace(slice, detail::TemporalSliceMeasure{
                                {1, 1}, window.begin.phase, window.end.phase, duration});
    rounded.emplace(
        slice, detail::TemporalSliceMeasure{{1, 1},
                                            window.begin.phase,
                                            window.end.phase,
                                            window.end.physical_time - window.begin.physical_time});
  }
  EXPECT_THROW((void)detail::authenticated_window(rounded, parent.begin.phase, parent.end.phase),
               std::runtime_error);
  const auto accepted =
      detail::authenticated_window(declared, parent.begin.phase, parent.end.phase);
  EXPECT_DOUBLE_EQ(accepted.duration_per_phase, macro_dt);
  EXPECT_DOUBLE_EQ(accepted.duration, macro_dt / 2.0);
  declared.rbegin()->second.substep_duration *= 1.001;
  EXPECT_THROW((void)detail::authenticated_window(declared, parent.begin.phase, parent.end.phase),
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
  EXPECT_THROW(ledger.accumulate(ledger.pending_entries(0).front().key,
                                 ledger.pending_entries(0).front().measure, ThrowingPayload{9.0}),
               std::runtime_error);
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
  EXPECT_EQ(ledger.discard_published_attempt(0), 1u);
  EXPECT_EQ(ledger.published_size(), 1u);
}

TEST(test_nd_flux_ledger, identity_index_preserves_insertion_order_and_exact_equivalence) {
  const auto query = sample_query<2>(0, 0);
  const FaceFluxFragmentMeasure measure{{1, 1}, {0, 1}, {1, 1}, 0.1, 1.0};
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  const std::array<std::string, 3> stages{"z", "a", "m"};
  const std::array<double, 3> values{1.e16, -1.e16, 1.0};
  for (std::size_t index = 0; index < stages.size(); ++index)
    ledger.accumulate(
        fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, stages[index], {0, 1}),
        measure, values[index]);
  auto same = ledger.pending_entries(0).front().key;
  // Physical time and payload/measure are not part of the original ordering identity.
  same.clock.physical_time = -0.0;
  EXPECT_THROW(ledger.accumulate(same, {{1, 2}, {0, 1}, {1, 1}, 0.2, 2.0}, 9.0),
               std::runtime_error);
  auto separate_family = same;
  separate_family.temporal_family = "another-occurrence";
  ledger.accumulate(separate_family, measure, 0.0);
  auto separate_axis = same;
  separate_axis.axis = 1;
  ledger.accumulate(separate_axis, measure, 7.0);
  ledger.commit();
  ASSERT_EQ(ledger.published_entries(0).size(), 4u);
  double sum = 0.0;
  for (std::size_t index = 0; index < stages.size(); ++index) {
    EXPECT_EQ(ledger.published_entries(0)[index].key.stage, stages[index]);
    sum += ledger.published_entries(0)[index].payload;
  }
  EXPECT_DOUBLE_EQ(sum, 1.0);
  EXPECT_DOUBLE_EQ(ledger.published_entries(1).front().payload, 7.0);
}

TEST(test_nd_flux_ledger, identity_index_follows_nested_rollback_copy_move_and_clear) {
  const auto query = sample_query<1>(0, 0);
  const FaceFluxFragmentMeasure measure{{1, 1}, {0, 1}, {1, 1}, 0.1, 1.0};
  const auto first =
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "first", {0, 1});
  const auto child =
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "child", {0, 1});
  TransactionalFaceFluxLedger<1, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  ledger.accumulate(first, measure, 1.0);
  ledger.begin(query.attempt);
  ledger.accumulate(child, measure, 2.0);
  auto copied = ledger;
  ledger.rollback();
  EXPECT_NO_THROW(ledger.accumulate(child, measure, 3.0));
  EXPECT_THROW(copied.accumulate(child, measure, 4.0), std::runtime_error);
  copied.rollback();
  EXPECT_NO_THROW(copied.accumulate(child, measure, 5.0));
  TransactionalFaceFluxLedger<1, double> assigned{ledger_budget()};
  assigned = copied;
  auto moved = std::move(assigned);
  EXPECT_THROW(moved.accumulate(first, measure, 6.0), std::runtime_error);
  moved.commit();
  EXPECT_EQ(moved.discard_published_attempt(query.attempt), 2u);
  moved.clear();
  moved.begin(query.attempt);
  EXPECT_NO_THROW(moved.accumulate(first, measure, 7.0));
  moved.commit();
  ASSERT_EQ(moved.published_size(), 1u);
  EXPECT_DOUBLE_EQ(moved.published_entries(0).front().payload, 7.0);
  EXPECT_DOUBLE_EQ(ledger.pending_entries(0).back().payload, 3.0);
  EXPECT_DOUBLE_EQ(copied.pending_entries(0).back().payload, 5.0);
}

TEST(test_nd_flux_ledger, failed_insertion_does_not_retain_an_identity) {
  const auto query = sample_query<1>(0, 0);
  const auto key = fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "retry", {0, 1});
  const FaceFluxFragmentMeasure measure{{1, 1}, {0, 1}, {1, 1}, 0.1, 1.0};
  for (int failed_move : {0, 1}) {
    TransactionalFaceFluxLedger<1, ThrowingMovePayload> ledger{ledger_budget()};
    ledger.begin(query.attempt);
    ThrowingMovePayload::moves_before_failure = failed_move;
    EXPECT_THROW(ledger.accumulate(key, measure, ThrowingMovePayload{2.0}), std::runtime_error);
    ThrowingMovePayload::moves_before_failure = -1;
    EXPECT_EQ(ledger.pending_size(), 0u);
    EXPECT_NO_THROW(ledger.accumulate(key, measure, ThrowingMovePayload{3.0}));
    EXPECT_THROW(ledger.accumulate(key, measure, ThrowingMovePayload{4.0}), std::runtime_error);
    ledger.commit();
    EXPECT_DOUBLE_EQ(ledger.published_entries(0).front().payload.value, 3.0);
  }
}

TEST(test_nd_flux_ledger, failed_copy_assignment_preserves_both_payload_and_identity) {
  const auto query = sample_query<1>(0, 0);
  const auto old_key =
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "old", {0, 1});
  const auto new_key =
      fragment_key(query, FaceLedgerRole::Coarse, query.coarse_face, "new", {0, 1});
  const FaceFluxFragmentMeasure measure{{1, 1}, {0, 1}, {1, 1}, 0.1, 1.0};
  TransactionalFaceFluxLedger<1, ThrowingPayload> source{ledger_budget()};
  TransactionalFaceFluxLedger<1, ThrowingPayload> destination{ledger_budget()};
  source.begin(query.attempt);
  destination.begin(query.attempt);
  source.accumulate(new_key, measure, ThrowingPayload{2.0});
  destination.accumulate(old_key, measure, ThrowingPayload{1.0});
  ThrowingPayload::fail_copy = true;
  EXPECT_THROW(destination = source, std::runtime_error);
  ThrowingPayload::fail_copy = false;
  ASSERT_EQ(destination.pending_size(), 1u);
  EXPECT_DOUBLE_EQ(destination.pending_entries(0).front().payload.value, 1.0);
  EXPECT_THROW(destination.accumulate(old_key, measure, ThrowingPayload{3.0}), std::runtime_error);
  EXPECT_NO_THROW(destination.accumulate(new_key, measure, ThrowingPayload{4.0}));
  destination.commit();
  EXPECT_EQ(destination.published_size(), 2u);
}

TEST(test_nd_flux_ledger, subnormal_cell_measure_fails_before_non_finite_axpy) {
  pops::amr::reflux::MetricFaceReflux<double> reflux;
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

TEST(test_nd_flux_ledger, authored_partial_integral_requires_its_independent_exact_contract) {
  using pops::amr::reflux::MetricRefluxTemporalContract;
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto query = sample_query<2>(0, 814);
  const auto budget = reflux_budget();
  // This comes from q + h/3 R(q), independently of any emitted face record.
  MetricRefluxTemporalContract expected;
  expected.coarse["diffusive-operator"][{{0, 1}, {1, 1}}] = pops::ExactWeight{1, 3};
  expected.fine = expected.coarse;
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(query.attempt);
  accumulate_stage(ledger, query, ratio, mapping, budget, "predictor", Rational{0, 1},
                   Rational{1, 3}, Rational{0, 1}, Rational{1, 1}, 2.0, 2.0, 1.0, 2.0, 5.0,
                   "diffusive-operator");
  ledger.commit();
  // The full-step API must continue to refuse this incomplete quadrature.
  EXPECT_THROW((void)metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy),
               std::runtime_error);
  const auto result = metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy, &expected);
  EXPECT_DOUBLE_EQ(result.coarse_integrated, 8.0 / 3.0);
  EXPECT_DOUBLE_EQ(result.fine_integrated, 20.0 / 3.0);
  EXPECT_NEAR(result.mismatch, 4.0, 1e-14);
  auto wrong = expected;
  wrong.fine["diffusive-operator"].begin()->second = pops::ExactWeight{1, 2};
  EXPECT_THROW((void)metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy, &wrong),
               std::runtime_error);
  wrong = expected;
  wrong.coarse["foreign-operator"] = wrong.coarse["diffusive-operator"];
  EXPECT_THROW((void)metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy, &wrong),
               std::runtime_error);
  wrong = expected;
  wrong.coarse["diffusive-operator"][{{0, 1}, {1, 2}}] = pops::ExactWeight{1, 3};
  EXPECT_THROW((void)metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy, &wrong),
               std::runtime_error);
  // Mutating both emitted sides together must not become its own authority.
  TransactionalFaceFluxLedger<2, double> mutated{ledger_budget()};
  mutated.begin(query.attempt);
  accumulate_stage(mutated, query, ratio, mapping, budget, "predictor", Rational{0, 1},
                   Rational{1, 2}, Rational{0, 1}, Rational{1, 1}, 2.0, 2.0, 1.0, 2.0, 5.0,
                   "diffusive-operator");
  mutated.commit();
  EXPECT_THROW((void)metric_reflux(mutated, query, ratio, mapping, budget, scalar_axpy, &expected),
               std::runtime_error);
}

TEST(test_nd_flux_ledger, coarse_face_index_reconciles_each_face_like_a_full_ledger_scan) {
  // Several coarse faces on both axes, their stages interleaved in the ledger, with fluxes whose
  // sums depend on the accumulation order: each indexed reconciliation must equal the full scan
  // bit for bit.
  const RefinementRatio<2> ratio{2, 2};
  const auto mapping = sample_mapping<2>();
  const auto budget = reflux_budget();
  std::vector<CoarseFaceRefluxKey<2>> queries;
  for (int axis = 0; axis < 2; ++axis)
    for (int face = 0; face < 3; ++face) {
      auto query = sample_query<2>(axis, 7);
      query.coarse_face[1 - axis] += 2 * face;
      queries.push_back(query);
    }
  TransactionalFaceFluxLedger<2, double> ledger{ledger_budget()};
  ledger.begin(7);
  const std::array<std::pair<const char*, Rational>, 3> stages{
      {{"ssp_a", Rational{0, 1}}, {"ssp_b", Rational{1, 2}}, {"ssp_c", Rational{1, 1}}}};
  for (std::size_t stage = 0; stage < stages.size(); ++stage)
    for (std::size_t face = 0; face < queries.size(); ++face) {
      const double flux = 0.1 * static_cast<double>(face + 1) + 0.3 * static_cast<double>(stage);
      accumulate_stage(ledger, queries[face], ratio, mapping, budget, stages[stage].first,
                       stages[stage].second, Rational{1, 3}, Rational{0, 1}, Rational{1, 1}, 0.7,
                       1.4, 0.7, flux, flux * 1.1);
    }
  ledger.commit();

  const PublishedCoarseFaceIndex index(ledger);
  EXPECT_EQ(&index.ledger(), &ledger);
  for (const auto& query : queries) {
    const auto& entries = index.entries(query.axis, query.coarse_face);
    ASSERT_EQ(entries.size(), stages.size() * 3);
    const auto& published = ledger.published_entries(query.axis);
    for (std::size_t entry = 0; entry < entries.size(); ++entry) {
      EXPECT_TRUE(entries[entry]->key.coarse_face == query.coarse_face);
      EXPECT_GE(entries[entry], published.data());
      EXPECT_LT(entries[entry], published.data() + published.size());
      if (entry > 0)
        EXPECT_LT(entries[entry - 1], entries[entry]);
    }
    const auto scanned = metric_reflux(ledger, query, ratio, mapping, budget, scalar_axpy);
    const auto indexed = metric_reflux(index, query, ratio, mapping, budget, scalar_axpy);
    EXPECT_EQ(indexed.coarse_integrated, scanned.coarse_integrated);
    EXPECT_EQ(indexed.fine_integrated, scanned.fine_integrated);
    EXPECT_EQ(indexed.mismatch, scanned.mismatch);
    EXPECT_EQ(indexed.coarse_weighted_measure, scanned.coarse_weighted_measure);
    EXPECT_EQ(indexed.fine_weighted_measure, scanned.fine_weighted_measure);
    EXPECT_EQ(indexed.fine_face_count, scanned.fine_face_count);
  }

  // A coarse face without fragments has an empty group and fails like the full scan.
  auto absent = queries.front();
  absent.coarse_face[1] += 101;
  EXPECT_TRUE(index.entries(absent.axis, absent.coarse_face).empty());
  EXPECT_THROW((void)metric_reflux(ledger, absent, ratio, mapping, budget, scalar_axpy),
               std::runtime_error);
  EXPECT_THROW((void)metric_reflux(index, absent, ratio, mapping, budget, scalar_axpy),
               std::runtime_error);
  EXPECT_THROW((void)index.entries(2, absent.coarse_face), std::invalid_argument);
}
