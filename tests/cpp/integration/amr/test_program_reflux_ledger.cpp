/// @file
/// @brief Exact-ranked Program reflux and accepted-checkpoint proofs.

#include <gtest/gtest.h>

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/reflux/face_flux_ledger.hpp>
#include <pops/amr/reflux/metric_reflux.hpp>
#include <pops/numerics/time/amr/levels/amr_patch_range.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace reflux = pops::amr::reflux;
namespace time_amr = pops::numerics::time::amr;
namespace program = pops::runtime::program;

constexpr pops::mesh::BoxArrayValidationBudget kLayoutBudget{8, 28};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{2, 8};
constexpr reflux::FaceFluxLedgerBudget kLedgerBudget{128, 128, 4};
constexpr reflux::MetricRefluxBudget kMetricBudget{16, 128, 64};

template <int Dim>
pops::Extent<Dim> filled_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::amr::RefinementRatio<Dim> ratio_two() {
  std::array<int, Dim> values{};
  values.fill(2);
  return pops::amr::RefinementRatio<Dim>(values);
}

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance() {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.program-reflux.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
pops::runtime::amr::AmrRuntime<Dim> make_runtime(std::string identity = "program-reflux") {
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = -2 + axis;
    upper[axis] = lower[axis] + 3;
  }
  const pops::Box<Dim> parent_domain{lower, upper};
  const auto ratio = ratio_two<Dim>();
  const pops::Box<Dim> child_domain = hierarchy::refine_box(parent_domain, ratio);
  const pops::mesh::BoxArray<Dim> parent_boxes(std::vector<pops::Box<Dim>>{parent_domain});
  const pops::mesh::BoxArray<Dim> child_boxes(std::vector<pops::Box<Dim>>{child_domain});
  const pops::mesh::RankSpace<Dim> ranks(pops::Index<Dim>{}, filled_extent<Dim>(1));
  const auto parent_distribution = pops::mesh::Distribution<Dim>::replicated(parent_boxes, ranks);
  const auto child_distribution = pops::mesh::Distribution<Dim>::replicated(child_boxes, ranks);
  hierarchy::LevelLayout<Dim> parent_layout(0, parent_domain, parent_boxes, parent_distribution,
                                            pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
  hierarchy::LevelLayout<Dim> child_layout(1, child_domain, child_boxes, child_distribution, ratio,
                                           kLayoutBudget);
  pops::MultiFab<Dim> parent(parent_boxes, parent_distribution, pops::Index<Dim>{}, 1,
                             filled_extent<Dim>(1));
  pops::MultiFab<Dim> child(child_boxes, child_distribution, pops::Index<Dim>{}, 1,
                            filled_extent<Dim>(1));
  std::vector<hierarchy::AmrLevelState<Dim>> levels;
  levels.emplace_back(std::move(parent_layout), std::move(parent));
  levels.emplace_back(std::move(child_layout), std::move(child));
  return pops::runtime::amr::AmrRuntime<Dim>(
      hierarchy::AmrHierarchy<Dim>(std::move(levels), kHierarchyBudget), load_balance<Dim>(),
      std::move(identity));
}

void payload_axpy(program::AmrProgramFacePayload& destination, double coefficient,
                  const program::AmrProgramFacePayload& source) {
  if (destination.empty())
    destination.assign(source.size(), pops::Real(0));
  if (destination.size() != source.size())
    throw std::invalid_argument("test Program face payload width mismatch");
  for (std::size_t component = 0; component < source.size(); ++component)
    destination[component] += static_cast<pops::Real>(coefficient) * source[component];
}

template <int Dim>
reflux::CoarseFaceRefluxKey<Dim> coarse_key(int axis, std::uint64_t attempt = 7) {
  reflux::CoarseFaceRefluxKey<Dim> key;
  key.owner = "program-reflux";
  key.state = "tracer.U";
  key.levels = {0, 1};
  key.axis = axis;
  key.coarse_face = pops::Index<Dim>{};
  key.attempt = attempt;
  key.macro_step = 3;
  return key;
}

template <int Dim>
reflux::FaceFluxFragment<Dim, program::AmrProgramFacePayload> fragment(
    const reflux::CoarseFaceRefluxKey<Dim>& query, pops::Index<Dim> face,
    reflux::FaceLedgerRole role, double face_measure, pops::Real value) {
  reflux::FaceFluxFragment<Dim, program::AmrProgramFacePayload> result;
  result.key.owner = query.owner;
  result.key.state = query.state;
  result.key.levels = query.levels;
  result.key.axis = query.axis;
  result.key.face = face;
  result.key.coarse_face = query.coarse_face;
  result.key.clock = {
      role == reflux::FaceLedgerRole::Coarse ? query.levels.coarse : query.levels.fine,
      query.macro_step, pops::amr::Rational(1, 2), 3.5};
  result.key.stage = "rk.accepted";
  result.key.attempt = query.attempt;
  result.key.role = role;
  result.measure.stage_weight = {1, 1};
  result.measure.substep_begin = {0, 1};
  result.measure.substep_end = {1, 1};
  result.measure.substep_duration = 1.0;
  result.measure.face_measure = face_measure;
  result.payload = {value};
  return result;
}

template <int Dim>
void publish_complete_face(
    reflux::TransactionalFaceFluxLedger<Dim, program::AmrProgramFacePayload>& ledger,
    const reflux::CoarseFaceRefluxKey<Dim>& query,
    const reflux::FaceRefinementMapping<Dim>& mapping,
    const pops::amr::RefinementRatio<Dim>& ratio) {
  const std::vector<pops::Index<Dim>> fine_faces =
      reflux::fine_faces_for_coarse_face(query, ratio, mapping, kMetricBudget);
  std::vector<reflux::FaceFluxFragment<Dim, program::AmrProgramFacePayload>> fragments;
  fragments.reserve(fine_faces.size() + 1);
  fragments.push_back(
      fragment(query, query.coarse_face, reflux::FaceLedgerRole::Coarse, 1.0, pops::Real(2)));
  const double fine_measure = 1.0 / static_cast<double>(fine_faces.size());
  for (const pops::Index<Dim>& face : fine_faces)
    fragments.push_back(
        fragment(query, face, reflux::FaceLedgerRole::Fine, fine_measure, pops::Real(4)));
  std::sort(fragments.begin(), fragments.end(),
            [](const auto& left, const auto& right) { return left.key < right.key; });
  ledger.begin(query.attempt);
  for (auto& entry : fragments)
    ledger.accumulate(std::move(entry.key), entry.measure, std::move(entry.payload));
  ledger.commit();
}

template <int Dim>
void prove_ranked_reflux_and_checkpoint() {
  auto runtime = make_runtime<Dim>();
  const std::array<int, 1> temporal_substeps{2};
  const auto plan = time_amr::PreparedAmrSubcyclePlan<Dim>::prepare(
      runtime, std::span<const int>(temporal_substeps), {1, kLayoutBudget});
  plan.require_live(runtime);
  const auto& transition = plan.transition(0);
  ASSERT_EQ(transition.parent_level(), 0U);
  ASSERT_EQ(transition.child_level(), 1U);
  ASSERT_EQ(transition.temporal_substeps(), 2);

  const auto ratio = runtime.hierarchy().layout(1).ratio_from_parent();
  const time_amr::PatchRange<Dim> patch(runtime.hierarchy().layout(1).patches()[0], ratio);
  EXPECT_EQ(patch.parent_footprint(), runtime.hierarchy().layout(0).domain());
  const reflux::FaceRefinementMapping<Dim> mapping{transition.interface_identity().parent.domain.lo,
                                                   transition.interface_identity().child.domain.lo};

  reflux::TransactionalFaceFluxLedger<Dim, program::AmrProgramFacePayload> ledger(kLedgerBudget);
  for (int axis = 0; axis < Dim; ++axis) {
    const auto query = coarse_key<Dim>(axis, 7 + static_cast<std::uint64_t>(axis));
    publish_complete_face(ledger, query, mapping, ratio);
    const auto reconciled = transition.reconcile_reflux(runtime, ledger, query, "tracer.U",
                                                        kMetricBudget, payload_axpy);
    ASSERT_EQ(reconciled.coarse_integrated.size(), 1U);
    ASSERT_EQ(reconciled.fine_integrated.size(), 1U);
    ASSERT_EQ(reconciled.mismatch.size(), 1U);
    EXPECT_EQ(reconciled.coarse_integrated[0], pops::Real(2));
    EXPECT_EQ(reconciled.fine_integrated[0], pops::Real(4));
    EXPECT_EQ(reconciled.mismatch[0], pops::Real(2));
    EXPECT_EQ(reconciled.fine_face_count, static_cast<std::size_t>(std::size_t{1} << (Dim - 1)));
    const auto lower = reflux::coarse_cell_reflux_correction(
        reconciled, 0.5, reflux::CoarseCellFaceSide::Lower, payload_axpy);
    const auto upper = reflux::coarse_cell_reflux_correction(
        reconciled, 0.5, reflux::CoarseCellFaceSide::Upper, payload_axpy);
    EXPECT_EQ(lower, (program::AmrProgramFacePayload{pops::Real(4)}));
    EXPECT_EQ(upper, (program::AmrProgramFacePayload{pops::Real(-4)}));
  }

  std::vector<pops::amr::ClockStamp> clocks{
      {0, 3, pops::amr::Rational(0, 1), 3.0},
      {1, 3, pops::amr::Rational(0, 1), 3.0},
  };
  program::CellTemporalPartitionAcceptedState temporal;
  auto accepted = program::accepted_amr_program_state<Dim>(
      std::string(runtime.spatial_contract()), runtime.topology_epoch(),
      runtime.materialization_generation(), std::move(clocks), temporal, ledger);
  accepted.logical_clock_ticks.emplace("clock.macro", 3);
  accepted.tagging_hysteresis_state = {1, 0, 1};
  const std::vector<std::uint8_t> bytes = program::serialize_amr_program_accepted_state(accepted);
  const auto decoded = program::deserialize_amr_program_accepted_state<Dim>(bytes);
  EXPECT_EQ(program::serialize_amr_program_accepted_state(decoded), bytes);
  EXPECT_NO_THROW(program::require_live_amr_program_checkpoint(decoded, runtime));

  auto restored = program::restore_amr_program_face_flux_ledger(
      decoded, reflux::FaceFluxLedgerBudget{256, 256, 4});
  EXPECT_EQ(restored.published_size(), ledger.published_size());
  const auto replay = transition.reconcile_reflux(runtime, restored, coarse_key<Dim>(0), "tracer.U",
                                                  kMetricBudget, payload_axpy);
  EXPECT_EQ(replay.mismatch, (program::AmrProgramFacePayload{pops::Real(2)}));

  auto other_runtime = make_runtime<Dim>("other-program-reflux");
  EXPECT_THROW(program::require_live_amr_program_checkpoint(decoded, other_runtime),
               std::invalid_argument);
  EXPECT_THROW(plan.require_live(other_runtime), std::invalid_argument);

  std::vector<std::uint8_t> corrupted = bytes;
  corrupted.front() ^= 0xffU;
  EXPECT_THROW((void)program::deserialize_amr_program_accepted_state<Dim>(corrupted),
               std::runtime_error);
}

void prove_checkpoint_rejections() {
  auto runtime = make_runtime<2>();
  reflux::TransactionalFaceFluxLedger<2, program::AmrProgramFacePayload> ledger(kLedgerBudget);
  ledger.begin(1);
  EXPECT_THROW((void)program::accepted_amr_program_state<2>(
                   std::string(runtime.spatial_contract()), runtime.topology_epoch(),
                   runtime.materialization_generation(),
                   {{0, 0, pops::amr::Rational(0, 1), 0.0}, {1, 0, pops::amr::Rational(0, 1), 0.0}},
                   {}, ledger),
               std::logic_error);
  ledger.rollback();

  const auto ratio = runtime.hierarchy().layout(1).ratio_from_parent();
  const reflux::FaceRefinementMapping<2> mapping{runtime.hierarchy().layout(0).domain().lo,
                                                 runtime.hierarchy().layout(1).domain().lo};
  const auto query = coarse_key<2>(0, 2);
  publish_complete_face(ledger, query, mapping, ratio);
  EXPECT_THROW(publish_complete_face(ledger, query, mapping, ratio), std::invalid_argument);
}

TEST(test_program_reflux_ledger,
     CanonicalMetricLedgerAndAcceptedCheckpointAreExactInOneTwoAndThreeDimensions) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  prove_ranked_reflux_and_checkpoint<1>();
  prove_ranked_reflux_and_checkpoint<2>();
  prove_ranked_reflux_and_checkpoint<3>();
  prove_checkpoint_rejections();
}

}  // namespace
