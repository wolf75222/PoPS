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
#include <limits>
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

void require_kokkos_runtime() {
#if defined(POPS_HAS_KOKKOS)
  static Kokkos::ScopeGuard guard;
#endif
}

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
  result.key.temporal_family =
      "pops.program-flux-family.v1:sha256:"
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
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
  ASSERT_TRUE(decoded.face_evidence_provenance);
  EXPECT_EQ(decoded.face_evidence_provenance->spatial_contract, runtime.spatial_contract());
  EXPECT_EQ(decoded.face_evidence_provenance->level_count, 2u);
  EXPECT_NO_THROW(program::require_live_amr_program_checkpoint(decoded, runtime));

  // POPSAND5 had no typed temporal-family string. Remove exactly those framed strings to obtain
  // the frozen legacy layout, then prove an AND6 rewrite retains the empty legacy family.
  std::vector<std::uint8_t> legacy5 = bytes;
  const std::string family = decoded.accepted_face_flux[0].front().key.temporal_family;
  for (;;) {
    const auto found = std::search(legacy5.begin(), legacy5.end(), family.begin(), family.end());
    if (found == legacy5.end())
      break;
    ASSERT_GE(std::distance(legacy5.begin(), found), 8);
    const auto frame = found - 8;
    std::uint64_t size = 0;
    for (int byte = 0; byte < 8; ++byte)
      size |= static_cast<std::uint64_t>(*(frame + byte)) << (8 * byte);
    ASSERT_EQ(size, family.size());
    legacy5.erase(frame, found + static_cast<std::ptrdiff_t>(family.size()));
  }
  legacy5[7] = '5';
  const auto decoded5 = program::deserialize_amr_program_accepted_state<Dim>(legacy5);
  for (const auto& axis : decoded5.accepted_face_flux)
    for (const auto& entry : axis)
      EXPECT_TRUE(entry.key.temporal_family.empty());
  const auto rewritten5 = program::serialize_amr_program_accepted_state(decoded5);
  const auto roundtrip5 = program::deserialize_amr_program_accepted_state<Dim>(rewritten5);
  EXPECT_EQ(program::serialize_amr_program_accepted_state(roundtrip5), rewritten5);
  for (const auto& axis : roundtrip5.accepted_face_flux)
    for (const auto& entry : axis)
      EXPECT_TRUE(entry.key.temporal_family.empty());

  // V4 ends immediately before the optional V5 origin suffix. Its evidence belonged to the
  // envelope geometry, so upgrade that exact old image without changing the accepted payload.
  std::vector<std::uint8_t> legacy = legacy5;
  legacy.resize(legacy.size() - (5 * sizeof(std::uint64_t) +
                                 decoded.face_evidence_provenance->spatial_contract.size()));
  legacy[7] = '4';
  const auto upgraded = program::deserialize_amr_program_accepted_state<Dim>(legacy);
  EXPECT_EQ(upgraded.face_evidence_provenance, decoded.face_evidence_provenance);
  for (const auto& axis : upgraded.accepted_face_flux)
    for (const auto& entry : axis)
      EXPECT_TRUE(entry.key.temporal_family.empty());

  // Regridding can remove the former child. Historical contributions remain tied to their
  // original two-level geometry, while the new accepted envelope describes one live level.
  auto remapped = decoded;
  remapped.spatial_contract = "new-single-level-geometry";
  ++remapped.topology_epoch;
  ++remapped.materialization_generation;
  remapped.level_clocks.resize(1);
  remapped.synchronization_events.push_back({0, 1, 0, "reflux", {0, 3, {0, 1}, 3.0}});
  const auto historical = program::deserialize_amr_program_accepted_state<Dim>(
      program::serialize_amr_program_accepted_state(remapped));
  EXPECT_EQ(historical.face_evidence_provenance, decoded.face_evidence_provenance);
  EXPECT_EQ(historical.level_clocks.size(), 1u);
  EXPECT_EQ(historical.accepted_face_flux[0].size(), decoded.accepted_face_flux[0].size());
  remapped.face_evidence_provenance->level_count = 1;
  EXPECT_THROW((void)program::serialize_amr_program_accepted_state(remapped),
               std::invalid_argument);

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
  corrupted = bytes;
  const auto family_bytes =
      std::search(corrupted.begin(), corrupted.end(), family.begin(), family.end());
  ASSERT_NE(family_bytes, corrupted.end());
  ASSERT_GE(std::distance(corrupted.begin(), family_bytes), 8);
  std::fill(family_bytes - 8, family_bytes, std::uint8_t{0xff});
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
  require_kokkos_runtime();
  prove_ranked_reflux_and_checkpoint<1>();
  prove_ranked_reflux_and_checkpoint<2>();
  prove_ranked_reflux_and_checkpoint<3>();
}

TEST(test_program_reflux_ledger, SingleWindowDensityAvoidsAnExtraTemporalRounding) {
  const auto query = coarse_key<1>(0);
  const pops::Real density =
      pops::Real(1) + pops::Real(21) * std::numeric_limits<pops::Real>::epsilon();
  for (const double dt : {0.001, 0.0005}) {
    const auto integrate = [&](pops::Real coarse_density) {
      reflux::TransactionalFaceFluxLedger<1, program::AmrProgramFacePayload> ledger(kLedgerBudget);
      auto coarse =
          fragment(query, query.coarse_face, reflux::FaceLedgerRole::Coarse, 1.0, coarse_density);
      auto fine = fragment(query, query.coarse_face, reflux::FaceLedgerRole::Fine, 1.0, density);
      coarse.measure.substep_duration = dt;
      fine.measure.substep_duration = dt;
      ledger.begin(query.attempt);
      ledger.accumulate(std::move(coarse.key), coarse.measure, std::move(coarse.payload));
      ledger.accumulate(std::move(fine.key), fine.measure, std::move(fine.payload));
      ledger.commit();
      return reflux::metric_reflux(ledger, query, ratio_two<1>(),
                                   reflux::FaceRefinementMapping<1>{}, kMetricBudget, payload_axpy);
    };
    const auto retained = integrate(density);
    ASSERT_EQ(retained.mismatch.size(), 1U);
    EXPECT_EQ(retained.mismatch[0], pops::Real(0));
    EXPECT_EQ(retained.coarse_integrated[0], static_cast<pops::Real>(dt) * density);

    // These volatile stores represent the two separate field kernels used by
    // the former full-window dt*F then (1/dt)*F path. The ledger must still apply
    // dt, so that unnecessary round trip can create reflux for identical fluxes.
    volatile pops::Real integrated = static_cast<pops::Real>(dt) * density;
    volatile pops::Real reconstructed = integrated * (pops::Real(1) / static_cast<pops::Real>(dt));
    const auto rounded_twice = integrate(reconstructed);
    EXPECT_NE(rounded_twice.mismatch[0], pops::Real(0));
  }
}

TEST(test_program_reflux_ledger, InvalidCheckpointAndDuplicateFacesRejectBeforeMutation) {
  require_kokkos_runtime();
  prove_checkpoint_rejections();
}

}  // namespace
