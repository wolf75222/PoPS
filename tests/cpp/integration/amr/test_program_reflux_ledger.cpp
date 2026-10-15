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
#include <pops/runtime/program/continuation_authority.hpp>
#include <pops/runtime/program/history_sample_identity_codec.hpp>
#include <pops/runtime/program/history_transfer_expression.hpp>
#include <pops/runtime/program/temporal_sample_source.hpp>
#include <pops/runtime/program/amr_history_flux_snapshot_codec.hpp>

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
      runtime.materialization_generation(), std::move(clocks), temporal, ledger,
      7 + static_cast<std::uint64_t>(Dim - 1));
  accepted.logical_clock_ticks.emplace("clock.macro", 3);
  accepted.tagging_hysteresis_state = {1, 0, 1};
  const std::vector<std::uint8_t> bytes = program::serialize_amr_program_accepted_state(accepted);
  const auto decoded = program::deserialize_amr_program_accepted_state<Dim>(bytes);
  EXPECT_EQ(program::serialize_amr_program_accepted_state(decoded), bytes);
  EXPECT_EQ(decoded.accepted_attempt, 7 + static_cast<std::uint64_t>(Dim - 1));
  ASSERT_TRUE(decoded.face_evidence_provenance);
  EXPECT_EQ(decoded.face_evidence_provenance->spatial_contract, runtime.spatial_contract());
  EXPECT_EQ(decoded.face_evidence_provenance->level_count, 2u);
  EXPECT_NO_THROW(program::require_live_amr_program_checkpoint(decoded, runtime));

  // AND7 and older are inspectable, but contain no committed attempt authority. Remove exactly
  // the new header word; never infer its value from the retained fragments or macro clock.
  std::vector<std::uint8_t> legacy7 = bytes;
  // ANDC's explicit source/publication relation is absent from legacy B and 7.
  // Remove each authenticated fragment's five new words, never relabel its wire.
  for (const auto& axis : accepted.accepted_face_flux)
    for (const auto& entry : axis) {
      program::checkpoint_detail::Writer writer;
      program::checkpoint_detail::write_face_fragment(writer, entry);
      const auto encoded = std::move(writer).take();
      const auto found =
          std::search(legacy7.begin(), legacy7.end(), encoded.begin(), encoded.end());
      ASSERT_NE(found, legacy7.end());
      program::checkpoint_detail::Writer previous_writer;
      program::checkpoint_detail::write_face_fragment(previous_writer, entry, false);
      auto previous = std::move(previous_writer).take();
      const auto offset = previous.size() - (entry.payload.size() + 1) * 8 - 40;
      previous.erase(previous.begin() + offset, previous.begin() + offset + 40);
      const auto position = std::distance(legacy7.begin(), found);
      legacy7.erase(found, found + encoded.size());
      legacy7.insert(legacy7.begin() + position, previous.begin(), previous.end());
    }
  legacy7[7] = 'B';
  const auto decodedB = program::deserialize_amr_program_accepted_state<Dim>(legacy7);
  EXPECT_EQ(program::serialize_amr_program_accepted_state(decodedB), bytes);
  const std::size_t attempt_offset = 5 * sizeof(std::uint64_t) + accepted.spatial_contract.size();
  legacy7.erase(legacy7.begin() + attempt_offset, legacy7.begin() + attempt_offset + 8);
  legacy7[7] = '7';
  const auto decoded7 = program::deserialize_amr_program_accepted_state<Dim>(legacy7);
  EXPECT_FALSE(decoded7.accepted_attempt);
  EXPECT_EQ(decoded7.accepted_face_flux[0].size(), decoded.accepted_face_flux[0].size());
  EXPECT_THROW(program::require_live_amr_program_checkpoint(decoded7, runtime),
               std::invalid_argument);
  EXPECT_THROW((void)program::serialize_amr_program_accepted_state(decoded7),
               std::invalid_argument);
  EXPECT_THROW((void)program::restore_amr_program_face_flux_ledger(decoded7, kLedgerBudget),
               std::invalid_argument);
  auto future_fragment = bytes;
  std::fill(future_fragment.begin() + attempt_offset, future_fragment.begin() + attempt_offset + 8,
            std::uint8_t{0});
  EXPECT_THROW((void)program::deserialize_amr_program_accepted_state<Dim>(future_fragment),
               std::invalid_argument);
  auto zero_fragment = decoded;
  zero_fragment.accepted_face_flux[0].front().key.attempt = 0;
  EXPECT_THROW((void)program::serialize_amr_program_accepted_state(zero_fragment),
               std::invalid_argument);

  // POPSAND5 also lacked typed temporal-family strings. Preserve inspection of the exact layout.
  std::vector<std::uint8_t> legacy5 = legacy7;
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
  EXPECT_FALSE(decoded5.accepted_attempt);
  EXPECT_THROW((void)program::serialize_amr_program_accepted_state(decoded5),
               std::invalid_argument);

  // V4 ends immediately before the optional V5 origin suffix. Its evidence belonged to the
  // envelope geometry, so upgrade that exact old image without changing the accepted payload.
  std::vector<std::uint8_t> legacy = legacy5;
  legacy.resize(legacy.size() - (5 * sizeof(std::uint64_t) +
                                 decoded.face_evidence_provenance->spatial_contract.size()));
  legacy[7] = '4';
  const auto upgraded = program::deserialize_amr_program_accepted_state<Dim>(legacy);
  EXPECT_FALSE(upgraded.accepted_attempt);
  EXPECT_THROW(program::require_live_amr_program_checkpoint(upgraded, runtime),
               std::invalid_argument);
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
                   {}, ledger, 0),
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

TEST(test_program_reflux_ledger, HistoryPublicationIdentityUsesPhysicalWindowsAndExactWire) {
  using Sample = program::HistorySampleIdentity;
  using Kind = program::HistorySampleKind;
  std::vector<Sample> ring(2, Sample::zero_start());
  const auto first = program::next_history_sample(ring, false, -0.5, .25);
  EXPECT_EQ(first.kind, Kind::Publication);
  EXPECT_EQ(first.ordinal, 1u);
  ring.assign(2, first);  // First-store cold fill is one authentic publication, not two.
  EXPECT_EQ(program::next_history_sample(ring, true, -0.5, .25), first);
  EXPECT_THROW((void)program::next_history_sample(ring, true, -.5, .125), std::invalid_argument);
  std::swap(ring[0], ring[1]);
  const auto second = program::next_history_sample(ring, false, -0.5, .25);
  EXPECT_EQ(second.ordinal, 2u);
  ring[0] = second;
  std::swap(ring[0], ring[1]);
  EXPECT_EQ(program::next_history_sample(ring, false, -0.5, .25).ordinal, 3u);
  // A fine substep has its own physical window; a new window resets its local ordinal.
  const auto substep = program::next_history_sample(ring, false, -.25, .125);
  EXPECT_EQ(substep.ordinal, 1u);
  EXPECT_NE(substep, first);
  const auto saved = ring;
  ring[1].ordinal = std::numeric_limits<std::uint64_t>::max();
  EXPECT_THROW((void)program::next_history_sample(ring, false, -.5, .25), std::overflow_error);
  EXPECT_EQ(ring[1].ordinal, std::numeric_limits<std::uint64_t>::max());
  ring = saved;  // Accepted HistoryManager snapshots copy the complete typed ledger.
  const auto bytes = program::encode_history_sample_identity("prior", 1, ring);
  EXPECT_EQ(program::decode_history_sample_identity(bytes, "prior", 1, 2), ring);
  EXPECT_THROW((void)program::decode_history_sample_identity(bytes, "foreign", 1, 2),
               std::invalid_argument);
  EXPECT_THROW((void)program::decode_history_sample_identity(bytes, "prior", 0, 2),
               std::invalid_argument);
  EXPECT_THROW((void)program::decode_history_sample_identity(bytes, "prior", 1, 3),
               std::invalid_argument);
  auto malformed_kind = bytes;
  malformed_kind[32 + std::string("prior").size()] = 3;
  EXPECT_THROW((void)program::decode_history_sample_identity(malformed_kind, "prior", 1, 2),
               std::invalid_argument);
  auto extra = bytes;
  extra.push_back(0);
  EXPECT_THROW((void)program::decode_history_sample_identity(extra, "prior", 1, 2),
               std::invalid_argument);
  const auto unknown = program::decode_history_sample_identity({}, "prior", 1, 2);
  EXPECT_EQ(unknown, std::vector<Sample>(2));
  EXPECT_FALSE(unknown.front().authenticated());
  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW((void)program::next_history_sample(ring, false, 0.0, invalid),
                 std::invalid_argument);
  auto invalid = Sample::zero_start();
  invalid.ordinal = 1;
  EXPECT_THROW(invalid.validate(), std::invalid_argument);
  invalid = first;
  invalid.start_bits = std::bit_cast<std::uint64_t>(std::numeric_limits<double>::infinity());
  EXPECT_THROW(invalid.validate(), std::invalid_argument);

  program::AmrProgramAcceptedState<2> state;
  state.spatial_contract = "tests.history-sample-wire";
  state.level_clocks = {{0, 3, {0, 1}, .75}, {1, 3, {0, 1}, .75}};
  state.histories = {{"prior", 0, "state", "cell", "clock", "linear", 2, 1}};
  Sample publication{std::bit_cast<std::uint64_t>(.5), std::bit_cast<std::uint64_t>(.25), 17,
                     Kind::Publication};
  for (int level = 0; level < 2; ++level)
    for (int slot = 0; slot < 2; ++slot)
      state.history_slots.push_back({"prior", level, slot, .25, true, 2, publication});
  const auto current = program::serialize_amr_program_accepted_state(state);
  const auto restored = program::deserialize_amr_program_accepted_state<2>(current);
  EXPECT_EQ(restored.history_slots, state.history_slots);
  // Freeze the actual AND6 layout: it predates committed attempts, declared history halos,
  // and the per-slot publication identities. Merely relabeling the current wire is invalid.
  auto legacy = current;
  const std::size_t attempt_offset = 5 * sizeof(std::uint64_t) + state.spatial_contract.size();
  legacy.erase(legacy.begin() + attempt_offset, legacy.begin() + attempt_offset + 8);
  ASSERT_TRUE(state.logical_clock_ticks.empty());
  ASSERT_EQ(state.histories.size(), 1u);
  const auto& history = state.histories.front();
  // Three container counts, five words per clock, and eight legacy descriptor words.
  // The five descriptor strings contribute their characters in addition to those words.
  const std::size_t halo_offset =
      attempt_offset + (3 + 5 * state.level_clocks.size() + 8) * sizeof(std::uint64_t) +
      history.name.size() + history.state_identity.size() + history.space_identity.size() +
      history.clock_identity.size() + history.interpolation_identity.size();
  ASSERT_GE(legacy.size(), halo_offset + 8);
  ASSERT_EQ(history.ghost_depth, -1);
  ASSERT_TRUE(std::all_of(legacy.begin() + halo_offset, legacy.begin() + halo_offset + 8,
                          [](std::uint8_t byte) { return byte == 0xff; }));
  legacy.erase(legacy.begin() + halo_offset, legacy.begin() + halo_offset + 8);
  const auto framed =
      program::encode_history_sample_identity("prior", 0, std::span(&publication, 1));
  const std::vector<std::uint8_t> record(framed.end() - 64, framed.end());
  int removed = 0;
  for (;;) {
    const auto found = std::search(legacy.begin(), legacy.end(), record.begin(), record.end());
    if (found == legacy.end())
      break;
    legacy.erase(found, found + 64);
    ++removed;
  }
  ASSERT_EQ(removed, 4);
  legacy[7] = '6';
  const auto old = program::deserialize_amr_program_accepted_state<2>(legacy);
  EXPECT_EQ(old.histories, state.histories);
  for (const auto& slot : old.history_slots)
    EXPECT_EQ(slot.sample, Sample{});
  EXPECT_FALSE(old.accepted_attempt);
  EXPECT_THROW((void)program::serialize_amr_program_accepted_state(old), std::invalid_argument);
  state.history_slots[0].sample = Sample::zero_start();
  EXPECT_THROW((void)program::serialize_amr_program_accepted_state(state), std::invalid_argument);
}

TEST(test_program_reflux_ledger, AuthoredHistoryMatrixBindsAllSourcesAndExactRestartIdentity) {
  using Sample = program::HistorySampleIdentity;
  using Kind = program::HistorySampleKind;
  std::vector<Sample> sources;
  for (double time : {0., 2., 1.})
    sources.push_back({std::bit_cast<std::uint64_t>(time), std::bit_cast<std::uint64_t>(1.), 1,
                       Kind::Publication});
  program::HistoryTransferRow row{1.5, .5, {-.125, .375, .75}};
  auto exact_row = row;
  exact_row.exact_weights = {pops::ExactWeight(-1, 8), pops::ExactWeight(3, 8),
                             pops::ExactWeight(3, 4)};
  EXPECT_NO_THROW(program::validate_history_transfer_weights(exact_row));
  EXPECT_EQ(program::history_transfer_exact_weight(exact_row, 0), pops::ExactWeight(-1, 8));
  auto mismatched = exact_row;
  mismatched.weights[0] = 0.;
  EXPECT_THROW(program::validate_history_transfer_weights(mismatched), std::invalid_argument);
  mismatched = exact_row;
  mismatched.exact_weights.pop_back();
  EXPECT_THROW(program::validate_history_transfer_weights(mismatched), std::invalid_argument);
  const auto exact_identity =
      program::reconstructed_history_sample(sources, exact_row, "closed-graph", "ring", 3, 4);
  // Distinct exact coefficients can round to exactly the same application
  // weight. Their reconstruction authorities must nevertheless differ.
  auto same_real = exact_row;
  same_real.exact_weights[0] = same_real.exact_weights[0] +
                               pops::ExactWeight::from_hex("0x1", "0x10000000000000000000000000");
  ASSERT_EQ(same_real.exact_weights[0].value(), exact_row.weights[0]);
  EXPECT_NO_THROW(program::validate_history_transfer_weights(same_real));
  EXPECT_NE(exact_identity, program::reconstructed_history_sample(sources, same_real,
                                                                  "closed-graph", "ring", 3, 4));
  const auto sample =
      program::reconstructed_history_sample(sources, row, "closed-graph", "ring", 3, 4);
  EXPECT_EQ(sample.kind, Kind::Reconstruction);
  EXPECT_EQ(std::bit_cast<double>(sample.start_bits), 1.5);
  EXPECT_NE(sample, sources[1]);
  EXPECT_EQ(sample,
            program::reconstructed_history_sample(sources, row, "closed-graph", "ring", 3, 4));
  auto changed = row;
  changed.weights[0] = 0.;
  EXPECT_NE(sample,
            program::reconstructed_history_sample(sources, changed, "closed-graph", "ring", 3, 4));
  EXPECT_NE(sample,
            program::reconstructed_history_sample(sources, row, "other-graph", "ring", 3, 4));
  auto changed_sources = sources;
  ++changed_sources[0].ordinal;
  EXPECT_NE(sample, program::reconstructed_history_sample(changed_sources, row, "closed-graph",
                                                          "ring", 3, 4));
  const auto bytes = program::encode_history_sample_identity("ring", 1, std::span(&sample, 1));
  EXPECT_EQ(std::string(bytes.begin(), bytes.begin() + 8), "POPSHID2");
  EXPECT_EQ(program::decode_history_sample_identity(bytes, "ring", 1, 1).front(), sample);
  auto no_digest = sample;
  no_digest.reconstruction = {};
  EXPECT_THROW(no_digest.validate(), std::invalid_argument);
  auto false_publication = sample;
  false_publication.kind = Kind::Publication;
  EXPECT_THROW(false_publication.validate(), std::invalid_argument);
  // The accepted-state wire carries the complete digest, including in capacity accounting.
  program::AmrProgramAcceptedState<2> state;
  state.spatial_contract = "history-transfer";
  state.level_clocks = {{0, 2, {0, 1}, 2.}};
  state.histories = {{"ring", 0, "state", "cell", "clock", "closed-graph", 2, 1}};
  state.history_slots = {{"ring", 0, 0, .5, true, 1, sample}, {"ring", 0, 1, .5, true, 1, sample}};
  state.histories.front().depth = 1;
  EXPECT_THROW((void)program::serialize_amr_program_accepted_state(state), std::invalid_argument);
  state.histories.front().depth = 2;
  const auto image = program::serialize_amr_program_accepted_state(state);
  EXPECT_EQ(program::deserialize_amr_program_accepted_state<2>(image).history_slots,
            state.history_slots);
  auto publication = sample;
  publication.kind = Kind::Publication;
  publication.reconstruction = {};
  for (auto& slot : state.history_slots)
    slot.sample = publication;
  auto old_image = program::serialize_amr_program_accepted_state(state);
  auto encoded = program::encode_history_sample_identity("ring", 0, std::span(&publication, 1));
  const std::vector<std::uint8_t> record(encoded.end() - 64, encoded.end());
  for (std::size_t slot = 0; slot < state.history_slots.size(); ++slot) {
    const auto found =
        std::search(old_image.begin(), old_image.end(), record.begin(), record.end());
    ASSERT_NE(found, old_image.end());
    old_image.erase(found + 32, found + 64);
  }
  old_image[7] = 'A';
  EXPECT_EQ(program::deserialize_amr_program_accepted_state<2>(old_image).history_slots,
            state.history_slots);
  encoded.resize(encoded.size() - 32);
  encoded[7] = '1';
  EXPECT_EQ(program::decode_history_sample_identity(encoded, "ring", 0, 1).front(), publication);
}

TEST(test_program_reflux_ledger, HistoryTransferNeverAliasesAttemptLocalBasisIdsAcrossSlots) {
  using Basis = program::history_flux::Basis<2>;
  struct Term {
    std::shared_ptr<const Basis> basis;
    std::map<int, pops::amr::Rational> coefficient;
  };
  using Expression = std::map<std::uint64_t, Term>;
  auto first = std::make_shared<Basis>();
  first->identity = 0;
  first->point.physical_time = 1.;
  auto second = std::make_shared<Basis>();
  second->identity = 0;
  second->point.physical_time = 2.;
  std::vector<Expression> input(2);
  input[0].emplace(0, Term{first, {{0, {1, 1}}}});
  input[1].emplace(0, Term{second, {{0, {1, 1}}}});
  int projections = 0;
  const auto projected = program::distinct_history_flux_sources(input, [&](const auto& basis) {
    ++projections;
    return std::make_shared<Basis>(*basis);
  });
  ASSERT_EQ(projected.size(), 2u);
  EXPECT_EQ(projections, 2);
  const auto& a = projected[0].begin()->second;
  const auto& b = projected[1].begin()->second;
  EXPECT_NE(projected[0].begin()->first, projected[1].begin()->first);
  EXPECT_NE(a.basis, b.basis);
  EXPECT_EQ(a.basis->point.physical_time, 1.);
  EXPECT_EQ(b.basis->point.physical_time, 2.);
  EXPECT_EQ(first->identity, 0u);
  EXPECT_EQ(second->identity, 0u);
  EXPECT_EQ(a.coefficient, input[0].begin()->second.coefficient);
  EXPECT_EQ(b.coefficient, input[1].begin()->second.coefficient);
}

TEST(test_program_reflux_ledger, InvalidCheckpointAndDuplicateFacesRejectBeforeMutation) {
  require_kokkos_runtime();
  prove_checkpoint_rejections();
}

TEST(test_program_reflux_ledger, CommittedAttemptWireAndCapacityIncludeEmptyLedgerAuthority) {
  program::AmrProgramAcceptedState<2> state;
  state.spatial_contract = "test.attempt-wire";
  state.level_clocks = {{0, 0, {0, 1}, 0.0}};
  state.logical_clock_ticks = {{"clock.macro", 0}};
  state.flux_budget_contract = "test.empty-flux-budget";
  state.coupling_contract = "test.no-coupling";
  program::AmrProgramAcceptedStateCapacity<2> capacity;
  capacity.spatial_contract_characters = state.spatial_contract.size();
  capacity.level_count = 1;
  capacity.logical_clock_identities = {"clock.macro"};
  capacity.temporal_provider_identity = state.temporal_partition.provider_identity;
  capacity.flux_budget_contract_characters = state.flux_budget_contract.size();
  capacity.coupling_contract_characters = state.coupling_contract.size();
  for (const std::uint64_t cursor :
       {std::uint64_t{0}, std::uint64_t{42}, std::numeric_limits<std::uint64_t>::max()}) {
    state.accepted_attempt = cursor;
    const auto bytes = program::serialize_amr_program_accepted_state(state);
    // Independent fixed wire count: one clock/tick, no histories, faces or events.
    EXPECT_EQ(bytes.size(), 264u + state.spatial_contract.size() +
                                std::string("clock.macro").size() +
                                state.temporal_partition.provider_identity.size() +
                                state.flux_budget_contract.size() + state.coupling_contract.size());
    EXPECT_EQ(program::serialized_amr_program_accepted_state_size(state), bytes.size());
    EXPECT_EQ(program::serialized_amr_program_accepted_state_capacity(capacity),
              bytes.size() + 32u + state.spatial_contract.size());  // Reserved geometry provenance.
    const auto decoded = program::deserialize_amr_program_accepted_state<2>(bytes);
    EXPECT_EQ(decoded.accepted_attempt, cursor);
    EXPECT_TRUE(decoded.accepted_face_flux[0].empty());
    EXPECT_TRUE(decoded.accepted_face_flux[1].empty());
    EXPECT_EQ(program::serialize_amr_program_accepted_state(decoded), bytes);
  }
  state.accepted_attempt = 9;
  if (pops::n_ranks() > 1) {
    auto asymmetric = state;
    if (pops::my_rank() == 0)
      asymmetric.accepted_attempt = 10;
    EXPECT_THROW(program::require_collective_amr_program_checkpoint_consensus(asymmetric),
                 std::runtime_error);
    if (pops::my_rank() == 0)
      asymmetric.accepted_attempt.reset();
    EXPECT_THROW(program::require_collective_amr_program_checkpoint_consensus(asymmetric),
                 std::runtime_error);
  }
  EXPECT_NO_THROW(program::require_collective_amr_program_checkpoint_consensus(state));
  state.accepted_attempt.reset();
  EXPECT_THROW((void)program::serialized_amr_program_accepted_state_size(state),
               std::invalid_argument);
}

TEST(test_program_reflux_ledger,
     AffinePublicationPreservesSourceQuadratureAndScalesAcceptedMeasure) {
  reflux::FaceFluxFragmentKey<2> key;
  key.owner = "field";
  key.state = "conserved_vector";
  key.levels = {0, 1};
  key.clock = {0, 3, {1, 2}, 3.5};
  key.stage = "sample";
  key.attempt = 7;
  const reflux::FaceFluxFragmentMeasure measure{{1, 1}, {0, 1}, {1, 1}, 1., .25};
  reflux::TransactionalFaceFluxLedger<2, program::AmrProgramFacePayload> ledger(kLedgerBudget);
  ledger.begin(7);
  ledger.accumulate(key, measure, {2., -3.});
  ledger.commit();
  EXPECT_THROW(ledger.finalize_affine_publication({3., 1., .9, .8}), std::invalid_argument);
  EXPECT_FALSE(ledger.published_entries(0).front().measure.affine_publication);
  EXPECT_THROW(ledger.finalize_affine_publication({2., 1., .8, .8}), std::invalid_argument);
  EXPECT_FALSE(ledger.published_entries(0).front().measure.affine_publication);
  ledger.finalize_affine_publication({3., 1., .8, .8});
  const auto& accepted = ledger.published_entries(0).front();
  EXPECT_EQ(accepted.key.clock, key.clock);
  EXPECT_EQ(accepted.measure.substep_duration, 1.);
  EXPECT_EQ(accepted.payload, (program::AmrProgramFacePayload{2., -3.}));
  EXPECT_EQ(reflux::weighted_face_flux_scale(accepted.measure), .2);
  EXPECT_THROW(ledger.finalize_affine_publication({3., 1., .8, .8}), std::logic_error);
  program::checkpoint_detail::Writer writer;
  program::checkpoint_detail::write_face_fragment(writer, accepted);
  const auto bytes = std::move(writer).take();
  program::checkpoint_detail::Reader reader(bytes);
  const auto restored = program::checkpoint_detail::read_face_fragment<2>(reader, true, true, true);
  ASSERT_TRUE(restored.measure.affine_publication);
  EXPECT_EQ(restored.measure.affine_publication->source_start, 3.);
  EXPECT_EQ(restored.measure.affine_publication->source_duration, 1.);
  EXPECT_EQ(restored.measure.affine_publication->accepted_duration, .8);
  EXPECT_EQ(restored.measure.affine_publication->increment_scale, .8);
  EXPECT_EQ(restored.key.clock, key.clock);
  EXPECT_EQ(restored.payload, accepted.payload);
  auto corrupt = bytes;
  const std::size_t tag = corrupt.size() - (accepted.payload.size() + 1) * 8 - 40;
  corrupt[tag] = 2;
  program::checkpoint_detail::Reader bad_tag(corrupt);
  EXPECT_THROW((void)program::checkpoint_detail::read_face_fragment<2>(bad_tag, true, true, true),
               std::invalid_argument);
  corrupt[tag] = 0;  // A zero tag may not hide nonzero source/publication metadata.
  program::checkpoint_detail::Reader hidden_relation(corrupt);
  EXPECT_THROW(
      (void)program::checkpoint_detail::read_face_fragment<2>(hidden_relation, true, true, true),
      std::invalid_argument);
}

TEST(test_program_reflux_ledger, SpatialFamilyAllowsDifferentLocalTemporalQuadratures) {
  const auto run = [](int corrupt) {
    const auto query = coarse_key<2>(0);
    const auto ratio = ratio_two<2>();
    const reflux::FaceRefinementMapping<2> mapping{};
    reflux::TransactionalFaceFluxLedger<2, program::AmrProgramFacePayload> ledger(kLedgerBudget);
    const auto fine_faces =
        reflux::fine_faces_for_coarse_face(query, ratio, mapping, kMetricBudget);
    const std::array<pops::amr::Rational, 3> coarse_weights{{{1, 6}, {2, 3}, {1, 6}}};
    const std::array<pops::amr::Rational, 3> fine_weights{{{23, 12}, {-4, 3}, {5, 12}}};
    ledger.begin(query.attempt);
    for (int stage = 0; stage < 3; ++stage) {
      auto entry = fragment(query, query.coarse_face, reflux::FaceLedgerRole::Coarse, 1., 2.);
      entry.key.stage = "coarse.rk3." + std::to_string(stage);
      entry.key.clock.phase = pops::amr::Rational(stage, 2);
      entry.key.clock.physical_time = 3. + entry.key.clock.phase.value();
      entry.measure.stage_weight = coarse_weights[stage];
      ledger.accumulate(std::move(entry.key), entry.measure, std::move(entry.payload));
    }
    for (int substep = 0; substep < 2; ++substep) {
      for (int stage = 0; stage < 3; ++stage) {
        for (const auto& face : fine_faces) {
          auto entry = fragment(query, face, reflux::FaceLedgerRole::Fine, .5, 4.);
          entry.key.stage = "fine.ab3." + std::to_string(stage);
          entry.key.clock.phase = pops::amr::Rational(substep, 2);
          entry.key.clock.physical_time = 3. + entry.key.clock.phase.value();
          entry.measure.stage_weight = fine_weights[stage];
          entry.measure.substep_begin = pops::amr::Rational(substep, 2);
          entry.measure.substep_end = pops::amr::Rational(substep + 1, 2);
          entry.measure.substep_duration = .5;
          if (corrupt == 1)
            entry.key.temporal_family += ".foreign";
          if (corrupt == 2 && stage == 2)
            entry.measure.stage_weight = pops::amr::Rational(6, 12);
          ledger.accumulate(std::move(entry.key), entry.measure, std::move(entry.payload));
        }
      }
    }
    ledger.commit();
    return reflux::metric_reflux(ledger, query, ratio, mapping, kMetricBudget, payload_axpy);
  };
  const auto result = run(0);
  ASSERT_EQ(result.coarse_integrated.size(), 1U);
  ASSERT_EQ(result.fine_integrated.size(), 1U);
  EXPECT_NEAR(result.coarse_integrated[0], 2., 1.e-14);
  EXPECT_NEAR(result.fine_integrated[0], 4., 1.e-14);
  EXPECT_NEAR(result.mismatch[0], 2., 1.e-14);
  EXPECT_THROW((void)run(1), std::runtime_error);
  EXPECT_THROW((void)run(2), std::runtime_error);
}

TEST(test_program_reflux_ledger, AffineInterfaceWireKeepsSourceAndRejectsHiddenRelations) {
  pops::amr::InterfaceFluxFragment<program::AmrProgramFacePayload> fragment;
  fragment.key = {"two-traces",
                  2,
                  0,
                  1,
                  {1, 7, {1, 2}, 3.5},
                  "stage",
                  "graph",
                  "rate",
                  "application",
                  {{1, 7, {0, 1}, 3.}, {1, 7, {1, 1}, 4.}},
                  pops::amr::InterfaceFluxOrientation::FineOutward,
                  0,
                  1};
  fragment.measure = {{1, 2}, .25, 1., true, reflux::AffineIntervalPublication{3., 1., .8, .8}};
  fragment.payload = {2., -3.};
  program::checkpoint_detail::Writer writer;
  program::checkpoint_detail::write_interface_fragment(writer, fragment);
  const auto bytes = std::move(writer).take();
  program::checkpoint_detail::Reader reader(bytes);
  const auto restored =
      program::checkpoint_detail::read_interface_fragment(reader, nullptr, true, true);
  ASSERT_TRUE(restored.measure.affine_publication);
  EXPECT_EQ(restored.key.clock, fragment.key.clock);
  EXPECT_EQ(restored.measure.substep_duration, 1.);
  EXPECT_EQ(restored.measure.affine_publication->accepted_duration, .8);
  EXPECT_EQ(restored.payload, fragment.payload);
  const std::size_t tag = bytes.size() - (fragment.payload.size() + 1) * 8 - 40;
  auto corrupt = bytes;
  corrupt[tag] = 2;
  program::checkpoint_detail::Reader bad_tag(corrupt);
  EXPECT_THROW(
      (void)program::checkpoint_detail::read_interface_fragment(bad_tag, nullptr, true, true),
      std::invalid_argument);
  corrupt[tag] = 0;
  program::checkpoint_detail::Reader hidden(corrupt);
  EXPECT_THROW(
      (void)program::checkpoint_detail::read_interface_fragment(hidden, nullptr, true, true),
      std::invalid_argument);
  program::checkpoint_detail::Writer previous_writer;
  program::checkpoint_detail::write_interface_fragment(previous_writer, fragment, false);
  auto legacy = std::move(previous_writer).take();
  const auto legacy_tag = legacy.size() - (fragment.payload.size() + 1) * 8 - 40;
  legacy.erase(legacy.begin() + legacy_tag, legacy.begin() + legacy_tag + 40);
  program::checkpoint_detail::Reader old(legacy);
  const auto previous = program::checkpoint_detail::read_interface_fragment(old);
  EXPECT_FALSE(previous.measure.affine_publication);
  EXPECT_EQ(previous.payload, fragment.payload);
}

TEST(test_program_reflux_ledger, TemporalPointDoesNotInventOutgoingHistoryAndSealsInvocation) {
  using Source = program::TemporalSampleSource;
  const std::vector<Source> sources{
      Source::history(program::HistorySampleIdentity::zero_start(), false, 0.),
      Source::point(.25, "native-state-event:attempt7:accepted")};
  EXPECT_DOUBLE_EQ(sources[1].interval, 0.);
  const program::HistoryTransferRow row{.25, .5, {0., 1.}};
  const auto first = program::reconstructed_temporal_sample(
      sources, row, "graph:parameters:gamma=.8", "ring", 2, 3);
  const auto second = program::reconstructed_temporal_sample(
      sources, row, "graph:parameters:gamma=.9", "ring", 2, 3);
  EXPECT_EQ(first.kind, program::HistorySampleKind::Reconstruction);
  EXPECT_NE(first, second);  // Even a parameter unused by the matrix remains authoritative.
  EXPECT_DOUBLE_EQ(std::bit_cast<double>(first.start_bits), .25);
  EXPECT_DOUBLE_EQ(std::bit_cast<double>(first.interval_bits), .5);
  auto invalid = row;
  invalid.weights = {1., 0.};
  EXPECT_THROW(
      (void)program::reconstructed_temporal_sample(sources, invalid, "invocation", "ring", 2, 3),
      std::invalid_argument);
  invalid = row;
  invalid.interval = 0.;
  EXPECT_THROW(
      (void)program::reconstructed_temporal_sample(sources, invalid, "invocation", "ring", 2, 3),
      std::invalid_argument);
  EXPECT_THROW((void)Source::point(.25, ""), std::invalid_argument);
}

}  // namespace

TEST(test_program_reflux_ledger, WideCoefficientCheckpointAndBudgetAreExact) {
  program::AmrProgramAcceptedState<2> state;
  state.accepted_attempt = 1;
  state.spatial_contract = "test.wide-weight-wire";
  state.level_clocks = {{0, 0, {0, 1}, 0.0}};
  state.logical_clock_ticks = {{"clock.macro", 0}};
  state.flux_budget_contract = "test.coefficient-budget/512";
  state.coupling_contract = "test.no-coupling";
  state.face_evidence_provenance =
      program::AmrProgramFaceEvidenceProvenance{state.spatial_contract, 0, 0, 2};
  auto& entry = state.accepted_face_flux[0].emplace_back();
  entry.key.attempt = 1;
  entry.key.owner = "exact-weight";
  entry.key.state = "tracer.U";
  entry.key.levels = {0, 1};
  entry.key.axis = 0;
  entry.key.clock = {0, 0, {0, 1}, 0.};
  entry.key.stage = "exact-weight-stage";
  entry.key.role = reflux::FaceLedgerRole::Coarse;
  entry.payload = {2.};
  for (double value : {1e-6, -1e-6, std::numeric_limits<double>::denorm_min()}) {
    entry.measure = {pops::ExactWeight::from_double(value, 512), {0, 1}, {1, 1}, 1., 1.};
    const auto bytes = program::serialize_amr_program_accepted_state(state);
    EXPECT_EQ(bytes[7], 'E');
    const auto restored = program::deserialize_amr_program_accepted_state<2>(bytes, nullptr, 512);
    EXPECT_EQ(restored.accepted_face_flux[0][0].measure.stage_weight, entry.measure.stage_weight);
    EXPECT_EQ(program::serialize_amr_program_accepted_state(restored), bytes);
    EXPECT_THROW((void)program::deserialize_amr_program_accepted_state<2>(bytes, nullptr, 8),
                 std::length_error);
    EXPECT_EQ(entry.measure.stage_weight.value(), value);
  }
}

namespace {
using ForestStore = program::ContinuationStore<2>;
using ForestValue = program::ContinuationValue<2>;

pops::MultiFab<2> continuation_forest_field() {
  const pops::Extent<2> cells{2, 2};
  const auto layout =
      pops::mesh::BoxArray<2>::from_domain(pops::Box<2>::from_extents(cells), cells);
  const pops::mesh::RankSpace<2> ranks(pops::Index<2>{0, 0}, pops::Extent<2>{pops::n_ranks(), 1});
  const auto distribution =
      pops::mesh::Distribution<2>::partitioned(layout, ranks, {pops::Index<2>{0, 0}});
  pops::MultiFab<2> field(layout, distribution, pops::Index<2>{pops::my_rank(), 0}, 1,
                          pops::Extent<2>{0, 0});
  field.set_val(pops::Real(3));
  return field;
}

ForestStore continuation_forest_store(const pops::MultiFab<2>& field, const std::string& identity,
                                      std::int64_t control = 11) {
  program::ContinuationType real, integer, field_type, tuple;
  real.kind = program::ContinuationKind::Float64;
  integer.kind = program::ContinuationKind::Int64;
  field_type.kind = program::ContinuationKind::Field;
  field_type.field_owner = 0;
  field_type.field_components = 1;
  field_type.scientific_contract = "test/forest/field";
  tuple.names = {"time", "step", "field", "control"};
  tuple.children = {real, integer, field_type, integer};
  program::ContinuationConfiguration configuration;
  configuration.type = tuple;
  configuration.identity = identity;
  configuration.byte_capacity = 65536;
  configuration.time_path = {0};
  configuration.step_path = {1};
  configuration.fields = {{{2}, 0, "clock.macro", "test/forest/read", "test/forest/write"}};
  ForestStore store;
  store.configure(configuration);
  auto prototypes = [&](int owner) -> const pops::MultiFab<2>& {
    if (owner != 0)
      throw std::invalid_argument("foreign forest owner");
    return field;
  };
  store.publish(
      store.prepare(ForestValue::product({ForestValue::real(.25), ForestValue::integer(2),
                                          ForestValue::field(field, 0, "test/forest/field"),
                                          ForestValue::integer(control)}),
                    prototypes, true));
  return store;
}

program::AmrProgramContinuationLevel continuation_forest_record(int level, const ForestStore& store,
                                                                const pops::MultiFab<2>& field) {
  auto prototypes = [&](int) -> const pops::MultiFab<2>& { return field; };
  return {level,
          3,
          7,
          store.domain_high_water(),
          store.configuration().identity,
          store.codec(prototypes).collective_identity(store.read(), store.publication()),
          store.checkpoint(prototypes)};
}

program::AmrProgramAcceptedState<2> continuation_forest_state(const pops::MultiFab<2>& field) {
  program::AmrProgramAcceptedState<2> state;
  state.spatial_contract = "test/forest/topology";
  state.topology_epoch = 3;
  state.materialization_generation = 7;
  state.level_clocks = {{0, 2, {0, 1}, .25}};
  state.logical_clock_ticks = {{"clock.macro", 2}};
  state.flux_budget_contract = "test/forest/flux";
  state.coupling_contract = "test/forest/coupling";
  const auto root = continuation_forest_store(field, "test/forest/root");
  const auto child = continuation_forest_store(field, "test/forest/child");
  state.continuation_levels = {continuation_forest_record(-1, root, field),
                               continuation_forest_record(0, child, field)};
  return state;
}
}  // namespace

TEST(test_program_reflux_ledger, ContinuationForestFramingRetainsCompleteNativeImages) {
  require_kokkos_runtime();
  const auto field = continuation_forest_field();
  auto state = continuation_forest_state(field);
  state.continuation_levels.front().domain_high_water = 1048576;
  const auto bytes = program::serialize_amr_program_accepted_state(state);
  EXPECT_EQ(std::string(bytes.begin(), bytes.begin() + 8), "POPSANDF");
  const auto decoded = program::deserialize_amr_program_accepted_state<2>(bytes);
  EXPECT_EQ(decoded.continuation_levels, state.continuation_levels);
  EXPECT_EQ(program::serialized_amr_program_accepted_state_size(state), bytes.size());
  for (const auto& image : decoded.continuation_levels) {
    auto store = continuation_forest_store(field, image.schema_identity);
    const auto restored =
        store.prepare_restart(image.image, [&](int) -> const pops::MultiFab<2>& { return field; });
    EXPECT_EQ(restored.value->members()[3]->integer(), 11);
    EXPECT_EQ(store.codec([&](int) -> const pops::MultiFab<2>& { return field; })
                  .collective_identity(restored.value, restored.publication),
              image.collective_identity);
    EXPECT_NO_THROW(
        program::require_continuation_projection<2>(restored.value->members()[2]->field(), field));
  }
  auto legacy = state;
  legacy.continuation_levels.clear();
  const auto legacy_bytes = program::serialize_amr_program_accepted_state(legacy);
  EXPECT_EQ(std::string(legacy_bytes.begin(), legacy_bytes.begin() + 8), "POPSANDE");
  EXPECT_TRUE(
      program::deserialize_amr_program_accepted_state<2>(legacy_bytes).continuation_levels.empty());
}

TEST(test_program_reflux_ledger, ContinuationForestRejectsMissingForeignAndTruncatedMembers) {
  require_kokkos_runtime();
  const auto field = continuation_forest_field();
  const auto accepted = continuation_forest_state(field);
  const auto accepted_bytes = program::serialize_amr_program_accepted_state(accepted);
  for (int mutation = 0; mutation != 8; ++mutation) {
    auto candidate = accepted;
    if (mutation == 0)
      candidate.continuation_levels.erase(candidate.continuation_levels.begin());
    if (mutation == 1)
      candidate.continuation_levels.pop_back();
    if (mutation == 2)
      candidate.continuation_levels[1].level = -1;
    if (mutation == 3)
      ++candidate.continuation_levels[1].topology_epoch;
    if (mutation == 4)
      ++candidate.continuation_levels[1].materialization_generation;
    if (mutation == 5)
      candidate.continuation_levels[1].domain_high_water = -1;
    if (mutation == 6)
      candidate.continuation_levels[1].collective_identity.clear();
    if (mutation == 7)
      candidate.continuation_levels[1].image.clear();
    EXPECT_THROW(program::serialize_amr_program_accepted_state(candidate), std::invalid_argument);
    EXPECT_EQ(program::serialize_amr_program_accepted_state(accepted), accepted_bytes);
  }
  auto truncated = accepted_bytes;
  truncated.pop_back();
  EXPECT_THROW(program::deserialize_amr_program_accepted_state<2>(truncated), std::runtime_error);
  auto trailing = accepted_bytes;
  trailing.push_back(0);
  EXPECT_THROW(program::deserialize_amr_program_accepted_state<2>(trailing), std::runtime_error);
}

TEST(test_program_reflux_ledger, ContinuationForestCapacityCountsCommonAndRankLocalBytes) {
  program::AmrProgramAcceptedStateCapacity<2> capacity;
  capacity.spatial_contract_characters = 12;
  capacity.level_count = 3;
  capacity.logical_clock_identities = {"clock"};
  capacity.temporal_provider_identity = "provider";
  capacity.flux_budget_contract_characters = 17;
  capacity.coupling_contract_characters = 13;
  const auto empty = program::serialized_amr_program_accepted_state_capacity(capacity);
  capacity.continuation_root_schema_characters = 11;
  capacity.continuation_root_image_bytes = 4096;
  capacity.continuation_level_schema_characters = 17;
  capacity.continuation_level_image_bytes = 8192;
  EXPECT_EQ(program::serialized_amr_program_accepted_state_capacity(capacity),
            empty + 8 + 56 + 11 + 2 * 4096 + 3 * (56 + 17 + 2 * 8192));
  capacity.continuation_root_image_bytes = 0;
  EXPECT_THROW(program::serialized_amr_program_accepted_state_capacity(capacity),
               std::invalid_argument);
  capacity.continuation_root_image_bytes = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(program::serialized_amr_program_accepted_state_capacity(capacity),
               std::length_error);
}

TEST(test_program_reflux_ledger,
     ContinuationForestConsensusChecksGlobalControlAndKeepsLocalShards) {
  require_kokkos_runtime();
  auto field = continuation_forest_field();
  const auto state = continuation_forest_state(field);
  EXPECT_NO_THROW(program::require_collective_amr_program_checkpoint_consensus(state));
  field.set_val(pops::Real(19));
  auto other = continuation_forest_state(field);
  for (std::size_t i = 0; i != state.continuation_levels.size(); ++i) {
    EXPECT_EQ(state.continuation_levels[i].collective_identity,
              other.continuation_levels[i].collective_identity);
    if (field.local_size() != 0)
      EXPECT_NE(state.continuation_levels[i].image, other.continuation_levels[i].image);
  }
  EXPECT_NO_THROW(program::require_collective_amr_program_checkpoint_consensus(other));
  if (pops::n_ranks() > 1) {
    if (pops::my_rank() == pops::n_ranks() - 1) {
      const auto changed = continuation_forest_store(field, "test/forest/root", 12);
      other.continuation_levels[0] = continuation_forest_record(-1, changed, field);
    }
    EXPECT_THROW(program::require_collective_amr_program_checkpoint_consensus(other),
                 std::runtime_error);
  }
}

TEST(test_program_reflux_ledger, ContinuationForestSealBindsRootAndEveryChildAuthority) {
  require_kokkos_runtime();
  const auto field = continuation_forest_field();
  const auto state = continuation_forest_state(field);
  const auto& records = state.continuation_levels;
  const auto root_identity = records.front().collective_identity;
  const auto seal = program::amr_continuation_forest_identity(records, root_identity);
  EXPECT_EQ(seal.size(), 32U);
  for (int mutation = 0; mutation < 6; ++mutation) {
    auto changed = records;
    if (mutation == 0)
      changed.back().collective_identity.push_back('x');
    else if (mutation == 1)
      ++changed.back().domain_high_water;
    else if (mutation == 2)
      ++changed.back().topology_epoch;
    else if (mutation == 3)
      ++changed.back().materialization_generation;
    else if (mutation == 4)
      changed.back().schema_identity.push_back('x');
    else
      changed.pop_back();
    EXPECT_NE(program::amr_continuation_forest_identity(changed, root_identity), seal);
  }
  EXPECT_NE(program::amr_continuation_forest_identity(records, root_identity + "x"), seal);
  auto other_rank = records;
  other_rank.back().image.push_back(17);
  EXPECT_EQ(program::amr_continuation_forest_identity(other_rank, root_identity), seal);
  std::swap(other_rank.front(), other_rank.back());
  EXPECT_THROW(program::amr_continuation_forest_identity(other_rank, root_identity),
               std::invalid_argument);
}

TEST(test_program_reflux_ledger, RankedContinuationSourceAuthorityAuthenticatesRankOrder) {
  const auto prefix = std::string(program::kProgramSourceAuthorityPrefix);
  const std::vector<std::string> local{prefix + std::string(64, 'a'), prefix + std::string(64, 'b'),
                                       prefix + std::string(64, 'c')};
  const auto wire = program::ranked_program_source_authority(local);
  EXPECT_EQ(wire.size(), program::ranked_program_source_authority_capacity(local.size()));
  EXPECT_EQ(std::string(wire.begin(), wire.begin() + 8), "POPSRSA1");
  for (std::size_t rank = 0; rank < local.size(); ++rank) {
    EXPECT_NO_THROW(program::require_ranked_program_source_authority(wire, 3, rank, local[rank]));
    EXPECT_THROW(
        program::require_ranked_program_source_authority(wire, 3, rank, local[(rank + 1) % 3]),
        std::invalid_argument);
  }
  auto swapped = local;
  std::swap(swapped[0], swapped[1]);
  const auto other = program::ranked_program_source_authority(swapped);
  EXPECT_THROW(program::require_ranked_program_source_authority(other, 3, 0, local[0]),
               std::invalid_argument);
  auto truncated = wire;
  truncated.pop_back();
  EXPECT_THROW(program::require_ranked_program_source_authority(truncated, 3, 0, local[0]),
               std::invalid_argument);
  EXPECT_THROW(program::require_ranked_program_source_authority(wire, 2, 0, local[0]),
               std::invalid_argument);
  EXPECT_THROW(program::require_ranked_program_source_authority(wire, 3, 3, local[0]),
               std::invalid_argument);
  EXPECT_THROW(program::ranked_program_source_authority({prefix + std::string(64, 'z')}),
               std::invalid_argument);
}
