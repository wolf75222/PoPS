/// @file
/// @brief MPI proof for exact-ranked AMR reflux and accepted-ledger restart.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/reflux/face_flux_ledger.hpp>
#include <pops/amr/reflux/metric_reflux.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/time/amr/levels/amr_patch_range.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace reflux = pops::amr::reflux;
namespace time_amr = pops::numerics::time::amr;
namespace program = pops::runtime::program;

constexpr pops::mesh::BoxArrayValidationBudget kLayoutBudget{64, 2'048};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{2, 4'096};
constexpr reflux::FaceFluxLedgerBudget kLedgerBudget{256, 256, 4};
constexpr reflux::MetricRefluxBudget kMetricBudget{32, 256, 64};

template <int Dim>
pops::Extent<Dim> filled_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Index<Dim> rank_coordinate(int rank) {
  pops::Index<Dim> result{};
  result[0] = rank;
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
          "space_filling_curve", "test.mpi-program-reflux.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
pops::runtime::amr::AmrRuntime<Dim> make_partitioned_runtime(int ranks, int rank) {
  pops::Index<Dim> coarse_lower{};
  pops::Index<Dim> coarse_upper{};
  coarse_lower[0] = -2;
  coarse_upper[0] = coarse_lower[0] + 4 * ranks - 1;
  for (int axis = 1; axis < Dim; ++axis) {
    coarse_lower[axis] = -1 - axis;
    coarse_upper[axis] = coarse_lower[axis] + 3;
  }
  const pops::Box<Dim> coarse_domain{coarse_lower, coarse_upper};
  const auto ratio = ratio_two<Dim>();
  const pops::Box<Dim> fine_domain = hierarchy::refine_box(coarse_domain, ratio);

  std::vector<pops::Box<Dim>> coarse_patches;
  std::vector<pops::Box<Dim>> fine_patches;
  std::vector<pops::Index<Dim>> owners;
  coarse_patches.reserve(static_cast<std::size_t>(ranks));
  fine_patches.reserve(static_cast<std::size_t>(ranks));
  owners.reserve(static_cast<std::size_t>(ranks));
  for (int owner = 0; owner < ranks; ++owner) {
    pops::Index<Dim> lower = coarse_lower;
    pops::Index<Dim> upper = coarse_upper;
    lower[0] = coarse_lower[0] + 4 * owner;
    upper[0] = lower[0] + 3;
    const pops::Box<Dim> patch{lower, upper};
    coarse_patches.push_back(patch);
    fine_patches.push_back(hierarchy::refine_box(patch, ratio));
    owners.push_back(rank_coordinate<Dim>(owner));
  }

  const pops::mesh::BoxArray<Dim> coarse_boxes(std::move(coarse_patches));
  const pops::mesh::BoxArray<Dim> fine_boxes(std::move(fine_patches));
  pops::Extent<Dim> rank_extent = filled_extent<Dim>(1);
  rank_extent[0] = ranks;
  const pops::mesh::RankSpace<Dim> rank_space(pops::Index<Dim>{}, rank_extent);
  const auto coarse_distribution =
      pops::mesh::Distribution<Dim>::partitioned(coarse_boxes, rank_space, owners);
  const auto fine_distribution =
      pops::mesh::Distribution<Dim>::partitioned(fine_boxes, rank_space, owners);
  const pops::Index<Dim> local_rank = rank_coordinate<Dim>(rank);

  hierarchy::LevelLayout<Dim> coarse_layout(0, coarse_domain, coarse_boxes, coarse_distribution,
                                            pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
  hierarchy::LevelLayout<Dim> fine_layout(1, fine_domain, fine_boxes, fine_distribution, ratio,
                                          kLayoutBudget);
  pops::MultiFab<Dim> coarse(coarse_boxes, coarse_distribution, local_rank, 1,
                             filled_extent<Dim>(1));
  pops::MultiFab<Dim> fine(fine_boxes, fine_distribution, local_rank, 1, filled_extent<Dim>(1));
  std::vector<hierarchy::AmrLevelState<Dim>> levels;
  levels.emplace_back(std::move(coarse_layout), std::move(coarse));
  levels.emplace_back(std::move(fine_layout), std::move(fine));
  return pops::runtime::amr::AmrRuntime<Dim>(
      hierarchy::AmrHierarchy<Dim>(std::move(levels), kHierarchyBudget), load_balance<Dim>(),
      "test.mpi-program-reflux.spatial");
}

void payload_axpy(program::AmrProgramFacePayload& destination, double coefficient,
                  const program::AmrProgramFacePayload& source) {
  if (destination.empty())
    destination.assign(source.size(), pops::Real(0));
  if (destination.size() != source.size())
    throw std::invalid_argument("MPI Program face payload width mismatch");
  for (std::size_t component = 0; component < source.size(); ++component)
    destination[component] += static_cast<pops::Real>(coefficient) * source[component];
}

template <int Dim>
reflux::CoarseFaceRefluxKey<Dim> coarse_key(int axis) {
  reflux::CoarseFaceRefluxKey<Dim> key;
  key.owner = "test.mpi-program-reflux.spatial";
  key.state = "tracer.U";
  key.levels = {0, 1};
  key.axis = axis;
  key.attempt = 17 + static_cast<std::uint64_t>(axis);
  key.macro_step = 4;
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
  result.key.clock = {role == reflux::FaceLedgerRole::Coarse ? 0 : 1, query.macro_step,
                      pops::amr::Rational(1, 2), 4.5};
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
  const auto fine_faces = reflux::fine_faces_for_coarse_face(query, ratio, mapping, kMetricBudget);
  std::vector<reflux::FaceFluxFragment<Dim, program::AmrProgramFacePayload>> entries;
  entries.reserve(fine_faces.size() + 1);
  entries.push_back(
      fragment(query, query.coarse_face, reflux::FaceLedgerRole::Coarse, 1.0, pops::Real(3)));
  const double fine_measure = 1.0 / static_cast<double>(fine_faces.size());
  for (const auto& face : fine_faces)
    entries.push_back(
        fragment(query, face, reflux::FaceLedgerRole::Fine, fine_measure, pops::Real(5)));
  std::sort(entries.begin(), entries.end(),
            [](const auto& left, const auto& right) { return left.key < right.key; });
  ledger.begin(query.attempt);
  for (auto& entry : entries)
    ledger.accumulate(std::move(entry.key), entry.measure, std::move(entry.payload));
  ledger.commit();
}

template <int Dim>
void prove_collective_reflux_checkpoint(int ranks, int rank) {
  auto runtime = make_partitioned_runtime<Dim>(ranks, rank);
  ASSERT_EQ(runtime.hierarchy().state(0).local_size(), 1);
  ASSERT_EQ(runtime.hierarchy().state(1).local_size(), 1);

  const std::array<int, 1> substeps{2};
  const auto plan = time_amr::PreparedAmrSubcyclePlan<Dim>::prepare(
      runtime, std::span<const int>(substeps), {1, kLayoutBudget});
  const auto& transition = plan.transition(0);
  const auto ratio = runtime.hierarchy().layout(1).ratio_from_parent();
  const time_amr::PatchRange<Dim> local_patch(runtime.hierarchy().state(1).box(0), ratio);
  EXPECT_FALSE(local_patch.parent_footprint().empty());
  const reflux::FaceRefinementMapping<Dim> mapping{transition.interface_identity().parent.domain.lo,
                                                   transition.interface_identity().child.domain.lo};

  reflux::TransactionalFaceFluxLedger<Dim, program::AmrProgramFacePayload> ledger(kLedgerBudget);
  for (int axis = 0; axis < Dim; ++axis) {
    const auto query = coarse_key<Dim>(axis);
    publish_complete_face(ledger, query, mapping, ratio);
    const auto reconciled = transition.reconcile_reflux(runtime, ledger, query, "tracer.U",
                                                        kMetricBudget, payload_axpy);
    EXPECT_EQ(reconciled.mismatch, (program::AmrProgramFacePayload{pops::Real(2)}));
    EXPECT_EQ(reconciled.fine_face_count, static_cast<std::size_t>(std::size_t{1} << (Dim - 1)));
  }

  program::CellTemporalPartitionAcceptedState temporal;
  auto accepted = program::accepted_amr_program_state<Dim>(
      std::string(runtime.spatial_contract()), runtime.topology_epoch(),
      runtime.materialization_generation(),
      {{0, 4, pops::amr::Rational(0, 1), 4.0}, {1, 4, pops::amr::Rational(0, 1), 4.0}}, temporal,
      ledger);
  accepted.logical_clock_ticks.emplace("clock.macro", 4);
  EXPECT_NO_THROW(program::require_collective_amr_program_checkpoint_consensus(accepted));
  EXPECT_NO_THROW(program::require_live_amr_program_checkpoint(accepted, runtime));

  auto restored = program::restore_amr_program_face_flux_ledger(accepted, kLedgerBudget);
  const auto replay = transition.reconcile_reflux(runtime, restored, coarse_key<Dim>(0), "tracer.U",
                                                  kMetricBudget, payload_axpy);
  EXPECT_EQ(replay.mismatch, (program::AmrProgramFacePayload{pops::Real(2)}));

  auto divergent = accepted;
  if (rank == 1)
    divergent.spatial_contract += ".rank-one-divergence";
  bool rejected = false;
  try {
    program::require_collective_amr_program_checkpoint_consensus(divergent);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  EXPECT_EQ(pops::all_reduce_sum(rejected ? 1L : 0L), static_cast<long>(ranks));
}

int run_exact_mpi_program_reflux(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      prove_collective_reflux_checkpoint<pops::kNativeDimension>(pops::n_ranks(), pops::my_rank());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact MPI Program reflux proof failed: %s\n", pops::my_rank(),
                   error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf("OK test_mpi_amr_program_reflux np=%d dim=%d exact-ledger-checkpoint\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_amr_program_reflux, ExactLedgerRestartAndCollectiveRefusal) {
  EXPECT_EQ(pops::test::RunTestBody(&run_exact_mpi_program_reflux, "test_mpi_amr_program_reflux"),
            0);
}
