/// @file
/// @brief MPI proof that exact AMR checkpoints cannot cross ownership materializations implicitly.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace program = pops::runtime::program;

constexpr pops::mesh::BoxArrayValidationBudget kLayoutBudget{64, 2'048};
constexpr hierarchy::HierarchyValidationBudget kHierarchyBudget{2, 4'096};

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
          "space_filling_curve", "test.mpi-checkpoint-migration.sfc",
          pops::PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
}

template <int Dim>
pops::runtime::amr::AmrRuntime<Dim> make_runtime(int ranks, int rank, int owner_shift) {
  pops::Index<Dim> coarse_lower{};
  pops::Index<Dim> coarse_upper{};
  coarse_upper[0] = 4 * ranks - 1;
  for (int axis = 1; axis < Dim; ++axis)
    coarse_upper[axis] = 3;
  const pops::Box<Dim> coarse_domain{coarse_lower, coarse_upper};
  const auto ratio = ratio_two<Dim>();
  const pops::Box<Dim> fine_domain = hierarchy::refine_box(coarse_domain, ratio);

  std::vector<pops::Box<Dim>> coarse_patches;
  std::vector<pops::Box<Dim>> fine_patches;
  std::vector<pops::Index<Dim>> owners;
  for (int patch = 0; patch < ranks; ++patch) {
    pops::Index<Dim> lower = coarse_lower;
    pops::Index<Dim> upper = coarse_upper;
    lower[0] = 4 * patch;
    upper[0] = lower[0] + 3;
    const pops::Box<Dim> box{lower, upper};
    coarse_patches.push_back(box);
    fine_patches.push_back(hierarchy::refine_box(box, ratio));
    owners.push_back(rank_coordinate<Dim>((patch + owner_shift) % ranks));
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
      "test.mpi-checkpoint-migration.spatial");
}

template <int Dim>
void prove_migration_requires_rematerialization(int ranks, int rank) {
  auto source = make_runtime<Dim>(ranks, rank, 0);
  auto target = make_runtime<Dim>(ranks, rank, 1);
  ASSERT_EQ(source.hierarchy().state(0).local_size(), 1);
  ASSERT_EQ(target.hierarchy().state(0).local_size(), 1);
  ASSERT_NE(source.spatial_contract(), target.spatial_contract());

  pops::amr::reflux::TransactionalFaceFluxLedger<Dim, program::AmrProgramFacePayload> empty_ledger(
      {16, 16, 2});
  program::CellTemporalPartitionAcceptedState temporal;
  auto accepted = program::accepted_amr_program_state<Dim>(
      std::string(source.spatial_contract()), source.topology_epoch(),
      source.materialization_generation(),
      {{0, 6, pops::amr::Rational(0, 1), 2.0}, {1, 6, pops::amr::Rational(0, 1), 2.0}}, temporal,
      empty_ledger);
  const auto bytes = program::serialize_amr_program_accepted_state(accepted);
  const auto decoded = program::deserialize_amr_program_accepted_state<Dim>(bytes);
  EXPECT_NO_THROW(program::require_collective_amr_program_checkpoint_consensus(decoded));
  EXPECT_NO_THROW(program::require_live_amr_program_checkpoint(decoded, source));
  EXPECT_THROW(program::require_live_amr_program_checkpoint(decoded, target),
               std::invalid_argument);

  auto rematerialized = decoded;
  rematerialized.spatial_contract = std::string(target.spatial_contract());
  rematerialized.topology_epoch = target.topology_epoch();
  rematerialized.materialization_generation = target.materialization_generation();
  EXPECT_NO_THROW(program::require_collective_amr_program_checkpoint_consensus(rematerialized));
  EXPECT_NO_THROW(program::require_live_amr_program_checkpoint(rematerialized, target));
  EXPECT_EQ(program::serialize_amr_program_accepted_state(
                program::deserialize_amr_program_accepted_state<Dim>(
                    program::serialize_amr_program_accepted_state(rematerialized))),
            program::serialize_amr_program_accepted_state(rematerialized));
}

int run_exact_checkpoint_migration(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      prove_migration_requires_rematerialization<pops::kNativeDimension>(pops::n_ranks(),
                                                                         pops::my_rank());
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact checkpoint migration proof failed: %s\n", pops::my_rank(),
                   error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf("OK test_mpi_amr_rebalance_migration np=%d dim=%d exact-rematerialization\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_amr_rebalance_migration, OwnershipChangeRequiresExactRematerialization) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_exact_checkpoint_migration, "test_mpi_amr_rebalance_migration"),
      0);
}
