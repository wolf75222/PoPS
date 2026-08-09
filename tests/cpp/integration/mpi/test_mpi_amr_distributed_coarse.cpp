/// @file
/// @brief MPI proof for exact-ranked transfers on a partitioned AMR hierarchy.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/time/amr/reflux/amr_flux_helpers.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

namespace hierarchy = pops::amr::hierarchy;
namespace transfer = pops::amr::transfer;
namespace time_amr = pops::numerics::time::amr;

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
pops::amr::RefinementRatio<Dim> refinement_ratio(int value) {
  std::array<int, Dim> values{};
  values.fill(value);
  return pops::amr::RefinementRatio<Dim>(values);
}

template <int Dim>
std::shared_ptr<const pops::PreparedLoadBalanceAuthority<Dim>> load_balance() {
  return std::make_shared<const pops::PreparedLoadBalanceAuthority<Dim>>(
      pops::prepare_load_balance_authority<Dim>(
          "space_filling_curve", "test.mpi-amr-distributed-coarse.sfc",
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
  const auto ratio = refinement_ratio<Dim>(2);
  const pops::Box<Dim> fine_domain = hierarchy::refine_box(coarse_domain, ratio);

  std::vector<pops::Box<Dim>> coarse_patches;
  std::vector<pops::Box<Dim>> fine_patches;
  std::vector<pops::Index<Dim>> owners;
  coarse_patches.reserve(static_cast<std::size_t>(ranks));
  fine_patches.reserve(static_cast<std::size_t>(ranks));
  owners.reserve(static_cast<std::size_t>(ranks));
  for (int owner = 0; owner < ranks; ++owner) {
    pops::Index<Dim> patch_lower = coarse_lower;
    pops::Index<Dim> patch_upper = coarse_upper;
    patch_lower[0] = coarse_lower[0] + 4 * owner;
    patch_upper[0] = patch_lower[0] + 3;
    const pops::Box<Dim> patch{patch_lower, patch_upper};
    coarse_patches.push_back(patch);
    fine_patches.push_back(hierarchy::refine_box(patch, ratio));
    owners.push_back(rank_coordinate<Dim>(owner));
  }

  const pops::mesh::BoxArray<Dim> coarse_layout(std::move(coarse_patches));
  const pops::mesh::BoxArray<Dim> fine_layout(std::move(fine_patches));
  pops::Extent<Dim> rank_extent = filled_extent<Dim>(1);
  rank_extent[0] = ranks;
  const pops::mesh::RankSpace<Dim> rank_space(pops::Index<Dim>{}, rank_extent);
  const auto coarse_distribution =
      pops::mesh::Distribution<Dim>::partitioned(coarse_layout, rank_space, owners);
  const auto fine_distribution =
      pops::mesh::Distribution<Dim>::partitioned(fine_layout, rank_space, owners);
  const pops::Index<Dim> local_rank = rank_coordinate<Dim>(rank);

  hierarchy::LevelLayout<Dim> coarse_level(0, coarse_domain, coarse_layout, coarse_distribution,
                                           pops::amr::RefinementRatio<Dim>{}, kLayoutBudget);
  hierarchy::LevelLayout<Dim> fine_level(1, fine_domain, fine_layout, fine_distribution, ratio,
                                         kLayoutBudget);
  pops::MultiFab<Dim> coarse(coarse_layout, coarse_distribution, local_rank, 1,
                             filled_extent<Dim>(2));
  pops::MultiFab<Dim> fine(fine_layout, fine_distribution, local_rank, 1, filled_extent<Dim>(2));
  std::vector<hierarchy::AmrLevelState<Dim>> levels;
  levels.emplace_back(std::move(coarse_level), std::move(coarse));
  levels.emplace_back(std::move(fine_level), std::move(fine));
  return pops::runtime::amr::AmrRuntime<Dim>(
      hierarchy::AmrHierarchy<Dim>(std::move(levels), kHierarchyBudget), load_balance<Dim>(),
      "test.mpi-amr-distributed-coarse.spatial");
}

template <int Dim>
POPS_HD pops::Real affine_parent(const pops::Index<Dim>& cell,
                                 const transfer::IndexMapping<Dim>& mapping) {
  pops::Real value = pops::Real(2.5);
  for (int axis = 0; axis < Dim; ++axis)
    value += pops::Real(0.125 * (axis + 1)) * pops::Real(cell[axis] - mapping.coarse_origin[axis]);
  return value;
}

template <int Dim>
POPS_HD pops::Real affine_child(const pops::Index<Dim>& cell,
                                const pops::amr::RefinementRatio<Dim>& ratio,
                                const transfer::IndexMapping<Dim>& mapping) {
  pops::Real value = pops::Real(2.5);
  for (int axis = 0; axis < Dim; ++axis) {
    const pops::Real relative = pops::Real(cell[axis] - mapping.fine_origin[axis]);
    const pops::Real parent_coordinate =
        (relative + pops::Real(0.5)) / pops::Real(ratio[axis]) - pops::Real(0.5);
    value += pops::Real(0.125 * (axis + 1)) * parent_coordinate;
  }
  return value;
}

template <int Dim>
struct FillAffineParent {
  pops::FieldView<pops::Real, Dim> values{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    values(cell) = affine_parent(cell, mapping);
  }
};

template <int Dim>
struct ParentError {
  pops::FieldView<const pops::Real, Dim> values{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    const pops::Real difference = values(cell) - affine_parent(cell, mapping);
    return difference < pops::Real(0) ? -difference : difference;
  }
};

template <int Dim>
struct ChildError {
  pops::FieldView<const pops::Real, Dim> values{};
  pops::amr::RefinementRatio<Dim> ratio{};
  transfer::IndexMapping<Dim> mapping{};

  POPS_HD pops::Real operator()(const pops::Index<Dim>& cell) const {
    const pops::Real difference = values(cell) - affine_child(cell, ratio, mapping);
    return difference < pops::Real(0) ? -difference : difference;
  }
};

template <int Dim>
double prove_partitioned_transfers(int ranks, int rank) {
  auto runtime = make_partitioned_runtime<Dim>(ranks, rank);
  auto& coarse = runtime.hierarchy().state(0);
  auto& fine = runtime.hierarchy().state(1);
  if (coarse.local_size() != 1 || fine.local_size() != 1)
    return std::numeric_limits<double>::infinity();

  const auto ratio = runtime.hierarchy().layout(1).ratio_from_parent();
  const transfer::IndexMapping<Dim> mapping{runtime.hierarchy().layout(0).domain().lo,
                                            runtime.hierarchy().layout(1).domain().lo};
  pops::for_each_cell(coarse.fab(0).grown_box(),
                      FillAffineParent<Dim>{coarse.fab(0).view(), mapping});

  const transfer::PreparedTransfer<Dim> prolongation = time_amr::prepare_linear_prolongation(
      runtime, 0, std::as_const(coarse.fab(0)).view(), fine.fab(0).view(), fine.box(0), mapping);
  if (prolongation.kind() != transfer::TransferKind::LinearProlongation ||
      prolongation.refinement_ratio() != ratio)
    return std::numeric_limits<double>::infinity();
  time_amr::execute_prepared_transfer(prolongation);
  pops::Real local_error = pops::for_each_cell_reduce_max(
      fine.box(0), ChildError<Dim>{std::as_const(fine.fab(0)).view(), ratio, mapping});

  pops::MultiFab<Dim> restricted(coarse.layout(), coarse.distribution(), coarse.local_rank(), 1,
                                 coarse.ghosts());
  const transfer::PreparedTransfer<Dim> restriction =
      time_amr::prepare_average_down(runtime, 1, std::as_const(fine.fab(0)).view(),
                                     restricted.fab(0).view(), coarse.box(0), mapping);
  if (restriction.kind() != transfer::TransferKind::ConservativeRestriction ||
      restriction.refinement_ratio() != ratio)
    return std::numeric_limits<double>::infinity();
  time_amr::execute_prepared_transfer(restriction);
  local_error = std::fmax(
      local_error,
      pops::for_each_cell_reduce_max(
          coarse.box(0), ParentError<Dim>{std::as_const(restricted.fab(0)).view(), mapping}));

  pops::Box<Dim> ghost_region = fine.box(0);
  ghost_region.hi[0] = ghost_region.lo[0] - 1;
  ghost_region.lo[0] = ghost_region.hi[0];
  const transfer::PreparedTransfer<Dim> fill_patch = time_amr::prepare_fill_patch(
      runtime, 0, std::as_const(coarse.fab(0)).view(), fine.fab(0).view(), ghost_region, mapping);
  if (fill_patch.kind() != transfer::TransferKind::CoarseFineGhostInterpolation ||
      fill_patch.refinement_ratio() != ratio)
    return std::numeric_limits<double>::infinity();
  time_amr::execute_prepared_transfer(fill_patch);
  local_error = std::fmax(
      local_error,
      pops::for_each_cell_reduce_max(
          ghost_region, ChildError<Dim>{std::as_const(fine.fab(0)).view(), ratio, mapping}));
  return pops::all_reduce_max(static_cast<double>(local_error));
}

int run_partitioned_transfer_proof(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    try {
      const double error =
          prove_partitioned_transfers<pops::kNativeDimension>(pops::n_ranks(), pops::my_rank());
      EXPECT_LT(error, 2e-13);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact transfer proof failed: %s\n", pops::my_rank(),
                   error.what());
      result = 1;
    }
    result = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(result || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && result == 0)
      std::printf(
          "POPS_MPI_PARITY_SIGNATURE_test_mpi_amr_distributed_coarse="
          "exact-ranked-partitioned-transfers\n");
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_amr_distributed_coarse, RunsExactRankedPartitionedTransfers) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_partitioned_transfer_proof, "test_mpi_amr_distributed_coarse"),
      0);
}
