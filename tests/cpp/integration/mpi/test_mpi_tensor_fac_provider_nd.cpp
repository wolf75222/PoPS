#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

template <int Dim>
pops::Extent<Dim> extents(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::RealVector<Dim> coordinates(pops::Real value) {
  pops::RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::PhysicalBoundaryConditions<Dim> homogeneous_dirichlet(const pops::Geometry<Dim>& geometry) {
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    spacing[axis] = geometry.spacing(axis);
    for (const pops::BoundarySide side : {pops::BoundarySide::lower, pops::BoundarySide::upper})
      faces[static_cast<std::size_t>(pops::Face<Dim>{axis, side}.ordinal())] =
          pops::PhysicalBoundaryFace{pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)};
  }
  return {pops::BoundaryTopology<Dim>::physical(), faces, spacing};
}

template <int Dim>
pops::runtime::program::HierarchyTensorSolverBuildRequest<Dim> partitioned_request() {
  using namespace pops;
  using namespace pops::runtime::program;

  Extent<Dim> rank_extents = extents<Dim>(1);
  rank_extents[0] = 2;
  const mesh::RankSpace<Dim> rank_space{Index<Dim>{}, rank_extents};
  Index<Dim> local_rank{};
  local_rank[0] = my_rank();
  std::array<int, Dim> ratio_values{};
  ratio_values.fill(2);
  const amr::RefinementRatio<Dim> ratio{ratio_values};

  Index<Dim> coarse_upper{};
  for (int axis = 0; axis < Dim; ++axis)
    coarse_upper[axis] = 3;
  const Box<Dim> coarse_domain{Index<Dim>{}, coarse_upper};
  const Geometry<Dim> coarse_geometry = Geometry<Dim>::from_bounds(
      coarse_domain, coordinates<Dim>(Real(0)), coordinates<Dim>(Real(1)));
  const mesh::BoxArray<Dim> coarse_layout(std::vector<Box<Dim>>{coarse_domain});
  const mesh::Distribution<Dim> coarse_distribution =
      mesh::Distribution<Dim>::replicated(coarse_layout, rank_space);

  const Geometry<Dim> middle_geometry = coarse_geometry.refine(extents<Dim>(2));
  Box<Dim> middle_left{Index<Dim>{}, middle_geometry.domain().hi};
  Box<Dim> middle_right = middle_left;
  middle_left.hi[0] = 3;
  middle_right.lo[0] = 4;
  const mesh::BoxArray<Dim> middle_layout(std::vector<Box<Dim>>{middle_left, middle_right});
  Index<Dim> rank_zero{};
  Index<Dim> rank_one{};
  rank_one[0] = 1;
  const mesh::Distribution<Dim> middle_distribution = mesh::Distribution<Dim>::partitioned(
      middle_layout, rank_space, std::vector<Index<Dim>>{rank_zero, rank_one});

  const Geometry<Dim> fine_geometry = middle_geometry.refine(extents<Dim>(2));
  Index<Dim> fine_lower{};
  Index<Dim> fine_upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine_lower[axis] = 4;
    fine_upper[axis] = 11;
  }
  Box<Dim> fine_left{fine_lower, fine_upper};
  Box<Dim> fine_right = fine_left;
  fine_left.hi[0] = 7;
  fine_right.lo[0] = 8;
  const mesh::BoxArray<Dim> fine_layout(std::vector<Box<Dim>>{fine_left, fine_right});
  const mesh::Distribution<Dim> fine_distribution = mesh::Distribution<Dim>::partitioned(
      fine_layout, rank_space, std::vector<Index<Dim>>{rank_one, rank_zero});

  HierarchyTensorSolverBuildRequest<Dim> result;
  result.block = 9;
  result.components = 1;
  result.levels.push_back(
      HierarchyTensorLevelBuildRequest<Dim>{coarse_geometry, homogeneous_dirichlet(coarse_geometry),
                                            coarse_layout, coarse_distribution, local_rank});
  result.levels.push_back(
      HierarchyTensorLevelBuildRequest<Dim>{middle_geometry, homogeneous_dirichlet(middle_geometry),
                                            middle_layout, middle_distribution, local_rank});
  result.levels.push_back(
      HierarchyTensorLevelBuildRequest<Dim>{fine_geometry, homogeneous_dirichlet(fine_geometry),
                                            fine_layout, fine_distribution, local_rank});
  result.ratios = {ratio, ratio};
  result.plan_identity = "pops.test.nd-tensor-fac.partitioned-mpi";
  result.operator_contract_identity =
      std::string(tensor_elliptic_detail::kScalarTensorEllipticContract);
  result.assembly_field_slots = tensor_elliptic_detail::assembly_slots<Dim>();
  result.solution_field_slot = "pops.tensor-elliptic.solution";
  result.options = tensor_elliptic_detail::default_options();
  return result;
}

template <int Dim>
void expect_partitioned_tensor_fac() {
  using namespace pops;
  using namespace pops::runtime::program;

  auto request = partitioned_request<Dim>();
  const ExecutionLane lane = ExecutionLane::world("pops.test.nd-tensor-fac.partitioned-mpi");
  const auto registry = make_default_hierarchy_tensor_solver_provider_registry<Dim>(lane);
  auto prepared = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, std::move(request), lane);
  auto* exact = dynamic_cast<AmrTensorElliptic<Dim>*>(prepared.get());
  ASSERT_NE(exact, nullptr);
  EXPECT_TRUE(exact->borrows_execution_lane());
  EXPECT_EQ(all_reduce_min(exact->has_remote_same_level_halo() ? 1L : 0L, lane), 1L);
  EXPECT_EQ(all_reduce_min(exact->has_remote_parent_gather() ? 1L : 0L, lane), 1L);
  EXPECT_EQ(all_reduce_min(exact->has_remote_fine_restriction() ? 1L : 0L, lane), 1L);
  EXPECT_EQ(all_reduce_min(exact->uses_replicated_parent_restriction() ? 1L : 0L, lane), 1L);

  for (int level = 0; level < prepared->level_count(); ++level) {
    for (int row = 0; row < Dim; ++row)
      for (int column = 0; column < Dim; ++column)
        prepared->assembly_target(tensor_elliptic_detail::coefficient_slot(row, column), level)
            .set_val(row == column ? Real(1) : Real(0));
    prepared->assembly_target("pops.tensor-elliptic.rhs", level).set_val(Real(0));
    prepared->stage_initial_guess(level, nullptr);
  }

  const SolveReport report =
      solve_prepared_hierarchy_tensor_collectively(
          *prepared, HierarchyTensorSolveControls{Real(1e-10), Real(0), 4}, lane)
          .consume(SolveConsumption::kAccept);
  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(all_reduce_min(static_cast<long>(report.iters), lane),
            all_reduce_max(static_cast<long>(report.iters), lane));
}

int run_partitioned_tensor_fac(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    EXPECT_EQ(pops::n_ranks(), 2);
    if (pops::n_ranks() == 2)
      expect_partitioned_tensor_fac<pops::kNativeDimension>();
    result = ::testing::Test::HasFailure() ? 1 : 0;
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_tensor_fac_provider_nd, SolvesPartitionedTensorHierarchyAtNativeDimension) {
  EXPECT_EQ(pops::test::RunTestBody(&run_partitioned_tensor_fac, "test_mpi_tensor_fac_provider_nd"),
            0);
}
