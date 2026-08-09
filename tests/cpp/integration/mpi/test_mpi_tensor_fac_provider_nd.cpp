#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/execution/for_each.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

pops::PhysicalBoundaryConditions<2> homogeneous_dirichlet(const pops::Geometry<2>& geometry) {
  std::array<pops::PhysicalBoundaryFace, 4> faces{};
  faces.fill(pops::PhysicalBoundaryFace{pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  return {pops::BoundaryTopology<2>::physical(), faces,
          pops::RealVector<2>{geometry.spacing(0), geometry.spacing(1)}};
}

pops::runtime::program::HierarchyTensorSolverBuildRequest<2> partitioned_request() {
  using namespace pops;
  using namespace pops::runtime::program;

  const mesh::RankSpace<2> rank_space{Index<2>{0, 0}, Extent<2>{2, 1}};
  const Index<2> local_rank{my_rank(), 0};
  const amr::RefinementRatio<2> ratio{std::array<int, 2>{2, 2}};

  const Box<2> coarse_domain{Index<2>{0, 0}, Index<2>{3, 3}};
  const Geometry<2> coarse_geometry = Geometry<2>::from_bounds(
      coarse_domain, RealVector<2>{Real(0), Real(0)}, RealVector<2>{Real(1), Real(1)});
  const mesh::BoxArray<2> coarse_layout(std::vector<Box<2>>{coarse_domain});
  const mesh::Distribution<2> coarse_distribution =
      mesh::Distribution<2>::replicated(coarse_layout, rank_space);

  const Geometry<2> middle_geometry = coarse_geometry.refine(Extent<2>{2, 2});
  const mesh::BoxArray<2> middle_layout(std::vector<Box<2>>{
      Box<2>{Index<2>{0, 0}, Index<2>{3, 7}},
      Box<2>{Index<2>{4, 0}, Index<2>{7, 7}},
  });
  const mesh::Distribution<2> middle_distribution = mesh::Distribution<2>::partitioned(
      middle_layout, rank_space, std::vector<Index<2>>{Index<2>{0, 0}, Index<2>{1, 0}});

  const Geometry<2> fine_geometry = middle_geometry.refine(Extent<2>{2, 2});
  const mesh::BoxArray<2> fine_layout(std::vector<Box<2>>{
      Box<2>{Index<2>{4, 4}, Index<2>{7, 11}},
      Box<2>{Index<2>{8, 4}, Index<2>{11, 11}},
  });
  // Reverse ownership relative to the middle level so both parent gather and fine restriction
  // necessarily cross the duplicated FAC communicator.
  const mesh::Distribution<2> fine_distribution = mesh::Distribution<2>::partitioned(
      fine_layout, rank_space, std::vector<Index<2>>{Index<2>{1, 0}, Index<2>{0, 0}});

  HierarchyTensorSolverBuildRequest<2> result;
  result.block = 9;
  result.components = 1;
  result.levels.push_back(
      HierarchyTensorLevelBuildRequest<2>{coarse_geometry, homogeneous_dirichlet(coarse_geometry),
                                          coarse_layout, coarse_distribution, local_rank});
  result.levels.push_back(
      HierarchyTensorLevelBuildRequest<2>{middle_geometry, homogeneous_dirichlet(middle_geometry),
                                          middle_layout, middle_distribution, local_rank});
  result.levels.push_back(
      HierarchyTensorLevelBuildRequest<2>{fine_geometry, homogeneous_dirichlet(fine_geometry),
                                          fine_layout, fine_distribution, local_rank});
  result.ratios = {ratio, ratio};
  result.plan_identity = "pops.test.rank2-tensor-fac.partitioned-mpi";
  result.operator_contract_identity =
      std::string(tensor_elliptic_detail::kScalarTensorEllipticRank2Contract);
  result.assembly_field_slots = tensor_elliptic_detail::assembly_slots<2>();
  result.solution_field_slot = "pops.tensor-elliptic.solution";
  result.options = tensor_elliptic_detail::default_options();
  result.options.values["fac.fine_sweeps"] = std::int64_t{36};
  result.options.values["fac.coarse_cycles"] = std::int64_t{128};
  result.options.values["fac.coarse_rel_tol"] = 1.0e-10;
  return result;
}

struct FillManufacturedRhs {
  pops::Geometry<2> geometry;
  pops::FieldView<pops::Real, 2> rhs{};

  POPS_HD void operator()(const pops::Index<2>& cell) const {
    const pops::Real x = geometry.cell_coordinate(0, cell[0]);
    const pops::Real y = geometry.cell_coordinate(1, cell[1]);
    constexpr pops::Real a_xx = pops::Real(2);
    constexpr pops::Real a_xy = pops::Real(0.3);
    constexpr pops::Real a_yx = pops::Real(0.3);
    constexpr pops::Real a_yy = pops::Real(1.5);
    rhs(cell, 0) =
        pops::Real(2) * a_xx * y * (pops::Real(1) - y) +
        pops::Real(2) * a_yy * x * (pops::Real(1) - x) -
        (a_xy + a_yx) * (pops::Real(1) - pops::Real(2) * x) * (pops::Real(1) - pops::Real(2) * y);
  }
};

struct ManufacturedError {
  pops::Geometry<2> geometry;
  pops::FieldView<const pops::Real, 2> solution{};
  pops::Box<2> region{};

  POPS_HD void operator()(std::int64_t linear, pops::Real& error) const {
    const int x_index = static_cast<int>(linear % region.length(0)) + region.lo[0];
    const int y_index = static_cast<int>(linear / region.length(0)) + region.lo[1];
    const pops::Index<2> cell{x_index, y_index};
    const pops::Real x = geometry.cell_coordinate(0, x_index);
    const pops::Real y = geometry.cell_coordinate(1, y_index);
    const pops::Real exact = x * (pops::Real(1) - x) * y * (pops::Real(1) - y);
    error = std::max(error, Kokkos::abs(solution(cell, 0) - exact));
  }
};

pops::Real finest_error(const pops::Geometry<2>& geometry,
                        pops::runtime::program::PreparedHierarchyTensorSolver<2>& prepared) {
  const auto& solution = prepared.solution(2);
  pops::Real local_error = pops::Real(0);
  for (std::size_t local = 0; local < solution.local_size(); ++local) {
    const pops::Box<2>& region = solution.box(local);
    pops::Real patch_error = pops::Real(0);
    Kokkos::parallel_reduce(
        "pops_rank2_tensor_fac_mpi_mms_error",
        Kokkos::RangePolicy<std::int64_t>(0, region.numPts()),
        ManufacturedError{geometry, std::as_const(solution.fab(local)).view(), region},
        Kokkos::Max<pops::Real>(patch_error));
    local_error = std::max(local_error, patch_error);
  }
  Kokkos::fence();
  return static_cast<pops::Real>(pops::all_reduce_max(static_cast<double>(local_error)));
}

void expect_partitioned_tensor_fac() {
  using namespace pops;
  using namespace pops::runtime::program;

  auto request = partitioned_request();
  const std::array<Geometry<2>, 3> geometries{
      request.levels[0].geometry, request.levels[1].geometry, request.levels[2].geometry};
  const auto registry = make_default_hierarchy_tensor_solver_provider_registry<2>();
  auto prepared = prepare_hierarchy_tensor_solver_collectively(
      *registry, tensor_elliptic_detail::kCompositeTensorProvider, std::move(request));
  auto* exact = dynamic_cast<AmrTensorElliptic<2>*>(prepared.get());
  ASSERT_NE(exact, nullptr);
  EXPECT_TRUE(exact->owns_execution_lane());
  EXPECT_EQ(all_reduce_min(exact->has_remote_same_level_halo() ? 1L : 0L), 1L);
  EXPECT_EQ(all_reduce_min(exact->has_remote_parent_gather() ? 1L : 0L), 1L);
  EXPECT_EQ(all_reduce_min(exact->has_remote_fine_restriction() ? 1L : 0L), 1L);
  EXPECT_EQ(all_reduce_min(exact->uses_replicated_parent_restriction() ? 1L : 0L), 1L);

  for (int level = 0; level < prepared->level_count(); ++level) {
    prepared->assembly_target("pops.tensor-elliptic.coefficient.0.0", level).set_val(Real(2));
    prepared->assembly_target("pops.tensor-elliptic.coefficient.0.1", level).set_val(Real(0.3));
    prepared->assembly_target("pops.tensor-elliptic.coefficient.1.0", level).set_val(Real(0.3));
    prepared->assembly_target("pops.tensor-elliptic.coefficient.1.1", level).set_val(Real(1.5));
    auto& rhs = prepared->assembly_target("pops.tensor-elliptic.rhs", level);
    for (std::size_t local = 0; local < rhs.local_size(); ++local)
      for_each_cell(rhs.box(local), FillManufacturedRhs{geometries[static_cast<std::size_t>(level)],
                                                        rhs.fab(local).view()});
    prepared->stage_initial_guess(level, nullptr);
  }
  Kokkos::fence();

  SolveOutcome outcome = solve_prepared_hierarchy_tensor_collectively(
      *prepared, HierarchyTensorSolveControls{Real(2e-6), Real(1e-12), 70});
  const SolveReport report = outcome.consume(SolveConsumption::kAccept);
  EXPECT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm
                               << " reference=" << report.reference_residual_norm;
  EXPECT_LT(report.residual_norm, report.reference_residual_norm);
  EXPECT_EQ(all_reduce_min(static_cast<long>(report.iters)),
            all_reduce_max(static_cast<long>(report.iters)));
  EXPECT_LT(finest_error(geometries[2], *prepared), Real(0.08));
}

int run_partitioned_tensor_fac(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int result = 0;
  {
    Kokkos::ScopeGuard kokkos(argc, argv);
    EXPECT_EQ(pops::n_ranks(), 2);
    if (pops::n_ranks() == 2)
      expect_partitioned_tensor_fac();
    result = ::testing::Test::HasFailure() ? 1 : 0;
  }
  pops::comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_tensor_fac_provider_nd, SolvesPartitionedCrossTensorHierarchy) {
  EXPECT_EQ(pops::test::RunTestBody(&run_partitioned_tensor_fac, "test_mpi_tensor_fac_provider_nd"),
            0);
}
