#include <gtest/gtest.h>

#include <pops/amr/refinement_ratio.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/parallel/comm.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = pops::kNativeDimension;

template <class Ranked, class Value>
Ranked filled(Value value) {
  Ranked result{};
  for (int axis = 0; axis < kDim; ++axis)
    result[axis] = value;
  return result;
}

pops::EllipticBuildRequest<kDim> request(int cells, pops::mesh::BoxArray<kDim> layout) {
  const pops::Box<kDim> domain{pops::Index<kDim>{}, filled<pops::Index<kDim>>(cells - 1)};
  const auto geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{}, filled<pops::RealVector<kDim>>(pops::Real(1)));
  pops::Extent<kDim> rank_extent = filled<pops::Extent<kDim>>(std::int64_t{1});
  rank_extent[0] = pops::n_ranks();
  pops::Index<kDim> local_rank{};
  local_rank[0] = pops::my_rank();
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{}, rank_extent};
  const auto distribution = pops::mesh::Distribution<kDim>::replicated(layout, ranks);
  std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  faces.fill(pops::PhysicalBoundaryFace{pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<kDim> spacing{};
  for (int axis = 0; axis < kDim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {geometry,
          layout,
          distribution,
          local_rank,
          pops::PhysicalBoundaryConditions<kDim>{pops::BoundaryTopology<kDim>::physical(), faces,
                                                 spacing},
          pops::Extent<kDim>{},
          filled<pops::Extent<kDim>>(std::int64_t{1}),
          {layout.size(), layout.size() * (layout.size() - 1) / 2}};
}

pops::EllipticBuildRequest<kDim> complete_request(int cells) {
  const pops::Box<kDim> domain{pops::Index<kDim>{}, filled<pops::Index<kDim>>(cells - 1)};
  return request(cells, pops::mesh::BoxArray<kDim>{std::vector<pops::Box<kDim>>{domain}});
}

void expect_report_is_structured(const pops::SolveReport& report, int maximum_iterations) {
  EXPECT_TRUE(report.valid());
  EXPECT_TRUE(pops::solve_report_is_publishable(report, maximum_iterations));
  EXPECT_FALSE(report.reason.empty());
  EXPECT_GE(report.iters, 0);
  EXPECT_GE(report.evaluations, 0);
  EXPECT_TRUE(std::isfinite(static_cast<double>(report.reference_residual_norm)));
  EXPECT_TRUE(std::isfinite(static_cast<double>(report.residual_norm)));
  EXPECT_TRUE(std::isfinite(static_cast<double>(report.rel_residual)));
}

}  // namespace

TEST(test_structured_solver_diagnostics, geometric_multigrid_publishes_the_unified_solve_report) {
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-8);
  options.absolute_tolerance = pops::Real(1e-11);
  options.maximum_cycles = 100;
  pops::elliptic::mg::GeometricMG<kDim> solver(complete_request(16), options);
  solver.install_nullspace(pops::FieldNullspacePlan<kDim>{},
                           pops::PreparedVectorDistribution<kDim>::replicated());
  solver.rhs().set_val(pops::Real(1));
  solver.phi().set_val(pops::Real(0));

  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  expect_report_is_structured(report, solver.maximum_iterations());
  EXPECT_EQ(report.reason, "geometric_mg_converged");
  EXPECT_EQ(solver.last_solve_report().status, report.status);
  EXPECT_EQ(solver.last_solve_report().residual_norm, report.residual_norm);
}

TEST(test_structured_solver_diagnostics,
     composite_fac_publishes_the_same_report_contract_for_the_exact_ranked_hierarchy) {
  auto coarse = complete_request(8);
  const pops::Box<kDim> fine_patch{filled<pops::Index<kDim>>(4), filled<pops::Index<kDim>>(11)};
  auto fine = request(16, pops::mesh::BoxArray<kDim>{std::vector<pops::Box<kDim>>{fine_patch}});
  pops::elliptic::mg::CompositeFacBuildRequest<kDim> build{
      {std::move(coarse), std::move(fine)},
      {pops::amr::RefinementRatio<kDim>{filled<std::array<int, kDim>>(2)}}};
  pops::elliptic::mg::CompositeFacPoisson<kDim> solver(std::move(build));
  solver.install_nullspace(pops::FieldNullspacePlan<kDim>{},
                           {pops::PreparedVectorDistribution<kDim>::replicated(),
                            pops::PreparedVectorDistribution<kDim>::replicated()});
  for (int level = 0; level < solver.n_levels(); ++level) {
    solver.rhs_level(level).set_val(pops::Real(0));
    solver.phi_level(level).set_val(pops::Real(0));
  }

  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  expect_report_is_structured(report, solver.maximum_iterations());
  EXPECT_EQ(report.reason, "composite_fac_initial_residual");
  EXPECT_EQ(report.iters, 0);
  EXPECT_EQ(solver.last_solve_report().status, report.status);
}
