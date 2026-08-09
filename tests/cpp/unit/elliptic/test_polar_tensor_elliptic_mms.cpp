#include <gtest/gtest.h>

#include <pops/numerics/elliptic/polar/polar_tensor_operator.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = 2;
constexpr pops::Real kRmin = pops::Real(0.4);
constexpr pops::Real kRmax = pops::Real(1.0);
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

std::size_t offset(const pops::Box<kDim>& storage, int i, int j) {
  return static_cast<std::size_t>(i - storage.lo[0]) +
         static_cast<std::size_t>(j - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

pops::PolarEllipticBuildRequest<kDim> request(int cells = 24) {
  const pops::Box<kDim> domain{pops::Index<kDim>{0, 0},
                              pops::Index<kDim>{cells - 1, cells - 1}};
  const auto geometry = pops::PolarGeometry<kDim>::annulus(domain, kRmin, kRmax);
  const pops::mesh::BoxArray<kDim> boxes(std::vector<pops::Box<kDim>>{domain});
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{0, 0},
                                         pops::Extent<kDim>{1, 1}};
  const auto distribution = pops::mesh::Distribution<kDim>::partitioned(
      boxes, ranks, std::vector<pops::Index<kDim>>{pops::Index<kDim>{0, 0}});
  std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  faces[0] = {pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)};
  faces[1] = {pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)};
  return {geometry,
          boxes,
          distribution,
          pops::Index<kDim>{0, 0},
          pops::PhysicalBoundaryConditions<kDim>{
              pops::BoundaryTopology<kDim>::axis_periodic({false, true}), faces,
              pops::RealVector<kDim>{geometry.dr(), geometry.dtheta()}},
          {1, 0}};
}

pops::Real exact(pops::Real radius) {
  return std::sin(kPi * (radius - kRmin) / (kRmax - kRmin));
}

pops::Real forcing(pops::Real radius) {
  const pops::Real wave = kPi / (kRmax - kRmin);
  const pops::Real phase = wave * (radius - kRmin);
  return -wave * wave * std::sin(phase) + wave * std::cos(phase) / radius;
}

void fill_rhs(pops::PolarTensorKrylovSolver<kDim>& solver) {
  auto& fab = solver.rhs().fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const auto& valid = fab.box();
  const auto& storage = fab.grown_box();
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
      host(offset(storage, i, j)) = forcing(solver.geom().r_cell(i));
  fab.copy_from_host(host);
}

pops::Real error_l2(const pops::PolarTensorKrylovSolver<kDim>& solver) {
  const auto& fab = solver.phi().fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const auto& valid = fab.box();
  const auto& storage = fab.grown_box();
  pops::Real sum = 0;
  pops::Real count = 0;
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
      const pops::Real difference =
          host(offset(storage, i, j)) - exact(solver.geom().r_cell(i));
      sum += difference * difference;
      count += pops::Real(1);
    }
  return std::sqrt(sum / count);
}

}  // namespace

TEST(test_polar_tensor_elliptic_mms, capabilities_are_exactly_rank_two) {
  static_assert(!pops::PolarTensorProvider<1>::available);
  static_assert(pops::PolarTensorProvider<2>::available);
  static_assert(!pops::PolarTensorProvider<3>::available);
  EXPECT_EQ(pops::PolarTensorProvider<1>::rejection_reason(),
            "polar tensor elliptic operator has exactly the axes (r, theta)");
  EXPECT_EQ(pops::PolarTensorProvider<3>::rejection_reason(),
            "polar tensor elliptic operator has exactly the axes (r, theta)");
}

TEST(test_polar_tensor_elliptic_mms, radial_line_solver_authenticates_the_final_residual) {
  pops::PolarTensorOptions options;
  options.relative_tolerance = pops::Real(1e-10);
  options.absolute_tolerance = pops::Real(1e-12);
  options.maximum_iterations = 500;
  options.preconditioner = pops::PolarPreconditioner::radial_line;
  auto solver = pops::PolarTensorProvider<kDim>::build(request(), options);
  fill_rhs(solver);
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_LT(report.residual_norm, pops::Real(1e-8));
  EXPECT_LT(error_l2(solver), pops::Real(2e-3));
}

TEST(test_polar_tensor_elliptic_mms, explicit_tensor_coefficients_use_the_same_exact_layout) {
  auto build = request(16);
  const auto boxes = build.boxes;
  const auto distribution = build.distribution;
  const auto rank = build.local_rank;
  pops::MultiFab<kDim> rr(boxes, distribution, rank, 1, pops::Extent<kDim>{1, 1});
  pops::MultiFab<kDim> tt(boxes, distribution, rank, 1, pops::Extent<kDim>{1, 1});
  pops::MultiFab<kDim> rt(boxes, distribution, rank, 1, pops::Extent<kDim>{1, 1});
  pops::MultiFab<kDim> tr(boxes, distribution, rank, 1, pops::Extent<kDim>{1, 1});
  rr.set_val(pops::Real(1));
  tt.set_val(pops::Real(1));
  rt.set_val(pops::Real(0));
  tr.set_val(pops::Real(0));
  pops::PolarTensorOptions options;
  options.maximum_iterations = 500;
  auto solver = pops::PolarTensorProvider<kDim>::build(std::move(build), options);
  solver.set_coefficients(rr, tt, &rt, &tr);
  fill_rhs(solver);
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_LT(report.residual_norm, pops::Real(1e-8));
}
