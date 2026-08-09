#include <gtest/gtest.h>

#include <pops/numerics/elliptic/polar/polar_poisson_solver.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = 2;
constexpr pops::Real kRmin = pops::Real(0.3);
constexpr pops::Real kRmax = pops::Real(1.0);
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

std::size_t offset(const pops::Box<kDim>& storage, int i, int j) {
  return static_cast<std::size_t>(i - storage.lo[0]) +
         static_cast<std::size_t>(j - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

pops::PolarEllipticBuildRequest<kDim> request(int radial_cells, int azimuthal_cells,
                                              pops::Real low_value,
                                              pops::Real high_value) {
  const pops::Box<kDim> domain{pops::Index<kDim>{0, 0},
                              pops::Index<kDim>{radial_cells - 1, azimuthal_cells - 1}};
  const auto geometry = pops::PolarGeometry<kDim>::annulus(domain, kRmin, kRmax);
  const pops::mesh::BoxArray<kDim> boxes(std::vector<pops::Box<kDim>>{domain});
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{0, 0},
                                         pops::Extent<kDim>{1, 1}};
  const auto distribution = pops::mesh::Distribution<kDim>::partitioned(
      boxes, ranks, std::vector<pops::Index<kDim>>{pops::Index<kDim>{0, 0}});
  std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  faces[static_cast<std::size_t>(
      pops::Face<kDim>{0, pops::BoundarySide::lower}.ordinal())] = {
      pops::PhysicalBoundaryKind::dirichlet, low_value};
  faces[static_cast<std::size_t>(
      pops::Face<kDim>{0, pops::BoundarySide::upper}.ordinal())] = {
      pops::PhysicalBoundaryKind::dirichlet, high_value};
  return {geometry,
          boxes,
          distribution,
          pops::Index<kDim>{0, 0},
          pops::PhysicalBoundaryConditions<kDim>{
              pops::BoundaryTopology<kDim>::axis_periodic({false, true}), faces,
              pops::RealVector<kDim>{geometry.dr(), geometry.dtheta()}},
          {1, 0}};
}

pops::Real radial_profile(pops::Real radius) {
  const pops::Real wave = kPi / (kRmax - kRmin);
  return pops::Real(1) + pops::Real(0.5) * (radius - kRmin) +
         std::sin(wave * (radius - kRmin));
}

pops::Real exact_solution(pops::Real radius, pops::Real theta) {
  const pops::Real wave = kPi / (kRmax - kRmin);
  return pops::Real(1) + pops::Real(0.5) * (radius - kRmin) +
         std::sin(wave * (radius - kRmin)) * std::cos(pops::Real(3) * theta);
}

pops::Real exact_rhs(pops::Real radius, pops::Real theta) {
  const pops::Real wave = kPi / (kRmax - kRmin);
  const pops::Real phase = wave * (radius - kRmin);
  const pops::Real mode = std::sin(phase) * std::cos(pops::Real(3) * theta);
  return pops::Real(0.5) / radius +
         (-(wave * wave) * std::sin(phase) + wave * std::cos(phase) / radius) *
             std::cos(pops::Real(3) * theta) -
         pops::Real(9) * mode / (radius * radius);
}

void fill_rhs(pops::PolarPoissonSolver<kDim>& solver) {
  auto& fab = solver.rhs().fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const auto& valid = fab.box();
  const auto& storage = fab.grown_box();
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
      host(offset(storage, i, j)) =
          exact_rhs(solver.geom().r_cell(i), solver.geom().theta_cell(j));
  fab.copy_from_host(host);
}

pops::Real error_l2(const pops::PolarPoissonSolver<kDim>& solver) {
  const auto& fab = solver.phi().fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const auto& valid = fab.box();
  const auto& storage = fab.grown_box();
  pops::Real error = 0;
  pops::Real measure = 0;
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
      const pops::Real radius = solver.geom().r_cell(i);
      const pops::Real weight = radius * solver.geom().dr() * solver.geom().dtheta();
      const pops::Real difference =
          host(offset(storage, i, j)) -
          exact_solution(radius, solver.geom().theta_cell(j));
      error += difference * difference * weight;
      measure += weight;
    }
  }
  return std::sqrt(error / measure);
}

pops::Real solve_error(int radial_cells) {
  auto build = request(radial_cells, 64, radial_profile(kRmin), radial_profile(kRmax));
  auto solver = pops::PolarPoissonProvider<kDim>::build(std::move(build));
  fill_rhs(solver);
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_LT(report.residual_norm, pops::Real(1e-9));
  EXPECT_NEAR(report.residual_norm, solver.residual(), pops::Real(1e-12));
  return error_l2(solver);
}

}  // namespace

TEST(test_polar_poisson_mms, capabilities_are_exactly_rank_two) {
  static_assert(!pops::PolarGeometryCapabilities<1>::available);
  static_assert(pops::PolarGeometryCapabilities<2>::available);
  static_assert(!pops::PolarGeometryCapabilities<3>::available);
  static_assert(!pops::PolarPoissonProvider<1>::available);
  static_assert(pops::PolarPoissonProvider<2>::available);
  static_assert(!pops::PolarPoissonProvider<3>::available);

  EXPECT_EQ(pops::PolarPoissonProvider<1>::rejection_reason(),
            "polar FFT/Thomas Poisson has exactly the axes (r, theta)");
  EXPECT_EQ(pops::PolarPoissonProvider<3>::rejection_reason(),
            "polar FFT/Thomas Poisson has exactly the axes (r, theta)");
}

TEST(test_polar_poisson_mms, direct_residual_is_authenticated_and_radial_error_is_second_order) {
  const pops::Real coarse = solve_error(24);
  const pops::Real medium = solve_error(48);
  const pops::Real fine = solve_error(96);
  EXPECT_GT(std::log2(coarse / medium), pops::Real(1.8));
  EXPECT_GT(std::log2(medium / fine), pops::Real(1.8));
}

TEST(test_polar_poisson_mms, rejects_a_nonperiodic_azimuthal_contract) {
  auto invalid = request(16, 16, radial_profile(kRmin), radial_profile(kRmax));
  std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  faces[0] = {pops::PhysicalBoundaryKind::dirichlet, radial_profile(kRmin)};
  faces[1] = {pops::PhysicalBoundaryKind::dirichlet, radial_profile(kRmax)};
  invalid.boundary = pops::PhysicalBoundaryConditions<kDim>{
      pops::BoundaryTopology<kDim>::physical(), faces,
      pops::RealVector<kDim>{invalid.geometry.dr(), invalid.geometry.dtheta()}};
  EXPECT_THROW((void)pops::PolarPoissonProvider<kDim>::build(std::move(invalid)),
               std::invalid_argument);
}
