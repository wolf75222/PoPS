#include <gtest/gtest.h>

#include <pops/numerics/elliptic/polar/polar_tensor_operator.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = 2;
constexpr int kRadial = 20;
constexpr int kTheta = 24;
constexpr pops::Real kRmin = pops::Real(0.5);
constexpr pops::Real kRmax = pops::Real(1.0);
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

std::size_t offset(const pops::Box<kDim>& storage, int i, int j) {
  return static_cast<std::size_t>(i - storage.lo[0]) +
         static_cast<std::size_t>(j - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

pops::PolarEllipticBuildRequest<kDim> theta_slab_request() {
  const pops::Box<kDim> domain{pops::Index<kDim>{0, 0},
                              pops::Index<kDim>{kRadial - 1, kTheta - 1}};
  const auto geometry = pops::PolarGeometry<kDim>::annulus(domain, kRmin, kRmax);
  const auto boxes = pops::mesh::BoxArray<kDim>::from_domain(
      domain, pops::Extent<kDim>{kRadial, kTheta / 4});
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{0, 0},
                                         pops::Extent<kDim>{1, 1}};
  std::vector<pops::Index<kDim>> owners(boxes.size(), pops::Index<kDim>{0, 0});
  const auto distribution =
      pops::mesh::Distribution<kDim>::partitioned(boxes, ranks, std::move(owners));
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
          {boxes.size(), boxes.size() * (boxes.size() - 1) / 2}};
}

pops::Real forcing(pops::Real radius, pops::Real theta) {
  const pops::Real radial_wave = kPi / (kRmax - kRmin);
  const pops::Real radial_phase = radial_wave * (radius - kRmin);
  const pops::Real radial = std::sin(radial_phase);
  const pops::Real angular = std::cos(pops::Real(2) * theta);
  return (-radial_wave * radial_wave * radial +
          radial_wave * std::cos(radial_phase) / radius -
          pops::Real(4) * radial / (radius * radius)) *
         angular;
}

void fill_rhs(pops::PolarTensorKrylovSolver<kDim>& solver) {
  for (std::size_t local = 0; local < solver.rhs().local_size(); ++local) {
    auto& fab = solver.rhs().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const auto& valid = fab.box();
    const auto& storage = fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        host(offset(storage, i, j)) =
            forcing(solver.geom().r_cell(i), solver.geom().theta_cell(j));
    fab.copy_from_host(host);
  }
}

}  // namespace

TEST(test_polar_schur_multibox, theta_slab_halos_and_radial_lines_form_one_operator) {
  pops::PolarTensorOptions options;
  options.relative_tolerance = pops::Real(1e-9);
  options.absolute_tolerance = pops::Real(1e-11);
  options.maximum_iterations = 800;
  options.preconditioner = pops::PolarPreconditioner::radial_line;
  auto solver = pops::PolarTensorProvider<kDim>::build(theta_slab_request(), options);
  ASSERT_EQ(solver.rhs().local_size(), 4U);
  fill_rhs(solver);
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_LT(report.residual_norm, pops::Real(1e-7));
}

TEST(test_polar_schur_multibox, radial_line_rejects_a_layout_that_cuts_radius) {
  auto invalid = theta_slab_request();
  const auto domain = invalid.geometry.domain();
  invalid.boxes = pops::mesh::BoxArray<kDim>::from_domain(
      domain, pops::Extent<kDim>{kRadial / 2, kTheta / 2});
  std::vector<pops::Index<kDim>> owners(invalid.boxes.size(), pops::Index<kDim>{0, 0});
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{0, 0},
                                         pops::Extent<kDim>{1, 1}};
  invalid.distribution = pops::mesh::Distribution<kDim>::partitioned(
      invalid.boxes, ranks, std::move(owners));
  invalid.layout_budget = {invalid.boxes.size(),
                           invalid.boxes.size() * (invalid.boxes.size() - 1) / 2};
  auto solver = pops::PolarTensorProvider<kDim>::build(std::move(invalid));
  const pops::SolveReport report = solver.solve();
  EXPECT_FALSE(report.solved());
  EXPECT_EQ(report.status, pops::SolveStatus::kInvalidEvaluation);
}
