#include <gtest/gtest.h>

#include <pops/numerics/elliptic/polar/polar_tensor_operator.hpp>
#include <pops/parallel/comm.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = 2;
constexpr int kRadial = 20;
constexpr int kTheta = 32;
constexpr pops::Real kRmin = pops::Real(0.5);
constexpr pops::Real kRmax = pops::Real(1.0);
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

std::size_t offset(const pops::Box<kDim>& storage, int i, int j) {
  return static_cast<std::size_t>(i - storage.lo[0]) +
         static_cast<std::size_t>(j - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

pops::PolarEllipticBuildRequest<kDim> request() {
  const int ranks = pops::n_ranks();
  if (kTheta % ranks != 0)
    throw std::invalid_argument("MPI polar test requires theta divisible by communicator size");
  const pops::Box<kDim> domain{pops::Index<kDim>{0, 0},
                              pops::Index<kDim>{kRadial - 1, kTheta - 1}};
  const auto geometry = pops::PolarGeometry<kDim>::annulus(domain, kRmin, kRmax);
  const auto boxes = pops::mesh::BoxArray<kDim>::from_domain(
      domain, pops::Extent<kDim>{kRadial, kTheta / ranks});
  const pops::mesh::RankSpace<kDim> rank_space{pops::Index<kDim>{0, 0},
                                              pops::Extent<kDim>{1, ranks}};
  std::vector<pops::Index<kDim>> owners;
  owners.reserve(boxes.size());
  for (int rank = 0; rank < ranks; ++rank)
    owners.push_back(pops::Index<kDim>{0, rank});
  const auto distribution = pops::mesh::Distribution<kDim>::partitioned(
      boxes, rank_space, std::move(owners));
  std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  faces[0] = {pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)};
  faces[1] = {pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)};
  return {geometry,
          boxes,
          distribution,
          pops::Index<kDim>{0, pops::my_rank()},
          pops::PhysicalBoundaryConditions<kDim>{
              pops::BoundaryTopology<kDim>::axis_periodic({false, true}), faces,
              pops::RealVector<kDim>{geometry.dr(), geometry.dtheta()}},
          {boxes.size(), boxes.size() * (boxes.size() - 1) / 2}};
}

pops::Real forcing(pops::Real radius, pops::Real theta) {
  const pops::Real radial_wave = kPi / (kRmax - kRmin);
  const pops::Real phase = radial_wave * (radius - kRmin);
  const pops::Real radial = std::sin(phase);
  return (-radial_wave * radial_wave * radial +
          radial_wave * std::cos(phase) / radius -
          pops::Real(9) * radial / (radius * radius)) *
         std::cos(pops::Real(3) * theta);
}

void fill_local_rhs(pops::PolarTensorKrylovSolver<kDim>& solver) {
  ASSERT_EQ(solver.rhs().local_size(), 1U);
  auto& fab = solver.rhs().fab(0);
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

}  // namespace

TEST(test_mpi_polar_schur, exact_theta_slabs_converge_collectively) {
  pops::PolarTensorOptions options;
  options.relative_tolerance = pops::Real(1e-8);
  options.absolute_tolerance = pops::Real(1e-10);
  options.maximum_iterations = 1000;
  options.preconditioner = pops::PolarPreconditioner::radial_line;
  auto solver = pops::PolarTensorProvider<kDim>::build(request(), options);
  fill_local_rhs(solver);
  const pops::SolveReport report = solver.solve();
  const long any_failed = pops::all_reduce_max(report.solved() ? 0L : 1L);
  ASSERT_EQ(any_failed, 0L) << report.reason;
  EXPECT_LT(report.residual_norm, pops::Real(1e-6));
}
