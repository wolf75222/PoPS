#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>
#include <pops/numerics/elliptic/polar/polar_poisson_solver.hpp>
#include <pops/numerics/elliptic/polar/polar_tensor_operator.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = 2;
constexpr int kCells = 64;
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

std::size_t offset(const pops::Box<kDim>& storage, int i, int j) {
  return static_cast<std::size_t>(i - storage.lo[0]) +
         static_cast<std::size_t>(j - storage.lo[1]) * static_cast<std::size_t>(storage.length(0));
}

pops::EllipticBuildRequest<kDim> request() {
  if (pops::n_ranks() != 1)
    throw std::logic_error("the unit operator identity test requires one rank");
  const pops::Box<kDim> domain{pops::Index<kDim>{0, 0}, pops::Index<kDim>{kCells - 1, kCells - 1}};
  const pops::Geometry<kDim> geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{0, 0}, pops::RealVector<kDim>{1, 1});
  const pops::mesh::BoxArray<kDim> layout(std::vector<pops::Box<kDim>>{domain});
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{0, 0}, pops::Extent<kDim>{1, 1}};
  const pops::mesh::Distribution<kDim> distribution = pops::mesh::Distribution<kDim>::partitioned(
      layout, ranks, std::vector<pops::Index<kDim>>{pops::Index<kDim>{0, 0}});
  const std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  return {geometry,
          layout,
          distribution,
          pops::Index<kDim>{0, 0},
          pops::PhysicalBoundaryConditions<kDim>{
              pops::BoundaryTopology<kDim>::axis_periodic({true, true}), faces,
              pops::RealVector<kDim>{geometry.spacing(0), geometry.spacing(1)}},
          pops::Extent<kDim>{0, 0},
          pops::Extent<kDim>{1, 1},
          {1, 0}};
}

void fill_mode(pops::MultiFab<kDim>& field, const pops::Geometry<kDim>& geometry) {
  auto& fab = field.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const pops::Box<kDim>& valid = fab.box();
  const pops::Box<kDim>& storage = fab.grown_box();
  pops::Real sum = 0;
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
      const pops::Real value = std::sin(pops::Real(2) * kPi * geometry.cell_coordinate(0, i)) *
                               std::sin(pops::Real(2) * kPi * geometry.cell_coordinate(1, j));
      host(offset(storage, i, j)) = value;
      sum += value;
    }
  }
  const pops::Real mean = sum / static_cast<pops::Real>(valid.numPts());
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
      host(offset(storage, i, j)) -= mean;
  fab.copy_from_host(host);
}

}  // namespace

TEST(test_elliptic_operator, fft_and_multigrid_invert_the_same_exact_ranked_operator) {
  auto fft_request = request();
  auto mg_request = fft_request;
  const auto layout = fft_request.boxes;
  const auto distribution = fft_request.distribution;
  const auto local_rank = fft_request.local_rank;
  const auto geometry = fft_request.geometry;
  const pops::Real measure = geometry.spacing(0) * geometry.spacing(1);

  pops::PoissonFFTSolver<kDim> fft = pops::make_elliptic_solver<pops::PoissonFFTSolver<kDim>>(
      std::move(fft_request), pops::PoissonFFTFactory<kDim>{});
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-11);
  options.absolute_tolerance = pops::Real(1e-13);
  options.maximum_cycles = 200;
  options.bottom_sweeps = 80;
  pops::elliptic::mg::GeometricMG<kDim> mg(std::move(mg_request), options);
  fft.install_nullspace(
      pops::constant_mean_zero_nullspace<kDim>("fft-constant", "unit-test", measure),
      pops::PreparedVectorDistribution<kDim>::distributed());
  mg.install_nullspace(
      pops::constant_mean_zero_nullspace<kDim>("mg-constant", "unit-test", measure),
      pops::PreparedVectorDistribution<kDim>::distributed());
  fill_mode(fft.rhs(), geometry);
  fill_mode(mg.rhs(), geometry);
  fft.phi().set_val(pops::Real(0));
  mg.phi().set_val(pops::Real(0));

  const pops::SolveReport fft_report = fft.solve();
  const pops::SolveReport mg_report = mg.solve();
  ASSERT_TRUE(fft_report.solved()) << fft_report.reason;
  ASSERT_TRUE(mg_report.solved()) << mg_report.reason;

  pops::MultiFab<kDim> fft_residual(layout, distribution, local_rank, 1, pops::Extent<kDim>{0, 0});
  pops::MultiFab<kDim> mg_residual(layout, distribution, local_rank, 1, pops::Extent<kDim>{0, 0});
  pops::elliptic::mg::poisson_residual_valid(fft.phi(), fft.rhs(), geometry, fft_residual);
  pops::elliptic::mg::poisson_residual_valid(mg.phi(), mg.rhs(), geometry, mg_residual);
  EXPECT_LT(pops::reduce_norm_inf(fft_residual), pops::Real(1e-9));
  EXPECT_LT(pops::reduce_norm_inf(mg_residual), pops::Real(1e-9));

  pops::MultiFab<kDim> difference(layout, distribution, local_rank, 1, pops::Extent<kDim>{0, 0});
  pops::lincomb(difference, pops::Real(1), fft.phi(), pops::Real(-1), mg.phi());
  EXPECT_LT(pops::reduce_norm_inf(difference), pops::Real(1e-9));
}

TEST(test_elliptic_operator, polar_providers_expose_only_their_exact_physical_rank) {
  static_assert(!pops::PolarGeometryCapabilities<1>::available);
  static_assert(pops::PolarGeometryCapabilities<2>::available);
  static_assert(!pops::PolarGeometryCapabilities<3>::available);
  static_assert(!pops::PolarPoissonProvider<1>::available);
  static_assert(pops::PolarPoissonProvider<2>::available);
  static_assert(!pops::PolarPoissonProvider<3>::available);
  static_assert(!pops::PolarTensorProvider<1>::available);
  static_assert(pops::PolarTensorProvider<2>::available);
  static_assert(!pops::PolarTensorProvider<3>::available);

  EXPECT_FALSE(pops::PolarPoissonProvider<1>::rejection_reason().empty());
  EXPECT_TRUE(pops::PolarPoissonProvider<2>::rejection_reason().empty());
  EXPECT_FALSE(pops::PolarPoissonProvider<3>::rejection_reason().empty());
  EXPECT_FALSE(pops::PolarTensorProvider<1>::rejection_reason().empty());
  EXPECT_TRUE(pops::PolarTensorProvider<2>::rejection_reason().empty());
  EXPECT_FALSE(pops::PolarTensorProvider<3>::rejection_reason().empty());
}
