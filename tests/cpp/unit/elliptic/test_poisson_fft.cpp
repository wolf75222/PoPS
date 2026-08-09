#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = 2;
constexpr int kFallbackCells = 48;
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

using Field = pops::MultiFab<kDim>;
using Layout = pops::mesh::BoxArray<kDim>;
using Distribution = pops::mesh::Distribution<kDim>;
using RankSpace = pops::mesh::RankSpace<kDim>;

std::size_t storage_ordinal(const pops::Box<kDim>& storage, const pops::Index<kDim>& index) {
  return static_cast<std::size_t>(index[0] - storage.lo[0]) +
         static_cast<std::size_t>(index[1] - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

pops::EllipticBuildRequest<kDim> fft_request(int cells) {
  const int ranks = pops::n_ranks();
  if (ranks < 1 || cells % ranks != 0)
    throw std::invalid_argument("FFT test extent must be divisible by communicator size");

  const pops::Box<kDim> domain{pops::Index<kDim>{0, 0}, pops::Index<kDim>{cells - 1, cells - 1}};
  const pops::Geometry<kDim> geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{0, 0}, pops::RealVector<kDim>{1, 1});
  const int local_y = cells / ranks;
  std::vector<pops::Box<kDim>> slabs;
  slabs.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank)
    slabs.emplace_back(pops::Index<kDim>{0, rank * local_y},
                       pops::Index<kDim>{cells - 1, (rank + 1) * local_y - 1});
  Layout layout(std::move(slabs));
  const RankSpace rank_space{pops::Index<kDim>{0, 0}, pops::Extent<kDim>{ranks, 1}};
  std::vector<pops::Index<kDim>> owners;
  owners.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank)
    owners.push_back(rank_space.coordinate(static_cast<std::size_t>(rank)));
  const Distribution distribution =
      Distribution::partitioned(layout, rank_space, std::move(owners));
  const pops::Index<kDim> local_rank =
      rank_space.coordinate(static_cast<std::size_t>(pops::my_rank()));
  const std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  const pops::RealVector<kDim> spacing{geometry.spacing(0), geometry.spacing(1)};
  const std::size_t pairs = layout.size() * (layout.size() - 1) / 2;
  return {geometry,
          std::move(layout),
          distribution,
          local_rank,
          pops::PhysicalBoundaryConditions<kDim>{
              pops::BoundaryTopology<kDim>::axis_periodic({true, true}), faces, spacing},
          pops::Extent<kDim>{0, 0},
          pops::Extent<kDim>{1, 1},
          {static_cast<std::size_t>(ranks), pairs}};
}

pops::PoissonFFTSolver<kDim> make_fft_solver(int cells) {
  auto request = fft_request(cells);
  const pops::Real measure = request.geometry.spacing(0) * request.geometry.spacing(1);
  pops::PoissonFFTSolver<kDim> solver = pops::make_elliptic_solver<pops::PoissonFFTSolver<kDim>>(
      std::move(request), pops::PoissonFFTFactory<kDim>{});
  solver.install_nullspace(
      pops::constant_mean_zero_nullspace<kDim>("periodic-fft", "unit-test", measure),
      pops::PreparedVectorDistribution<kDim>::distributed());
  return solver;
}

void fill_mode(Field& field, const pops::Geometry<kDim>& geometry, pops::Real offset) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const pops::Box<kDim>& valid = fab.box();
    const pops::Box<kDim>& storage = fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const pops::Index<kDim> index{i, j};
        const pops::Real x = geometry.cell_coordinate(0, i);
        const pops::Real y = geometry.cell_coordinate(1, j);
        host(storage_ordinal(storage, index)) =
            offset + std::sin(pops::Real(2) * kPi * x) * std::sin(pops::Real(2) * kPi * y);
      }
    }
    fab.copy_from_host(host);
  }
}

void subtract_global_mean(Field& field) {
  pops::Real cells = 0;
  for (const pops::Box<kDim>& box : field.layout().boxes())
    cells += static_cast<pops::Real>(box.numPts());
  const pops::Real mean = pops::reduce_sum(field) / cells;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const pops::Box<kDim>& valid = fab.box();
    const pops::Box<kDim>& storage = fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        host(storage_ordinal(storage, pops::Index<kDim>{i, j})) -= mean;
    fab.copy_from_host(host);
  }
}

pops::Real maximum_difference(const Field& left, const Field& right) {
  pops::Real local_max = 0;
  for (std::size_t local = 0; local < left.local_size(); ++local) {
    const auto& left_fab = left.fab(local);
    const auto& right_fab = right.fab(local);
    auto left_host = left_fab.create_host_mirror();
    auto right_host = right_fab.create_host_mirror();
    left_fab.copy_to_host(left_host);
    right_fab.copy_to_host(right_host);
    const pops::Box<kDim>& valid = left_fab.box();
    const pops::Box<kDim>& left_storage = left_fab.grown_box();
    const pops::Box<kDim>& right_storage = right_fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const pops::Index<kDim> index{i, j};
        local_max =
            std::max(local_max, std::abs(left_host(storage_ordinal(left_storage, index)) -
                                         right_host(storage_ordinal(right_storage, index))));
      }
    }
  }
  return static_cast<pops::Real>(pops::all_reduce_max(static_cast<double>(local_max)));
}

}  // namespace

TEST(test_poisson_fft, capability_is_exact_rank_two_and_spectral_fails_closed) {
  static_assert(!pops::PoissonFFTCapabilities<1>::available);
  static_assert(pops::PoissonFFTCapabilities<2>::available);
  static_assert(!pops::PoissonFFTCapabilities<3>::available);
  static_assert(pops::PoissonFFTCapabilities<2>::discrete_symbol);
  static_assert(!pops::PoissonFFTCapabilities<2>::continuous_spectral_symbol);

  EXPECT_FALSE(
      pops::PoissonFFTCapabilities<2>::supports(pops::PoissonFFTSymbol::continuous_spectral));
  EXPECT_EQ(pops::PoissonFFTCapabilities<2>::rejection_reason(
                pops::PoissonFFTSymbol::continuous_spectral),
            "continuous spectral FFT has no exact apply/residual provider");
  EXPECT_THROW((void)pops::PoissonFFTFactory<2>{pops::PoissonFFTSymbol::continuous_spectral},
               std::invalid_argument);
}

TEST(test_poisson_fft, direct_dft_fallback_rejects_nonzero_mean_then_solves_exactly) {
  pops::PoissonFFT slow_probe(kFallbackCells, kFallbackCells, 1.0, 1.0);
  pops::PoissonFFT fast_probe(32, 32, 1.0, 1.0);
  EXPECT_TRUE(slow_probe.uses_direct_dft_fallback());
  EXPECT_FALSE(fast_probe.uses_direct_dft_fallback());

  pops::PoissonFFTSolver<kDim> solver = make_fft_solver(kFallbackCells);
  solver.phi().set_val(pops::Real(7));
  fill_mode(solver.rhs(), solver.geom(), pops::Real(1));
  pops::reset_poisson_fft_direct_dft_fallback_count();
  const pops::SolveReport incompatible = solver.solve();
  EXPECT_EQ(incompatible.status, pops::SolveStatus::kIncompatibleRhs);
  EXPECT_EQ(pops::poisson_fft_direct_dft_fallback_count(), 0);
  EXPECT_EQ(pops::reduce_norm_inf(solver.phi()), pops::Real(7));

  subtract_global_mean(solver.rhs());
  const pops::SolveReport solved = solver.solve();
  ASSERT_TRUE(solved.solved()) << solved.reason;
  EXPECT_GT(pops::poisson_fft_direct_dft_fallback_count(), 0);
  EXPECT_TRUE(std::isfinite(static_cast<double>(pops::reduce_norm_inf(solver.phi()))));
  EXPECT_LT(solved.residual_norm, pops::Real(1e-9));
}

TEST(test_poisson_fft, discrete_provider_matches_exact_rank_geometric_mg) {
  auto mg_request = fft_request(kFallbackCells);
  auto fft_request_value = mg_request;
  const pops::Real measure = mg_request.geometry.spacing(0) * mg_request.geometry.spacing(1);

  pops::PoissonFFTSolver<kDim> fft = pops::make_elliptic_solver<pops::PoissonFFTSolver<kDim>>(
      std::move(fft_request_value), pops::PoissonFFTFactory<kDim>{});
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-11);
  options.absolute_tolerance = pops::Real(1e-13);
  options.maximum_cycles = 200;
  options.bottom_sweeps = 80;
  pops::elliptic::mg::GeometricMG<kDim> mg(std::move(mg_request), options);
  fft.install_nullspace(
      pops::constant_mean_zero_nullspace<kDim>("periodic-fft", "unit-test", measure),
      pops::PreparedVectorDistribution<kDim>::distributed());
  mg.install_nullspace(
      pops::constant_mean_zero_nullspace<kDim>("periodic-mg", "unit-test", measure),
      pops::PreparedVectorDistribution<kDim>::distributed());

  fill_mode(fft.rhs(), fft.geom(), pops::Real(0));
  fill_mode(mg.rhs(), mg.geom(), pops::Real(0));
  subtract_global_mean(fft.rhs());
  subtract_global_mean(mg.rhs());
  fft.phi().set_val(pops::Real(0));
  mg.phi().set_val(pops::Real(0));
  const pops::SolveReport fft_report = fft.solve();
  const pops::SolveReport mg_report = mg.solve();
  ASSERT_TRUE(fft_report.solved()) << fft_report.reason;
  ASSERT_TRUE(mg_report.solved()) << mg_report.reason;
  EXPECT_LT(fft_report.residual_norm, pops::Real(1e-9));
  const pops::Real reference = pops::reduce_norm_inf(mg.phi());
  ASSERT_GT(reference, pops::Real(0));
  EXPECT_LT(maximum_difference(fft.phi(), mg.phi()) / reference, pops::Real(1e-6));
}
