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

constexpr int kRegressionDim = 2;
constexpr int kFallbackCells = 48;
constexpr pops::Real kPi = pops::Real(3.141592653589793238462643383279502884L);

using Field = pops::MultiFab<kRegressionDim>;
using Layout = pops::mesh::BoxArray<kRegressionDim>;
using Distribution = pops::mesh::Distribution<kRegressionDim>;
using RankSpace = pops::mesh::RankSpace<kRegressionDim>;

std::size_t storage_ordinal(const pops::Box<kRegressionDim>& storage,
                            const pops::Index<kRegressionDim>& index) {
  return static_cast<std::size_t>(index[0] - storage.lo[0]) +
         static_cast<std::size_t>(index[1] - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

pops::EllipticBuildRequest<kRegressionDim> fft_request(int cells) {
  const int ranks = pops::n_ranks();
  if (ranks < 1 || cells % ranks != 0)
    throw std::invalid_argument("FFT test extent must be divisible by communicator size");

  const pops::Box<kRegressionDim> domain{pops::Index<kRegressionDim>{0, 0},
                                         pops::Index<kRegressionDim>{cells - 1, cells - 1}};
  const pops::Geometry<kRegressionDim> geometry = pops::Geometry<kRegressionDim>::from_bounds(
      domain, pops::RealVector<kRegressionDim>{0, 0}, pops::RealVector<kRegressionDim>{1, 1});
  const int local_y = cells / ranks;
  std::vector<pops::Box<kRegressionDim>> slabs;
  slabs.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank)
    slabs.emplace_back(pops::Index<kRegressionDim>{0, rank * local_y},
                       pops::Index<kRegressionDim>{cells - 1, (rank + 1) * local_y - 1});
  Layout layout(std::move(slabs));
  const RankSpace rank_space{pops::Index<kRegressionDim>{0, 0},
                             pops::Extent<kRegressionDim>{ranks, 1}};
  std::vector<pops::Index<kRegressionDim>> owners;
  owners.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank)
    owners.push_back(rank_space.coordinate(static_cast<std::size_t>(rank)));
  const Distribution distribution =
      Distribution::partitioned(layout, rank_space, std::move(owners));
  const pops::Index<kRegressionDim> local_rank =
      rank_space.coordinate(static_cast<std::size_t>(pops::my_rank()));
  const std::array<pops::PhysicalBoundaryFace, 2 * kRegressionDim> faces{};
  const pops::RealVector<kRegressionDim> spacing{geometry.spacing(0), geometry.spacing(1)};
  const std::size_t pairs = layout.size() * (layout.size() - 1) / 2;
  return {geometry,
          std::move(layout),
          distribution,
          local_rank,
          pops::PhysicalBoundaryConditions<kRegressionDim>{
              pops::BoundaryTopology<kRegressionDim>::axis_periodic({true, true}), faces, spacing},
          pops::Extent<kRegressionDim>{0, 0},
          pops::Extent<kRegressionDim>{1, 1},
          {static_cast<std::size_t>(ranks), pairs}};
}

pops::PoissonFFTSolver<kRegressionDim> make_fft_solver(int cells) {
  auto request = fft_request(cells);
  const pops::Real measure = request.geometry.spacing(0) * request.geometry.spacing(1);
  pops::PoissonFFTSolver<kRegressionDim> solver =
      pops::make_elliptic_solver<pops::PoissonFFTSolver<kRegressionDim>>(
          std::move(request), pops::PoissonFFTFactory<kRegressionDim>{});
  solver.install_nullspace(
      pops::constant_mean_zero_nullspace<kRegressionDim>("periodic-fft", "unit-test", measure),
      pops::PreparedVectorDistribution<kRegressionDim>::distributed());
  return solver;
}

void fill_mode(Field& field, const pops::Geometry<kRegressionDim>& geometry, pops::Real offset) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const pops::Box<kRegressionDim>& valid = fab.box();
    const pops::Box<kRegressionDim>& storage = fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const pops::Index<kRegressionDim> index{i, j};
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
  for (const pops::Box<kRegressionDim>& box : field.layout().boxes())
    cells += static_cast<pops::Real>(box.numPts());
  const pops::Real mean = pops::reduce_sum(field) / cells;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const pops::Box<kRegressionDim>& valid = fab.box();
    const pops::Box<kRegressionDim>& storage = fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        host(storage_ordinal(storage, pops::Index<kRegressionDim>{i, j})) -= mean;
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
    const pops::Box<kRegressionDim>& valid = left_fab.box();
    const pops::Box<kRegressionDim>& left_storage = left_fab.grown_box();
    const pops::Box<kRegressionDim>& right_storage = right_fab.grown_box();
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
        const pops::Index<kRegressionDim> index{i, j};
        local_max =
            std::max(local_max, std::abs(left_host(storage_ordinal(left_storage, index)) -
                                         right_host(storage_ordinal(right_storage, index))));
      }
    }
  }
  return static_cast<pops::Real>(pops::all_reduce_max(static_cast<double>(local_max)));
}

template <int Dim>
void expect_device_cartesian_fft() {
  constexpr int kExtent = 8;
  std::array<int, Dim> cells{};
  std::array<double, Dim> lengths{};
  cells.fill(kExtent);
  lengths.fill(1.0);
  pops::PoissonFFT<Dim> fft(cells, lengths, "pops.unit.nd-device-fft");
  typename pops::PoissonFFT<Dim>::device_view rhs("fft_nd_rhs", fft.local_cell_count());
  typename pops::PoissonFFT<Dim>::device_view phi("fft_nd_phi", fft.local_cell_count());
  auto host_rhs = Kokkos::create_mirror_view(rhs);
  for (std::size_t ordinal = 0; ordinal < fft.local_cell_count(); ++ordinal) {
    const int x = static_cast<int>(ordinal % kExtent);
    host_rhs[ordinal] = typename pops::PoissonFFT<Dim>::complex_type(
        std::sin(pops::Real(2) * kPi * (x + pops::Real(0.5)) / kExtent), 0.0);
  }
  Kokkos::deep_copy(rhs, host_rhs);
  fft.solve(rhs, phi);
  auto host_phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, phi);
  const pops::Real h = pops::Real(1) / kExtent;
  const pops::Real lambda =
      (pops::Real(2) * std::cos(pops::Real(2) * kPi / kExtent) - pops::Real(2)) / (h * h);
  for (std::size_t ordinal = 0; ordinal < fft.local_cell_count(); ++ordinal)
    EXPECT_NEAR(host_phi[ordinal].real(), host_rhs[ordinal].real() / lambda, 1e-11);
}

template <int Dim>
void expect_cartesian_solver() {
  constexpr int kExtent = 8;
  pops::Index<Dim> lo{};
  pops::Index<Dim> hi{};
  pops::RealVector<Dim> lower{};
  pops::RealVector<Dim> upper{};
  pops::Extent<Dim> one{};
  std::array<bool, Dim> periodic{};
  std::array<pops::PhysicalBoundaryFace, 2 * Dim> faces{};
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis) {
    hi[axis] = kExtent - 1;
    upper[axis] = pops::Real(1);
    one[axis] = 1;
    periodic[axis] = true;
  }
  const pops::Box<Dim> domain{lo, hi};
  const pops::Geometry<Dim> geometry = pops::Geometry<Dim>::from_bounds(domain, lower, upper);
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  const pops::mesh::BoxArray<Dim> layout(std::vector<pops::Box<Dim>>{domain});
  const pops::mesh::RankSpace<Dim> ranks{lo, one};
  const auto distribution =
      pops::mesh::Distribution<Dim>::partitioned(layout, ranks, std::vector<pops::Index<Dim>>{lo});
  pops::Extent<Dim> phi_ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    phi_ghosts[axis] = 1;
  pops::EllipticBuildRequest<Dim> request{
      geometry,
      layout,
      distribution,
      lo,
      pops::PhysicalBoundaryConditions<Dim>{pops::BoundaryTopology<Dim>::axis_periodic(periodic),
                                            faces, spacing},
      pops::Extent<Dim>{},
      phi_ghosts,
      {1, 0}};
  pops::PoissonFFTSolver<Dim> solver = pops::make_elliptic_solver<pops::PoissonFFTSolver<Dim>>(
      std::move(request), pops::PoissonFFTFactory<Dim>{});
  pops::Real measure = pops::Real(1);
  for (int axis = 0; axis < Dim; ++axis)
    measure *= geometry.spacing(axis);
  solver.install_nullspace(
      pops::constant_mean_zero_nullspace<Dim>("periodic-unit-fft", "unit-nd-test", measure),
      pops::PreparedVectorDistribution<Dim>::distributed());
  auto rhs = solver.rhs().fab(0).view();
  const auto solver_geometry = solver.geom();
  pops::for_each_cell(solver.rhs().box(0), [=](const pops::CellIndex<Dim>& cell) {
    rhs(cell, 0) = Kokkos::sin(pops::Real(2) * kPi * solver_geometry.cell_coordinate(0, cell[0]));
  });
  const pops::SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved()) << report.reason;
  EXPECT_LT(report.residual_norm, pops::Real(1e-9));
}

}  // namespace

TEST(test_poisson_fft, capability_is_cartesian_nd_and_spectral_fails_closed) {
  static_assert(pops::PoissonFFTCapabilities<1>::available);
  static_assert(pops::PoissonFFTCapabilities<2>::available);
  static_assert(pops::PoissonFFTCapabilities<3>::available);
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

TEST(test_poisson_fft, device_engine_executes_same_cartesian_trace_in_one_and_three_dimensions) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP() << "MPI target qualifies the pairwise phases";
  expect_device_cartesian_fft<1>();
  expect_device_cartesian_fft<3>();
}

TEST(test_poisson_fft, solver_executes_in_one_two_and_three_dimensions) {
  if (pops::n_ranks() != 1)
    GTEST_SKIP() << "MPI target owns the partitioned solver matrix";
  expect_cartesian_solver<1>();
  expect_cartesian_solver<2>();
  expect_cartesian_solver<3>();
}

TEST(test_poisson_fft, radix_last_axis_has_no_direct_dft_fallback_but_nonpow2_is_diagnosed) {
  auto execute = [](int extent) {
    pops::PoissonFFT<1> fft({extent}, {1.0}, "pops.unit.last-axis-radix");
    pops::PoissonFFT<1>::device_view rhs("fft_last_rhs", fft.local_cell_count());
    pops::PoissonFFT<1>::device_view phi("fft_last_phi", fft.local_cell_count());
    Kokkos::deep_copy(rhs, pops::PoissonFFT<1>::complex_type(0.0, 0.0));
    fft.solve(rhs, phi);
  };
  pops::reset_poisson_fft_direct_dft_fallback_count();
  execute(8);
  EXPECT_EQ(pops::poisson_fft_direct_dft_fallback_count(), 0u);
  execute(6);
  EXPECT_GT(pops::poisson_fft_direct_dft_fallback_count(), 0u);
}

TEST(test_poisson_fft, rejects_invalid_cartesian_workspace_before_lane_materialization) {
  EXPECT_THROW((pops::PoissonFFT<1>({0}, {1.0}, "pops.unit.invalid-fft")), std::invalid_argument);
  EXPECT_THROW((pops::PoissonFFT<2>({8, 8}, {1.0, 0.0}, "pops.unit.invalid-fft")),
               std::invalid_argument);
}

TEST(test_poisson_fft, device_dft_rejects_nonzero_mean_then_solves_exactly) {
  pops::PoissonFFT<2> slow_probe({kFallbackCells, kFallbackCells}, {1.0, 1.0});
  pops::PoissonFFT<2> fast_probe({32, 32}, {1.0, 1.0});
  EXPECT_TRUE(slow_probe.uses_direct_dft_fallback());
  EXPECT_FALSE(fast_probe.uses_direct_dft_fallback());

  pops::PoissonFFTSolver<kRegressionDim> solver = make_fft_solver(kFallbackCells);
  solver.phi().set_val(pops::Real(7));
  fill_mode(solver.rhs(), solver.geom(), pops::Real(1));
  const pops::SolveReport incompatible = solver.solve();
  EXPECT_EQ(incompatible.status, pops::SolveStatus::kIncompatibleRhs);
  EXPECT_EQ(pops::reduce_norm_inf(solver.phi()), pops::Real(7));

  subtract_global_mean(solver.rhs());
  const pops::SolveReport solved = solver.solve();
  ASSERT_TRUE(solved.solved()) << solved.reason;
  EXPECT_TRUE(std::isfinite(static_cast<double>(pops::reduce_norm_inf(solver.phi()))));
  EXPECT_LT(solved.residual_norm, pops::Real(1e-9));
}

TEST(test_poisson_fft, discrete_provider_matches_exact_rank_geometric_mg) {
  auto mg_request = fft_request(kFallbackCells);
  auto fft_request_value = mg_request;
  const pops::Real measure = mg_request.geometry.spacing(0) * mg_request.geometry.spacing(1);

  pops::PoissonFFTSolver<kRegressionDim> fft =
      pops::make_elliptic_solver<pops::PoissonFFTSolver<kRegressionDim>>(
          std::move(fft_request_value), pops::PoissonFFTFactory<kRegressionDim>{});
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-11);
  options.absolute_tolerance = pops::Real(1e-13);
  options.maximum_cycles = 200;
  options.bottom_sweeps = 80;
  pops::elliptic::mg::GeometricMG<kRegressionDim> mg(std::move(mg_request), options);
  fft.install_nullspace(
      pops::constant_mean_zero_nullspace<kRegressionDim>("periodic-fft", "unit-test", measure),
      pops::PreparedVectorDistribution<kRegressionDim>::distributed());
  mg.install_nullspace(
      pops::constant_mean_zero_nullspace<kRegressionDim>("periodic-mg", "unit-test", measure),
      pops::PreparedVectorDistribution<kRegressionDim>::distributed());

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
