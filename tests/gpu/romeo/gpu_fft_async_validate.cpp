// Device-to-host ordering gate for the exact-ranked host FFT solver.

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = 2;
constexpr int kCells = 64;

struct MeanZeroRhsKernel {
  pops::FieldView<pops::Real, kDim> rhs{};

  POPS_HD void operator()(const pops::CellIndex<kDim>& cell) const {
    rhs(cell, 0) = ((cell[0] + cell[1]) & 1) == 0 ? pops::Real(1) : pops::Real(-1);
  }
};

pops::EllipticBuildRequest<kDim> request() {
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

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    auto build = request();
    const pops::Real measure = build.geometry.spacing(0) * build.geometry.spacing(1);
    pops::PoissonFFTSolver<kDim> solver = pops::make_elliptic_solver<pops::PoissonFFTSolver<kDim>>(
        std::move(build), pops::PoissonFFTFactory<kDim>{});
    solver.install_nullspace(
        pops::constant_mean_zero_nullspace<kDim>("gpu-fft-constant", "gpu-test", measure),
        pops::PreparedVectorDistribution<kDim>::distributed());

    // Deliberately no fence: the host mirror inside solve() owns the ordering boundary.
    pops::for_each_cell(solver.rhs().box(0), MeanZeroRhsKernel{solver.rhs().fab(0).view()});
    const pops::SolveReport report = solver.solve();

    const pops::Real phi_norm = pops::reduce_norm_inf(solver.phi());
    const pops::Real rhs_mean =
        pops::reduce_sum(solver.rhs()) / static_cast<pops::Real>(kCells * kCells);
    const bool finite = std::isfinite(static_cast<double>(phi_norm));
    const bool mean_zero = std::abs(static_cast<double>(rhs_mean)) < 1e-14;
    const bool solved = report.solved() && report.residual_norm < pops::Real(1e-9);

    std::printf("[fft-order] exec=%s n=%d rhs_mean=%.17g phi_norm=%.17g residual=%.3e\n",
                Kokkos::DefaultExecutionSpace::name(), kCells, static_cast<double>(rhs_mean),
                static_cast<double>(phi_norm), static_cast<double>(report.residual_norm));
    if (!finite || !mean_zero || !solved) {
      std::printf("FAIL gpu_fft_async_validate ordering_or_residual\n");
      rc = 1;
    } else {
      std::printf("OK gpu_fft_async_validate\n");
    }
  }
  Kokkos::finalize();
  return rc;
}
