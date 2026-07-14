// Targeted ordering gate for the host FFT solver's RHS hand-off.
//
// The periodic, mean-zero RHS is produced by an asynchronous device kernel and
// PoissonFFTSolver::solve() is called immediately.  solve() must own the
// device-to-host ordering boundary; the residual reduction then verifies both
// the finite solution and the discrete equation.

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct MeanZeroRhsKernel {
  pops::Array4 rhs;

  POPS_HD void operator()(int i, int j) const {
    // Even-sized periodic domain: this checkerboard has an exactly zero mean
    // while avoiding device transcendental functions in the ordering probe.
    rhs(i, j, 0) = ((i + j) & 1) == 0 ? pops::Real(1) : pops::Real(-1);
  }
};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    constexpr int n = 64;  // radix-2 path; keeps the probe short on the allocated compute GPU.
    const pops::Box2D domain = pops::Box2D::from_extents(n, n);
    const pops::Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
    const pops::BoxArray boxes(std::vector<pops::Box2D>{domain});
    pops::BCRec periodic;
    periodic.xlo = periodic.xhi = periodic.ylo = periodic.yhi = pops::BCType::Periodic;
    pops::PoissonFFTSolver solver(geometry, boxes, periodic);

    // No fence is inserted here: solve() must make the async RHS visible to
    // its host-side FFT implementation through rhs().sync_host().
    const pops::Array4 rhs = solver.rhs().fab(0).array();
    pops::for_each_cell(domain, MeanZeroRhsKernel{rhs});
    solver.solve();

    // residual() uses a blocking norm reduction, so the scalar is safe to
    // inspect on the host without an extra device fence in this harness.
    const pops::Real residual = solver.residual();
    const pops::Real phi_norm = pops::norm_inf(solver.phi());
    const pops::Real rhs_mean = pops::sum(solver.rhs()) / static_cast<pops::Real>(n * n);
    const bool finite = std::isfinite(static_cast<double>(phi_norm));
    const bool mean_zero = std::abs(static_cast<double>(rhs_mean)) < 1e-14;
    const bool residual_small = residual < pops::Real(1e-9);

    std::printf("[fft-order] exec=%s n=%d rhs_mean=%.17g phi_norm=%.17g residual=%.3e\n",
                Kokkos::DefaultExecutionSpace::name(), n, static_cast<double>(rhs_mean),
                static_cast<double>(phi_norm), static_cast<double>(residual));
    if (!finite || !mean_zero || !residual_small) {
      std::printf("FAIL gpu_fft_async_validate ordering_or_residual\n");
      rc = 1;
    } else {
      std::printf("OK gpu_fft_async_validate\n");
    }
  }
  Kokkos::finalize();
  return rc;
}
