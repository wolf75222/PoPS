// C++ equivalent of the ExB row of tests/python/unit/runtime/test_seam_combinations.py
// (test_system_generated_seam_advances): the Python smoke drives the native System engine seam with
// model=pops.Model(state=Scalar(), transport=ExB(B0=...), source=NoSource(),
// elliptic=BackgroundDensity(...)) and asserts step_cfl returns a finite, positive dt. There was no C++
// counterpart exercising the SAME (transport="exb", flux=None) seam through System::add_block(ModelSpec)
// -- this closes that gap.
//
// ModelSpec mirrors the python brick combination one-to-one: transport="exb" (divergence-free E x B
// drift), source="none" (no force), elliptic="background" (f = alpha (n - n0), neutralizing background).
// Same vocabulary as test_amr_potential.cpp's exb_background() helper, but exercised on a plain System
// (not AmrSystem) and driven through step_cfl (the CFL-limited step the python seam test calls), plus a
// conservation check (ExB transport is divergence-free: total mass should not drift beyond round-off).
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"  // pops::test::Checker
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

// Smooth density bump, periodic: same shape as the python seam test's _seed_density (a sin*sin
// perturbation around 1), so the CFL wave speed is finite and step_cfl advances.
std::vector<double> seed_density(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  const double pi = 3.14159265358979323846;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      rho[static_cast<std::size_t>(j) * n + i] =
          1.0 + 0.1 * std::sin(2 * pi * x) * std::sin(2 * pi * y);
    }
  return rho;
}

// Same ModelSpec combination as the python seam's ("exb", None) row: transport=exb, source=none,
// elliptic=background (neutralizing fond, so the periodic Poisson source has zero mean).
ModelSpec exb_seam_model(double n0) {
  ModelSpec spec;
  spec.transport = "exb";
  spec.source = "none";
  spec.elliptic = "background";
  spec.B0 = 1.0;
  spec.alpha = 1.0;
  spec.n0 = n0;
  return spec;
}

}  // namespace

static int pops_run_test_exb_seam(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  pops::test::Checker chk;

  const int n = 32;
  const std::vector<double> rho = seed_density(n);
  double n0 = 0;
  for (double v : rho)
    n0 += v;
  n0 /= (static_cast<double>(n) * n);  // background = mean density -> neutral source (periodic Poisson)

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  System sys(cfg);
  sys.add_block("blk", exb_seam_model(n0));  // defaults: minmod / rusanov / conservative / explicit
  sys.set_density("blk", rho);

  const double m0 = sys.mass("blk");
  chk(std::isfinite(m0), "initial mass finite");

  // Same assertion as the python seam test: step_cfl returns a finite, POSITIVE dt.
  const double dt = sys.step_cfl(0.4);
  chk(std::isfinite(dt), "exb seam: step_cfl returns a finite dt");
  chk(dt > 0.0, "exb seam: step_cfl returns a positive dt");

  const double m1 = sys.mass("blk");
  chk(std::isfinite(m1), "post-step mass finite");
  // Conservation check: ExB E x B drift is divergence-free (pure advection of the density), so a single
  // CFL-limited step should not drift the total mass beyond round-off / flux-limiter truncation. Bound
  // matched to the AMR mass-conservation checks elsewhere (test_mpi_amr_distributed_coarse: < 1e-10
  // after many steps); here a SINGLE step, so an even tighter bound is safe.
  const double dm = std::fabs(m1 - m0);
  chk(dm < 1e-9 * (std::fabs(m0) + 1.0), "exb seam: mass conserved across one CFL step");

  std::printf("EXBSEAM dt=%.3e m0=%.17e m1=%.17e dm=%.3e\n", dt, m0, m1, dm);

  if (chk.fails() == 0)
    std::printf("OK test_exb_seam (System + ModelSpec exb/none/background: step_cfl advances, mass "
               "conserved)\n");
  return chk.failed();
}

TEST(test_exb_seam, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_exb_seam, "test_exb_seam"), 0);
}
