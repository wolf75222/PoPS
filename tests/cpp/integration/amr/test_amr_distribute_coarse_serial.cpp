// SERIAL regression lock for ADC-620 (build_amr_compiled paired the single-box fine seed with the
// COARSE DistributionMapping; with distribute_coarse=true the coarse dmap has one entry per coarse box,
// so a 1-box fine BoxArray met a 4-entry mapping and the MultiFab layout check added by ADC-590/#416
// aborted -- test_mpi_amr_distributed_coarse_np{1,2,4} ALL terminated, np1 included, since it is a
// metadata mismatch, not a rank-count issue). This is the np1 case of that regression, run WITHOUT the
// MPI harness (no comm_init / comm_finalize, no MPI ranks): a plain GoogleTest binary that builds the
// compiled-AMR hierarchy with distribute_coarse=true on a single rank and checks it does not abort.
//
// Setup mirrors tests/cpp/integration/mpi/test_mpi_amr_distributed_coarse.cpp (four density bubbles,
// euler_poisson compiled model, geometric_mg coarse solve) minus comm_init/comm_finalize/all_reduce: on
// a single rank the distributed-coarse round-robin dmap degenerates to a single owner (rank 0), so the
// serial run exercises exactly the code path ADC-620 fixed (coupler_make_coarse_layout splits the coarse
// into coarse_max_grid tiles while the fine seed used to carry the COARSE dmap).
//
// What we verify (honesty criteria of this regression lock):
//   (1) the hierarchy BUILDS: constructing the distribute_coarse=true AmrSystem and stepping it does not
//       throw/abort (pre-fix this aborted via the MultiFab layout check, a hard std::runtime_error).
//   (2) a CFL step advances by a FINITE, POSITIVE dt (no NaN/Inf from a corrupted layout).
//   (3) mass/density DIGESTS match the replicated-coarse baseline (distribute_coarse=false) BIT-
//       IDENTICALLY where test_mpi_amr_distributed_coarse asserts the same at np=1: at a single rank the
//       distributed coarse layout owns every tile on rank 0, so the trajectory is IDENTICAL to the
//       replicated mode (same ownership, same order of operations, no cross-rank reduction to blur bits).
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"  // pops::test::Checker, checksum
#include <pops/physics/bricks/bricks.hpp>        // CompositeModel, GravityForce, GravityCoupling
#include <pops/physics/fluids/euler.hpp>         // Euler
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem, ...)
#include <pops/runtime/amr_system.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

namespace {

std::vector<double> four_bubbles(int n) {
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  const double cx[4] = {0.25, 0.75, 0.25, 0.75};
  const double cy[4] = {0.25, 0.25, 0.75, 0.75};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      double r = 1.0;
      for (int b = 0; b < 4; ++b) {
        const double dx = x - cx[b], dy = y - cy[b];
        r += 0.5 * std::exp(-(dx * dx + dy * dy) / 0.004);
      }
      rho[static_cast<std::size_t>(j) * n + i] = r;
    }
  return rho;
}

struct Result {
  std::vector<double> dens;
  double mass, m0;
  int npf;
};

// Builds an AmrSystem (4 bubbles, euler_poisson compiled), advances nsteps, returns the coarse density
// (n*n, single-rank so already global), the final mass and m0. distribute=true exercises the ADC-620
// path (coupler_make_coarse_layout splits the coarse, the fine seed used to borrow that dmap).
Result run(int n, int nsteps, double dt, bool distribute) {
  const std::vector<double> rho = four_bubbles(n);
  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;
  cfg.regrid_every = 4;
  cfg.distribute_coarse = distribute;  // ADC-620: coarse split into tiles, fine seed needs its OWN dmap

  AmrSystem sys(cfg);
  add_compiled_model(sys, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                     "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_refinement(1.2);
  sys.set_density("gas", rho);

  Result R;
  R.m0 = sys.mass();
  for (int s = 0; s < nsteps; ++s)
    sys.step(dt);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::fence();
#endif
  R.dens = sys.density();
  R.mass = sys.mass();
  R.npf = sys.n_patches();
  return R;
}

}  // namespace

static int pops_run_test_amr_distribute_coarse_serial(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  pops::test::Checker chk;  // style terse: n'imprime que les echecs (FAIL <libelle>)

  const int n = 64;
  const int nsteps = 16;
  const double dt = 1e-3;

  // (1) the hierarchy BUILDS: pre-ADC-620, this constructor + the first step() aborted via the
  // MultiFab layout check (box_array.size=1, dmap.size=4) even at a single rank (np1 aborted too: it is
  // a metadata mismatch, not a rank-count issue). If it still aborts, the process terminates here rather
  // than reaching the assertions below -- the regression lock is the process surviving construction and
  // stepping at all, on top of the checks that follow.
  const Result dis = run(n, nsteps, dt, /*distribute=*/true);
  const Result rep = run(n, nsteps, dt, /*distribute=*/false);  // oracle: replicated coarse (unaffected)

  chk(dis.dens.size() == static_cast<std::size_t>(n) * n, "distributed coarse density has size n*n");
  chk(rep.dens.size() == dis.dens.size(), "replicated coarse density same size as distributed");

  // (2) a CFL step advances by a finite, positive dt (no NaN/Inf from a corrupted layout).
  AmrSystemConfig probe_cfg;
  probe_cfg.n = n;
  probe_cfg.L = 1.0;
  probe_cfg.periodic = true;
  probe_cfg.regrid_every = 4;
  probe_cfg.distribute_coarse = true;
  AmrSystem probe(probe_cfg);
  add_compiled_model(probe, "gas",
                     Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}}, "minmod",
                     "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  probe.set_poisson("charge_density", "geometric_mg");
  probe.set_refinement(1.2);
  probe.set_density("gas", four_bubbles(n));
  const double dt_cfl = probe.step_cfl(0.4);
  chk(std::isfinite(dt_cfl), "distribute_coarse step_cfl returns a finite dt");
  chk(dt_cfl > 0.0, "distribute_coarse step_cfl returns a positive dt");

  // (3) mass/density digests match the REPLICATED baseline bit-identically: on a single rank the
  // distributed coarse layout owns every tile on rank 0 (round-robin of N tiles onto 1 rank), so
  // ownership and the order of operations are IDENTICAL to the replicated mono-box mode -- same as
  // test_mpi_amr_distributed_coarse asserts (dist_vs_repl_dmax) at np=1, just without the MPI harness.
  double dmax = 0;
  for (std::size_t k = 0; k < dis.dens.size() && k < rep.dens.size(); ++k)
    dmax = std::fmax(dmax, std::fabs(dis.dens[k] - rep.dens[k]));
  const double csum_dis = pops::test::checksum(dis.dens);
  const double csum_rep = pops::test::checksum(rep.dens);

  std::printf(
      "AMRDISTSERIAL npf_dist=%d npf_repl=%d | dmax=%.17e | csum_dist=%.17e csum_repl=%.17e | "
      "mass_dist=%.17e mass_repl=%.17e\n",
      dis.npf, rep.npf, dmax, csum_dis, csum_rep, dis.mass, rep.mass);

  chk(dmax == 0.0, "distributed coarse density bit-identical to replicated coarse (single rank)");
  chk(csum_dis == csum_rep, "checksum(distributed) == checksum(replicated) (single rank)");
  chk(std::isfinite(dis.mass) && std::isfinite(rep.mass), "final mass finite in both modes");
  chk(std::fabs(dis.mass - dis.m0) < 1e-10, "mass conserved (distribute_coarse=true)");
  chk(std::fabs(rep.mass - rep.m0) < 1e-10, "mass conserved (distribute_coarse=false, oracle)");
  chk(dis.mass == rep.mass, "final mass bit-identical distributed vs replicated (single rank)");

  if (chk.fails() == 0)
    std::printf(
        "OK test_amr_distribute_coarse_serial (ADC-620: distribute_coarse=true hierarchy builds and "
        "steps on a single rank, bit-identical to replicated coarse)\n");
  return chk.failed();
}

TEST(test_amr_distribute_coarse_serial, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_amr_distribute_coarse_serial,
                                    "test_amr_distribute_coarse_serial"),
            0);
}
