// SERIAL regression lock for ADC-620 (the old single-block builder paired the single-box fine seed with the
// COARSE DistributionMapping; with distribute_coarse=true the coarse dmap has one entry per coarse box,
// so a 1-box fine BoxArray met a 4-entry mapping and the MultiFab layout check added by ADC-590/#416
// aborted -- test_mpi_amr_distributed_coarse_np{1,2,4} ALL terminated, np1 included, since it is a
// metadata mismatch, not a rank-count issue). This is the np1 case of that regression, run WITHOUT the
// MPI harness (no comm_init / comm_finalize, no MPI ranks): a plain GoogleTest binary that builds the
// compiled-AMR hierarchy with distribute_coarse=true on a single rank and checks it does not abort.
//
// Setup mirrors tests/cpp/integration/mpi/test_mpi_amr_distributed_coarse.cpp (2^Dim corner density
// bubbles, a compiled Euler/gravity model, geometric_mg coarse solve) minus
// comm_init/comm_finalize/all_reduce: on a single rank the distributed-coarse round-robin dmap
// degenerates to a single owner (rank 0), so the serial run exercises exactly the code path ADC-620 fixed
// (coupler_make_coarse_layout splits the coarse into coarse_max_grid tiles while the fine seed used to
// carry the COARSE dmap).
//
// What we verify (honesty criteria of this regression lock):
//   (1) the two-level hierarchy BUILDS in both ownership modes (pre-fix the distributed layout aborted
//       via the MultiFab layout check, a hard std::runtime_error).
//   (2) without the public conservative multi-level synchronization provider, the program rejects its
//       step before publication and leaves the accepted density byte-identical.
//   (3) accepted mass/density match the replicated-coarse baseline (distribute_coarse=false) to a strict
//       machine-roundoff bound. Both layouts have the same owner at np=1, but the multi-tile field solver
//       has a different local reduction tree than the replicated mono-box layout.
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include "test_harness.hpp"                // pops::test::Checker, checksum
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, GravityCoupling, NoSource
#include <pops/physics/fluids/euler.hpp>   // Euler
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem, ...)
#include <pops/runtime/amr_system.hpp>

#include "amr_tagging_test_authority.hpp"

#include <cmath>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

constexpr int Dim = kNativeDimension;
using NativeAmrSystem = AmrSystem<Dim>;
using NativeAmrSystemConfig = AmrSystemConfig<Dim>;
using Model = CompositeModel<EulerND<Dim>, NoSource, GravityCoupling>;
const std::string kProviderConsumerQid = "test.amr.distribute-coarse.gas";
constexpr const char* kMissingSynchronizationProvider =
    "AmrProgramContext::advance_hierarchy requires a prepared exact-ranked "
    "conservative multi-level synchronization provider before any level state is advanced";

NativeAmrSystemConfig native_config(int n, bool distribute) {
  NativeAmrSystemConfig config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = n;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  config.regrid_every = 4;
  config.distribute_coarse = distribute;
  return config;
}

Model gravity_model() {
  return {{},
          EulerND<Dim>::prepare(Real(1.4)),
          NoSource{},
          GravityCoupling{Real(-1.0), Real(1.0), Real(1.0)}};
}

std::size_t cell_count(int n) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(n);
  return result;
}

std::vector<double> corner_bubbles(int n) {
  std::vector<double> rho(cell_count(n));
  constexpr int bubble_count = 1 << Dim;
  for (std::size_t cell = 0; cell < rho.size(); ++cell) {
    std::size_t quotient = cell;
    std::array<double, Dim> coordinate{};
    for (int axis = 0; axis < Dim; ++axis) {
      coordinate[static_cast<std::size_t>(axis)] =
          (static_cast<double>(quotient % static_cast<std::size_t>(n)) + 0.5) / n;
      quotient /= static_cast<std::size_t>(n);
    }
    double value = 1.0;
    for (int bubble = 0; bubble < bubble_count; ++bubble) {
      double radius_squared = 0.0;
      for (int axis = 0; axis < Dim; ++axis) {
        const double center = (bubble & (1 << axis)) == 0 ? 0.25 : 0.75;
        const double offset = coordinate[static_cast<std::size_t>(axis)] - center;
        radius_squared += offset * offset;
      }
      value += 0.5 * std::exp(-radius_squared / 0.004);
    }
    rho[cell] = value;
  }
  // A periodic Poisson operator has a constant nullspace. Keep the non-trivial corner peaks while
  // authoring an exactly neutralizing background instead of relying on the solver's former silent
  // RHS projection.
  double mean = 0.0;
  for (double value : rho)
    mean += value;
  mean /= static_cast<double>(rho.size());
  for (double& value : rho)
    value += 1.0 - mean;
  return rho;
}

struct Result {
  std::vector<double> dens;
  double mass = 0.0;
  int npf;
  int levels = 0;
  std::string advance_error;
  bool density_unchanged = false;
};

// Builds a two-level AmrSystem, attempts the deliberately unsupported program advance, and returns the
// accepted coarse density. distribute=true exercises the ADC-620 path
// (coupler_make_coarse_layout splits the coarse, while the fine seed needs its own dmap).
Result run(int n, bool distribute) {
  const std::vector<double> rho = corner_bubbles(n);
  NativeAmrSystemConfig cfg = native_config(n, distribute);

  NativeAmrSystem sys(cfg);
  sys.install_block_state_route("gas", "test.amr.distribute-coarse/gas/state");
  // Temporal subcycling is an independent execution authority: spell it out even though this
  // regression happens to use the same ratio as the spatial hierarchy.  The runtime must never
  // infer a clock relation from mesh refinement.
  sys.set_temporal_relations({2}, {1}, {"integral_only"});
  add_compiled_model(sys, "gas", gravity_model(), "minmod", "rusanov", "conservative", "explicit",
                     /*gamma=*/1.4, /*substeps=*/1, /*stride=*/1, {}, {},
                     /*positivity_floor=*/0.0, static_cast<double>(kWenoEpsilon),
                     /*wave_speed_cache=*/false, kProviderConsumerQid);
  sys.set_poisson("charge_density", "geometric_mg");
  test::install_prepared_threshold_union(sys, {{"gas", "rho", 1.2}});
  sys.set_density("gas", rho);
  test::install_forward_euler_program(sys);

  Result R;
  const std::vector<double> accepted = sys.density();
  try {
    sys.step(1e-3);
  } catch (const std::exception& error) {
    R.advance_error = error.what();
  }
#if defined(POPS_HAS_KOKKOS)
  Kokkos::fence();
#endif
  R.dens = sys.density();
  R.mass = sys.mass();
  R.npf = sys.n_patches();
  R.levels = sys.n_levels();
  R.density_unchanged = R.dens.size() == accepted.size() &&
                        (R.dens.empty() ||
                         std::memcmp(R.dens.data(), accepted.data(), R.dens.size() * sizeof(double)) ==
                             0);
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

  // The hierarchy must build in both ownership modes. Program advancement is deliberately refused
  // until the public facade receives its conservative multi-level synchronization provider.
  const Result dis = run(n, /*distribute=*/true);
  const Result rep = run(n, /*distribute=*/false);

  chk(dis.dens.size() == cell_count(n), "distributed coarse density has exact-ranked size");
  chk(rep.dens.size() == dis.dens.size(), "replicated coarse density same size as distributed");
  chk(dis.levels == 2 && rep.levels == 2, "both ownership modes materialize two exact levels");
  chk(dis.advance_error == kMissingSynchronizationProvider &&
          rep.advance_error == kMissingSynchronizationProvider,
      "unprepared multi-level advance reports the exact missing synchronization-provider diagnostic");
  chk(dis.density_unchanged && rep.density_unchanged,
      "unprepared multi-level advance leaves accepted density byte-identical");

  // (3) The distributed and replicated layouts share the same owner at np=1 but not the same local
  // reduction tree (multi-tile versus mono-box). Require a tight scale-aware machine-roundoff bound,
  // substantially stronger than the 1e-9 cross-layout threshold in the MPI companion test.
  double dmax = 0;
  double density_scale = 1.0;
  for (std::size_t k = 0; k < dis.dens.size() && k < rep.dens.size(); ++k) {
    dmax = std::fmax(dmax, std::fabs(dis.dens[k] - rep.dens[k]));
    density_scale =
        std::fmax(density_scale, std::fmax(std::fabs(dis.dens[k]), std::fabs(rep.dens[k])));
  }
  const double csum_dis = pops::test::checksum(dis.dens);
  const double csum_rep = pops::test::checksum(rep.dens);
  const double density_roundoff = 16.0 * std::numeric_limits<double>::epsilon() * density_scale;

  std::printf(
      "AMRDISTSERIAL npf_dist=%d npf_repl=%d | dmax=%.17e | csum_dist=%.17e csum_repl=%.17e | "
      "mass_dist=%.17e mass_repl=%.17e\n",
      dis.npf, rep.npf, dmax, csum_dis, csum_rep, dis.mass, rep.mass);

  chk(dmax <= density_roundoff,
      "distributed coarse density matches replicated coarse to machine round-off (single rank)");
  chk(std::isfinite(dis.mass) && std::isfinite(rep.mass), "accepted mass finite in both modes");
  chk(std::fabs(dis.mass - rep.mass) < 1e-12,
      "accepted mass distributed == replicated to round-off (single rank)");

  if (chk.fails() == 0)
    std::printf(
        "OK test_amr_distribute_coarse_serial (ADC-620: distribute_coarse=true hierarchy builds "
        "and rejects unprepared multi-level advance without mutation)\n");
  return chk.failed();
}

TEST(test_amr_distribute_coarse_serial, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_amr_distribute_coarse_serial,
                                    "test_amr_distribute_coarse_serial"),
            0);
}
