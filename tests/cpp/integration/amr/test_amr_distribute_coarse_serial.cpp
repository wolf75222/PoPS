// SERIAL regression lock for ADC-620 (the old single-block builder paired the single-box fine seed with the
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
//   (3) mass/density DIGESTS match the replicated-coarse baseline (distribute_coarse=false) to a
//       strict machine-roundoff bound. Both layouts have the same owner at np=1, but the multi-tile
//       field solver has a different local reduction tree than the replicated mono-box layout.
#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include "test_harness.hpp"  // pops::test::Checker, checksum
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, GravityForce, GravityCoupling
#include <pops/physics/fluids/euler.hpp>   // EulerND
#include <pops/runtime/amr/field_solver_options.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>  // add_compiled_model(AmrSystem, ...)
#include <pops/runtime/amr_system.hpp>

#include "amr_tagging_test_authority.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

template <int Dim>
std::size_t cell_count(const Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
std::vector<double> four_bubbles(const Extent<Dim>& shape) {
  std::vector<double> rho(cell_count(shape));
  for (std::size_t ordinal = 0; ordinal < rho.size(); ++ordinal) {
    std::size_t remainder = ordinal;
    double radius_squared[4]{};
    for (int axis = 0; axis < Dim; ++axis) {
      const auto extent = static_cast<std::size_t>(shape[axis]);
      const double coordinate =
          (static_cast<double>(remainder % extent) + 0.5) / static_cast<double>(extent);
      remainder /= extent;
      for (int bubble = 0; bubble < 4; ++bubble) {
        const double center = ((bubble >> (axis % 2)) & 1) == 0 ? 0.25 : 0.75;
        const double delta = coordinate - center;
        radius_squared[bubble] += delta * delta;
      }
    }
    double value = 1.0;
    for (double radius : radius_squared)
      value += 0.5 * std::exp(-radius / 0.004);
    rho[ordinal] = value;
  }
  // A periodic Poisson operator has a constant nullspace. Keep the four non-trivial peaks while
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

template <int Dim>
using Model = CompositeModel<EulerND<Dim>, GravityForceND<Dim>, GravityCoupling>;

template <int Dim>
Model<Dim> gravity_model() {
  Model<Dim> model{};
  model.hyp = EulerND<Dim>::prepare(Real(1.4));
  model.src = GravityForceND<Dim>{};
  model.ell = GravityCoupling{Real(-1.0), Real(1.0), Real(1.0)};
  return model;
}

template <int Dim>
std::vector<double> gravity_state(const std::vector<double>& density) {
  using Gas = EulerND<Dim>;
  const std::size_t cells = density.size();
  std::vector<double> state(static_cast<std::size_t>(Gas::n_vars) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[static_cast<std::size_t>(Gas::density_component) * cells + cell] = density[cell];
    state[static_cast<std::size_t>(Gas::energy_component) * cells + cell] = 2.5;
  }
  return state;
}

constexpr const char* kGravityConsumerQid = "tests.amr-distribute-coarse/gas/physical-flux@1";

AmrFieldSolverOptions gravity_field_solver_options() {
  CompositeFacOptions fac;
  fac.abs_tol = Real(1e-10);
  fac.coarse_abs_tol = Real(1e-12);
  return geometric_mg_amr_field_solver_options(GeometricMgOptions{}, fac);
}

template <int Dim>
void install_default_gravity_field_output(AmrSystem<Dim>& system) {
  using namespace runtime::system;
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "field", "scalar"};
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 2;

  std::vector<AuxiliaryComponentKey> keys;
  std::vector<AuxiliaryOutput<Dim>> outputs;
  keys.reserve(static_cast<std::size_t>(Dim + 1));
  outputs.reserve(static_cast<std::size_t>(Dim + 1));
  for (int component = 0; component <= Dim; ++component) {
    AuxiliaryComponentKey key{
        "tests.amr-distribute-coarse/gravity-output@1", "field", "fields_from_state",
        component == 0 ? "potential" : "gradient-" + std::to_string(component - 1)};
    keys.push_back(key);
    outputs.push_back({std::move(key), contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "tests.amr-distribute-coarse/gravity-field-output@1",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      std::move(outputs),
      {}});
  AuxiliaryConsumerProviderPlan<Dim> consumer{kGravityConsumerQid, {}};
  consumer.values.reserve(static_cast<std::size_t>(Dim));
  for (int axis = 0; axis < Dim; ++axis)
    consumer.values.push_back({{keys[static_cast<std::size_t>(axis + 1)], contract, shape},
                               static_cast<std::size_t>(axis)});
  system.install_auxiliary_consumer_plan(std::move(consumer));
  system.seal_auxiliary_providers();
  system.register_default_elliptic_field_output(keys, 1);
}

template <int Dim>
AmrSystemConfig<Dim> make_config(int n, bool distribute) {
  AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = n;
    config.periodicity[axis] = true;
  }
  config.regrid_every = 4;
  config.distribute_coarse = distribute;
  return config;
}

struct Result {
  std::vector<double> dens;
  double mass, m0;
  int npf;
};

// Builds an AmrSystem (4 bubbles, euler_poisson compiled), advances nsteps, returns the coarse density
// (one flattened exact-rank coarse image, single-rank so already global), the final mass and m0.
// distribute=true exercises the ADC-620 path (coupler_make_coarse_layout splits the coarse, the fine
// seed used to borrow that dmap).
template <int Dim>
Result run(int n, int nsteps, double dt, bool distribute) {
  const AmrSystemConfig<Dim> cfg = make_config<Dim>(n, distribute);
  const std::vector<double> rho = four_bubbles(cfg.shape);

  AmrSystem<Dim> sys(cfg);
  test::install_amr_runtime_authority(sys, distribute
                                               ? "test.amr-distribute-coarse.distributed-runtime"
                                               : "test.amr-distribute-coarse.replicated-runtime");
  // Temporal subcycling is an independent execution authority: spell it out even though this
  // regression happens to use the same ratio as the spatial hierarchy.  The runtime must never
  // infer a clock relation from mesh refinement.
  sys.set_temporal_relations({2}, {1}, {"integral_only"});
  sys.install_block_state_route("gas", "tests.amr-distribute-coarse/gas/state@1");
  add_compiled_model<Dim>(sys, "gas", gravity_model<Dim>(), "minmod", "rusanov", "conservative",
                          "explicit", /*gamma=*/1.4, 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false, kGravityConsumerQid);
  sys.set_poisson("charge_density", "geometric_mg", "periodic", gravity_field_solver_options());
  install_default_gravity_field_output(sys);
  test::install_prepared_threshold_union(sys, {{"gas", "rho", 1.2}});
  sys.set_conservative_state("gas", gravity_state<Dim>(rho));
  test::install_forward_euler_program(sys, true);

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
  constexpr int Dim = kNativeDimension;
  const int nsteps = 16;
  const double dt = 1e-3;

  // (1) the hierarchy BUILDS: pre-ADC-620, this constructor + the first step() aborted via the
  // MultiFab layout check (box_array.size=1, dmap.size=4) even at a single rank (np1 aborted too: it is
  // a metadata mismatch, not a rank-count issue). If it still aborts, the process terminates here rather
  // than reaching the assertions below -- the regression lock is the process surviving construction and
  // stepping at all, on top of the checks that follow.
  const Result dis = run<Dim>(n, nsteps, dt, /*distribute=*/true);
  const Result rep =
      run<Dim>(n, nsteps, dt, /*distribute=*/false);  // oracle: replicated coarse (unaffected)

  const AmrSystemConfig<Dim> probe_cfg = make_config<Dim>(n, true);
  chk(dis.dens.size() == cell_count(probe_cfg.shape),
      "distributed coarse density has the exact-rank cell count");
  chk(rep.dens.size() == dis.dens.size(), "replicated coarse density same size as distributed");

  // (2) a CFL step advances by a finite, positive dt (no NaN/Inf from a corrupted layout).
  AmrSystem<Dim> probe(probe_cfg);
  test::install_amr_runtime_authority(probe, "test.amr-distribute-coarse.probe-runtime");
  probe.set_temporal_relations({2}, {1}, {"integral_only"});
  probe.install_block_state_route("gas", "tests.amr-distribute-coarse/gas/state@1");
  add_compiled_model<Dim>(probe, "gas", gravity_model<Dim>(), "minmod", "rusanov", "conservative",
                          "explicit", /*gamma=*/1.4, 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false, kGravityConsumerQid);
  probe.set_poisson("charge_density", "geometric_mg", "periodic", gravity_field_solver_options());
  install_default_gravity_field_output(probe);
  test::install_prepared_threshold_union(probe, {{"gas", "rho", 1.2}});
  probe.set_conservative_state("gas", gravity_state<Dim>(four_bubbles(probe_cfg.shape)));
  test::install_forward_euler_program(probe, true);
  const double dt_cfl = probe.step_cfl(0.4);
  chk(std::isfinite(dt_cfl), "distribute_coarse step_cfl returns a finite dt");
  chk(dt_cfl > 0.0, "distribute_coarse step_cfl returns a positive dt");

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
  chk(std::isfinite(dis.mass) && std::isfinite(rep.mass), "final mass finite in both modes");
  chk(std::fabs(dis.mass - dis.m0) < 1e-10, "mass conserved (distribute_coarse=true)");
  chk(std::fabs(rep.mass - rep.m0) < 1e-10, "mass conserved (distribute_coarse=false, oracle)");
  // The scalar mass is a separate reduction whose tile-traversal order also differs between the
  // multi-tile distributed-coarse layout and the mono-box replicated layout; non-associative float
  // addition makes the two masses agree only to round-off, not bit-for-bit.
  chk(std::fabs(dis.mass - rep.mass) < 1e-12,
      "final mass distributed == replicated to round-off (single rank)");

  if (chk.fails() == 0)
    std::printf(
        "OK test_amr_distribute_coarse_serial (ADC-620: distribute_coarse=true hierarchy builds "
        "and "
        "steps on a single rank, round-off equivalent to replicated coarse)\n");
  return chk.failed();
}

TEST(test_amr_distribute_coarse_serial, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_amr_distribute_coarse_serial,
                                    "test_amr_distribute_coarse_serial"),
            0);
}
