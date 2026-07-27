// Harness Kokkos de la VALIDATION INTEGREE AmrSystem + MPI (Cuda sur ROMEO, OpenMP pour la preuve
// hote) + MESURE DE STRONG-SCALING. Superset du test de regression
// tests/cpp/integration/mpi/test_mpi_amr_compiled_parity.cpp : meme cas
// (4 bulles, modele euler_poisson COMPILE via add_compiled_model, hierarchie AMR avec regrid +
// reflux + Poisson, niveau fin multi-patch distribue sur n_ranks()) MAIS instrumente pour la PERF du
// backend actif et la COMPARAISON grossier REPLIQUE vs REPARTI (le coeur du strong-scaling AMR) :
//   - imprime mass / csum / csumsq / cmax + verifie la consistance cross-rang (cmax, max insensible
//     a l'ordre, doit etre bit-identique a tous les np dans les DEUX modes) ;
//   - mesure le temps PAR PAS (apres warmup + Kokkos::fence pour capturer le travail device async)
//     pour le mode REPLIQUE puis le mode REPARTI, dans le MEME run -> le script compare per_step_ms
//     np=1/2/4 reparti vs replique et conclut sur le scaling.
//
// Argument : amrmpi_integrated [n] (defaut 128). Le grossier reparti utilise coarse_max_grid = n/2
// (2x2), le decoupage le moins agressif pour le MG geometrique (cf. la mesure de convergence : 2x2
// converge en autant de cycles que le mono-box, l'aggressif 8x8+ degrade).
//
// Lance par amrmpi_romeo_build.sh en srun -n 1/2/4 --gpus-per-task=1 (un GH200 par rang). Sous Cuda,
// for_each_cell est async ; density()/mass() de l'AmrSystem fencent en interne avant la lecture hote,
// et on encadre la mesure de temps par Kokkos::fence() pour ne pas sous-estimer le cout device.
#include <pops/physics/bricks/bricks.hpp>
#include <pops/physics/fluids/euler.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace pops;
using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

struct ZeroElliptic {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};

using MagneticModel = CompositeModel<Euler, MagneticLorentzForce, ZeroElliptic>;

static void install_forward_euler_program(AmrSystem& system) {
  std::vector<int> block_map(static_cast<std::size_t>(system.n_blocks()));
  std::iota(block_map.begin(), block_map.end(), 0);
  system.set_program_block_map(block_map);
  system.install_program_step([](double) {});
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    throw std::runtime_error("AMR MPI Kokkos harness requires the materialized runtime engine");

  auto context = std::make_shared<runtime::program::AmrProgramContext>(system.engine(), &system);
  context->configure_primary_clock("romeo.amrmpi.macro");
  context->install([context](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context](double level_dt) {
      context->set_stage_time(0, 1);
      (void)consume_solve_outcome(context->solve_fields());

      std::vector<MultiFab*> states;
      std::vector<MultiFab*> residuals;
      states.reserve(static_cast<std::size_t>(context->n_blocks()));
      residuals.reserve(static_cast<std::size_t>(context->n_blocks()));
      for (int block = 0; block < context->n_blocks(); ++block) {
        MultiFab& state = context->state(block);
        MultiFab& residual = context->rhs_scratch(1000 + block, 0, state);
        context->rhs_into(block, state, residual, 3000 + block);
        states.push_back(&state);
        residuals.push_back(&residual);
      }
      for (std::size_t block = 0; block < states.size(); ++block)
        context->axpy(*states[block], Real(level_dt), *residuals[block], Real(level_dt),
                      {{1, 1, 1}});
    });
  });
}

static std::vector<double> four_bubbles(int n) {
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
  // Periodic self-gravity has the constant nullspace. Preserve the four non-trivial peaks while
  // authoring their exact discrete neutralizing background (mean density = GravityCoupling rho0);
  // the prepared field solver must never project an incompatible RHS silently.
  const double mean =
      std::accumulate(rho.begin(), rho.end(), 0.0) / static_cast<double>(rho.size());
  for (double& value : rho)
    value += 1.0 - mean;
  return rho;
}

// Small ProgramGraph regression that keeps the former CUDA/CUDA+MPI B_z proof on the final
// temporal path. Both runs have the same genuinely refined hierarchy and initial state; only the
// prepared B_z field changes. The coarse and fine trajectories must both rotate +m_x toward -m_y.
static int run_bz_program_probe(int n) {
  const int me = my_rank();
  const std::vector<double> rho = four_bubbles(n);
  const std::size_t cells = static_cast<std::size_t>(n) * n;

  std::vector<double> state(4 * cells, 0.0);
  std::vector<double> bz(cells, 0.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const std::size_t cell = static_cast<std::size_t>(j) * n + i;
      const double x = (i + 0.5) / n;
      const double y = (j + 0.5) / n;
      state[cell] = rho[cell];
      state[cells + cell] = 1.0;
      state[3 * cells + cell] = 3.0;
      bz[cell] = 2.0 + 0.25 * std::sin(2.0 * 3.14159265358979323846 * x) *
                           std::cos(2.0 * 3.14159265358979323846 * y);
    }

  auto run = [&](const std::vector<double>& field) {
    AmrSystemConfig cfg;
    cfg.n = n;
    cfg.L = 1.0;
    cfg.periodicity = {true, true};
    cfg.regrid_every = 0;
    cfg.distribute_coarse = true;
    cfg.coarse_max_grid = n / 2;

    AmrSystem system(cfg);
    system.set_temporal_relations({2}, {1}, {"integral_only"});
    add_compiled_model(
        system, "magnetic",
        MagneticModel{Euler{Real(1.4)}, MagneticLorentzForce{Real(1)}, ZeroElliptic{}}, "none",
        "rusanov", "conservative", "euler", /*gamma=*/1.4);
    system.set_refinement(1.2);
    system.set_conservative_state("magnetic", state);
    system.set_magnetic_field(field);
    install_forward_euler_program(system);

    const int levels = system.n_levels();
    system.advance(0.01, 1);
    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(levels));
    for (int level = 0; level < levels; ++level)
      result.push_back(system.block_level_state_global("magnetic", level));
    return result;
  };

  const std::vector<double> zero_bz(cells, 0.0);
  const auto baseline = run(zero_bz);
  const auto magnetized = run(bz);

  int fails = 0;
  if (me == 0) {
    if (baseline.size() < 2 || magnetized.size() != baseline.size()) {
      std::printf("FAIL ProgramGraph B_z probe did not materialize the refined hierarchy\n");
      return 1;
    }
    for (std::size_t level = 0; level < baseline.size(); ++level) {
      if (baseline[level].size() != magnetized[level].size() || baseline[level].empty() ||
          baseline[level].size() % 4 != 0) {
        std::printf("FAIL ProgramGraph B_z probe invalid level %zu state\n", level);
        ++fails;
        continue;
      }
      const std::size_t level_cells = baseline[level].size() / 4;
      double max_delta = 0.0;
      double transverse_delta = 0.0;
      for (std::size_t cell = 0; cell < level_cells; ++cell) {
        for (int component = 0; component < 4; ++component) {
          const std::size_t index = static_cast<std::size_t>(component) * level_cells + cell;
          max_delta =
              std::max(max_delta, std::fabs(magnetized[level][index] - baseline[level][index]));
        }
        transverse_delta +=
            magnetized[level][2 * level_cells + cell] - baseline[level][2 * level_cells + cell];
      }
      transverse_delta /= static_cast<double>(level_cells);
      if (!(max_delta > 1e-3 && transverse_delta < -1e-3)) {
        std::printf(
            "FAIL ProgramGraph B_z level=%zu max_delta=%.3e transverse_delta=%.3e "
            "(B_z not consumed on device)\n",
            level, max_delta, transverse_delta);
        ++fails;
      }
    }
    if (fails == 0)
      std::printf("OK ProgramGraph B_z probe (coarse+fine, MPI-owned patches, exec=%s)\n",
                  Kokkos::DefaultExecutionSpace::name());
  }
  return fails;
}

// Un run complet pour un mode d'ownership donne (replique ou reparti). Imprime la signature du champ
// + per_step_ms (max sur les rangs). Renvoie le nombre d'echecs (rang 0). cmax (max, insensible a
// l'ordre) doit etre bit-identique cross-rang dans les DEUX modes ; les sommes additives ne le sont
// pas forcement quand le grossier est reparti (ordre de reduction FMA, documente pour #59).
static int run_mode(int n, bool distribute, const char* tag) {
  const int me = my_rank(), np = n_ranks();
  const std::vector<double> rho = four_bubbles(n);

  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};
  cfg.regrid_every = 8;
  cfg.distribute_coarse = distribute;  // reparti => grossier multi-box reparti (strong-scaling)
  cfg.coarse_max_grid = distribute ? n / 2 : 0;  // 2x2 : le decoupage qui ne degrade pas le MG

  AmrSystem sys(cfg);
  sys.set_temporal_relations({2}, {1}, {"integral_only"});
  add_compiled_model(sys, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                     "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
  sys.set_poisson("charge_density", "geometric_mg");
  sys.set_refinement(1.2);
  sys.set_density("gas", rho);
  install_forward_euler_program(sys);

  const double m0 = sys.mass();  // build paresseux (regrid initial distribue)
  const int np0 = sys.n_patches();

  const double dt = 5e-4;
  const int warmup = 4, measured = 40;
  for (int s = 0; s < warmup; ++s)
    sys.step(dt);  // warmup (JIT/cache/alloc)
  Kokkos::fence();
  const auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < measured; ++s)
    sys.step(dt);
  Kokkos::fence();  // capturer le travail device async avant de stopper le chrono
  const auto t1 = std::chrono::steady_clock::now();
  const double per_step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / measured;

  Kokkos::fence();
  const std::vector<double> dens = sys.density();  // reconstruit n*n (reparti : all_reduce interne)
  const double mass = sys.mass();
  const int npf = sys.n_patches();

  double csum = 0, csumsq = 0, cmax = 0;
  for (double v : dens) {
    csum += v;
    csumsq += v * v;
    const double a = std::fabs(v);
    if (a > cmax)
      cmax = a;
  }
  // cmax cross-rang : max insensible a l'ordre -> bit-identique attendu dans les deux modes.
  const double xmax = all_reduce_max(cmax), xmin = -all_reduce_max(-cmax);
  const double cmax_spread = xmax - xmin;
  const double maxstep = all_reduce_max(per_step_ms);  // pas le plus lent (le mur)

  int fails = 0;
  if (me == 0) {
    std::printf(
        "AMRMPI[%s] np=%d patches0=%d patchesF=%d | mass=%.17e | csum=%.17e csumsq=%.17e "
        "cmax=%.17e | cmax_crossrank_spread=%.3e\n",
        tag, np, np0, npf, mass, csum, csumsq, cmax, cmax_spread);
    std::printf(
        "AMRMPI[%s] exec=%s m0=%.17e (conservation: dm=%.3e) | per_step_ms=%.4f "
        "(max over ranks, n=%d, measured=%d)\n",
        tag, Kokkos::DefaultExecutionSpace::name(), m0, std::fabs(mass - m0), maxstep, n, measured);
    if (!(dens.size() == static_cast<std::size_t>(n) * n)) {
      std::printf("FAIL taille\n");
      ++fails;
    }
    if (!(cmax > 1e-6)) {
      std::printf("FAIL densite triviale\n");
      ++fails;
    }
    if (!(npf >= 2)) {
      std::printf("FAIL < 2 patchs fins\n");
      ++fails;
    }
    if (!(std::fabs(mass - m0) < 1e-9)) {
      std::printf("FAIL conservation (dm)\n");
      ++fails;
    }
    if (!(cmax_spread == 0.0)) {
      std::printf("FAIL cmax non bit-identique cross-rang\n");
      ++fails;
    }
    if (fails == 0)
      std::printf(
          "OK amrmpi_integrated[%s] np=%d (AmrSystem+MPI, exec=%s: cmax bit-identique "
          "cross-rang)\n",
          tag, np, Kokkos::DefaultExecutionSpace::name());
  }
  return fails;
}

int main(int argc, char** argv) {
  comm_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fails = 0;
  {
    int n = 128;  // grossier 128^2, fin 256^2 sous les patchs : charge GPU non triviale
    if (argc > 1)
      n = std::atoi(argv[1]);
    fails += run_bz_program_probe(std::max(16, n / 4));
    fails += run_mode(n, /*distribute=*/false, "replique");
    fails += run_mode(n, /*distribute=*/true, "reparti");
  }
  Kokkos::finalize();
  comm_finalize();
  return fails ? 1 : 0;
}
