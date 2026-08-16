// Harness Kokkos de la VALIDATION INTEGREE AmrSystem + MPI (Cuda sur ROMEO, OpenMP pour la preuve
// hote) + MESURE DE STRONG-SCALING. Superset du test de regression
// tests/cpp/integration/mpi/test_mpi_amr_compiled_parity.cpp : meme cas
// (lattice de pics exacte-rang, modele euler_poisson COMPILE via add_compiled_model, hierarchie AMR avec regrid +
// reflux + Poisson, niveau fin multi-patch distribue sur n_ranks()) MAIS instrumente pour la PERF du
// backend actif et la COMPARAISON grossier REPLIQUE vs REPARTI (le coeur du strong-scaling AMR) :
//   - imprime mass / csum / csumsq / cmax + verifie la consistance cross-rang (cmax, max insensible
//     a l'ordre, doit etre bit-identique a tous les np dans les DEUX modes) ;
//   - mesure le temps PAR PAS (apres warmup + Kokkos::fence pour capturer le travail device async)
//     pour le mode REPLIQUE puis le mode REPARTI, dans le MEME run -> le script compare per_step_ms
//     np=1/2/4 reparti vs replique et conclut sur le scaling.
//
// Argument : amrmpi_integrated [n] (defaut 128). Le grossier reparti utilise coarse_max_grid = n/2
// sur chaque axe, le decoupage le moins agressif pour le MG geometrique.
//
// Lance par amrmpi_romeo_build.sh en srun -n 1/2/4 --gpus-per-task=1 (un GH200 par rang). Sous Cuda,
// for_each_cell est async ; density()/mass() de l'AmrSystem fencent en interne avant la lecture hote,
// et on encadre la mesure de temps par Kokkos::fence() pour ne pas sous-estimer le cout device.
#include <pops/physics/bricks/bricks.hpp>
#include <pops/physics/fluids/euler.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include "amr_tagging_test_authority.hpp"
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pops;
constexpr int kDim = kNativeDimension;
using NativeAmrSystem = AmrSystem<kDim>;
using NativeMultiFab = MultiFab<kDim>;
using Model = CompositeModel<EulerND<kDim>, GravityForceND<kDim>, GravityCoupling>;
using MagneticModel = CompositeModel<EulerND<kDim>, MagneticLorentzForceND<kDim>, NoElliptic>;

namespace aux = runtime::system;

static std::size_t cell_count(int n) {
  std::size_t result = 1;
  for (int axis = 0; axis < kDim; ++axis)
    result *= static_cast<std::size_t>(n);
  return result;
}

static void configure_domain(AmrSystemConfig<kDim>& cfg, int n) {
  for (int axis = 0; axis < kDim; ++axis) {
    cfg.shape[axis] = n;
    cfg.lower[axis] = Real(0);
    cfg.upper[axis] = Real(1);
    cfg.periodicity[axis] = true;
  }
}

static void install_forward_euler_program(NativeAmrSystem& system) {
  std::vector<int> block_map(static_cast<std::size_t>(system.n_blocks()));
  std::iota(block_map.begin(), block_map.end(), 0);
  system.set_program_block_map(block_map);
  system.install_program_step([](double) {});
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    throw std::runtime_error("AMR MPI Kokkos harness requires the materialized runtime engine");

  auto context =
      std::make_shared<runtime::program::AmrProgramContext<kDim>>(system.engine(), &system);
  context->configure_primary_clock("romeo.amrmpi.macro");
  context->install([context](double macro_dt) {
    context->advance_hierarchy(macro_dt, [context](double level_dt) {
      context->set_stage_time(0, 1);
      if (context->level() == 0)
        (void)consume_solve_outcome(context->solve_default_field_on_coarse_level());

      std::vector<NativeMultiFab*> states;
      std::vector<NativeMultiFab*> residuals;
      states.reserve(static_cast<std::size_t>(context->n_blocks()));
      residuals.reserve(static_cast<std::size_t>(context->n_blocks()));
      for (int block = 0; block < context->n_blocks(); ++block) {
        NativeMultiFab& state = context->state(block);
        NativeMultiFab& residual = context->rhs_scratch(1000 + block, 0, state);
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

static std::vector<double> gaussian_lattice(int n) {
  std::vector<double> rho(cell_count(n));
  constexpr int bubble_count = 1 << kDim;
  for (std::size_t ordinal = 0; ordinal < rho.size(); ++ordinal) {
    std::array<double, kDim> position{};
    std::size_t encoded = ordinal;
    for (int axis = 0; axis < kDim; ++axis) {
      const int coordinate = static_cast<int>(encoded % static_cast<std::size_t>(n));
      encoded /= static_cast<std::size_t>(n);
      position[static_cast<std::size_t>(axis)] = (coordinate + 0.5) / n;
    }
    double value = 1.0;
    for (int bubble = 0; bubble < bubble_count; ++bubble) {
      double radius_squared = 0.0;
      for (int axis = 0; axis < kDim; ++axis) {
        const double center = ((bubble >> axis) & 1) == 0 ? 0.25 : 0.75;
        const double delta = position[static_cast<std::size_t>(axis)] - center;
        radius_squared += delta * delta;
      }
      value += 0.5 * std::exp(-radius_squared / 0.004);
    }
    rho[ordinal] = value;
  }
  // Periodic self-gravity has the constant nullspace. Preserve the non-trivial lattice while
  // authoring their exact discrete neutralizing background (mean density = GravityCoupling rho0);
  // the prepared field solver must never project an incompatible RHS silently.
  const double mean =
      std::accumulate(rho.begin(), rho.end(), 0.0) / static_cast<double>(rho.size());
  for (double& value : rho)
    value += 1.0 - mean;
  return rho;
}

static std::array<aux::AuxiliaryComponentKey, 3> install_magnetic_provider(
    NativeAmrSystem& system) {
  const aux::AuxiliaryComponentContract contract{"cell-average", "cell", "magnetic-field", "input",
                                                 "scalar"};
  aux::AuxiliaryStorageShape<kDim> shape;
  for (int axis = 0; axis < kDim; ++axis)
    shape.halo[axis] = 2;
  std::array<aux::AuxiliaryComponentKey, 3> keys{
      aux::AuxiliaryComponentKey{"romeo.amrmpi", "input", "magnetic", "B-x"},
      aux::AuxiliaryComponentKey{"romeo.amrmpi", "input", "magnetic", "B-y"},
      aux::AuxiliaryComponentKey{"romeo.amrmpi", "input", "magnetic", "B-z"}};
  std::vector<aux::AuxiliaryOutput<kDim>> outputs;
  outputs.reserve(keys.size());
  for (const auto& key : keys)
    outputs.push_back({key, contract, shape});
  system.install_prepared_auxiliary_provider(aux::PreparedAuxiliaryProvider<kDim>{
      "romeo.amrmpi.magnetic-input@1",
      aux::AuxiliaryProviderKind::input,
      {aux::AuxiliaryEvaluationEvent::initialization, aux::AuxiliaryFreshness::once},
      std::move(outputs),
      {}});
  aux::AuxiliaryConsumerProviderPlan<kDim> plan;
  plan.consumer_qid = "romeo.amrmpi.magnetic-consumer@1";
  for (std::size_t slot = 0; slot < keys.size(); ++slot)
    plan.values.push_back({{keys[slot], contract, shape}, slot});
  system.install_auxiliary_consumer_plan(std::move(plan));
  system.seal_auxiliary_providers();
  return keys;
}

static std::vector<aux::AuxiliaryComponentKey> install_gravity_field_provider(
    NativeAmrSystem& system) {
  const aux::AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "amr-field",
                                                 "scalar"};
  aux::AuxiliaryStorageShape<kDim> shape;
  for (int axis = 0; axis < kDim; ++axis)
    shape.halo[axis] = 2;

  std::vector<aux::AuxiliaryComponentKey> keys;
  keys.reserve(static_cast<std::size_t>(kDim + 1));
  keys.emplace_back("pops.amr.default-field", "field", "fields_from_state", "potential");
  for (int axis = 0; axis < kDim; ++axis)
    keys.emplace_back("pops.amr.default-field", "field", "fields_from_state",
                      "gradient-" + std::to_string(axis));

  std::vector<aux::AuxiliaryOutput<kDim>> outputs;
  outputs.reserve(keys.size());
  for (const auto& key : keys)
    outputs.push_back({key, contract, shape});
  system.install_prepared_auxiliary_provider(aux::PreparedAuxiliaryProvider<kDim>{
      "romeo.amrmpi.default-field-output@1",
      aux::AuxiliaryProviderKind::field_output,
      {aux::AuxiliaryEvaluationEvent::before_field_solve, aux::AuxiliaryFreshness::evaluation},
      std::move(outputs),
      {}});

  aux::AuxiliaryConsumerProviderPlan<kDim> plan;
  plan.consumer_qid = "romeo.amrmpi.gravity-consumer@1";
  for (int axis = 0; axis < kDim; ++axis)
    plan.values.push_back({{keys[static_cast<std::size_t>(axis + 1)], contract, shape},
                           static_cast<std::size_t>(axis)});
  system.install_auxiliary_consumer_plan(std::move(plan));
  system.seal_auxiliary_providers();
  return keys;
}

// ProgramGraph regression on the final CUDA/CUDA+MPI path. Both runs own the same refined
// hierarchy and state; only the three-component, owner-qualified magnetic provider changes. The
// observed response is the exact native-rank projection of momentum x B.
static int run_magnetic_provider_program_probe(int n) {
  const int me = my_rank();
  const std::vector<double> rho = gaussian_lattice(n);
  const std::size_t cells = cell_count(n);

  constexpr int components = kDim + 2;
  std::vector<double> state(static_cast<std::size_t>(components) * cells, 0.0);
  std::array<std::vector<double>, 3> magnetic;
  for (auto& component : magnetic)
    component.assign(cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const double x = (static_cast<double>(cell % static_cast<std::size_t>(n)) + 0.5) / n;
    state[cell] = rho[cell];
    for (int axis = 0; axis < kDim; ++axis)
      state[static_cast<std::size_t>(axis + 1) * cells + cell] = 0.25 * (axis + 1);
    state[static_cast<std::size_t>(kDim + 1) * cells + cell] = 3.0;
    magnetic[0][cell] = 0.5 + 0.05 * std::cos(2.0 * 3.14159265358979323846 * x);
    magnetic[1][cell] = -0.75 + 0.05 * std::sin(2.0 * 3.14159265358979323846 * x);
    magnetic[2][cell] = 2.0 + 0.25 * std::sin(2.0 * 3.14159265358979323846 * x);
  }

  auto run = [&](const std::array<std::vector<double>, 3>& field) {
    AmrSystemConfig<kDim> cfg;
    configure_domain(cfg, n);
    cfg.regrid_every = 0;
    cfg.distribute_coarse = true;
    for (int axis = 0; axis < kDim; ++axis)
      cfg.coarse_max_grid[axis] = n / 2;

    NativeAmrSystem system(cfg);
    system.set_temporal_relations({2}, {1}, {"integral_only"});
    const auto keys = install_magnetic_provider(system);
    add_compiled_model<kDim>(
        system, "magnetic",
        MagneticModel{
            {}, EulerND<kDim>{Real(1.4)}, MagneticLorentzForceND<kDim>{Real(1)}, NoElliptic{}},
        "none", "rusanov", "conservative", "euler", /*gamma=*/1.4, /*substeps=*/1,
        /*stride=*/1, {}, {}, /*positivity_floor=*/0.0,
        /*weno_epsilon=*/static_cast<double>(kWenoEpsilon), /*wave_speed_cache=*/false,
        "romeo.amrmpi.magnetic-consumer@1");
    test::install_prepared_threshold_union(system, {{"magnetic", "rho", 1.2}});
    system.set_conservative_state("magnetic", state);
    for (std::size_t component = 0; component < keys.size(); ++component)
      system.stage_auxiliary_input(keys[component], field[component]);
    install_forward_euler_program(system);
    system.refresh_auxiliary(aux::AuxiliaryEvaluationPoint{
        "romeo.amrmpi", 0, 0, 0, 0, 0, 0, aux::AuxiliaryEvaluationEvent::initialization});

    const int levels = system.n_levels();
    system.advance(0.01, 1);
    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(levels));
    for (int level = 0; level < levels; ++level)
      result.push_back(system.block_level_state_global("magnetic", level));
    return result;
  };

  std::array<std::vector<double>, 3> zero_magnetic;
  for (auto& component : zero_magnetic)
    component.assign(cells, 0.0);
  const auto baseline = run(zero_magnetic);
  const auto magnetized = run(magnetic);

  int fails = 0;
  if (me == 0) {
    if (baseline.size() < 2 || magnetized.size() != baseline.size()) {
      std::printf("FAIL ProgramGraph magnetic-provider probe did not refine the hierarchy\n");
      return 1;
    }
    for (std::size_t level = 0; level < baseline.size(); ++level) {
      if (baseline[level].size() != magnetized[level].size() || baseline[level].empty() ||
          baseline[level].size() % components != 0) {
        std::printf("FAIL ProgramGraph magnetic-provider probe invalid level %zu state\n", level);
        ++fails;
        continue;
      }
      const std::size_t level_cells = baseline[level].size() / components;
      double max_delta = 0.0;
      double transverse_delta = 0.0;
      for (std::size_t cell = 0; cell < level_cells; ++cell) {
        for (int component = 0; component < components; ++component) {
          const std::size_t index = static_cast<std::size_t>(component) * level_cells + cell;
          max_delta =
              std::max(max_delta, std::fabs(magnetized[level][index] - baseline[level][index]));
        }
        const int observed_momentum = kDim > 1 ? 2 : 1;
        transverse_delta +=
            magnetized[level][static_cast<std::size_t>(observed_momentum) * level_cells + cell] -
            baseline[level][static_cast<std::size_t>(observed_momentum) * level_cells + cell];
      }
      transverse_delta /= static_cast<double>(level_cells);
      const bool has_transverse_projection = kDim > 1;
      const bool projection_matches =
          has_transverse_projection ? max_delta > 1e-3 && transverse_delta < -1e-3
                                    : max_delta < 1e-12 && std::fabs(transverse_delta) < 1e-12;
      if (!projection_matches) {
        std::printf(
            "FAIL ProgramGraph magnetic-provider level=%zu max_delta=%.3e "
            "projected_delta=%.3e (three-component provider projection incorrect on device)\n",
            level, max_delta, transverse_delta);
        ++fails;
      }
    }
    if (fails == 0)
      std::printf(
          "OK ProgramGraph magnetic-provider probe Dim=%d (coarse+fine, MPI-owned patches, "
          "exec=%s)\n",
          kDim, Kokkos::DefaultExecutionSpace::name());
  }
  return fails;
}

// Un run complet pour un mode d'ownership donne (replique ou reparti). Imprime la signature du champ
// + per_step_ms (max sur les rangs). Renvoie le nombre d'echecs (rang 0). cmax (max, insensible a
// l'ordre) doit etre bit-identique cross-rang dans les DEUX modes ; les sommes additives ne le sont
// pas forcement quand le grossier est reparti (ordre de reduction FMA, documente pour #59).
static int run_mode(int n, bool distribute, const char* tag) {
  const int me = my_rank(), np = n_ranks();
  const std::vector<double> rho = gaussian_lattice(n);

  AmrSystemConfig<kDim> cfg;
  configure_domain(cfg, n);
  cfg.regrid_every = 8;
  cfg.distribute_coarse = distribute;  // reparti => grossier multi-box reparti (strong-scaling)
  for (int axis = 0; axis < kDim; ++axis)
    cfg.coarse_max_grid[axis] = distribute ? n / 2 : 0;

  NativeAmrSystem sys(cfg);
  sys.set_temporal_relations({2}, {1}, {"integral_only"});
  const auto gravity_field_keys = install_gravity_field_provider(sys);
  add_compiled_model<kDim>(
      sys, "gas",
      Model{{}, EulerND<kDim>{1.4}, GravityForceND<kDim>{}, GravityCoupling{-1.0, 1.0, 1.0}},
      "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4, /*substeps=*/1,
      /*stride=*/1, {}, {}, /*positivity_floor=*/0.0,
      /*weno_epsilon=*/static_cast<double>(kWenoEpsilon), /*wave_speed_cache=*/false,
      "romeo.amrmpi.gravity-consumer@1");
  sys.set_poisson("charge_density", "geometric_mg");
  sys.register_elliptic_field("gas", "fields_from_state", gravity_field_keys, -1);
  test::install_prepared_threshold_union(sys, {{"gas", "rho", 1.2}});
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
  const std::vector<double> dens = sys.density();  // reconstruit n^Dim (all_reduce si reparti)
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
    if (!(dens.size() == cell_count(n))) {
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
    int n = 128;  // n^Dim grossier, (2n)^Dim fin sous les patchs : charge GPU non triviale
    if (argc > 1)
      n = std::atoi(argv[1]);
    fails += run_magnetic_provider_program_probe(std::max(16, n / 4));
    fails += run_mode(n, /*distribute=*/false, "replique");
    fails += run_mode(n, /*distribute=*/true, "reparti");
  }
  Kokkos::finalize();
  comm_finalize();
  return fails ? 1 : 0;
}
