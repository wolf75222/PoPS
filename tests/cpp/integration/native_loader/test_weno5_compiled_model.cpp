// WENO5 sur le chemin COMPILE NATIF (add_compiled_model, le gabarit en-tete qu'inline le loader .so
// de production / add_native_block). Le verrou : add_compiled_model alloue desormais l'etat du bloc a
// block_n_ghost(limiter) APRES install_block (3 ghosts pour weno5), comme add_block (PR #88) ; sans
// cette largeur, assemble_rhs lirait hors bornes le stencil 5 points de WENO5.
//
// On exige que le vrai paquet compile installe WENO5 avec exactement trois ghosts et consomme les
// gradients de potentiel exact-rank publies par le solveur cartesian_cg. L'etat a pression uniforme
// rend la divergence Euler nulle : les residus de quantite de mouvement non triviaux prouvent donc
// la voie WENO + GravityForceND, et non une comparaison tautologique avec le meme preparateur.
#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, GravityForceND, GravityCoupling
#include <pops/physics/fluids/euler.hpp>   // EulerND
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int Dim = kNativeDimension;
using NativeSystem = System<Dim>;
using NativeSystemConfig = SystemConfig<Dim>;
using Model = CompositeModel<EulerND<Dim>, GravityForceND<Dim>, GravityCoupling>;

constexpr const char* kGravityFieldSlot = "test.weno5-compiled/gravity-field";
constexpr const char* kGravityField = "gravity-potential";

template <int RuntimeDim>
void install_runtime_authority(System<RuntimeDim>& system, std::string_view identity) {
  auto lane =
      std::make_shared<ExecutionLane>(ExecutionLane::duplicate_world_collectively(identity));
  system.install_prepared_boundary_execution_lane(std::move(lane));
}

Model gravity_model(double rho0) {
  Model model{};
  model.hyp = EulerND<Dim>::prepare(Real(1.4));
  model.src = GravityForceND<Dim>{};
  model.ell = GravityCoupling{Real(-1.0), Real(1.0), static_cast<Real>(rho0)};
  return model;
}

NativeSystemConfig native_config(int n, double length) {
  NativeSystemConfig config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = n;
    config.lower[axis] = Real(0);
    config.upper[axis] = static_cast<Real>(length);
    config.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  config.boxes = {Box<Dim>::from_extents(config.shape)};
  return config;
}

std::size_t cell_count(int n) {
  std::size_t count = 1;
  for (int axis = 0; axis < Dim; ++axis)
    count *= static_cast<std::size_t>(n);
  return count;
}

std::vector<double> hydrostatic_initial_state(const std::vector<double>& rho) {
  std::vector<double> state(static_cast<std::size_t>(Model::n_vars) * rho.size(), 0.0);
  std::copy(rho.begin(), rho.end(), state.begin());
  const double energy = 1.0 / (1.4 - 1.0);
  for (std::size_t cell = 0; cell < rho.size(); ++cell)
    state[static_cast<std::size_t>(Dim + 1) * rho.size() + cell] = energy;
  return state;
}

std::vector<runtime::system::AuxiliaryComponentKey> install_gravity_field_authority(
    NativeSystem& system) {
  using namespace runtime::system;

  constexpr const char* backend = "test.weno5-compiled/cartesian-cg";
  system.register_configured_field_solver_provider(
      "cartesian_cg", backend,
      PreparedProviderOptions{
          "pops.system.cartesian-cg-options@1",
          {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{200}}, {"rel_tol", 1.0e-8}}});
  system.set_field_solver_plan(kGravityFieldSlot, "test.weno5-compiled/gravity-plan@1",
                               "test.weno5-compiled/gravity-provider@1",
                               "test.weno5-compiled/gravity-output@1", "gas", kGravityField,
                               {"test.weno5-compiled/gas/gravity-rhs@1"}, {"gas"}, {kGravityField},
                               {1.0}, backend);
  system.set_field_topology_authority(kGravityFieldSlot, "builtin_rectangular_cell_graph_v1",
                                      "test.weno5-compiled/periodic-cartesian@1",
                                      "test.weno5-compiled/periodic-cartesian:v1");
  const std::vector<std::string> periodic_faces(static_cast<std::size_t>(2 * Dim), "periodic");
  const std::vector<double> zero_faces(static_cast<std::size_t>(2 * Dim), 0.0);
  system.set_field_boundary_plan(kGravityFieldSlot, periodic_faces, zero_faces, zero_faces,
                                 zero_faces);

  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "field", "scalar"};
  std::vector<AuxiliaryOutput<Dim>> outputs;
  std::vector<AuxiliaryComponentKey> keys;
  outputs.reserve(static_cast<std::size_t>(Dim + 1));
  keys.reserve(static_cast<std::size_t>(Dim + 1));
  for (int component = 0; component <= Dim; ++component) {
    AuxiliaryComponentKey key{
        "test.weno5-compiled/gravity-output@1", "field", kGravityField,
        component == 0 ? "potential" : "gradient-" + std::to_string(component - 1)};
    keys.push_back(key);
    outputs.push_back({std::move(key), contract, shape});
  }
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "test.weno5-compiled/gravity-field-output@1",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      std::move(outputs),
      {}});

  AuxiliaryConsumerProviderPlan<Dim> gas_plan{"gas", {}};
  gas_plan.values.reserve(static_cast<std::size_t>(Dim));
  for (int axis = 0; axis < Dim; ++axis)
    gas_plan.values.push_back({{keys[static_cast<std::size_t>(axis + 1)], contract, shape},
                               static_cast<std::size_t>(axis)});
  system.install_auxiliary_consumer_plan(std::move(gas_plan));
  system.seal_auxiliary_providers();
  return keys;
}

// Densite lisse (bulle gaussienne) : transport regulier ou WENO5 est pleinement actif.
std::vector<double> smooth_rho(int n) {
  std::vector<double> rho(cell_count(n));
  for (std::size_t cell = 0; cell < rho.size(); ++cell) {
    std::size_t remaining = cell;
    double radius_squared = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const int index = static_cast<int>(remaining % static_cast<std::size_t>(n));
      remaining /= static_cast<std::size_t>(n);
      const double coordinate = (static_cast<double>(index) + 0.5) / n - 0.5;
      radius_squared += coordinate * coordinate;
    }
    rho[cell] = 1.0 + 0.3 * std::exp(-radius_squared / 0.02);
  }
  return rho;
}

struct CompiledRun {
  Extent<Dim> ghosts{};
  std::vector<double> residual;
  std::vector<double> potential;
  std::vector<std::vector<double>> gradients;
};

// Real public compiled-package route: prepare, install, field-output publication, then residual.
CompiledRun run_compiled(int n, double L, const std::vector<double>& rho, double rho0) {
  NativeSystem sys(native_config(n, L));
  install_runtime_authority(sys, "test.weno5-compiled/runtime@1");
  sys.install_block_state_route("gas", "test.weno5-compiled/gas/state");
  sys.set_poisson("charge_density", "cartesian_cg");
  const auto field_outputs = install_gravity_field_authority(sys);
  const Model model = gravity_model(rho0);
  add_compiled_model(sys, "gas", model, "weno5", "rusanov", "conservative", "explicit",
                     /*gamma=*/1.4);
  sys.register_elliptic_field("gas", kGravityField, field_outputs, 1);
  sys.set_block_elliptic_field("gas", kGravityField,
                               [model](const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
                                 add_model_elliptic_rhs(model, state, rhs);
                               });
  sys.set_state("gas", hydrostatic_initial_state(rho));
  // Materialize the diagnostic input before invoking the public field publication route.  The
  // latter is a writer transaction and must not run while an accepted read view is held.
  const MultiFab<Dim> state = [&] {
    const auto state_view = sys.block_state(0);
    return MultiFab<Dim>(*state_view.get());
  }();
  (void)pops::consume_solve_outcome(sys.solve_fields_from_state(kGravityField, 0, state));

  CompiledRun result;
  result.ghosts = state.ghosts();
  result.residual = sys.eval_rhs("gas");
  result.potential = sys.field_potential_global(kGravityFieldSlot);
  result.gradients.reserve(static_cast<std::size_t>(Dim));
  for (int axis = 0; axis < Dim; ++axis)
    result.gradients.push_back(
        sys.auxiliary_component(field_outputs[static_cast<std::size_t>(axis + 1)]));
  return result;
}

double max_abs(const std::vector<double>& values) {
  double result = 0.0;
  for (double value : values)
    result = std::fmax(result, std::fabs(value));
  return result;
}

double max_abs_component(const std::vector<double>& values, int component, std::size_t cells) {
  double result = 0.0;
  const std::size_t first = static_cast<std::size_t>(component) * cells;
  for (std::size_t cell = 0; cell < cells; ++cell)
    result = std::fmax(result, std::fabs(values[first + cell]));
  return result;
}

}  // namespace

static int pops_run_test_weno5_compiled_model(int argc, char** argv) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int n = 48;
  const double L = 1.0;
  const std::vector<double> rho = smooth_rho(n);
  // The periodic Poisson operator has the constant nullspace.  Author its physical neutralizing
  // background from the exact discrete state used by both routes; the prepared solver must never
  // project the positive Gaussian mean silently.
  const double rho0 =
      std::accumulate(rho.begin(), rho.end(), 0.0) / static_cast<double>(rho.size());

  const CompiledRun run = run_compiled(n, L, rho, rho0);
  int fails = 0;
  for (int axis = 0; axis < Dim; ++axis) {
    if (run.ghosts[axis] != Weno5::n_ghost || run.ghosts[axis] != 3) {
      std::printf("FAIL WENO5 ghost contract on axis %d: got %lld, expected 3\n", axis,
                  static_cast<long long>(run.ghosts[axis]));
      ++fails;
    }
  }
  if (!(max_abs(run.potential) > 1e-6)) {
    std::printf("FAIL cartesian_cg gravity potential is trivial\n");
    ++fails;
  }
  for (int axis = 0; axis < Dim; ++axis) {
    if (!(max_abs(run.gradients[static_cast<std::size_t>(axis)]) > 1e-6)) {
      std::printf("FAIL gravity gradient provider is trivial on axis %d\n", axis);
      ++fails;
    }
    if (!(max_abs_component(run.residual, axis + 1, rho.size()) > 1e-6)) {
      std::printf("FAIL WENO5 gravity momentum residual is trivial on axis %d\n", axis);
      ++fails;
    }
  }

  if (fails == 0)
    std::printf("OK test_weno5_compiled_model (WENO5 ghosts=3, %d gravity gradients)\n", Dim);
  return fails == 0 ? 0 : 1;
}

TEST(test_weno5_compiled_model, Runs) {
  EXPECT_EQ(
      pops::test::RunTestBody(&pops_run_test_weno5_compiled_model, "test_weno5_compiled_model"), 0);
}
