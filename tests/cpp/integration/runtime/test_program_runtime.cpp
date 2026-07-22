// Compiled time-program runtime seam (epic ADC-399 / ADC-401 Phase 2b): a Forward-Euler Program,
// installed as a macro-step closure via pops::runtime::program::ProgramContext, runs C++-side during
// sim.step(dt). This test proves the seam end-to-end WITHOUT codegen or a .so: it builds the closure
// in C++ (the role the generated problem.so will later fill) and checks bit-parity against a reference
// Forward-Euler step computed from the SAME existing primitives (solve_fields + eval_rhs + U + dt*R).
//
// Model: a compressible Euler gas with a NON-UNIFORM pressure IC (u = v = 0), so -div F has a non-zero
// momentum component -> the step actually changes the state (parity is not vacuous). No source, no
// charge (NoEll), so the result is pure gas dynamics and deterministic across two System instances.

#include <gtest/gtest.h>

#include <pops/mesh/storage/multifab.hpp>
#include <pops/physics/bricks/source.hpp>                // NoSource
#include <pops/physics/composition/composite.hpp>        // CompositeModel
#include <pops/physics/fluids/euler.hpp>                 // Euler
#include <pops/runtime/builders/compiled/dsl_block.hpp>  // add_compiled_model
#include <pops/runtime/program/program_context.hpp>      // ProgramContext (the seam under test)
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

#if defined(POPS_HAS_KOKKOS)
static void ensure_kokkos() {
  static Kokkos::ScopeGuard guard;
  (void)guard;
}
#endif

// Elliptic brick that contributes nothing (no charge): the Poisson RHS stays zero, phi = 0, and the
// Euler flux ignores aux -> the residual is pure gas dynamics.
struct NoEll {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};
using GasModel = CompositeModel<Euler, NoSource, NoEll>;

struct UnitDensitySource {
  template <class State>
  POPS_HD State apply(const State&, const Aux&) const {
    State source{};
    source[0] = Real(1);
    return source;
  }
};
using SourcedGasModel = CompositeModel<Euler, UnitDensitySource, NoEll>;

struct ProjectingEuler : Euler {
  POPS_HD State project(const State& input, const Aux&) const {
    State output = input;
    output[0] = Real(2);
    return output;
  }
};
using ProjectingGasModel = CompositeModel<ProjectingEuler, NoSource, NoEll>;

struct DiffusiveGasModel : GasModel {
  POPS_HD Real diffusivity() const { return Real(0.1); }
};

static void fill_ic(std::vector<double>& U, int n, double gamma) {
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  const double pi = 3.14159265358979323846;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const std::size_t k =
          static_cast<std::size_t>(j) * n + i;  // j slow, i fast (get_state layout)
      const double x = (i + 0.5) / n, y = (j + 0.5) / n;
      const double p =
          3.0 + 0.5 * std::cos(2 * pi * x) * std::cos(2 * pi * y);  // periodic, non-uniform
      U[0 * nn + k] = 1.0;                                          // rho
      U[1 * nn + k] = 0.0;                                          // rho u
      U[2 * nn + k] = 0.0;                                          // rho v
      U[3 * nn + k] = p / (gamma - 1.0);                            // E (u = v = 0)
    }
}

static void add_gas(System& s, double gamma, const std::string& limiter = "minmod") {
  add_compiled_model(s, "gas", GasModel{Euler{gamma}, NoSource{}, NoEll{}}, limiter, "rusanov",
                     "conservative", "explicit", gamma);
  s.set_poisson("charge_density", "geometric_mg");
}

static void add_sourced_gas(System& system, double gamma) {
  add_compiled_model(system, "gas", SourcedGasModel{Euler{gamma}, UnitDensitySource{}, NoEll{}},
                     "none", "rusanov", "conservative", "explicit", gamma);
}

static void add_imex_sourced_gas(System& system, double gamma) {
  add_compiled_model(system, "gas", SourcedGasModel{Euler{gamma}, UnitDensitySource{}, NoEll{}},
                     "none", "rusanov", "conservative", "imex", gamma);
}

static void add_projecting_gas(System& system, double gamma) {
  ProjectingEuler transport;
  transport.gamma = gamma;
  add_compiled_model(system, "gas", ProjectingGasModel{transport, NoSource{}, NoEll{}}, "none",
                     "rusanov", "conservative", "explicit", gamma);
}

static void add_ssprk3_gas(System& system, double gamma) {
  add_compiled_model(system, "gas", GasModel{Euler{gamma}, NoSource{}, NoEll{}}, "none", "rusanov",
                     "conservative", "ssprk3", gamma);
}

static void add_diffusive_gas(System& system, double gamma) {
  DiffusiveGasModel model;
  model.hyp.gamma = gamma;
  add_compiled_model(system, "gas", model, "none", "rusanov", "conservative", "explicit", gamma);
}

TEST(ProgramRuntime, ForwardEulerProgramContextMatchesEvalRhsReferenceAndCountsKernels) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  const int n = 16;
  const double gamma = 1.4, dt = 1e-3;
  const std::size_t nn = static_cast<std::size_t>(n) * n;

  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  std::vector<double> U0(4 * nn);
  fill_ic(U0, n, gamma);

  // Reference: one Forward-Euler step via the existing primitives, combined on the host.
  System ref(cfg);
  add_gas(ref, gamma);
  ref.set_state("gas", U0);
  ref.solve_fields();
  const std::vector<double> R0 = ref.eval_rhs("gas");
  std::vector<double> Uref(4 * nn);
  for (std::size_t k = 0; k < Uref.size(); ++k)
    Uref[k] = U0[k] + dt * R0[k];

  // Program: the SAME step expressed as a ProgramContext closure and driven by sim.step(dt).
  System sim(cfg);
  add_gas(sim, gamma);
  sim.set_state("gas", U0);
  sim.set_program_block_map({0});

  runtime::program::ProgramContext ctx(&sim);
  ctx.configure_primary_clock("macro");
  ctx.install([ctx](double h) {
    ctx.begin_step(h);
    ctx.set_stage_time(0, 1);
    ctx.solve_fields();
    for (int b = 0; b < ctx.n_blocks(); ++b) {
      MultiFab& U = ctx.state(b);
      MultiFab R = ctx.rhs_scratch_like(U);
      ctx.rhs_into(b, U, R, 0);
      ctx.axpy(U, Real(h), R);  // U <- U + h * R  (Forward Euler)
    }
  });

  // Profiling counters (ADC-459, Spec 3 section 29): enable the System Profiler, so the ProgramContext
  // seam ops the step body calls (solve_fields, rhs_into, axpy) bump "kernels" and rhs_scratch_like
  // records the scratch peak. This is the HOST-validatable path (a ProgramContext built directly in
  // C++, no compiled .so); the cache hit/skip counters need a held schedule the codegen emits, so they
  // are exercised on the Kokkos/ROMEO compiled-.so runtime, not here.
  sim.enable_profiling();
  const int step0 = sim.macro_step();
  sim.step(dt);
  const std::vector<double> Up = sim.get_state("gas");

  double err = 0, change = 0;
  for (std::size_t k = 0; k < Up.size(); ++k) {
    err = std::fmax(err, std::fabs(Up[k] - Uref[k]));
    change = std::fmax(change, std::fabs(Up[k] - U0[k]));
  }
  EXPECT_TRUE(err < 1e-12) << "parity: max|Up - Uref| = " << err;
  EXPECT_TRUE(sim.macro_step() == step0 + 1)
      << "macro_step not advanced (" << step0 << " -> " << sim.macro_step() << ")";
  EXPECT_TRUE(change > 1e-9) << "program step did not change the state (change = " << change << ")";

  // ADC-459 counters: one step ran solve_fields + (1 block) rhs_into + axpy = EXACTLY 3 kernel-
  // dispatching seam ops (no double-count: solve_fields counts once, via Impl::solve_fields). Pinning
  // the exact value guards against a seam double-counting (a >0 check would not).
  const runtime::program::Profiler& prof = sim.profiler();
  EXPECT_TRUE(prof.counter("kernels") == 3)
      << "kernels counter = " << static_cast<long long>(prof.counter("kernels"))
      << ", expected 3 (solve_fields + rhs_into + axpy, no double)";
  EXPECT_TRUE(prof.counter("scratch_allocs") > 0)
      << "scratch_allocs counter not incremented (= "
      << static_cast<long long>(prof.counter("scratch_allocs")) << ")";
  EXPECT_TRUE(prof.counter("scratch_peak_bytes") > 0)
      << "scratch_peak_bytes not recorded (= "
      << static_cast<long long>(prof.counter("scratch_peak_bytes")) << ")";
  // The cache hit/skip counters never fire on this native ProgramContext step (no held schedule); they
  // exist as counters only after the compiled scheduler emits cache_should_update. Assert they read 0.
  EXPECT_TRUE(prof.counter("cache_hits") == 0 && prof.counter("cache_misses") == 0)
      << "cache counters moved on the native path (hits="
      << static_cast<long long>(prof.counter("cache_hits"))
      << " misses=" << static_cast<long long>(prof.counter("cache_misses")) << ")";
  {
    const std::string report = sim.profile_report();
    EXPECT_TRUE(report.find("kernels=") != std::string::npos)
        << "profile_report omits the kernels counter line";
  }
}

TEST(ProgramRuntime, ForwardEulerProgramContextHonorsEmbeddedBoundaryResidualMetrics) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 16;
  constexpr double gamma = 1.4;
  constexpr double dt = 1e-3;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  std::vector<double> initial(4 * cells);
  fill_ic(initial, n, gamma);

  const auto install_forward_euler = [](System& system) {
    system.set_program_block_map({0});
    runtime::program::ProgramContext context(&system);
    context.configure_primary_clock("macro");
    context.install([context](double step) {
      context.begin_step(step);
      context.set_stage_time(0, 1);
      MultiFab& state = context.state(0);
      MultiFab residual = context.rhs_scratch_like(state);
      context.rhs_into(0, state, residual, 0);
      context.axpy(state, Real(step), residual);
    });
  };

  System cartesian(cfg);
  add_gas(cartesian, gamma, "none");
  cartesian.set_state("gas", initial);
  install_forward_euler(cartesian);
  cartesian.step(dt);
  const std::vector<double> cartesian_state = cartesian.get_state("gas");

  System staircase(cfg);
  add_gas(staircase, gamma, "none");
  staircase.set_state("gas", initial);
  staircase.set_disc_domain(0.5, 0.5, 0.34, "staircase");
  const std::vector<double> mask = staircase.disc_mask();
  install_forward_euler(staircase);
  staircase.step(dt);
  const std::vector<double> staircase_state = staircase.get_state("gas");

  System cutcell(cfg);
  add_gas(cutcell, gamma, "none");
  cutcell.set_state("gas", initial);
  cutcell.set_disc_domain(0.5, 0.5, 0.34, "cutcell");
  install_forward_euler(cutcell);
  cutcell.step(dt);
  const std::vector<double> cutcell_state = cutcell.get_state("gas");

  double inactive_change = 0.0;
  double cutcell_inactive_change = 0.0;
  double active_change = 0.0;
  double cutcell_active_change = 0.0;
  double cartesian_inactive_change = 0.0;
  double route_difference = 0.0;
  int active_cells = 0;
  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = mask[cell] >= 0.5;
    active_cells += active ? 1 : 0;
    inactive_cells += active ? 0 : 1;
    for (int component = 0; component < 4; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      const double change = std::fabs(staircase_state[index] - initial[index]);
      if (active)
        active_change = std::fmax(active_change, change);
      else {
        inactive_change = std::fmax(inactive_change, change);
        cartesian_inactive_change = std::fmax(cartesian_inactive_change,
                                              std::fabs(cartesian_state[index] - initial[index]));
      }
      route_difference =
          std::fmax(route_difference, std::fabs(staircase_state[index] - cartesian_state[index]));
      const double cutcell_change = std::fabs(cutcell_state[index] - initial[index]);
      if (active)
        cutcell_active_change = std::fmax(cutcell_active_change, cutcell_change);
      else
        cutcell_inactive_change = std::fmax(cutcell_inactive_change, cutcell_change);
      route_difference =
          std::fmax(route_difference, std::fabs(cutcell_state[index] - cartesian_state[index]));
    }
  }

  ASSERT_GT(active_cells, 0);
  ASSERT_GT(inactive_cells, 0);
  EXPECT_EQ(inactive_change, 0.0)
      << "the Program wrote a non-zero staircase RHS outside the active set";
  EXPECT_EQ(cutcell_inactive_change, 0.0)
      << "the Program wrote a non-zero cut-cell RHS outside the active set";
  EXPECT_GT(active_change, 1e-10) << "the active Program residual was vacuous";
  EXPECT_GT(cutcell_active_change, 1e-10) << "the active cut-cell Program residual was vacuous";
  EXPECT_GT(cartesian_inactive_change, 1e-10)
      << "the Cartesian oracle did not exercise cells excluded by the staircase";
  EXPECT_GT(route_difference, 1e-10)
      << "the Program silently evaluated the Cartesian residual under staircase geometry";
  for (const double value : cutcell_state)
    EXPECT_TRUE(std::isfinite(value));
}

TEST(ProgramRuntime, SourceOnlyProgramStagePreservesEmbeddedBoundaryInactiveCells) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  constexpr double dt = 0.125;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};
  std::vector<double> initial(4 * cells);
  fill_ic(initial, n, gamma);

  const auto install_source_step = [](System& system) {
    system.set_program_block_map({0});
    runtime::program::ProgramContext context(&system);
    context.configure_primary_clock("macro");
    context.install([context](double step) {
      context.begin_step(step);
      MultiFab& state = context.state(0);
      MultiFab source = context.rhs_scratch_like(state);
      context.source_default_into(0, state, source);
      context.axpy(state, Real(step), source);
    });
  };

  System cartesian(cfg);
  add_sourced_gas(cartesian, gamma);
  cartesian.set_state("gas", initial);
  install_source_step(cartesian);
  cartesian.step(dt);
  const auto cartesian_state = cartesian.get_state("gas");

  System staircase(cfg);
  add_sourced_gas(staircase, gamma);
  staircase.set_state("gas", initial);
  staircase.set_disc_domain(0.5, 0.5, 0.31, "staircase");
  const auto mask = staircase.disc_mask();
  try {
    staircase.require_cartesian_generated_operator(0, "named_source");
    FAIL() << "a generated source must fail before evaluating inactive storage";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("named_source"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("embedded-boundary"), std::string::npos);
  }
  install_source_step(staircase);
  staircase.step(dt);
  const auto staircase_state = staircase.get_state("gas");

  int active_cells = 0;
  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = mask[cell] >= 0.5;
    active_cells += active ? 1 : 0;
    inactive_cells += active ? 0 : 1;
    const double expected = initial[cell] + (active ? dt : 0.0);
    EXPECT_DOUBLE_EQ(staircase_state[cell], expected);
    EXPECT_DOUBLE_EQ(cartesian_state[cell], initial[cell] + dt);
    for (int component = 1; component < 4; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      EXPECT_DOUBLE_EQ(staircase_state[index], initial[index]);
    }
  }
  EXPECT_GT(active_cells, 0);
  EXPECT_GT(inactive_cells, 0);
}

TEST(ProgramRuntime, NativeImexSourcePreservesEmbeddedBoundaryInactiveCells) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  constexpr double dt = 1e-3;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};
  std::vector<double> initial(4 * cells);
  fill_ic(initial, n, gamma);

  System system(cfg);
  add_imex_sourced_gas(system, gamma);
  system.set_state("gas", initial);
  system.set_disc_domain(0.5, 0.5, 0.31, "staircase");
  const auto mask = system.disc_mask();
  system.step(dt);
  const auto result = system.get_state("gas");

  double active_change = 0.0;
  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = mask[cell] >= 0.5;
    inactive_cells += active ? 0 : 1;
    for (int component = 0; component < 4; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      if (active)
        active_change = std::fmax(active_change, std::fabs(result[index] - initial[index]));
      else
        EXPECT_DOUBLE_EQ(result[index], initial[index]);
    }
  }
  EXPECT_GT(inactive_cells, 0);
  EXPECT_GT(active_change, 1e-10);
}

TEST(ProgramRuntime, EmbeddedBoundaryCflReductionIgnoresInactiveCells) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  System system(cfg);
  add_gas(system, gamma, "none");
  system.set_disc_domain(0.5, 0.5, 0.31, "staircase");
  const auto mask = system.disc_mask();
  std::vector<double> state(4 * cells);
  fill_ic(state, n, gamma);
  for (std::size_t cell = 0; cell < cells; ++cell)
    if (mask[cell] < 0.5)
      state[3 * cells + cell] = 1.0e12;
  system.set_state("gas", state);

  const double embedded_speed = system.block_max_speed(0, system.block_state(0));
  system.set_geometry_mode("none");
  const double cartesian_speed = system.block_max_speed(0, system.block_state(0));
  EXPECT_GT(embedded_speed, 0.0);
  EXPECT_GT(cartesian_speed, embedded_speed * 100.0)
      << "inactive high-speed cells still constrained the embedded-boundary CFL";
}

TEST(ProgramRuntime, CutCellCflIncludesPreparedInverseVolumeFraction) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 18;
  constexpr double gamma = 1.4;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};
  std::vector<double> uniform(4 * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    uniform[cell] = 1.0;
    uniform[3 * cells + cell] = 2.5;
  }

  System staircase(cfg);
  add_gas(staircase, gamma, "none");
  staircase.set_state("gas", uniform);
  staircase.set_disc_domain(0.5, 0.5, 0.34, "staircase", 0.1);
  const double staircase_speed = staircase.block_max_speed(0, staircase.block_state(0));

  System cutcell(cfg);
  add_gas(cutcell, gamma, "none");
  cutcell.set_state("gas", uniform);
  cutcell.set_disc_domain(0.5, 0.5, 0.34, "cutcell", 0.1);
  const double cutcell_speed = cutcell.block_max_speed(0, cutcell.block_state(0));

  EXPECT_GT(staircase_speed, 0.0);
  EXPECT_GT(cutcell_speed, staircase_speed)
      << "the cut-cell CFL ignored the residual's inverse-volume metric";
}

TEST(ProgramRuntime, PhysicalReductionsUsePreparedEmbeddedBoundaryMeasure) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 16;
  constexpr double gamma = 1.4;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  System staircase(cfg);
  add_gas(staircase, gamma, "none");
  staircase.set_disc_domain(0.5, 0.5, 0.31, "staircase");
  const std::vector<double> staircase_mask = staircase.disc_mask();
  std::vector<double> staircase_state(4 * cells, 0.0);
  int staircase_active = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = staircase_mask[cell] >= 0.5;
    staircase_active += active ? 1 : 0;
    staircase_state[cell] = active ? 2.0 : 1000.0;
    staircase_state[cells + cell] = active ? 3.0 : -1000.0;
  }
  ASSERT_GT(staircase_active, 0);
  ASSERT_LT(staircase_active, static_cast<int>(cells));
  staircase.set_state("gas", staircase_state);
  staircase.set_program_block_map({0});
  runtime::program::ProgramContext staircase_context(&staircase);
  MultiFab& staircase_field = staircase_context.state(0);
  const Real staircase_sum = staircase_context.sum_component(0, staircase_field, 0);
  const Real staircase_abs_sum = staircase_context.abs_sum_component(0, staircase_field, 0);
  const Real staircase_dot = staircase_context.dot(0, staircase_field, staircase_field);
  EXPECT_EQ(staircase_sum, Real(2 * staircase_active));
  EXPECT_EQ(staircase_abs_sum, staircase_sum);
  EXPECT_EQ(staircase.mass("gas"), static_cast<double>(staircase_sum));
  EXPECT_EQ(staircase.reduce_component("gas", "sum", 0),
            static_cast<double>(staircase_sum));
  EXPECT_NEAR(staircase_dot, Real(2) * staircase_sum, 1e-12);
  EXPECT_NEAR(staircase_context.norm2(0, staircase_field), std::sqrt(staircase_dot), 1e-12);
  EXPECT_EQ(staircase_context.max_component(0, staircase_field, 0), Real(2));
  EXPECT_EQ(staircase_context.min_component(0, staircase_field, 1), Real(3));
  EXPECT_EQ(staircase_context.norm_inf(0, staircase_field), Real(2));
  EXPECT_THROW((void)staircase_context.sum(staircase_field), std::runtime_error)
      << "an embedded-boundary Program reduction without an explicit owner was accepted";

  System cutcell(cfg);
  add_gas(cutcell, gamma, "none");
  cutcell.set_disc_domain(0.5, 0.5, 0.31, "cutcell");
  const std::vector<double> cutcell_mask = cutcell.disc_mask();
  std::vector<double> cutcell_state(4 * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = cutcell_mask[cell] >= 0.5;
    cutcell_state[cell] = active ? 2.0 : 1000.0;
    cutcell_state[cells + cell] = active ? 3.0 : -1000.0;
  }
  cutcell.set_state("gas", cutcell_state);
  cutcell.set_program_block_map({0});
  runtime::program::ProgramContext cutcell_context(&cutcell);
  MultiFab& cutcell_field = cutcell_context.state(0);
  const Real cutcell_sum = cutcell_context.sum_component(0, cutcell_field, 0);
  const Real cutcell_dot = cutcell_context.dot(0, cutcell_field, cutcell_field);
  EXPECT_GT(cutcell_sum, Real(0));
  EXPECT_LT(cutcell_sum, staircase_sum)
      << "the cut-cell integral ignored the prepared relative volume fraction";
  EXPECT_NEAR(cutcell.mass("gas"), static_cast<double>(cutcell_sum), 1e-12);
  EXPECT_NEAR(cutcell_dot, Real(2) * cutcell_sum, 1e-10);
  EXPECT_EQ(cutcell_context.max_component(0, cutcell_field, 0), Real(2));
  EXPECT_EQ(cutcell_context.min_component(0, cutcell_field, 1), Real(3));
  EXPECT_EQ(cutcell_context.norm_inf(0, cutcell_field), Real(2));
}

TEST(ProgramRuntime, PointwiseDomainUsesThePreparedBlockMaskForValidation) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  for (const std::string mode : {"staircase", "cutcell"}) {
    System system(cfg);
    add_gas(system, gamma, "none");
    system.set_disc_domain(0.5, 0.5, 0.32, mode);
    const std::vector<double> mask = system.disc_mask();
    system.set_state("gas", std::vector<double>(4 * cells, 2.0));
    system.set_program_block_map({0});

    runtime::program::ProgramContext context(&system);
    MultiFab& state = context.state(0);
    const MultiFab* prepared = context.pointwise_active_mask(0, state);
    ASSERT_NE(prepared, nullptr) << mode;
    MultiFab status = context.alloc_scalar_field(1, 0);
    int active = 0;
    int inactive = 0;
    for (int li = 0; li < status.local_size(); ++li) {
      Fab2D& fab = status.fab(li);
      const Box2D box = fab.box();
      for (int j = box.lo[1]; j <= box.hi[1]; ++j) {
        for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
          const bool is_active = mask[static_cast<std::size_t>(j * n + i)] >= 0.5;
          active += is_active ? 1 : 0;
          inactive += is_active ? 0 : 1;
          fab(i, j, 0) = is_active ? Real(0) : Real(1);
        }
      }
    }
    ASSERT_GT(active, 0) << mode;
    ASSERT_GT(inactive, 0) << mode;
    EXPECT_EQ(context.pointwise_status_max(0, status, prepared), Real(0)) << mode;
    EXPECT_THROW((void)context.pointwise_status_max(0, status, &status),
                 std::invalid_argument)
        << mode;
  }
}

TEST(ProgramRuntime, Ssprk3AndProgramAlgebraPreserveInactiveBits) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 14;
  constexpr double gamma = 1.4;
  constexpr double inactive_value = 0.9;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};
  std::vector<double> initial(4 * cells);
  fill_ic(initial, n, gamma);

  System native(cfg);
  add_ssprk3_gas(native, gamma);
  native.set_disc_domain(0.5, 0.5, 0.32, "staircase");
  const auto mask = native.disc_mask();
  for (std::size_t cell = 0; cell < cells; ++cell)
    if (mask[cell] < 0.5)
      for (int component = 0; component < 4; ++component)
        initial[static_cast<std::size_t>(component) * cells + cell] = inactive_value;
  native.set_state("gas", initial);
  native.step(1.0e-4);
  const auto native_result = native.get_state("gas");

  System program(cfg);
  add_gas(program, gamma, "none");
  program.set_state("gas", initial);
  program.set_disc_domain(0.5, 0.5, 0.32, "staircase");
  program.set_program_block_map({0});
  runtime::program::ProgramContext context(&program);
  context.configure_primary_clock("macro");
  context.install([context](double) {
    MultiFab& state = context.state(0);
    MultiFab identical = state;
    context.lincomb(state, Real(1) / Real(3), state, Real(2) / Real(3), identical);
  });
  program.step(1.0e-4);
  const auto program_result = program.get_state("gas");

  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    if (mask[cell] >= 0.5)
      continue;
    ++inactive_cells;
    for (int component = 0; component < 4; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      EXPECT_EQ(std::bit_cast<std::uint64_t>(native_result[index]),
                std::bit_cast<std::uint64_t>(initial[index]));
      EXPECT_EQ(std::bit_cast<std::uint64_t>(program_result[index]),
                std::bit_cast<std::uint64_t>(initial[index]));
    }
  }
  EXPECT_GT(inactive_cells, 0);
}

TEST(ProgramRuntime, EmbeddedBoundaryCapabilitiesRejectUnsupportedProvidersBeforePublication) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  SystemConfig cfg;
  cfg.n = 10;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  System reconstructed(cfg);
  add_gas(reconstructed, 1.4, "minmod");
  EXPECT_THROW(reconstructed.set_disc_domain(0.5, 0.5, 0.3, "staircase"), std::runtime_error);
  const auto reconstructed_mask = reconstructed.disc_mask();
  EXPECT_TRUE(std::all_of(reconstructed_mask.begin(), reconstructed_mask.end(),
                          [](double value) { return value == 1.0; }));

  System diffusive(cfg);
  add_diffusive_gas(diffusive, 1.4);
  EXPECT_THROW(diffusive.set_disc_domain(0.5, 0.5, 0.3, "cutcell"), std::runtime_error);
  const auto diffusive_mask = diffusive.disc_mask();
  EXPECT_TRUE(std::all_of(diffusive_mask.begin(), diffusive_mask.end(),
                          [](double value) { return value == 1.0; }));
}

TEST(ProgramRuntime, PointwiseProjectionPreservesEmbeddedBoundaryInactiveCells) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 12;
  constexpr double gamma = 1.4;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};
  std::vector<double> initial(4 * cells);
  fill_ic(initial, n, gamma);

  const auto install_projection_step = [](System& system) {
    system.set_program_block_map({0});
    runtime::program::ProgramContext context(&system);
    context.configure_primary_clock("macro");
    context.install([context](double step) {
      context.begin_step(step);
      context.apply_projection(0, context.state(0));
    });
  };

  System cartesian(cfg);
  add_projecting_gas(cartesian, gamma);
  cartesian.set_state("gas", initial);
  install_projection_step(cartesian);
  cartesian.step(0.1);
  const auto cartesian_state = cartesian.get_state("gas");

  System cutcell(cfg);
  add_projecting_gas(cutcell, gamma);
  cutcell.set_state("gas", initial);
  cutcell.set_disc_domain(0.5, 0.5, 0.31, "cutcell");
  const auto mask = cutcell.disc_mask();
  install_projection_step(cutcell);
  cutcell.step(0.1);
  const auto cutcell_state = cutcell.get_state("gas");

  int active_cells = 0;
  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    const bool active = mask[cell] >= 0.5;
    active_cells += active ? 1 : 0;
    inactive_cells += active ? 0 : 1;
    EXPECT_DOUBLE_EQ(cartesian_state[cell], 2.0);
    EXPECT_DOUBLE_EQ(cutcell_state[cell], active ? 2.0 : initial[cell]);
    for (int component = 1; component < 4; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
      EXPECT_DOUBLE_EQ(cutcell_state[index], initial[index]);
    }
  }
  EXPECT_GT(active_cells, 0);
  EXPECT_GT(inactive_cells, 0);
}

TEST(ProgramRuntime, EmbeddedBoundaryRejectsUnqualifiedBoundaryLinearizationEntryPoints) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  SystemConfig cfg;
  cfg.n = 8;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};
  System system(cfg);
  add_gas(system, 1.4, "none");
  system.set_disc_domain(0.5, 0.5, 0.3, "staircase");
  system.set_program_block_map({0});
  runtime::program::ProgramContext context(&system);
  MultiFab& state = context.state(0);
  MultiFab output = context.rhs_scratch_like(state);
  const runtime::multiblock::BoundaryEvaluationPoint point{
      "clock.boundary-linearization", 0, 0, 0, 0, amr::Rational(0, 1), 0.1, 0.0};

  const auto expect_metric_rejection = [](auto&& operation) {
    try {
      operation();
      FAIL() << "embedded-boundary boundary linearization was accepted";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find("signed-mask or cut-cell metric contract"),
                std::string::npos);
    }
  };
  expect_metric_rejection([&] { context.boundary_residual_into_at(point, 0, state, output); });
  expect_metric_rejection([&] { context.boundary_jvp_into_at(point, 0, state, output, output); });
}

TEST(ProgramRuntime, RejectedAttemptRestoresStateHistoryCacheDiagnosticsAndClock) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  System sim(cfg);
  add_gas(sim, gamma);
  std::vector<double> initial(4 * static_cast<std::size_t>(n) * n);
  fill_ic(initial, n, gamma);
  sim.set_state("gas", initial);
  sim.register_history("gas.U", 2, 4);
  sim.set_program_block_map({0});

  runtime::program::ProgramContext ctx(&sim);
  ctx.install([ctx](double dt) {
    MultiFab& state = ctx.state(0);
    MultiFab bump = state;
    bump.set_val(Real(dt));
    ctx.axpy(state, Real(1), bump);
    ctx.store_history("gas.U", state);
    ctx.rotate_histories();
    ctx.cache_store_scratch(17, state);
    ctx.record_scalar("provisional", Real(42));
    throw runtime::program::StepAttemptRejected(SolveStatus::kIterationLimit, "solve",
                                                "fault injection after provisional publications");
  });

  EXPECT_THROW(sim.step(1e-3), runtime::program::StepAttemptRejected);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_DOUBLE_EQ(sim.time(), 0.0);
  EXPECT_EQ(sim.get_state("gas"), initial);
  EXPECT_FALSE(sim.history_initialized("gas.U"));
  EXPECT_FALSE(sim.program_cache().has(17));
  EXPECT_TRUE(sim.program_diagnostics().empty());
}
