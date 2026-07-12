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

#include <cmath>
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

static void add_gas(System& s, double gamma) {
  add_compiled_model(s, "gas", GasModel{Euler{gamma}, NoSource{}, NoEll{}}, "minmod", "rusanov",
                     "conservative", "explicit", gamma);
  s.set_poisson("charge_density", "geometric_mg");
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
  cfg.periodic = true;

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
  ctx.install([ctx](double h) {
    ctx.solve_fields();
    for (int b = 0; b < ctx.n_blocks(); ++b) {
      MultiFab& U = ctx.state(b);
      MultiFab R = ctx.rhs_scratch_like(U);
      ctx.rhs_into(b, U, R);
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

TEST(ProgramRuntime, RejectedAttemptRestoresStateHistoryCacheDiagnosticsAndClock) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

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
    throw runtime::program::StepAttemptRejected(
        SolveStatus::kIterationLimit, "solve", "fault injection after provisional publications");
  });

  EXPECT_THROW(sim.step(1e-3), runtime::program::StepAttemptRejected);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_DOUBLE_EQ(sim.time(), 0.0);
  EXPECT_EQ(sim.get_state("gas"), initial);
  EXPECT_FALSE(sim.history_initialized("gas.U"));
  EXPECT_FALSE(sim.program_cache().has(17));
  EXPECT_TRUE(sim.program_diagnostics().empty());
}
