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
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/program/program_context.hpp>  // ProgramContext (the seam under test)
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
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

struct DrainingDensitySource {
  template <class State>
  POPS_HD State apply(const State&, const Aux&) const {
    State source{};
    source[0] = Real(-1);
    return source;
  }
};
using DrainingGasModel = CompositeModel<Euler, DrainingDensitySource, NoEll>;

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

static void add_draining_gas(System& system, double gamma) {
  add_compiled_model(
      system, "gas", DrainingGasModel{Euler{gamma}, DrainingDensitySource{}, NoEll{}}, "none",
      "rusanov", "conservative", "explicit", gamma);
}

static void add_projecting_gas(System& system, double gamma) {
  ProjectingEuler transport;
  transport.gamma = gamma;
  add_compiled_model(system, "gas", ProjectingGasModel{transport, NoSource{}, NoEll{}}, "none",
                     "rusanov", "conservative", "explicit", gamma);
}

static void add_diffusive_gas(System& system, double gamma) {
  DiffusiveGasModel model;
  model.hyp.gamma = gamma;
  add_compiled_model(system, "gas", model, "none", "rusanov", "conservative", "explicit", gamma);
}

TEST(ProgramRuntime, BalanceDueWindowUsesTheOuterAcceptedStepAndCleansUpOnFailure) {
  runtime::program::ProgramRuntimeState state;
  const std::string contract = "pops.balance-due-contract.v1:sha256:" + std::string(64, '1');
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '2');

  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 3, "test"), std::logic_error);
  state.run_balance_due_window(2, "test", [&] {
    EXPECT_TRUE(state.balance_consumer_is_due(contract, route, 3, "test"));
    EXPECT_FALSE(state.balance_consumer_is_due(contract, route, 2, "test"));
    EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 0, "test"),
                 std::invalid_argument);
    EXPECT_THROW((void)state.balance_consumer_is_due("forged", route, 3, "test"),
                 std::invalid_argument);
  });
  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 3, "test"), std::logic_error);

  EXPECT_THROW(
      state.run_balance_due_window(3, "test", [] { throw std::runtime_error("attempt rejected"); }),
      std::runtime_error);
  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 4, "test"), std::logic_error);
}

TEST(ProgramRuntime, AutomaticBalanceDueMarkerIsAttemptLocalMonotoneAndReplaySafe) {
  runtime::program::ProgramRuntimeState state;

  EXPECT_FALSE(state.automatic_balance_capture_due());
  EXPECT_THROW(state.note_automatic_balance_capture_due(true, "test"), std::logic_error);
  state.run_balance_due_window(0, "test", [&] {
    state.note_automatic_balance_capture_due(false, "test");
    EXPECT_FALSE(state.automatic_balance_capture_due());
    state.note_automatic_balance_capture_due(true, "test");
    EXPECT_TRUE(state.automatic_balance_capture_due());
    state.note_automatic_balance_capture_due(false, "test");
    EXPECT_TRUE(state.automatic_balance_capture_due());
  });
  EXPECT_TRUE(state.automatic_balance_capture_due());

  state.begin_step_projection_report();
  EXPECT_FALSE(state.automatic_balance_capture_due());
  state.run_balance_replay("test", [&] {
    state.note_automatic_balance_capture_due(false, "test");
    EXPECT_FALSE(state.automatic_balance_capture_due());
    EXPECT_THROW(state.note_automatic_balance_capture_due(true, "test"), std::logic_error);
  });
  EXPECT_FALSE(state.automatic_balance_capture_due());
}

TEST(ProgramRuntime, SelectedAutomaticBalanceTermsRequireCompleteQualifiedEvidence) {
  runtime::program::ProgramRuntimeState state;
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '5');
  state.begin_step_projection_report();
  state.run_balance_due_window(0, "test", [&] {
    state.note_automatic_balance_capture_due(true, "test");
    state.record_balance_term(route, "storage_change", 1.0, "test");
    state.record_balance_term(route, "outward_boundary_flux", 2.0, "test");
    state.record_balance_term(route, "sources", 3.0, "test");
    state.record_automatic_balance_term(2, 0, 1, "projection", 0.25, "test");
    state.record_automatic_balance_term(2, 1, 1, "projection", 0.75, "test");
    state.record_automatic_balance_term(2, 0, 1, "reflux", 0.5, "test");
  });
  state.complete_balance_step(true);

  const auto selected =
      state.selected_accepted_balance_terms(route, 2, 1, {0, 1}, {"projection", "reflux"}, "test");
  EXPECT_EQ(selected.at("storage_change"), 1.0);
  EXPECT_EQ(selected.at("outward_boundary_flux"), 2.0);
  EXPECT_EQ(selected.at("sources"), 3.0);
  EXPECT_EQ(selected.at("projection"), 1.0);
  EXPECT_EQ(selected.at("reflux"), 0.5);

  EXPECT_THROW((void)state.selected_accepted_balance_terms(route, 2, 1, {0, 1, 2},
                                                           {"projection", "reflux"}, "test"),
               std::runtime_error);
  EXPECT_THROW((void)state.selected_accepted_balance_terms(route, 2, 1, {0, 2},
                                                           {"projection", "reflux"}, "test"),
               std::invalid_argument);
}

TEST(ProgramRuntime, SelectiveReplayCompilesBalanceOffAndRestoresTheGuard) {
  runtime::program::ProgramRuntimeState state;
  const std::string contract = "pops.balance-due-contract.v1:sha256:" + std::string(64, '3');
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '4');

  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 2, "test"), std::logic_error);
  state.run_balance_replay("test", [&] {
    EXPECT_FALSE(state.balance_consumer_is_due(contract, route, 2, "test"));
    EXPECT_THROW((void)state.balance_consumer_is_due("forged", route, 2, "test"),
                 std::invalid_argument);
    EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 0, "test"),
                 std::invalid_argument);
    EXPECT_THROW(state.run_balance_replay("nested", [] {}), std::logic_error);
    EXPECT_THROW(state.run_balance_due_window(1, "nested", [] {}), std::logic_error);
  });
  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 2, "test"), std::logic_error);
  state.run_balance_due_window(1, "test", [&] {
    EXPECT_THROW(state.run_balance_replay("window", [] {}), std::logic_error);
  });

  EXPECT_THROW(state.run_balance_replay("test", [] { throw std::runtime_error("replay failed"); }),
               std::runtime_error);
  EXPECT_THROW((void)state.balance_consumer_is_due(contract, route, 2, "test"), std::logic_error);
}

TEST(ProgramRuntime, ReplayAuthorityRequiresAnArtifactAndAnExactRingDepthPair) {
  runtime::program::ProgramRuntimeState state;
  state.history_replay_authorities_ = {{"gas.previous", 3}};

  EXPECT_FALSE(state.authorizes_history_replay("gas.previous", 3))
      << "a table without an authenticated artifact marker is not replay authority";

  state.artifact_backed_ = true;
  state.operator_authorities_ = {{{1, 2, 3, 4}}};
  state.installed_hash_ = "old-artifact";
  state.block_map_ = {0};
  state.seed_params(0, {2.0});
  state.dt_bound_ = [](Real cfl) { return cfl; };
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));
  EXPECT_FALSE(state.authorizes_history_replay("gas.previous", 2));
  EXPECT_FALSE(state.authorizes_history_replay("other.previous", 3));

  state.install_unverified_step([](double) {});
  EXPECT_FALSE(state.artifact_backed_);
  EXPECT_TRUE(state.operator_authorities_.empty());
  EXPECT_TRUE(state.history_replay_authorities_.empty());
  EXPECT_TRUE(state.installed_hash_.empty());
  EXPECT_TRUE(state.block_map_.empty());
  EXPECT_TRUE(state.block_params_.empty());
  EXPECT_FALSE(state.dt_bound_);
  EXPECT_FALSE(state.restart_regrid_preflight_);
  EXPECT_FALSE(state.restart_regrid_);
  EXPECT_FALSE(state.restart_resync_);
  EXPECT_FALSE(state.authorizes_history_replay("gas.previous", 3))
      << "a direct native step must revoke every earlier artifact authority";
}

TEST(ProgramRuntime, ArtifactStepInstallRequiresOneNewStepAndRollsBackExactly) {
  runtime::program::ProgramRuntimeState state;
  int old_steps = 0;
  int new_steps = 0;
  int restart_preflights = 0;
  int restart_regrids = 0;
  int restart_resyncs = 0;
  state.install_unverified_step([&](double) { ++old_steps; });
  state.install_restart_hooks([&] { ++restart_preflights; }, [&] { ++restart_regrids; },
                              [&] { ++restart_resyncs; }, "test");
  state.operator_authorities_ = {{{1, 2, 3, 4}}};
  state.history_replay_authorities_ = {{"gas.previous", 3}};
  state.installed_hash_ = "accepted-artifact";
  state.block_map_ = {2};
  state.seed_params(0, {3.0});
  state.dt_bound_ = [](Real cfl) { return Real(0.5) * cfl; };
  state.artifact_backed_ = true;
  const auto accepted_generation = state.step_install_generation_;

  auto interrupted = state.capture_artifact_step_install();
  state.operator_authorities_ = {{{9, 8, 7, 6}}};
  state.install_unverified_step([&](double) { ++new_steps; });
  state.rollback_artifact_step_install(std::move(interrupted));
  state.step_(0.1);
  EXPECT_EQ(old_steps, 1);
  EXPECT_EQ(new_steps, 0);
  EXPECT_EQ(state.step_install_generation_, accepted_generation);
  EXPECT_EQ(state.operator_authorities_,
            (std::vector<std::array<std::uint64_t, 4>>{{{1, 2, 3, 4}}}));
  EXPECT_EQ(state.installed_hash_, "accepted-artifact");
  EXPECT_EQ(state.block_map_, (std::vector<int>{2}));
  EXPECT_EQ(state.block_params_.size(), 1u);
  ASSERT_TRUE(state.dt_bound_);
  EXPECT_DOUBLE_EQ(state.dt_bound_(0.4), 0.2);
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));
  EXPECT_NO_THROW(state.preflight_regrid_on_restart("test"));
  EXPECT_NO_THROW(state.regrid_on_restart("test"));
  EXPECT_NO_THROW(state.resync_after_restart_rollback("test"));
  EXPECT_EQ(restart_preflights, 1);
  EXPECT_EQ(restart_regrids, 1);
  EXPECT_EQ(restart_resyncs, 1);

  const auto no_op = state.capture_artifact_step_install();
  EXPECT_THROW(state.require_exact_artifact_step_install(no_op, "test"), std::runtime_error);

  auto exact = state.capture_artifact_step_install();
  state.install_unverified_step([&](double) { ++new_steps; });
  EXPECT_NO_THROW(state.require_exact_artifact_step_install(exact, "test"));
  state.rollback_artifact_step_install(std::move(exact));
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));

  auto duplicate = state.capture_artifact_step_install();
  state.install_unverified_step([&](double) { ++new_steps; });
  state.install_unverified_step([&](double) { ++new_steps; });
  EXPECT_THROW(state.require_exact_artifact_step_install(duplicate, "test"), std::runtime_error);
  state.rollback_artifact_step_install(std::move(duplicate));
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));

  const auto generation_before_empty = state.step_install_generation_;
  EXPECT_THROW(state.install_unverified_step({}), std::invalid_argument);
  EXPECT_EQ(state.step_install_generation_, generation_before_empty);
  EXPECT_TRUE(state.authorizes_history_replay("gas.previous", 3));
}

TEST(ProgramRuntime, FacadeTemporalOperationsRequireProgramBeforeMutation) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System system(SystemConfig{8, 1.0, Periodicity{true, true}});
  system.enable_profiling();
  const double initial_time = system.time();
  const int initial_step = system.macro_step();
  const std::string initial_profile = system.profile_report();

  const auto expect_program_required = [&](auto&& operation, const char* name) {
    try {
      operation();
      ADD_FAILURE() << name << " accepted a program-less temporal operation";
    } catch (const std::logic_error& error) {
      EXPECT_NE(std::string(error.what()).find(name), std::string::npos);
      EXPECT_NE(std::string(error.what()).find("installed whole-system Program"),
                std::string::npos);
    }
    EXPECT_DOUBLE_EQ(system.time(), initial_time);
    EXPECT_EQ(system.macro_step(), initial_step);
    EXPECT_EQ(system.profile_report(), initial_profile);
  };

  expect_program_required([&] { system.step(0.01); }, "System::step");
  expect_program_required([&] { system.advance(0.01, 0); }, "System::advance");
  expect_program_required([&] { (void)system.step_cfl(0.4); }, "System::step_cfl");
}

TEST(ProgramRuntime, GlobalCadencePublishesExactSubstepAndStrideWindowTimes) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  SystemConfig config;
  config.n = 4;
  config.L = 1.0;
  config.periodicity = {true, true};

  System subcycled(config);
  subcycled.set_clock(1.0, 10);
  runtime::program::ProgramContext subcycled_context(&subcycled);
  std::vector<double> subcycled_times;
  std::vector<int> subcycled_macro_steps;
  subcycled_context.install([&](double) {
    subcycled_times.push_back(static_cast<double>(subcycled_context.physical_time()));
    subcycled_macro_steps.push_back(subcycled.macro_step());
  });
  subcycled.set_program_cadence(/*substeps=*/2, /*stride=*/1);
  subcycled.step(0.2);

  ASSERT_EQ(subcycled_times.size(), 2u);
  EXPECT_NEAR(subcycled_times[0], 1.0, 1.0e-14);
  EXPECT_NEAR(subcycled_times[1], 1.1, 1.0e-14);
  EXPECT_EQ(subcycled_macro_steps, (std::vector<int>{10, 10}));
  EXPECT_NEAR(subcycled.time(), 1.2, 1.0e-14);
  EXPECT_EQ(subcycled.macro_step(), 11);

  System catchup(config);
  runtime::program::ProgramContext catchup_context(&catchup);
  catchup_context.configure_primary_clock("macro");
  std::vector<double> catchup_times;
  std::vector<double> catchup_steps;
  std::vector<int> catchup_macro_steps;
  std::vector<bool> catchup_every_one_due;
  catchup_context.install([&](double h) {
    catchup_times.push_back(static_cast<double>(catchup_context.physical_time()));
    catchup_steps.push_back(h);
    catchup_macro_steps.push_back(catchup.macro_step());
    catchup_every_one_due.push_back(catchup_context.schedule_is_due(
        41, 1, runtime::program::ScheduleDomainKind::kAcceptedStep, "macro", "", -1));
  });
  catchup.set_program_cadence(/*substeps=*/3, /*stride=*/2);

  catchup.step(0.1);
  EXPECT_TRUE(catchup_times.empty());
  EXPECT_NEAR(catchup.time(), 0.1, 1.0e-14);
  EXPECT_EQ(catchup.macro_step(), 1);
  EXPECT_DOUBLE_EQ(catchup.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(catchup.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(catchup.program_cadence_window_start_time(), 0.0);

  catchup.step(0.2);
  ASSERT_EQ(catchup_times.size(), 3u);
  EXPECT_NEAR(catchup_times[0], 0.0, 1.0e-14);
  EXPECT_NEAR(catchup_times[1], 0.1, 1.0e-14);
  EXPECT_NEAR(catchup_times[2], 0.2, 1.0e-14);
  ASSERT_EQ(catchup_steps.size(), 3u);
  for (const double h : catchup_steps)
    EXPECT_NEAR(h, 0.1, 1.0e-14);
  EXPECT_DOUBLE_EQ(catchup_times.back() + catchup_steps.back(), 0.1 + 0.2);
  EXPECT_EQ(catchup_macro_steps, (std::vector<int>{0, 0, 0}));
  EXPECT_EQ(catchup_every_one_due, (std::vector<bool>{true, true, true}));
  EXPECT_NEAR(catchup.time(), 0.3, 1.0e-14);
  EXPECT_EQ(catchup.macro_step(), 2);
  EXPECT_DOUBLE_EQ(catchup.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(catchup.program_cadence_window_steps(), 0);
  EXPECT_DOUBLE_EQ(catchup.program_cadence_window_start_time(), 0.0);
}

TEST(ProgramRuntime, StrideHeldStepsPublishTheExactZeroBalance) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  SystemConfig config;
  config.n = 4;
  config.L = 1.0;
  config.periodicity = {true, true};

  System system(config);
  runtime::program::ProgramContext context(&system);
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '7');
  const std::array<std::pair<const char*, double>, 5> records{{
      {"storage_change", 1.0},
      {"outward_boundary_flux", 2.0},
      {"sources", 3.0},
      {"reflux", 4.0},
      {"projection", 5.0},
  }};
  context.install([&](double) {
    for (const auto& [name, value] : records)
      context.record_balance_term(route, name, value);
  });
  system.set_program_cadence(/*substeps=*/1, /*stride=*/3);

  const auto step_and_read = [&]() {
    system.begin_step_transaction();
    system.step(0.1);
    const auto balance = system.accepted_balance_terms(route);
    system.commit_step_transaction();
    system.finalize_step_transaction();
    return balance;
  };

  for (int held = 0; held < 2; ++held) {
    const auto balance = step_and_read();
    ASSERT_EQ(balance.size(), records.size());
    for (const auto& [name, _value] : records)
      EXPECT_DOUBLE_EQ(balance.at(name), 0.0);
  }

  system.begin_step_transaction();
  system.step(0.1);
  const auto rejected_due = system.accepted_balance_terms(route);
  for (const auto& [name, value] : records)
    EXPECT_DOUBLE_EQ(rejected_due.at(name), value);
  system.rollback_step_transaction();
  system.begin_step_transaction();
  const auto restored_held = system.accepted_balance_terms(route);
  for (const auto& [name, _value] : records)
    EXPECT_DOUBLE_EQ(restored_held.at(name), 0.0);
  system.rollback_step_transaction();

  const auto due = step_and_read();
  ASSERT_EQ(due.size(), records.size());
  for (const auto& [name, value] : records)
    EXPECT_DOUBLE_EQ(due.at(name), value);
}

TEST(ProgramRuntime,
     CadenceUsesThePreparedFacadeEndpointWhenFloatingPointAdditionIsNonAssociative) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  SystemConfig config;
  config.n = 4;
  config.L = 1.0;
  config.periodicity = {true, true};

  System system(config);
  system.set_clock(0.1, 0);
  runtime::program::ProgramContext context(&system);
  std::vector<double> starts;
  std::vector<double> steps;
  context.install([&](double h) {
    starts.push_back(static_cast<double>(context.physical_time()));
    steps.push_back(h);
  });
  system.set_program_cadence(/*substeps=*/3, /*stride=*/3);

  const double after_first = 0.1 + 0.1;
  const double after_second = after_first + 0.1;
  const double accepted_endpoint = after_second + 0.3;
  const double effective_dt = (0.1 + 0.1) + 0.3;
  const double reconstructed_endpoint = 0.1 + effective_dt;
  const double numerical_dt = accepted_endpoint - 0.1;
  ASSERT_NE(std::bit_cast<std::uint64_t>(accepted_endpoint),
            std::bit_cast<std::uint64_t>(reconstructed_endpoint))
      << "fixture must exercise floating-point non-associativity";
  ASSERT_NE(std::bit_cast<std::uint64_t>(numerical_dt), std::bit_cast<std::uint64_t>(effective_dt))
      << "fixture must distinguish dt provenance from the representable facade interval";

  system.step(0.1);
  system.step(0.1);
  EXPECT_TRUE(starts.empty());
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1 + 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 2);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.1);

  system.step(0.3);
  ASSERT_EQ(starts.size(), 3u);
  ASSERT_EQ(steps.size(), 3u);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(starts.front()), std::bit_cast<std::uint64_t>(0.1));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(starts.back() + steps.back()),
            std::bit_cast<std::uint64_t>(accepted_endpoint));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(system.time()),
            std::bit_cast<std::uint64_t>(accepted_endpoint));
  EXPECT_NE(std::bit_cast<std::uint64_t>(system.time()),
            std::bit_cast<std::uint64_t>(reconstructed_endpoint));
  EXPECT_EQ(system.macro_step(), 3);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
}

TEST(ProgramRuntime, CadenceWindowRestartAndRejectedDueStepAreTransactional) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  SystemConfig config;
  config.n = 4;
  config.L = 1.0;
  config.periodicity = {true, true};

  System system(config);
  std::vector<double> accepted_steps;
  bool reject = true;
  system.install_program_step([&](double h) {
    if (reject)
      throw runtime::program::StepAttemptRejected(SolveStatus::kIterationLimit, "cadence",
                                                  "fault injection in due stride window");
    accepted_steps.push_back(h);
  });
  system.set_program_cadence(/*substeps=*/1, /*stride=*/2);
  system.step(0.1);

  EXPECT_THROW(system.step(0.2), runtime::program::StepAttemptRejected);
  EXPECT_DOUBLE_EQ(system.time(), 0.1);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.1);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);

  reject = false;
  system.step(0.2);
  ASSERT_EQ(accepted_steps.size(), 1u);
  EXPECT_NEAR(accepted_steps[0], 0.3, 1.0e-14);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);

  System restarted(config);
  std::vector<double> restarted_times;
  restarted.install_program_step([&](double) { restarted_times.push_back(restarted.time()); });
  restarted.set_program_cadence(/*substeps=*/1, /*stride=*/2);
  restarted.restore_program_cadence_window(/*accumulated_dt=*/0.1, /*held_steps=*/1,
                                           /*window_start_time=*/0.0, /*accepted_last_dt=*/0.07,
                                           /*accepted_time=*/0.1,
                                           /*macro_step=*/1);
  restarted.set_clock(/*t=*/0.1, /*macro_step=*/1);
  EXPECT_DOUBLE_EQ(restarted.program_last_dt(), 0.07);
  restarted.step(0.2);
  EXPECT_EQ(restarted_times, (std::vector<double>{0.0}));
  EXPECT_NEAR(restarted.time(), 0.3, 1.0e-14);
  EXPECT_EQ(restarted.macro_step(), 2);
}

TEST(ProgramRuntime, CadenceRestoreCommitsOnlyForTheExactAcceptedClockPair) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  SystemConfig config;
  config.n = 4;
  config.L = 1.0;
  config.periodicity = {true, true};

  System system(config);
  system.install_program_step([](double) {});
  system.set_program_cadence(/*substeps=*/1, /*stride=*/2);

  // The accepted time is part of the native restore preflight. A malformed image cannot touch the
  // accepted window or arm a transaction.
  EXPECT_THROW(system.restore_program_cadence_window(
                   /*accumulated_dt=*/0.1, /*held_steps=*/1, /*window_start_time=*/0.1,
                   /*accepted_last_dt=*/0.075, /*accepted_time=*/0.1, /*macro_step=*/1),
               std::runtime_error);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);
  EXPECT_THROW(system.restore_program_cadence_window(
                   /*accumulated_dt=*/0.1, /*held_steps=*/1, /*window_start_time=*/0.0,
                   /*accepted_last_dt=*/std::numeric_limits<double>::quiet_NaN(),
                   /*accepted_time=*/0.1, /*macro_step=*/1),
               std::runtime_error);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.0);

  // A valid image is only staged. One-ulp clock drift rejects and discards the transaction while the
  // accepted facade clock and cadence image remain byte-for-byte unchanged.
  system.restore_program_cadence_window(/*accumulated_dt=*/0.1, /*held_steps=*/1,
                                        /*window_start_time=*/0.0, /*accepted_last_dt=*/0.075,
                                        /*accepted_time=*/0.1,
                                        /*macro_step=*/1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.0);
  EXPECT_THROW(system.set_clock(std::nextafter(0.1, 1.0), /*macro_step=*/1), std::runtime_error);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);

  // The mismatch did not strand a pending token: a direct clean-boundary restore remains usable.
  system.set_clock(/*t=*/0.25, /*macro_step=*/0);
  system.restore_program_cadence_window(/*accumulated_dt=*/0.25, /*held_steps=*/1,
                                        /*window_start_time=*/0.25, /*accepted_last_dt=*/0.075,
                                        /*accepted_time=*/0.5,
                                        /*macro_step=*/1);
  system.set_clock(/*t=*/0.5, /*macro_step=*/1);
  EXPECT_DOUBLE_EQ(system.time(), 0.5);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.25);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.25);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.075);

  // Replacing a non-empty accepted window is transactional too: a bad replacement leaves both that
  // window and the facade cursor intact, and ordinary stepping remains possible afterwards.
  system.restore_program_cadence_window(/*accumulated_dt=*/0.4, /*held_steps=*/1,
                                        /*window_start_time=*/1.0, /*accepted_last_dt=*/0.2,
                                        /*accepted_time=*/1.4,
                                        /*macro_step=*/3);
  EXPECT_THROW(system.set_clock(std::nextafter(1.4, 2.0), /*macro_step=*/3), std::runtime_error);
  EXPECT_DOUBLE_EQ(system.time(), 0.5);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.25);
  EXPECT_EQ(system.program_cadence_window_steps(), 1);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.25);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), 0.075);
  system.step(0.25);
  EXPECT_DOUBLE_EQ(system.time(), 0.75);
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), 0.0);
  EXPECT_EQ(system.program_cadence_window_steps(), 0);
}

TEST(ProgramRuntime, CadenceRejectsDtAbsorbedByThePhysicalClock) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  SystemConfig config;
  config.n = 4;
  config.L = 1.0;
  config.periodicity = {true, true};

  System system(config);
  int calls = 0;
  system.install_program_step([&](double) { ++calls; });
  system.set_clock(1.0e16, 0);

  EXPECT_THROW(system.step(0.5), std::overflow_error);
  EXPECT_DOUBLE_EQ(system.time(), 1.0e16);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_EQ(calls, 0);
}

TEST(ProgramRuntime, CadenceFailsBeforeMutationWhenSubstepsCollapseTheRepresentableInterval) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  SystemConfig config;
  config.n = 4;
  config.L = 1.0;
  config.periodicity = {true, true};

  System system(config);
  int calls = 0;
  system.install_program_step([&](double) { ++calls; });
  system.set_program_cadence(/*substeps=*/3, /*stride=*/1);
  system.set_clock(1.0, 0);
  const double one_ulp = std::nextafter(1.0, 2.0) - 1.0;

  EXPECT_THROW(system.step(one_ulp), std::overflow_error);
  EXPECT_DOUBLE_EQ(system.time(), 1.0);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_EQ(calls, 0);
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
  (void)pops::consume_solve_outcome(ref.solve_fields());
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
    auto field_outcome = ctx.solve_fields();
    (void)field_outcome.consume(SolveConsumption::kAccept);
    for (int b = 0; b < ctx.n_blocks(); ++b) {
      MultiFab& U = ctx.state(b);
      MultiFab R = ctx.rhs_scratch_like(U);
      ctx.rhs_into(b, U, R, 0);
      ctx.axpy(U, Real(h), R);  // U <- U + h * R  (Forward Euler)
    }
  });
  sim.set_program_block_map({0});

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
    system.set_program_block_map({0});
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
    system.set_program_block_map({0});
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

TEST(ProgramRuntime, TerminalSourcePublicationConsumesPreparedRecoveryBeforeCommit) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  constexpr double gamma = 1.4;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  SystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodicity = {true, true};

  System system(cfg);
  add_draining_gas(system, gamma);
  std::vector<double> initial(4 * cells);
  fill_ic(initial, n, gamma);
  system.set_state("gas", initial);
  system.set_program_block_map({0});
  runtime::program::ProgramContext context(&system);
  context.configure_primary_clock("test.clock.source-recovery");
  context.install([context](double step) {
    context.begin_step(step);
    MultiFab& live = context.state(0);
    MultiFab& source = context.rhs_scratch(920001, 0, live);
    MultiFab& candidate = context.scratch_state(920002, 0, live);
    context.source_default_into(0, live, source);
    context.lincomb(candidate, Real(1), live, Real(0), live);
    context.axpy(candidate, Real(step), source);
    context.commit_many({{&live, &candidate}});
  });
  system.set_program_block_map({0});

  system.step(0.25);
  const std::vector<double> accepted = system.get_state("gas");
  for (std::size_t cell = 0; cell < cells; ++cell)
    EXPECT_DOUBLE_EQ(accepted[cell], 0.75);
  EXPECT_DOUBLE_EQ(system.time(), 0.25);
  EXPECT_EQ(system.macro_step(), 1);

  // A second source update reaches rho=0 exactly. Euler recovery would divide momentum by rho;
  // commit_many must therefore refuse the whole candidate before copying one live component.
  try {
    system.step(0.75);
    FAIL() << "an unrecoverable model-source endpoint must not publish";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("prepared variable recovery rejected"),
              std::string::npos);
  }
  EXPECT_EQ(system.get_state("gas"), accepted);
  EXPECT_DOUBLE_EQ(system.time(), 0.25);
  EXPECT_EQ(system.macro_step(), 1);
}

TEST(ProgramRuntime, ExplicitSourceProgramPreservesEmbeddedBoundaryInactiveCells) {
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
  add_sourced_gas(system, gamma);
  system.set_state("gas", initial);
  system.set_disc_domain(0.5, 0.5, 0.31, "staircase");
  const auto mask = system.disc_mask();
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
  system.set_program_block_map({0});
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
  EXPECT_EQ(staircase.reduce_component("gas", "sum", 0), static_cast<double>(staircase_sum));
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
    EXPECT_THROW((void)context.pointwise_status_max(0, status, &status), std::invalid_argument)
        << mode;
  }
}

TEST(ProgramRuntime, Ssprk3ProgramAlgebraPreservesInactiveBits) {
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

  System program(cfg);
  add_gas(program, gamma, "none");
  program.set_disc_domain(0.5, 0.5, 0.32, "staircase");
  const auto mask = program.disc_mask();
  for (std::size_t cell = 0; cell < cells; ++cell)
    if (mask[cell] < 0.5)
      for (int component = 0; component < 4; ++component)
        initial[static_cast<std::size_t>(component) * cells + cell] = inactive_value;
  program.set_state("gas", initial);
  program.set_program_block_map({0});
  runtime::program::ProgramContext context(&program);
  context.configure_primary_clock("macro");
  context.install([context](double step) {
    context.begin_step(step);
    MultiFab& state = context.state(0);
    MultiFab initial_state = state;
    MultiFab stage = state;
    MultiFab residual = context.rhs_scratch_like(state);

    context.set_stage_time(0, 1);
    context.rhs_into(0, state, residual, 100);
    context.axpy(stage, Real(step), residual);

    context.set_stage_time(1, 1);
    residual.set_val(Real(0));
    context.rhs_into(0, stage, residual, 101);
    context.axpy(stage, Real(step), residual);
    context.lincomb(stage, Real(3) / Real(4), initial_state, Real(1) / Real(4), stage);

    context.set_stage_time(1, 2);
    residual.set_val(Real(0));
    context.rhs_into(0, stage, residual, 102);
    context.axpy(stage, Real(step), residual);
    context.lincomb(state, Real(1) / Real(3), initial_state, Real(2) / Real(3), stage);
  });
  program.set_program_block_map({0});
  program.step(1.0e-4);
  const auto program_result = program.get_state("gas");

  int inactive_cells = 0;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    if (mask[cell] >= 0.5)
      continue;
    ++inactive_cells;
    for (int component = 0; component < 4; ++component) {
      const std::size_t index = static_cast<std::size_t>(component) * cells + cell;
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
    system.set_program_block_map({0});
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

TEST(ProgramRuntime, ProjectAndRecheckConsumesSolveAndCommitsProjectedCandidate) {
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
  add_projecting_gas(sim, gamma);
  sim.set_poisson("charge_density", "geometric_mg");
  std::vector<double> initial(4 * static_cast<std::size_t>(n) * n);
  fill_ic(initial, n, gamma);
  sim.set_state("gas", initial);
  sim.set_program_block_map({0});

  int consumed_solves = 0;
  runtime::program::ProgramContext ctx(&sim);
  ctx.install([ctx, &consumed_solves](double dt) {
    ctx.begin_step(dt);
    MultiFab& state = ctx.state(0);
    MultiFab& candidate = ctx.scratch_state(666001, 0, state);
    candidate.set_val(Real(-1));

    auto field_outcome = ctx.solve_fields();
    const SolveReport field_report = field_outcome.consume(SolveConsumption::kAccept);
    if (!field_report.solved_value_available())
      throw std::logic_error("ProjectAndRecheck test did not receive a solved field value");
    ++consumed_solves;

    if (ctx.min(0, candidate) <= Real(0)) {
      ctx.apply_projection(0, candidate);
      if (ctx.min(0, candidate) <= Real(0))
        throw runtime::program::StepAttemptRejected(
            SolveStatus::kIterationLimit, "guard recheck",
            "ProjectAndRecheck projection did not repair the candidate");
    }
    ctx.commit_many({{&state, &candidate}});
  });
  sim.set_program_block_map({0});

  sim.step(1e-3);

  EXPECT_EQ(consumed_solves, 1);
  EXPECT_EQ(sim.macro_step(), 1);
  EXPECT_DOUBLE_EQ(sim.time(), 1e-3);
  const std::vector<double> accepted = sim.get_state("gas");
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  for (std::size_t cell = 0; cell < cells; ++cell) {
    EXPECT_DOUBLE_EQ(accepted[cell], 2.0);
    for (int component = 1; component < 4; ++component)
      EXPECT_DOUBLE_EQ(accepted[static_cast<std::size_t>(component) * cells + cell], -1.0);
  }
}

TEST(ProgramRuntime, ProjectAndRecheckFailureConsumesSolveAndRollsBackWithoutPublication) {
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
  add_projecting_gas(sim, gamma);
  sim.set_poisson("charge_density", "geometric_mg");
  std::vector<double> initial(4 * static_cast<std::size_t>(n) * n);
  fill_ic(initial, n, gamma);
  sim.set_state("gas", initial);
  sim.register_history("gas.guard_candidate", 2, 4);
  sim.set_program_block_map({0});

  int consumed_solves = 0;
  runtime::program::ProgramContext ctx(&sim);
  ctx.install([ctx, &consumed_solves](double dt) {
    ctx.begin_step(dt);
    MultiFab& state = ctx.state(0);
    MultiFab& candidate = ctx.scratch_state(666002, 0, state);
    candidate.set_val(Real(-1));

    auto field_outcome = ctx.solve_fields();
    const SolveReport field_report = field_outcome.consume(SolveConsumption::kAccept);
    if (!field_report.solved_value_available())
      throw std::logic_error("ProjectAndRecheck test did not receive a solved field value");
    ++consumed_solves;

    if (ctx.min(0, candidate) < Real(3)) {
      ctx.apply_projection(0, candidate);
      ctx.store_history("gas.guard_candidate", candidate);
      ctx.rotate_histories();
      ctx.cache_store_scratch(666002, candidate);
      ctx.record_scalar("project_and_recheck.provisional", Real(1));
      if (ctx.min(0, candidate) < Real(3))
        throw runtime::program::StepAttemptRejected(
            SolveStatus::kIterationLimit, "guard recheck",
            "ProjectAndRecheck candidate remained inadmissible");
    }
    ctx.commit_many({{&state, &candidate}});
  });
  sim.set_program_block_map({0});

  EXPECT_THROW(sim.step(1e-3), runtime::program::StepAttemptRejected);

  EXPECT_EQ(consumed_solves, 1);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_DOUBLE_EQ(sim.time(), 0.0);
  EXPECT_EQ(sim.get_state("gas"), initial);
  EXPECT_FALSE(sim.history_initialized("gas.guard_candidate"));
  EXPECT_FALSE(sim.program_cache().has(666002));
  EXPECT_TRUE(sim.program_diagnostics().empty());
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

TEST(ProgramRuntime, AnalyticInitialStatePublishesOnlyAfterPreparedRecoveryAcceptsEveryCell) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  System system(SystemConfig{n, 1.0, Periodicity{true, true}});
  ModelSpec scalar;
  scalar.transport = "exb";
  scalar.source = "none";
  scalar.elliptic = "charge";
  system.add_block("tracer", scalar);

  const std::vector<double> accepted(static_cast<std::size_t>(n) * n, 0.25);
  system.set_state("tracer", accepted);
  system.set_block_conversion(
      "tracer", [](const double* in, double* out) { out[0] = in[0]; },
      [](const double* in, double* out) {
        RecoveryReport report;
        if (!std::isfinite(in[0]) || in[0] > 0.75) {
          report.status = RecoveryStatus::kRejected;
          report.cause = RecoveryCause::kInadmissibleCandidate;
          report.failing_component = 0;
          return report;
        }
        out[0] = in[0];
        report.status = RecoveryStatus::kRecovered;
        report.cause = RecoveryCause::kNone;
        return report;
      });

  EXPECT_THROW(system.set_analytic_expression_state(
                   "tracer", "cell", "cell", "conservative_cell_average", {{"constant"}}, {{1.0}}),
               std::runtime_error);
  EXPECT_EQ(system.get_state("tracer"), accepted);

  EXPECT_THROW(system.set_analytic_mapped_state("tracer", {{"input", "constant", "add"}},
                                                {{0.0, 1.0, 0.0}}, {"state:0"}),
               std::runtime_error);
  EXPECT_EQ(system.get_state("tracer"), accepted);

  EXPECT_THROW(system.set_analytic_gaussian_state("tracer", 0.5, 0.5, 1.0, 0.0, 16.0),
               std::runtime_error);
  EXPECT_EQ(system.get_state("tracer"), accepted);

  EXPECT_EQ(system.set_analytic_expression_state(
                "tracer", "cell", "cell", "conservative_cell_average", {{"constant"}}, {{0.5}}),
            static_cast<std::int64_t>(n) * n);
  EXPECT_EQ(system.get_state("tracer"), std::vector<double>(static_cast<std::size_t>(n) * n, 0.5));
}

TEST(ProgramRuntime, AnalyticInitialStatePublishesWhenPreparedRecoveryAcceptsEveryCell) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  System system(SystemConfig{n, 1.0, Periodicity{true, true}});
  ModelSpec scalar;
  scalar.transport = "exb";
  scalar.source = "none";
  scalar.elliptic = "charge";
  system.add_block("tracer", scalar);
  system.set_block_conversion(
      "tracer", [](const double* in, double* out) { out[0] = in[0]; },
      [](const double* in, double* out) {
        RecoveryReport report;
        out[0] = in[0];
        report.status = RecoveryStatus::kRecovered;
        report.cause = RecoveryCause::kNone;
        return report;
      });

  EXPECT_EQ(system.set_analytic_expression_state(
                "tracer", "cell", "cell", "conservative_cell_average", {{"constant"}}, {{0.5}}),
            static_cast<std::int64_t>(n) * n);
  EXPECT_EQ(system.get_state("tracer"), std::vector<double>(static_cast<std::size_t>(n) * n, 0.5));
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
  sim.set_program_block_map({0});

  EXPECT_THROW(sim.step(1e-3), runtime::program::StepAttemptRejected);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_DOUBLE_EQ(sim.time(), 0.0);
  EXPECT_EQ(sim.get_state("gas"), initial);
  EXPECT_FALSE(sim.history_initialized("gas.U"));
  EXPECT_FALSE(sim.program_cache().has(17));
  EXPECT_TRUE(sim.program_diagnostics().empty());
}
