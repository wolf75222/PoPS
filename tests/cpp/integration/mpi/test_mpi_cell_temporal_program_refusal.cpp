#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/physics/fluids/euler.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

template <int Dim>
void install_explicit_amr_callback_program_rank_isolated(
    AmrSystem<Dim>& system, std::string_view identity, std::string_view clock,
    test::explicit_amr_program_detail::callback_type callback,
    const std::optional<test::program_v5::CallbackProgramCellTemporalAuthority>& cell_temporal =
        std::nullopt) {
  static_assert(Dim == kNativeDimension,
                "the ABI-v5 explicit AMR fixture is compiled for POPS_NATIVE_DIM");
  if (identity.empty() || clock.empty() || !callback)
    throw std::invalid_argument(
        "explicit AMR callback Program requires exact callback authorities");
  auto& callbacks = test::explicit_amr_program_detail::callbacks();
  const auto callback_identifier = static_cast<std::uint64_t>(callbacks.size());
  callbacks.emplace_back(std::move(callback));
#if !defined(POPS_TEST_TMPDIR)
  throw std::runtime_error("explicit AMR ABI-v5 fixture requires POPS_TEST_TMPDIR");
#else
  static std::size_t fixture_index = 0;
  const std::string prefix = std::string(POPS_TEST_TMPDIR) + "/explicit_amr_callback_refusal_rank" +
                             std::to_string(my_rank()) + "_" + std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  std::ofstream source(source_path);
  if (!source)
    throw std::runtime_error("cannot create rank-isolated explicit AMR callback source");
  source << test::program_v5::callback_program_source(
      callback_identifier, identity, clock, system.block_names(), {},
      "pops_test_explicit_amr_program_callback", "amr", {}, {}, {}, {}, std::nullopt,
      cell_temporal);
  source.close();
  const auto compiled = test::native_dso::compile_shared(source_path, library_path);
  if (!compiled.ok) {
    test::native_dso::report_compile_failure("rank-isolated explicit_amr_callback_program",
                                             compiled);
    throw std::runtime_error("rank-isolated explicit AMR ABI-v5 callback compilation failed");
  }
  system.install_program(library_path);
#endif
}

int run_collective_refusal() {
  constexpr int Dim = kNativeDimension;
  AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 4;
    config.periodicity[axis] = true;
  }
  // The physical-boundary exclusion is the intended refusal witness. A default-device build
  // reaches the host-only preparation guard first with the same otherwise valid route.
  config.periodicity[0] = false;
  // The declared 2:1 temporal relation below is qualified against one configured AMR
  // transition, even though this refusal fixture never publishes a fine level.
  config.level_count = 2;
  config.regrid_every = 0;

  AmrSystem<Dim> system(config);
  test::install_amr_runtime_authority(system, "tests.cell-temporal-refusal/runtime@1");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  using Transport = CompositeModel<EulerND<Dim>, NoSource, NoElliptic>;
  system.install_block_state_route("tracer", "tests.cell-temporal-refusal/tracer/state@1");
  std::vector<std::string> face_types;
  std::vector<std::string> face_identities;
  for (int axis = 0; axis < Dim; ++axis)
    for (int side = 0; side < 2; ++side) {
      face_types.push_back(config.periodicity[axis] ? "periodic" : "foextrap");
      face_identities.push_back("tests.cell-temporal-refusal/tracer/face/" +
                                std::to_string(2 * axis + side));
    }
  std::vector<std::string> component_roles{"density"};
  for (int axis = 0; axis < Dim; ++axis)
    component_roles.push_back("momentum:" + std::to_string(axis));
  component_roles.push_back("energy");
  system.install_hyperbolic_boundary(
      "tracer", "tests.cell-temporal-refusal/tracer/boundary@1", 1, face_types,
      std::vector<double>(component_roles.size() * static_cast<std::size_t>(2 * Dim), 0.0),
      face_identities, component_roles, "tests.cell-temporal-refusal/tracer/state@1");
  add_compiled_model<Dim>(
      system, "tracer",
      Transport{{}, {}, EulerND<Dim>::prepare(Real(1.4)), NoSource{}, NoElliptic{}}, "minmod",
      "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
      static_cast<double>(kWenoEpsilon), false,
      "tests.cell-temporal-refusal/tracer/physical-flux@1");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  std::vector<double> state(static_cast<std::size_t>(EulerND<Dim>::n_vars) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[static_cast<std::size_t>(EulerND<Dim>::density_component) * cells + cell] = 1.0;
    state[static_cast<std::size_t>(EulerND<Dim>::energy_component) * cells + cell] = 2.5;
  }
  system.set_conservative_state("tracer", state);
  system.set_program_block_map({0});

  const std::string clock = "test.clock.cell-local-mpi-refusal";
  const auto accepted_before = system.program_accepted_state();
  const ExecutionLane lane = ExecutionLane::world("tests.cell-temporal-refusal/lane");
  const test::program_v5::CallbackProgramCellTemporalAuthority cell_temporal{
      clock, 100, 0, {{0, -1, 0}}};
  bool refused = false;
  try {
    install_explicit_amr_callback_program_rank_isolated<Dim>(
        system, "tests.cell-temporal-refusal/program@1", clock,
        [](auto& context, double dt) { context.begin_step(dt); }, cell_temporal);
  } catch (const std::runtime_error& error) {
    refused =
        std::string(error.what()) == "AmrSystem::install_program installation failed collectively";
  }
  const long refusing_ranks = all_reduce_sum(refused ? 1L : 0L, lane);
  const bool unchanged = system.program_accepted_state() == accepted_before;
  return refusing_ranks == lane.size() && unchanged ? 0 : 1;
}

int run_collective_unqualified_dt_refusal() {
  constexpr int Dim = kNativeDimension;
  AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 4;
    config.periodicity[axis] = true;
  }
  config.level_count = 1;
  config.regrid_every = 0;
  config.distribute_coarse = true;
  for (int axis = 0; axis < Dim; ++axis)
    config.coarse_max_grid[axis] = 2;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();

  AmrSystem<Dim> system(config);
  test::install_amr_runtime_authority(system, "tests.cell-temporal-dt-refusal/runtime@1");
  system.install_block_state_route("tracer", "tests.cell-temporal-dt-refusal/tracer/state@1");
  using Transport = CompositeModel<EulerND<Dim>, NoSource, NoElliptic>;
  add_compiled_model<Dim>(
      system, "tracer",
      Transport{{}, {}, EulerND<Dim>::prepare(Real(1.4)), NoSource{}, NoElliptic{}}, "minmod",
      "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
      static_cast<double>(kWenoEpsilon), false,
      "tests.cell-temporal-dt-refusal/tracer/physical-flux@1");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  std::vector<double> state(static_cast<std::size_t>(EulerND<Dim>::n_vars) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[static_cast<std::size_t>(EulerND<Dim>::density_component) * cells + cell] = 1.0;
    state[static_cast<std::size_t>(EulerND<Dim>::energy_component) * cells + cell] = 2.5;
  }
  system.set_conservative_state("tracer", state);
  system.set_program_block_map({0});
  const std::string clock = "tests.cell-temporal-dt-refusal/clock";
  const test::program_v5::CallbackProgramCellTemporalAuthority cell_temporal{
      clock, 100, 0, {{0, -1, 0}}};
  install_explicit_amr_callback_program_rank_isolated<Dim>(
      system, "tests.cell-temporal-dt-refusal/program@1", clock,
      [](auto& context, double dt) {
        context.begin_step(dt);
        context.advance_same_level_cell_temporal(dt);
      },
      cell_temporal);
  const auto accepted_before = system.program_accepted_state();
  const ExecutionLane lane = ExecutionLane::world("tests.cell-temporal-dt-refusal/lane");
  bool refused = false;
  try {
    system.step(0.015);
  } catch (const std::invalid_argument& error) {
    refused = std::string(error.what()).find("cell-local AMR dt") != std::string::npos;
  }
  const long refusals = all_reduce_sum(refused ? 1L : 0L, lane);
  const bool unchanged = system.program_accepted_state() == accepted_before;
  return refusals == lane.size() && unchanged ? 0 : 1;
}

int run_collective_future_ratio_refusal() {
  constexpr int Dim = kNativeDimension;
  AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 4;
    config.periodicity[axis] = true;
  }
  // The first installed hierarchy is deliberately coarse-only.  The 5/2 relation is therefore
  // future-only at this point and must still refuse the cell-local artifact at installation.
  config.level_count = 3;
  config.transition_ratios.resize(2);
  config.transition_buffers.resize(2);
  config.transition_lookaheads.resize(2);
  config.transition_ratios[1] = config.transition_ratios[0];
  config.transition_buffers[1] = config.transition_buffers[0];
  config.transition_lookaheads[1] = config.transition_lookaheads[0];
  config.explicit_bootstrap = true;
  config.regrid_every = 0;

  AmrSystem<Dim> system(config);
  test::install_amr_runtime_authority(system, "tests.cell-temporal-future-ratio/runtime@1");
  system.set_temporal_relations({2, 5}, {1, 2}, {"integral_only", "explicit_final_substep"});
  system.install_block_state_route("tracer", "tests.cell-temporal-future-ratio/tracer/state@1");
  using Transport = CompositeModel<EulerND<Dim>, NoSource, NoElliptic>;
  add_compiled_model<Dim>(
      system, "tracer",
      Transport{{}, {}, EulerND<Dim>::prepare(Real(1.4)), NoSource{}, NoElliptic{}}, "minmod",
      "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
      static_cast<double>(kWenoEpsilon), false,
      "tests.cell-temporal-future-ratio/tracer/physical-flux@1");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  std::vector<double> state(static_cast<std::size_t>(EulerND<Dim>::n_vars) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[static_cast<std::size_t>(EulerND<Dim>::density_component) * cells + cell] = 1.0;
    state[static_cast<std::size_t>(EulerND<Dim>::energy_component) * cells + cell] = 2.5;
  }
  system.set_conservative_state("tracer", state);
  system.set_program_block_map({0});

  const auto accepted_before = system.program_accepted_state();
  const std::string clock = "tests.cell-temporal-future-ratio/clock";
  const test::program_v5::CallbackProgramCellTemporalAuthority cell_temporal{
      clock, 100, 0, {{0, -1, 0}}};
  const ExecutionLane lane = ExecutionLane::world("tests.cell-temporal-future-ratio/lane");
  bool refused = false;
  try {
    install_explicit_amr_callback_program_rank_isolated<Dim>(
        system, "tests.cell-temporal-future-ratio/program@1", clock,
        [](auto& context, double dt) { context.begin_step(dt); }, cell_temporal);
  } catch (const std::runtime_error& error) {
    refused =
        std::string(error.what()) == "AmrSystem::install_program installation failed collectively";
  }
  const long refusing_ranks = all_reduce_sum(refused ? 1L : 0L, lane);
  return refusing_ranks == lane.size() && system.program_accepted_state() == accepted_before ? 0
                                                                                             : 1;
}

int run_collective_regrid_resident_refusal() {
  constexpr int Dim = kNativeDimension;
  AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    config.periodicity[axis] = true;
  }
  config.level_count = 2;
  config.regrid_every = 0;
  config.explicit_bootstrap = true;
  config.distribute_coarse = true;
  for (int axis = 0; axis < Dim; ++axis)
    config.coarse_max_grid[axis] = 2;

  AmrSystem<Dim> system(config);
  test::install_amr_runtime_authority(system, "tests.cell-temporal-regrid/runtime@1");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  const std::string state_route = "tests.cell-temporal-regrid/tracer/state@1";
  system.install_block_state_route("tracer", state_route);
  using Transport = CompositeModel<EulerND<Dim>, NoSource, NoElliptic>;
  add_compiled_model<Dim>(
      system, "tracer",
      Transport{{}, {}, EulerND<Dim>::prepare(Real(1.4)), NoSource{}, NoElliptic{}}, "minmod",
      "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
      static_cast<double>(kWenoEpsilon), false,
      "tests.cell-temporal-regrid/tracer/physical-flux@1");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  std::vector<double> state(static_cast<std::size_t>(EulerND<Dim>::n_vars) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[static_cast<std::size_t>(EulerND<Dim>::density_component) * cells + cell] = 1.0;
    state[static_cast<std::size_t>(EulerND<Dim>::energy_component) * cells + cell] = 2.5;
  }
  system.set_conservative_state("tracer", state);
  system.set_program_block_map({0});
  system.bind_bootstrap_subject(state_route, "tracer", "bound_level_zero");
  system.stage_bootstrap_array(state_route, "tracer", "cell", "cell", EulerND<Dim>::n_vars,
                               config.shape, state);
  test::install_prepared_threshold_union(
      system, {{"tracer", "rho", 0.5, test::PreparedThresholdRelation::Above, state_route}},
      "tests.cell-temporal-regrid/tagging@1");

  const std::string clock = "tests.cell-temporal-regrid/clock";
  const test::program_v5::CallbackProgramCellTemporalAuthority cell_temporal{
      clock, 100, 0, {{0, -1, 0}}};
  install_explicit_amr_callback_program_rank_isolated<Dim>(
      system, "tests.cell-temporal-regrid/program@1", clock,
      [](auto& context, double dt) { context.begin_step(dt); }, cell_temporal);
  system.begin_bootstrap_plan();
  (void)system.materialize_bootstrap_action(state_route, "initialize_level_zero",
                                            "bound_level_zero", 0);
  system.commit_bootstrap_level();

  const auto accepted_before = system.program_accepted_state();
  const auto topology_before = system.checkpoint_topology_epoch();
  const int regrids_before = system.checkpoint_regrid_count();
  const ExecutionLane lane = ExecutionLane::world("tests.cell-temporal-regrid/lane");
  bool refused = false;
  try {
    system.execute_prepared_tagging(0);
    (void)system.regrid_from_prepared_tagging(0);
  } catch (const std::exception& error) {
    const std::string message = error.what();
    refused =
        message ==
            "AMR topology regrid has no declared transfer provider for the cell-local resident "
            "executor" ||
        message == "AMR cell-local topology-transfer preflight failed collectively";
  }
  const long refusing_ranks = all_reduce_sum(refused ? 1L : 0L, lane);
  const bool unchanged = system.program_accepted_state() == accepted_before &&
                         system.checkpoint_topology_epoch() == topology_before &&
                         system.checkpoint_regrid_count() == regrids_before &&
                         system.n_levels() == 1;
  return refusing_ranks == lane.size() && unchanged ? 0 : 1;
}

int run_collective_step_change_block_refusal_and_retry() {
  constexpr int Dim = kNativeDimension;
  AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.regrid_every = 0;
  config.distribute_coarse = true;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 4;
    config.coarse_max_grid[axis] = 2;
    config.periodicity[axis] = true;
  }

  AmrSystem<Dim> system(config);
  test::install_amr_runtime_authority(system, "tests.step-change-name-refusal/runtime@1");
  system.install_block_state_route("tracer", "tests.step-change-name-refusal/tracer/state@1");
  using Transport = CompositeModel<EulerND<Dim>, NoSource, NoElliptic>;
  add_compiled_model<Dim>(
      system, "tracer",
      Transport{{}, {}, EulerND<Dim>::prepare(Real(1.4)), NoSource{}, NoElliptic{}}, "minmod",
      "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
      static_cast<double>(kWenoEpsilon), false,
      "tests.step-change-name-refusal/tracer/physical-flux@1");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  std::vector<double> state(static_cast<std::size_t>(EulerND<Dim>::n_vars) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[static_cast<std::size_t>(EulerND<Dim>::density_component) * cells + cell] = 1.0;
    state[static_cast<std::size_t>(EulerND<Dim>::energy_component) * cells + cell] = 2.5;
  }
  system.set_conservative_state("tracer", state);
  system.set_program_block_map({0});
  const std::string clock = "tests.step-change-name-refusal/clock";
  install_explicit_amr_callback_program_rank_isolated<Dim>(
      system, "tests.step-change-name-refusal/program@1", clock,
      [](auto& context, double dt) { context.begin_step(dt); });

  const ExecutionLane lane = ExecutionLane::world("tests.step-change-name-refusal/lane");
  std::uint64_t expected_dispatches = 0;
  {
    const auto accepted = system.accepted_amr_runtime();
    for (std::size_t level = 0; level < accepted->hierarchy().num_levels(); ++level) {
      const auto& values = accepted->hierarchy().state(level);
      expected_dispatches += values.local_size() * static_cast<std::uint64_t>(values.ncomp());
    }
  }
  const auto accepted_before = system.program_accepted_state();
  system.begin_step_transaction();
  system.step(0.01);
  bool refused = false;
  {
    auto scope = system._provisional_read_scope();
    try {
      (void)system.step_change_l2_for_block(my_rank() == 0 ? "tracer" : "rank-local-missing");
    } catch (const std::invalid_argument& error) {
      refused =
          std::string_view(error.what()).find("block request differs") != std::string_view::npos;
    }
  }
  const long refusal_count = all_reduce_sum(refused ? 1L : 0L, lane);
  system.rollback_step_transaction();
  const bool unchanged_after_rollback = system.program_accepted_state() == accepted_before;

  system.begin_step_transaction();
  system.step(0.01);
  bool retry_succeeded = false;
  {
    auto scope = system._provisional_read_scope();
    try {
      const double measured = system.step_change_l2_for_block("tracer");
      // This callback only opens its candidate step, so U^{n+1}=U^n.  The zero is an exact
      // multi-patch, multi-component composite oracle; the dispatch count proves every Euler
      // component/patch was folded through the prepared workspace.
      retry_succeeded = measured == 0.0 && std::isfinite(measured) &&
                        system._step_change_l2_last_dispatches() == expected_dispatches;
    } catch (...) {
      retry_succeeded = false;
    }
  }
  system.rollback_step_transaction();
  const long retry_count = all_reduce_sum(retry_succeeded ? 1L : 0L, lane);
  return refusal_count == lane.size() && retry_count == lane.size() && unchanged_after_rollback ? 0
                                                                                                : 1;
}

int pops_run_test_mpi_cell_temporal_program_refusal(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int result = run_collective_refusal() == 0
                         ? (run_collective_unqualified_dt_refusal() == 0
                                ? (run_collective_future_ratio_refusal() == 0
                                       ? (run_collective_regrid_resident_refusal() == 0
                                              ? run_collective_step_change_block_refusal_and_retry()
                                              : 1)
                                       : 1)
                                : 1)
                         : 1;
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_cell_temporal_program_refusal, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_cell_temporal_program_refusal,
                                    "test_mpi_cell_temporal_program_refusal"),
            0);
}
