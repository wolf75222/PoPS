#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/physics/bricks/bricks.hpp>
#include <pops/physics/fluids/euler.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

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
  if (!system.uses_runtime_engine())
    return 1;

  const std::string clock = "test.clock.cell-local-mpi-refusal";
  test::install_explicit_amr_callback_program<Dim>(
      system, "tests.cell-temporal-refusal/program@1", clock, {}, {},
      [clock](auto& context, double dt) {
        context.begin_step(dt);
        const std::array route{runtime::program::SameLevelCellTemporalForwardEulerRoute{0, 0, 0}};
        context.prepare_same_level_cell_temporal_execution(clock, 100, 0, route);
      });
  const auto accepted_before = system.program_accepted_state();
  const ExecutionLane lane = ExecutionLane::world("tests.cell-temporal-refusal/lane");

  bool refused = false;
  try {
    system.step(0.01);
  } catch (const std::runtime_error& error) {
    refused = std::string(error.what()) == "cell-local AMR route preparation failed collectively";
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
  test::install_explicit_amr_callback_program<Dim>(
      system, "tests.cell-temporal-dt-refusal/program@1", clock, {}, {},
      [clock](auto& context, double dt) {
        context.begin_step(dt);
        const std::array route{runtime::program::SameLevelCellTemporalForwardEulerRoute{0, 0, 0}};
        context.prepare_same_level_cell_temporal_execution(clock, 100, 0, route);
        context.advance_same_level_cell_temporal(dt);
      });
  const auto accepted_before = system.program_accepted_state();
  const ExecutionLane lane = ExecutionLane::world("tests.cell-temporal-dt-refusal/lane");
  bool refused = false;
  try {
    system.step(0.015);
  } catch (const std::runtime_error& error) {
    refused = std::string(error.what()).find("cell-local AMR dt") != std::string::npos;
  }
  const long refusals = all_reduce_sum(refused ? 1L : 0L, lane);
  const bool unchanged = system.program_accepted_state() == accepted_before;
  return refusals == lane.size() && unchanged ? 0 : 1;
}

int pops_run_test_mpi_cell_temporal_program_refusal(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int result = run_collective_refusal() == 0 ? run_collective_unqualified_dt_refusal() : 1;
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_cell_temporal_program_refusal, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_cell_temporal_program_refusal,
                                    "test_mpi_cell_temporal_program_refusal"),
            0);
}
