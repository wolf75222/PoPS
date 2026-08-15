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
#include <pops/runtime/program/amr_program_context.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops::runtime::program::detail {

// This witness is intentionally a friend of AmrProgramContext rather than a public test API: it
// replaces only the already-installed remap callback, after the outer transaction snapshot exists.
// The callback corrupts the just-swapped candidate on rank zero, then invokes the real private
// refresh path.  Consequently the enclosing AmrSystem regrid transaction, rather than the test,
// owns rollback.
template <int Dim>
struct AmrProgramHistoryRemapCollectiveTestAccess {
  using context_type = AmrProgramContext<Dim>;

  static void install_rank_zero_candidate_metadata_corruption(context_type& context,
                                                              std::string history_name, int level) {
    auto& program = context.runtime_state();
    program.history_remap_accepted_ = [&context, history_name = std::move(history_name),
                                       level]() mutable {
      if (my_rank() == 0) {
        auto& histories = context.runtime_state().hist_;
        const std::string key = context.history_key_(history_name, level);
        histories.fill_count.at(key) = 0;
        histories.initialized.at(key) = true;
      }
      context.refresh_accepted_hierarchy_state_after_remap_();
    };
  }
};

}  // namespace pops::runtime::program::detail

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
  add_compiled_model<Dim>(system, "tracer",
                          Transport{{}, EulerND<Dim>::prepare(Real(1.4)), NoSource{}, NoElliptic{}},
                          "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
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
  system.install_program_step([](double) {});
  system.set_program_block_map({0});
  (void)system.mass("tracer");
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    return 1;
  auto context =
      std::make_shared<runtime::program::AmrProgramContext<Dim>>(system.engine(), &system);
  context->configure_primary_clock("test.clock.cell-local-mpi-refusal");
  const ExecutionLane& lane = context->prepared_execution_lane();

  bool refused = false;
  try {
    const std::array route{runtime::program::SameLevelCellTemporalForwardEulerRoute{0, -1, 0}};
    context->prepare_same_level_cell_temporal_execution("test.clock.cell-local-mpi-refusal", 100, 0,
                                                        route);
  } catch (const std::runtime_error& error) {
    refused = std::string(error.what()) == "cell-local AMR route preparation failed collectively";
  }
  const long refusing_ranks = all_reduce_sum(refused ? 1L : 0L, lane);
  const bool unchanged = system.program_accepted_state().empty();
  return refusing_ranks == lane.size() && unchanged ? 0 : 1;
}

int run_collective_history_remap_refusal() {
  constexpr int Dim = kNativeDimension;
  AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    config.periodicity[axis] = true;
  }
  config.level_count = 2;
  config.regrid_every = 1;

  AmrSystem<Dim> system(config);
  test::install_amr_runtime_authority(system, "tests.history-remap-refusal/runtime@1");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("tracer", "tests.history-remap-refusal/tracer/state@1");
  using Transport = CompositeModel<EulerND<Dim>, NoSource, NoElliptic>;
  add_compiled_model<Dim>(system, "tracer",
                          Transport{{}, EulerND<Dim>::prepare(Real(1.4)), NoSource{}, NoElliptic{}},
                          "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false,
                          "tests.history-remap-refusal/tracer/physical-flux@1");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  std::vector<double> state(static_cast<std::size_t>(EulerND<Dim>::n_vars) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    state[static_cast<std::size_t>(EulerND<Dim>::density_component) * cells + cell] = 1.0;
    state[static_cast<std::size_t>(EulerND<Dim>::energy_component) * cells + cell] = 2.5;
  }
  system.set_conservative_state("tracer", state);
  test::install_prepared_threshold_union(system, {{"tracer", "rho", 0.5}},
                                         "tests.history-remap-refusal/tagging@1");
  if (system.engine() == nullptr || system.engine()->hierarchy().num_levels() != 2)
    return 1;

  auto context =
      std::make_shared<runtime::program::AmrProgramContext<Dim>>(system.engine(), &system);
  context->configure_primary_clock("tests.history-remap-refusal/clock@1");
  context->install([](double) {}, context);
  system.set_program_block_map({0});
  using FluxBudget = typename AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.history-remap-refusal/flux@1", std::vector<FluxBudget>(1, FluxBudget{1, 1}), 0, 0);
  context->for_each_program_resource_level([&](int) {
    context->register_history("tracer.rate", 1, -1, 0, "tracer.U", "cell.conservative",
                              "tests.history-remap-refusal/clock@1", "dense.linear");
  });
  for (const double dt : {0.1, 0.2, 0.3}) {
    context->begin_step(dt);
    context->for_each_program_resource_level([&](int) {
      MultiFab<Dim> sample = context->scratch_state_like(context->state(0));
      sample.set_val(Real(dt));
      context->store_history("tracer.rate", sample, 0);
    });
    context->for_each_program_resource_level(
        [&](int) { context->rotate_histories("tests.history-remap-refusal/clock@1"); });
  }

  auto* engine = system.engine();
  const ExecutionLane& lane = context->prepared_execution_lane();
  const std::uint64_t topology_before = engine->topology_epoch();
  const std::uint64_t materialization_before = engine->materialization_generation();
  const std::string spatial_before{engine->spatial_contract()};
  const auto patches_before = system.patch_boxes();
  const auto state_before = system.block_level_state_global("tracer", 0);
  const double mass_before = system.mass("tracer");
  const auto names_before = system.history_names();
  std::array<bool, 2> initialized_before{};
  std::array<int, 2> fill_before{};
  std::array<std::array<double, 2>, 2> slot_dt_before{};
  std::array<std::array<std::vector<double>, 2>, 2> history_before{};
  for (int level : {0, 1}) {
    initialized_before[static_cast<std::size_t>(level)] =
        system.history_initialized("tracer.rate", level);
    fill_before[static_cast<std::size_t>(level)] = system.history_fill_count("tracer.rate", level);
    for (int slot : {0, 1}) {
      slot_dt_before[static_cast<std::size_t>(level)][static_cast<std::size_t>(slot)] =
          system.history_slot_dt("tracer.rate", level, slot);
      history_before[static_cast<std::size_t>(level)][static_cast<std::size_t>(slot)] =
          system.history_global("tracer.rate", level, slot);
    }
  }

  runtime::program::detail::AmrProgramHistoryRemapCollectiveTestAccess<
      Dim>::install_rank_zero_candidate_metadata_corruption(*context, "tracer.rate", 0);
  bool refused = false;
  try {
    (void)system.regrid_from_prepared_tagging(0);
  } catch (const std::runtime_error& error) {
    refused =
        std::string(error.what()) == "AMR Program hierarchy-state publication failed collectively";
  }
  bool unchanged = engine->topology_epoch() == topology_before &&
                   engine->materialization_generation() == materialization_before &&
                   engine->spatial_contract() == spatial_before &&
                   system.patch_boxes() == patches_before &&
                   system.block_level_state_global("tracer", 0) == state_before &&
                   system.mass("tracer") == mass_before && system.history_names() == names_before;
  for (int level : {0, 1}) {
    unchanged = unchanged &&
                system.history_initialized("tracer.rate", level) ==
                    initialized_before[static_cast<std::size_t>(level)] &&
                system.history_fill_count("tracer.rate", level) ==
                    fill_before[static_cast<std::size_t>(level)];
    for (int slot : {0, 1})
      unchanged =
          unchanged &&
          system.history_slot_dt("tracer.rate", level, slot) ==
              slot_dt_before[static_cast<std::size_t>(level)][static_cast<std::size_t>(slot)] &&
          system.history_global("tracer.rate", level, slot) ==
              history_before[static_cast<std::size_t>(level)][static_cast<std::size_t>(slot)];
  }
  return all_reduce_sum(refused ? 1L : 0L, lane) == lane.size() &&
                 all_reduce_sum(unchanged ? 1L : 0L, lane) == lane.size()
             ? 0
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
  const int result = run_collective_refusal() == 0 ? run_collective_history_remap_refusal() : 1;
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_cell_temporal_program_refusal, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_cell_temporal_program_refusal,
                                    "test_mpi_cell_temporal_program_refusal"),
            0);
}
