#include <gtest/gtest.h>

#include <pops/core/model/physical_model.hpp>
#include <pops/core/state/state.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;
using namespace pops::runtime::program;

namespace {

struct LinearTransportModel {
  using State = StateVec<1>;
  using Prim = State;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;

  Real velocity_x = Real(0.7);
  Real velocity_y = Real(-0.2);

  POPS_HD State flux(const State& state, const auto&, int axis) const {
    return State{(axis == 0 ? velocity_x : velocity_y) * state[0]};
  }
  POPS_HD Real max_wave_speed(const State&, const auto&, int axis) const {
    const Real velocity = axis == 0 ? velocity_x : velocity_y;
    return velocity < Real(0) ? -velocity : velocity;
  }
  POPS_HD State source(const State&, const Aux&) const { return State{Real(0)}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }
  POPS_HD Prim to_primitive(const State& state) const { return state; }
  POPS_HD State to_conservative(const Prim& primitive) const { return primitive; }

  [[nodiscard]] static constexpr PreparedProviderIdentity
  transport_model_provider_identity() noexcept {
    return {"pops.test.program-cell-local-transport", 1};
  }
  void serialize_exact_transport_parameters(ExactContractBuilder& contract) const {
    contract.scalar(velocity_x).scalar(velocity_y);
  }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  }
};

static_assert(PhysicalModel<LinearTransportModel>);
static_assert(detail::ExactAmrTransportModelProvider<LinearTransportModel>);

std::vector<double> initial_state(int n) {
  std::vector<double> state(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(n);
      const double y = (static_cast<double>(j) + 0.5) / static_cast<double>(n);
      state[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
            static_cast<std::size_t>(i)] =
          1.0 + 0.1 * std::sin(2.0 * std::numbers::pi * x) * std::cos(2.0 * std::numbers::pi * y);
    }
  return state;
}

std::shared_ptr<AmrProgramContext> install_cell_local_program(AmrSystem& system) {
  system.install_program_step([](double) {});
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    throw std::runtime_error("cell-local Program test requires a materialized AMR runtime");
  auto context = std::make_shared<AmrProgramContext>(system.engine(), &system);
  context->configure_primary_clock("test.clock.cell-local");
  context->prepare_same_level_cell_temporal_execution("test.clock.cell-local", 100, 0);
  context->install([context](double dt) { context->advance_same_level_cell_temporal(dt); },
                   context);
  system.set_program_block_map({0});
  return context;
}

double state_sum(AmrSystem& system) {
  const std::vector<double> values = system.density("tracer");
  return std::accumulate(values.begin(), values.end(), 0.0);
}

}  // namespace

TEST(test_cell_temporal_program_route,
     installed_program_commits_exact_ticks_state_and_conservative_face_ledger) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  constexpr int n = 8;
  AmrSystemConfig config;
  config.n = n;
  config.L = 1.0;
  config.level_count = 1;
  config.regrid_every = 0;
  config.periodicity = {true, true};

  AmrSystem system(config);
  add_compiled_model(system, "tracer", LinearTransportModel{}, "none", "rusanov", "conservative",
                     "euler");
  system.set_density("tracer", initial_state(n));
  const auto context = install_cell_local_program(system);
  const double sum_before = state_sum(system);
  const std::vector<double> state_before = system.density("tracer");

  system.step(0.01);

  EXPECT_NE(system.density("tracer"), state_before);
  EXPECT_NEAR(state_sum(system), sum_before,
              64.0 * std::numeric_limits<double>::epsilon() * std::abs(sum_before));
  const auto manifest = system.program_temporal_partition_manifest();
  ASSERT_FALSE(manifest.empty());
  EXPECT_EQ(manifest.front()[1], "cell_local");
  EXPECT_EQ(manifest.front()[2], kSameLevelTransportEulerStageFluxProvider);
  EXPECT_EQ(manifest.front()[4], "1");
  EXPECT_EQ(manifest.front()[5], "100");

  const SameLevelCellIntegratedFluxLedger& ledger = context->accepted_same_level_cell_flux_ledger();
  EXPECT_EQ(ledger.begin_tick(), 0);
  EXPECT_EQ(ledger.end_tick(), 1);
  EXPECT_EQ(ledger.publication_generation(), 1u);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const std::size_t cell = static_cast<std::size_t>(j * n + i);
      const std::size_t right = static_cast<std::size_t>(j * n + (i + 1) % n);
      const std::size_t upper = static_cast<std::size_t>(((j + 1) % n) * n + i);
      EXPECT_DOUBLE_EQ(ledger.integrated_flux(cell, SameLevelCellFace::XHigh, 0),
                       ledger.integrated_flux(right, SameLevelCellFace::XLow, 0));
      EXPECT_DOUBLE_EQ(ledger.integrated_flux(cell, SameLevelCellFace::YHigh, 0),
                       ledger.integrated_flux(upper, SameLevelCellFace::YLow, 0));
    }
}

TEST(test_cell_temporal_program_route,
     invalid_tick_outer_rollback_and_same_topology_restart_remain_atomic) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  constexpr int n = 8;
  AmrSystemConfig config;
  config.n = n;
  config.L = 1.0;
  config.level_count = 1;
  config.regrid_every = 0;
  config.periodicity = {true, true};

  AmrSystem system(config);
  add_compiled_model(system, "tracer", LinearTransportModel{}, "none", "rusanov", "conservative",
                     "euler");
  system.set_density("tracer", initial_state(n));
  const auto context = install_cell_local_program(system);
  system.step(0.01);

  const std::vector<double> accepted_state = system.density("tracer");
  const std::vector<std::uint8_t> accepted_bytes = system.program_accepted_state();
  const double accepted_time = system.time();
  const int accepted_step = system.macro_step();
  const auto accepted_ledger = context->accepted_same_level_cell_flux_ledger().accepted_state();

  EXPECT_THROW(system.step(0.015), std::invalid_argument);
  EXPECT_EQ(system.density("tracer"), accepted_state);
  EXPECT_EQ(system.program_accepted_state(), accepted_bytes);
  EXPECT_DOUBLE_EQ(system.time(), accepted_time);
  EXPECT_EQ(system.macro_step(), accepted_step);
  EXPECT_EQ(context->accepted_same_level_cell_flux_ledger().publication_generation(),
            accepted_ledger.publication_generation);

  system.begin_step_transaction();
  system.step(0.01);
  system.rollback_step_transaction();
  EXPECT_EQ(system.density("tracer"), accepted_state);
  EXPECT_EQ(system.program_accepted_state(), accepted_bytes);
  EXPECT_DOUBLE_EQ(system.time(), accepted_time);
  EXPECT_EQ(system.macro_step(), accepted_step);
  EXPECT_THROW(context->accepted_same_level_cell_flux_ledger(), std::logic_error);

  system.step(0.01);
  EXPECT_EQ(context->accepted_same_level_cell_flux_ledger().publication_generation(), 1u);
  const std::vector<std::uint8_t> restart_bytes = system.program_accepted_state();
  system.begin_restart_transaction();
  system.restore_checkpoint_accepted_state(restart_bytes);
  system.commit_restart_transaction();
  EXPECT_THROW(context->accepted_same_level_cell_flux_ledger(), std::logic_error);
  EXPECT_NO_THROW(system.step(0.01));
  EXPECT_EQ(context->accepted_same_level_cell_flux_ledger().publication_generation(), 1u);
}
