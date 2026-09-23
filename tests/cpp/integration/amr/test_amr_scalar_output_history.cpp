#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/program/history_sample_identity_codec.hpp>

#include <cmath>
#include <iostream>
#include <vector>

namespace {
template <int Dim>
struct ScalarModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  Law law{};
  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.scalar-output-history.model", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& builder) const {
    for (int axis = 0; axis < Dim; ++axis)
      builder.scalar(law.velocity()[axis]);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  POPS_HD auto recover(const State& state) const { return law.recover(state); }
  POPS_HD auto make_conservative(const Primitive& state) const { return law.make_conservative(state); }
  POPS_HD auto admissibility(const State& state) const { return law.admissibility(state); }
  template <int Axis> POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis> POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};
}  // namespace

// 0/3: ordinary 1:1/2:1 histories; 1: equal-clock output; 2: unequal-clock output.
class AmrScalarOutputHistory : public ::testing::TestWithParam<int> {};

TEST_P(AmrScalarOutputHistory, ExpansionPreservesOnlyAuthenticatedOutputOverlap) {
  constexpr int Dim = pops::kNativeDimension;
  const bool output_only = GetParam() == 1 || GetParam() == 2;
  const int temporal_ratio = GetParam() >= 2 ? 2 : 1;
  pops::AmrSystemConfig<Dim> config;
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 16;
    config.transition_buffers.front()[axis] = 0;
    config.transition_lookaheads.front()[axis] = 0;
    cells *= 16;
  }
  config.regrid_every = 0;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.scalar-output-history/runtime");
  system.set_temporal_relations({temporal_ratio}, {1}, {"integral_only"});
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(
      system, "tracer", ScalarModel<Dim>{}, "minmod", "rusanov", "conservative", "explicit",
      static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.,
      static_cast<double>(pops::kWenoEpsilon), false, "tests.scalar-output-history/flux");
  system.set_conservative_state("tracer", std::vector<double>(cells, 1.));
  pops::test::install_prepared_refine_coarsen_threshold(
      system, {"tracer", "u", .5, pops::test::PreparedThresholdRelation::Above},
      {"tracer", "u", .5, pops::test::PreparedThresholdRelation::Below},
      "tests.scalar-output-history/tagging");
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("clock.macro");
  context->declare_clock_relation("clock.macro", "clock.level.1", temporal_ratio);
  context->install([context](double dt) { context->advance_hierarchy(dt, [](double) {}); }, context);
  system.set_program_block_map({0});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.scalar-output-history/flux-budget", std::vector<FluxBudget>(1, FluxBudget{2, 1}), 0, 0);
  context->for_each_program_resource_level([&](int) {
    context->register_history("tracer.potential", 1, 1, 0, "scalar-history:tracer.potential",
                             output_only ? std::string(pops::runtime::program::kScalarOutputHistorySpace)
                                         : "scalar-field", "clock.macro", "none");
  });
  int step = 0;
  for (double dt : {.1, .2}) {
    ++step;
    context->advance_hierarchy(dt, [&](double) {
      auto value = context->rhs_scratch_like(context->state(0));
      const bool fine = context->level() == 1;
      for (std::size_t patch = 0; patch < value.local_size(); ++patch) {
        const auto view = value.fab(patch).view();
        pops::for_each_cell(value.box(patch), [=] POPS_HD(const pops::Index<Dim>& index) {
          view(index, 0) = pops::Real(step * (fine ? 100 : 10));
          if (fine)
            view(index, 0) += pops::Real(.25) * cos(pops::Real(.31) * index[0]);
        });
      }
      context->store_history("tracer.potential", value, 0);
      context->rotate_histories("clock.macro");
    });
  }
  std::size_t center = 0, stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    center += 8 * stride;
    stride *= 16;
  }
  std::vector<double> partial(cells, .25);
  partial[center] = 1.;
  system.set_conservative_state("tracer", partial);
  system.execute_prepared_tagging(0);
  ASSERT_TRUE(system.regrid_from_prepared_tagging(0));
  const auto before0 = system.history_global("tracer.potential", 1, 0);
  const auto before1 = system.history_global("tracer.potential", 1, 1);
  const auto old_boxes = system.patch_boxes();
  const double time_before = system.time();
  const auto steps_before = system.macro_step();
  system.set_conservative_state("tracer", std::vector<double>(cells, 1.));
  if (output_only) {
    context->for_each_program_resource_level([&](int) {
      EXPECT_THROW(context->history("tracer.potential", 1, 0), std::invalid_argument);
      EXPECT_THROW(context->history_zero_start("tracer.potential", 1, 1, 0),
                   std::invalid_argument);
    });
    const auto expect_atomic_refusal = [&]() {
      system.execute_prepared_tagging(0);
      const auto accepted = system.program_accepted_state();
      EXPECT_THROW(system.regrid_from_prepared_tagging(0), std::exception);
      EXPECT_EQ(system.patch_boxes(), old_boxes);
      EXPECT_EQ(system.program_accepted_state(), accepted);
      EXPECT_EQ(system.history_global("tracer.potential", 1, 0), before0);
      EXPECT_EQ(system.history_global("tracer.potential", 1, 1), before1);
      EXPECT_EQ(system.time(), time_before);
      EXPECT_EQ(system.macro_step(), steps_before);
    };
    if (temporal_ratio != 1) {
      expect_atomic_refusal();
      return;
    }
    const auto parent_identity = system.history_sample_identity("tracer.potential", 0);
    const auto child_identity = system.history_sample_identity("tracer.potential", 1);
    const double dt = system.history_slot_dt("tracer.potential", 1, 0);
    system.restore_history_slot_dt("tracer.potential", 1, 0, 2 * dt);
    expect_atomic_refusal();
    system.restore_history_slot_dt("tracer.potential", 1, 0, dt);
    // Matching unknown IDs do not authenticate a common physical sample time.
    system.restore_history_sample_identity("tracer.potential", 0, {});
    system.restore_history_sample_identity("tracer.potential", 1, {});
    expect_atomic_refusal();
    system.restore_history_sample_identity("tracer.potential", 0, parent_identity);
    system.restore_history_sample_identity("tracer.potential", 1, child_identity);
  }
  system.execute_prepared_tagging(0);
  ASSERT_TRUE(system.regrid_from_prepared_tagging(0));
  EXPECT_EQ(system.time(), time_before);
  EXPECT_EQ(system.macro_step(), steps_before);
  const auto after0 = system.history_global("tracer.potential", 1, 0);
  const auto after1 = system.history_global("tracer.potential", 1, 1);
  std::size_t retained = 0, added = 0, changed = 0;
  double maximum_change = 0.;
  for (std::size_t cell = 0; cell < before0.size(); ++cell) {
    if (before0[cell] != 0.) {
      ++retained;
      if (before0[cell] != after0[cell] || before1[cell] != after1[cell])
        ++changed;
      maximum_change = std::max(maximum_change, std::abs(after1[cell]-before1[cell]));
    } else {
      ++added;
      EXPECT_EQ(after0[cell], 10.);
      EXPECT_EQ(after1[cell], 20.);
    }
  }
  std::cout << "scalar_history_overlap retained=" << retained << " added=" << added
            << " changed=" << changed << " max_change=" << maximum_change
            << " time_change=" << system.time()-time_before << '\n';
  EXPECT_GT(retained, 0u);
  EXPECT_GT(added, 0u);
  EXPECT_EQ(changed, output_only ? 0u : retained);
  EXPECT_EQ(system.history_slot_dt("tracer.potential", 1, 0), .1);
  EXPECT_EQ(system.history_slot_dt("tracer.potential", 1, 1), .2);
  // The archive envelopes deliberately differ by level; compare their authenticated samples.
  EXPECT_EQ(pops::runtime::program::decode_history_sample_identity(
                system.history_sample_identity("tracer.potential", 1), "tracer.potential", 1, 2),
            pops::runtime::program::decode_history_sample_identity(
                system.history_sample_identity("tracer.potential", 0), "tracer.potential", 0, 2));
  const auto accepted = system.program_accepted_state();
  const auto restored = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(accepted);
  EXPECT_EQ(restored.pending_history_remaps.empty(), output_only);
  if (!output_only) {
    // The integrator route still requires a current store before a deferred lag read.
    context->for_each_program_resource_level([&](int level) {
      if (level == 1)
        EXPECT_THROW(context->history("tracer.potential", 1, 0), std::runtime_error);
    });
    return;
  }
  EXPECT_NO_THROW(system.restore_checkpoint_accepted_state(accepted));
  EXPECT_EQ(system.program_accepted_state(), accepted);

  // An output marker cannot grant permission to project a lagged interface flux.
  system.set_conservative_state("tracer", partial);
  system.execute_prepared_tagging(0);
  ASSERT_TRUE(system.regrid_from_prepared_tagging(0));
  context->advance_hierarchy(.3, [&](double) {
    auto& state = context->state(0);
    auto rate = context->rhs_scratch_like(state);
    context->rhs_into(0, state, rate, 0);
    context->store_history("tracer.potential", rate, 0);
    context->rotate_histories("clock.macro");
  });
  const auto flux_boxes = system.patch_boxes();
  const auto flux0 = system.history_global("tracer.potential", 1, 0);
  const auto flux1 = system.history_global("tracer.potential", 1, 1);
  system.set_conservative_state("tracer", std::vector<double>(cells, 1.));
  system.execute_prepared_tagging(0);
  const auto flux_accepted = system.program_accepted_state();
  EXPECT_THROW(system.regrid_from_prepared_tagging(0), std::runtime_error);
  EXPECT_EQ(system.patch_boxes(), flux_boxes);
  EXPECT_EQ(system.program_accepted_state(), flux_accepted);
  EXPECT_EQ(system.history_global("tracer.potential", 1, 0), flux0);
  EXPECT_EQ(system.history_global("tracer.potential", 1, 1), flux1);
}

INSTANTIATE_TEST_SUITE_P(Contracts, AmrScalarOutputHistory, ::testing::Values(0, 1, 2, 3));
