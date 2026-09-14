#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/mesh/storage/mf_arith.hpp>

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr int Dim = pops::kNativeDimension;
using System = pops::AmrSystem<Dim>;
using Context = pops::runtime::program::AmrProgramContext<Dim>;
using Field = pops::MultiFab<Dim>;
constexpr const char* kProgram = "tests.amr-hierarchy-barrier/program@1";
struct Model : pops::nd::ScalarAdvection<Dim> {
  static constexpr int n_providers = 0;
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};
struct Evidence {
  int mode = 0;
  int gathered = 0, continued = 0, solves = 0, publications = 0, completed = 0;
};
struct Fixture {
  std::unique_ptr<System> system;
  std::shared_ptr<Context> context;
  std::shared_ptr<Evidence> evidence;
};

void expect_value(const Field& field, pops::Real value) {
  if (field.local_size() != 0) {
    EXPECT_EQ(pops::reduce_min_local(field), value);
    EXPECT_EQ(pops::reduce_max_local(field), value);
  }
}

Fixture prepare_fixture() {
  pops::comm_init();
  pops::AmrSystemConfig<Dim> config;
  config.explicit_bootstrap = true;
  config.regrid_every = 0;
  config.distribute_coarse = true;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    config.periodicity[axis] = true;
    config.coarse_max_grid[axis] = axis == 0 ? 2 : 8;
  }
  auto system = std::make_unique<System>(config);
  pops::test::install_amr_runtime_authority(*system, "tests.amr-hierarchy-barrier/runtime");
  system->set_temporal_relations({1}, {1}, {"integral_only"});
  system->install_block_state_route("scalar", "tests.amr-hierarchy-barrier/state");
  pops::add_compiled_model<Dim>(*system, "scalar", Model{}, "minmod", "rusanov", "conservative",
                                "explicit", 1.4, 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.amr-hierarchy-barrier/physical-flux");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= 8;
  system->set_conservative_state("scalar", std::vector<double>(cells, 7.0));
  auto context = pops::runtime::program::make_program_execution_provider(system.get());
#ifdef POPS_HAS_KOKKOS
  EXPECT_TRUE(pops::kokkos_initialized_by_pops());
  EXPECT_TRUE(pops::kokkos_atexit_finalize_registered());
#endif
  // Explicit bootstrap stages the source before materialization; initialize its accepted buffers.
  system->set_conservative_state("scalar", std::vector<double>(cells, 7.0));
  auto evidence = std::make_shared<Evidence>();
  System* facade = system.get();
  context->configure_primary_clock("clock.barrier");
  context->install(
      [context, evidence, facade](double dt) {
        context->advance_mapping_hierarchy(
            dt,
            [context, evidence, facade](double) {
              ++evidence->gathered;
              context->set_stage_time(1, 2);
              const auto rank = context->prepared_execution_lane().rank();
              auto kind = Context::HierarchyBarrierKind::linear_solve;
              if (evidence->mode == 4 && rank == 0)
                kind = static_cast<Context::HierarchyBarrierKind>(99);
              const std::string hash =
                  evidence->mode == 2 && rank == 0 ? "foreign-program" : kProgram;
              const std::string dependencies = evidence->mode == 1 && context->level() == 1
                                                   ? "different-level-inputs"
                                                   : "gather/state/half-step";
              if (evidence->mode == 3 && rank == 0)
                context->set_stage_time(0, 1);
              context->suspend_hierarchy_barrier(
                  kind, 101, hash, dependencies, 1, 2,
                  [context, evidence, facade] {
                    ++evidence->solves;
                    EXPECT_EQ(context->level(), 0);
                    EXPECT_EQ(evidence->gathered, facade->n_levels());
                    EXPECT_EQ(facade->time(), 0.0);
                    for (int level = 0; level < facade->n_levels(); ++level)
                      expect_value(facade->engine()->hierarchy().state(level), pops::Real(7));
                    if (evidence->mode == 5) {
                      context->state(0).set_val(pops::Real(19));
                      if (context->prepared_execution_lane().rank() == 0)
                        throw std::runtime_error("injected unique barrier callback failure 91");
                    }
                  },
                  [context, evidence, facade] {
                    ++evidence->continued;
                    EXPECT_EQ(evidence->solves, 1);
                    context->set_stage_time(3, 4);
                    context->suspend_hierarchy_barrier(
                        Context::HierarchyBarrierKind::field_publication, 102, kProgram,
                        "field/publication/three-quarter-step", 3, 4,
                        [context, evidence, facade] {
                          ++evidence->publications;
                          EXPECT_EQ(context->level(), 0);
                          EXPECT_EQ(evidence->continued, facade->n_levels());
                          EXPECT_EQ(facade->time(), 0.0);
                        },
                        [context] { context->state(0).set_val(pops::Real(11)); });
                  });
            },
            context, [evidence] { ++evidence->completed; });
      },
      context);
  system->set_program_block_map({0});
  using Budget = System::PreparedAmrProgramFluxExpressionBlockBudget;
  system->install_prepared_amr_program_flux_expression_budget(kProgram, {Budget{0, 0}}, 0, 0);

  const auto& parent = system->engine()->hierarchy().layout(0);
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = 1;
  const pops::mesh::BoxArray<Dim> boxes(std::vector<pops::Box<Dim>>{{{}, upper}});
  pops::amr::tagging::ClusterOptions<Dim> options;
  options.min_efficiency = 0.7;
  options.min_box_size.fill(1);
  options.max_box_size.fill(16);
  options.budget = {16, 256, 8192, 64, 1U << 20};
  pops::amr::tagging::ClusterResultIdentity<Dim> identity{
      "tests.amr-hierarchy-barrier/cluster", parent.exact_identity(), options, {}, boxes.boxes()};
  std::array<int, Dim> ratio{};
  ratio.fill(2);
  auto prepared = context->prepare_regrid(
      0, pops::amr::RefinementRatio<Dim>(ratio), {boxes, std::move(identity)},
      {.clustered_parent_layout = {16, 120},
       .fine_layout = {16, 120},
       .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()}});
  const auto& state = system->engine()->hierarchy().state(0);
  Field child(prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
              state.local_rank(), state.ncomp(), state.ghosts());
  child.set_val(pops::Real(7));
  context->publish_regrid(std::move(prepared), std::move(child));
  return {std::move(system), std::move(context), std::move(evidence)};
}

void expect_success(const Fixture& fixture) {
  EXPECT_EQ(fixture.evidence->solves, 1);
  EXPECT_EQ(fixture.evidence->publications, 1);
  EXPECT_EQ(fixture.evidence->completed, 1);
  EXPECT_EQ(fixture.evidence->gathered, 2);
  EXPECT_EQ(fixture.evidence->continued, 2);
  EXPECT_DOUBLE_EQ(fixture.system->time(), 0.125);
  for (int level = 0; level < fixture.system->n_levels(); ++level)
    expect_value(fixture.system->engine()->hierarchy().state(level), pops::Real(11));
}
}  // namespace

TEST(AmrHierarchyBarrierContinuation, AllLevelsGatherBeforeOneCallbackAndResumeBeforePublication) {
  auto fixture = prepare_fixture();
  ASSERT_EQ(fixture.system->n_levels(), 2);
  fixture.system->step(0.125);
  expect_success(fixture);
}

TEST(AmrHierarchyBarrierContinuation, DependencyHashFractionAndKindFailBeforeCallbackThenRetry) {
  auto fixture = prepare_fixture();
  const auto accepted = fixture.system->program_accepted_state();
  for (int mode = 1; mode <= 4; ++mode) {
    *fixture.evidence = Evidence{};
    fixture.evidence->mode = mode;
    EXPECT_THROW(fixture.system->step(0.125), std::exception);
    EXPECT_EQ(fixture.evidence->solves, 0);
    EXPECT_EQ(fixture.evidence->publications, 0);
    EXPECT_EQ(fixture.evidence->completed, 0);
    EXPECT_EQ(fixture.system->time(), 0.0);
    EXPECT_EQ(fixture.system->program_accepted_state(), accepted);
    for (int level = 0; level < fixture.system->n_levels(); ++level)
      expect_value(fixture.system->engine()->hierarchy().state(level), pops::Real(7));
  }
  *fixture.evidence = Evidence{};
  fixture.system->step(0.125);
  expect_success(fixture);
}

TEST(AmrHierarchyBarrierContinuation, RankLocalUniqueCallbackFailureRollsBackAndCanRetry) {
  auto fixture = prepare_fixture();
  const auto accepted = fixture.system->program_accepted_state();
  fixture.evidence->mode = 5;
  EXPECT_THROW(fixture.system->step(0.125), std::exception);
  EXPECT_EQ(fixture.evidence->solves, 1);
  EXPECT_EQ(fixture.evidence->publications, 0);
  EXPECT_EQ(fixture.evidence->completed, 0);
  EXPECT_EQ(fixture.system->time(), 0.0);
  EXPECT_EQ(fixture.system->program_accepted_state(), accepted);
  for (int level = 0; level < fixture.system->n_levels(); ++level)
    expect_value(fixture.system->engine()->hierarchy().state(level), pops::Real(7));
  *fixture.evidence = Evidence{};
  fixture.system->step(0.125);
  expect_success(fixture);
}
