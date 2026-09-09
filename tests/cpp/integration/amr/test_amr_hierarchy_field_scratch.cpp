#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/nd/prepared_composite_general_field.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

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
namespace general = pops::elliptic::nd::general_composite_detail;
struct Model : pops::nd::ScalarAdvection<Dim> {
  static constexpr int n_providers = 0;
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};
struct Fixture {
  std::unique_ptr<System> system;
  std::shared_ptr<Context> context;
};

void expect_value(const Field& field, pops::Real value) {
  for (int component = 0; component < field.ncomp(); ++component)
    if (field.local_size() != 0) {
      EXPECT_EQ(pops::reduce_min_local(field, component), value);
      EXPECT_EQ(pops::reduce_max_local(field, component), value);
    }
}

void replace_child(Fixture& fixture, int upper_bound) {
  const auto& parent = fixture.system->engine()->hierarchy().layout(0);
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis)
    upper[axis] = upper_bound;
  const pops::mesh::BoxArray<Dim> boxes(std::vector<pops::Box<Dim>>{{{}, upper}});
  pops::amr::tagging::ClusterOptions<Dim> options;
  options.min_efficiency = 0.7;
  options.min_box_size.fill(1);
  options.max_box_size.fill(16);
  options.budget = {16, 256, 8192, 64, 1U << 20};
  pops::amr::tagging::ClusterResultIdentity<Dim> identity{
      "tests.hierarchy-field-scratch/cluster", parent.exact_identity(), options, {}, boxes.boxes()};
  std::array<int, Dim> ratio{};
  ratio.fill(2);
  auto prepared = fixture.context->prepare_regrid(
      0, pops::amr::RefinementRatio<Dim>(ratio), {boxes, std::move(identity)},
      {.clustered_parent_layout = {16, 120},
       .fine_layout = {16, 512},
       .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()}});
  const auto& state = fixture.system->engine()->hierarchy().state(0);
  Field child(prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
              state.local_rank(), state.ncomp(), state.ghosts());
  child.set_val(pops::Real(7));
  fixture.context->publish_regrid(std::move(prepared), std::move(child));
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
  pops::test::install_amr_runtime_authority(*system, "tests.hierarchy-field-scratch/runtime");
  system->set_temporal_relations({1}, {1}, {"integral_only"});
  system->install_block_state_route("scalar", "tests.hierarchy-field-scratch/state");
  pops::add_compiled_model<Dim>(*system, "scalar", Model{}, "minmod", "rusanov", "conservative",
                                "explicit", 1.4, 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.hierarchy-field-scratch/physical-flux");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= 8;
  system->set_conservative_state("scalar", std::vector<double>(cells, 7.0));
  auto context = pops::runtime::program::make_program_execution_provider(system.get());
  // Explicit bootstrap stages the source before materialization, then initializes its live storage.
  system->set_conservative_state("scalar", std::vector<double>(cells, 7.0));
  system->register_program_hierarchy_tensor_solver_provider(
      pops::elliptic::nd::make_composite_general_field_provider<Dim>());
  context->install([](double) {}, context);
  system->set_program_block_map({0});
  using Budget = System::PreparedAmrProgramFluxExpressionBlockBudget;
  system->install_prepared_amr_program_flux_expression_budget(
      "tests.hierarchy-field-scratch/program@1", {Budget{0, 0}}, 0, 0);
  context->configure_primary_clock("clock.field-scratch");
  Fixture fixture{std::move(system), std::move(context)};
  replace_child(fixture, 1);
  pops::PreparedProviderOptions options;
  options.schema_identity = std::string(general::options_id);
  options.values.emplace("coefficient_components", std::int64_t{1});
  options.values.emplace("reaction.0.0", 1.0);
  options.values.emplace("restart", std::int64_t{2});
  for (int axis = 0; axis < Dim; ++axis)
    for (int side = 0; side < 2; ++side)
      options.values.emplace("physical." + std::to_string(axis) + "." + std::to_string(side),
                             std::string("periodic"));
  for (const std::int64_t key : {701, 702})
    fixture.context->configure_hierarchy_field_solver(
        key, 1, "tests.field/" + std::to_string(key), std::string(general::provider_id),
        "tests.field/plan", "tests.field/screened-periodic@1",
        {std::string(general::coefficient_slot), std::string(general::rhs_slot)},
        std::string(general::solution_slot), options);
  return fixture;
}
}  // namespace

TEST(AmrHierarchyFieldScratch, SolverOwnershipAndExactLevelSlotsReuseWithoutAliasing) {
  auto fixture = prepare_fixture();
  auto& context = *fixture.context;
  ASSERT_EQ(context.nlev(), 2);
  std::array<Field*, 2> levels{};
  context.for_each_program_resource_level([&](int level) {
    auto& solution = context.hierarchy_field_solution(701);
    auto& assembly = context.hierarchy_field_assembly(701, general::rhs_slot);
    EXPECT_THROW(context.scalar_scratch(800, 0, solution), std::invalid_argument);
    EXPECT_THROW(context.scalar_scratch(800, 0, assembly), std::invalid_argument);
    auto& scratch = context.hierarchy_field_scratch(701, 800, 0, 2, 1);
    levels[level] = &scratch;
    EXPECT_EQ(scratch.layout(), solution.layout());
    EXPECT_EQ(scratch.distribution(), solution.distribution());
    EXPECT_EQ(scratch.local_rank(), solution.local_rank());
    EXPECT_EQ(scratch.ncomp(), 2);
    EXPECT_FALSE(scratch.shares_storage_with(solution));
    EXPECT_FALSE(scratch.shares_storage_with(assembly));
    expect_value(scratch, pops::Real(0));
    scratch.set_val(pops::Real(17));
    auto& same = context.hierarchy_field_scratch(701, 800, 0, 2, 1);
    EXPECT_EQ(&same, &scratch);
    expect_value(same, pops::Real(0));
    auto& other_value = context.hierarchy_field_scratch(701, 801, 0, 2, 1);
    auto& other_subslot = context.hierarchy_field_scratch(701, 800, 1, 2, 1);
    auto& other_solve = context.hierarchy_field_scratch(702, 800, 0, 2, 1);
    for (auto* other : {&other_value, &other_subslot, &other_solve}) {
      EXPECT_NE(other, &scratch);
      EXPECT_FALSE(other->shares_storage_with(scratch));
    }
    other_value.set_val(pops::Real(3));
    other_subslot.set_val(pops::Real(5));
    other_solve.set_val(pops::Real(7));
    expect_value(scratch, pops::Real(0));
  });
  EXPECT_NE(levels[0], levels[1]);
  EXPECT_FALSE(levels[0]->shares_storage_with(*levels[1]));
}

TEST(AmrHierarchyFieldScratch, RankLocalUnknownSolveAndShapeRejectWithoutResetThenRetry) {
  auto fixture = prepare_fixture();
  auto& context = *fixture.context;
  const int rank = context.prepared_execution_lane().rank();
  auto& scratch = context.hierarchy_field_scratch(701, 800, 0, 1, 1);
  scratch.set_val(pops::Real(13));
  EXPECT_THROW(context.hierarchy_field_scratch(rank == 0 ? 999 : 701, 800, 0, 1, 1),
               std::exception);
  expect_value(scratch, pops::Real(13));
  EXPECT_THROW(context.hierarchy_field_scratch(701, 800, 0, rank == 0 ? 2 : 1, 1), std::exception);
  expect_value(scratch, pops::Real(13));
  EXPECT_THROW(context.hierarchy_field_scratch(701, 800, 0, 1, rank == 0 ? 0 : 1), std::exception);
  expect_value(scratch, pops::Real(13));
  auto& retry = context.hierarchy_field_scratch(701, 800, 0, 1, 1);
  EXPECT_EQ(&retry, &scratch);
  expect_value(retry, pops::Real(0));
}

TEST(AmrHierarchyFieldScratch, RegridRepreparesSolverAndScratchAgainstTheNewFineLayout) {
  auto fixture = prepare_fixture();
  auto& context = *fixture.context;
  const auto before = context.program_resource_topology();
  context.with_program_resource_level(
      1, [&] { context.hierarchy_field_scratch(701, 800, 0, 1, 1).set_val(pops::Real(19)); });
  const auto old_layout = context.layout(1).exact_identity();
  replace_child(fixture, 2);
  const auto after = context.program_resource_topology();
  EXPECT_NE(before.epoch, after.epoch);
  EXPECT_NE(old_layout, context.layout(1).exact_identity());
  context.with_program_resource_level(1, [&] {
    auto& solution = context.hierarchy_field_solution(701);
    // A new generation permits this slot to acquire a new shape; an old cache would reject it.
    auto& scratch = context.hierarchy_field_scratch(701, 800, 0, 3, 0);
    EXPECT_EQ(scratch.layout(), solution.layout());
    EXPECT_EQ(scratch.distribution(), solution.distribution());
    EXPECT_EQ(scratch.ncomp(), 3);
    expect_value(scratch, pops::Real(0));
    EXPECT_FALSE(scratch.shares_storage_with(solution));
  });
}

TEST(AmrHierarchyFieldScratch, RollbackDiscardsProviderScratchBeforeReconstruction) {
  auto fixture = prepare_fixture();
  auto& context = *fixture.context;
  const auto accepted = fixture.system->program_accepted_state();
  context.for_each_program_resource_level(
      [&](int) { context.hierarchy_field_scratch(701, 800, 0, 1, 1).set_val(pops::Real(19)); });
  fixture.system->begin_step_transaction();
  context.for_each_program_resource_level(
      [&](int) { context.hierarchy_field_scratch(701, 800, 0, 1, 1).set_val(pops::Real(23)); });
  fixture.system->rollback_step_transaction();
  EXPECT_EQ(fixture.system->program_accepted_state(), accepted);
  context.for_each_program_resource_level([&](int level) {
    auto& scratch = context.hierarchy_field_scratch(701, 800, 0, 2, 0);
    EXPECT_EQ(scratch.ncomp(), 2);
    EXPECT_EQ(scratch.layout(), context.hierarchy_field_solution(701).layout());
    expect_value(scratch, pops::Real(0));
    expect_value(fixture.system->engine()->hierarchy().state(level), pops::Real(7));
  });
}
