#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include "native_dso_compiler.hpp"
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/nd/prepared_composite_general_field.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>

#include <array>
#include <cstdio>
#include <ctime>
#include <fstream>
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

std::string rollback_program_source() {
  // This artifact owns its actual Program/context. The real loader alone grants snapshot
  // enrollment; a direct context.install callback deliberately has no artifact authority.
  return R"CPP(
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <memory>
#include <stdexcept>
    using Context = pops::runtime::program::AmrProgramContext<pops::kNativeDimension>;
    static std::weak_ptr<Context> installed_context;
    static int visits = 0;
    extern "C" void pops_test_field_context(std::shared_ptr<Context>* output) {
      *output = installed_context.lock();
    }
    extern "C" int pops_test_field_visits() {
      return visits;
    }
    extern "C" const char* pops_program_abi_key() {
      return POPS_ABI_KEY_LITERAL;
    }
    extern "C" const char* pops_program_route_manifest() {
      return pops::kRouteRegistrySignature;
    }
    extern "C" const char* pops_program_name() {
      return "authenticated-field-scratch-rollback";
    }
    extern "C" const char* pops_program_hash() {
      return "tests.field-scratch/authenticated-rollback@1";
    }
    extern "C" int pops_program_operator_authority_count() {
      return 0;
    }
    extern "C" std::uint64_t pops_program_operator_authority_word(int, int) {
      return 0;
    }
    extern "C" int pops_program_block_count() {
      return 1;
    }
    extern "C" const char* pops_program_block_name(int block) {
      return block == 0 ? "scalar" : "";
    }
    extern "C" bool pops_program_has_flux_expression() {
      return false;
    }
    extern "C" int pops_program_flux_expression_budget_count() {
      return 1;
    }
    extern "C" std::uint64_t pops_program_interface_coupling_application_bound() {
      return 0;
    }
    extern "C" std::uint64_t pops_program_interface_coupling_identity_character_bound() {
      return 0;
    }
    extern "C" std::uint64_t pops_program_flux_rhs_basis_bound(int) {
      return 0;
    }
    extern "C" std::uint64_t pops_program_flux_coefficient_term_bound(int) {
      return 0;
    }
    extern "C" int pops_program_checkpoint_history_count() {
      return 0;
    }
    extern "C" const char* pops_program_checkpoint_history_name(int) {
      return "";
    }
    extern "C" int pops_program_checkpoint_history_owner(int) {
      return 0;
    }
    extern "C" const char* pops_program_checkpoint_history_state_identity(int) {
      return "";
    }
    extern "C" const char* pops_program_checkpoint_history_space_identity(int) {
      return "";
    }
    extern "C" const char* pops_program_checkpoint_history_clock_identity(int) {
      return "";
    }
    extern "C" const char* pops_program_checkpoint_history_interpolation_identity(int) {
      return "";
    }
    extern "C" int pops_program_checkpoint_history_depth(int) {
      return 0;
    }
    extern "C" int pops_program_checkpoint_history_components(int) {
      return 0;
    }
    extern "C" int pops_program_checkpoint_logical_clock_count() {
      return 1;
    }
    extern "C" const char* pops_program_checkpoint_logical_clock_identity(int clock) {
      return clock == 0 ? "clock.field-scratch" : "";
    }
    extern "C" const char* pops_program_checkpoint_temporal_provider_identity() {
      return "pops.temporal-partition.global@1";
    }
    extern "C" std::uint64_t pops_program_checkpoint_temporal_cell_capacity() {
      return 0;
    }
    extern "C" std::uint64_t pops_program_checkpoint_temporal_cells_per_topology_cell() {
      return 0;
    }
    extern "C" int pops_module_operator_count() {
      return 0;
    }
    extern "C" const char* pops_module_operator_owner(int) {
      return "";
    }
    extern "C" const char* pops_module_operator_name(int) {
      return "";
    }
    extern "C" const char* pops_module_operator_kind(int) {
      return "";
    }
    extern "C" const char* pops_module_operator_signature(int) {
      return "";
    }
    extern "C" const char* pops_module_operator_requirements(int) {
      return "";
    }
    extern "C" int pops_module_state_space_count() {
      return 1;
    }
    extern "C" const char* pops_module_state_space_name(int space) {
      return space == 0 ? "U" : "";
    }
    extern "C" const char* pops_module_state_space_owner(int space) {
      return space == 0 ? "scalar" : "";
    }
    extern "C" int pops_module_field_space_count() {
      return 0;
    }
    extern "C" const char* pops_module_field_space_name(int) {
      return "";
    }
    extern "C" const char* pops_module_field_space_owner(int) {
      return "";
    }
    extern "C" void pops_install_program_amr(pops::AmrSystem<pops::kNativeDimension>* system) {
      auto context = pops::runtime::program::make_program_execution_provider(system);
      installed_context = context;
      context->configure_primary_clock("clock.field-scratch");
      context->install(
          [context](double dt) {
            const bool reject = dt > 0.1;
            context->advance_mapping_hierarchy(
                dt,
                [context, reject](double) {
                  const int rank = context->prepared_execution_lane().rank();
                  auto& scratch =
                      context->hierarchy_field_scratch(701, 800, 0, reject ? 1 : 2, reject ? 1 : 0);
                  if (scratch.local_size() != 0 && (pops::reduce_min_local(scratch) != 0 ||
                                                    pops::reduce_max_local(scratch) != 0))
                    throw std::runtime_error("field scratch was not reconstructed/reset");
                  scratch.set_val(pops::Real(23));
                  ++visits;
                  if (reject && rank == 0)
                    throw std::runtime_error("injected authenticated field scratch failure");
                },
                context, [] {});
          },
          context);
    }
  )CPP";
}

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

Fixture prepare_fixture(const std::string& artifact = {}) {
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
  std::shared_ptr<Context> context;
  if (artifact.empty()) {
    context = pops::runtime::program::make_program_execution_provider(system.get());
    context->install([](double) {}, context);
    system->set_program_block_map({0});
    using Budget = System::PreparedAmrProgramFluxExpressionBlockBudget;
    system->install_prepared_amr_program_flux_expression_budget(
        "tests.hierarchy-field-scratch/program@1", {Budget{0, 0}}, 0, 0);
    context->configure_primary_clock("clock.field-scratch");
  } else {
    system->install_program(artifact);
    const auto handle = pops::dynlib::open(artifact);
    if (!pops::dynlib::valid(handle))
      throw std::runtime_error("cannot inspect the installed field scratch fixture artifact");
    auto get_context = reinterpret_cast<void (*)(std::shared_ptr<Context>*)>(
        pops::dynlib::sym(handle, "pops_test_field_context"));
    if (get_context)
      get_context(&context);
    pops::dynlib::close(handle);
    if (!context)
      throw std::runtime_error("field scratch fixture did not install its exact context");
  }
  // Explicit bootstrap stages the source before materialization, then initializes its live storage.
  system->set_conservative_state("scalar", std::vector<double>(cells, 7.0));
  system->register_program_hierarchy_tensor_solver_provider(
      pops::elliptic::nd::make_composite_general_field_provider<Dim>());
  // Topology refresh restores accepted Program state, whose capacity is sealed at bind.
  if (!artifact.empty())
    system->mark_bound();
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
  pops::comm_init();
  const std::string stem = std::string(POPS_TEST_TMPDIR) + "/amr_field_scratch_" +
                           std::to_string(pops::my_rank()) + "_" + std::to_string(std::clock());
  const std::string source = stem + ".cpp", library = stem + ".so";
  {
    std::ofstream output(source);
    output << rollback_program_source();
  }
  const auto package = pops::test::native_dso::compile_shared(
      source, library, "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI");
  if (!package.ok)
    pops::test::native_dso::report_compile_failure("AmrHierarchyFieldScratch", package);
  ASSERT_EQ(pops::all_reduce_min(package.ok ? 1L : 0L), 1L);
  auto fixture = prepare_fixture(library);
  auto& context = *fixture.context;
  ASSERT_EQ(fixture.system->installed_program_hash(),
            "tests.field-scratch/authenticated-rollback@1");
  const auto handle = pops::dynlib::open(library);
  ASSERT_TRUE(pops::dynlib::valid(handle));
  auto visits = reinterpret_cast<int (*)()>(pops::dynlib::sym(handle, "pops_test_field_visits"));
  ASSERT_NE(visits, nullptr);
  const auto accepted = fixture.system->program_accepted_state();
  context.for_each_program_resource_level(
      [&](int) { context.hierarchy_field_scratch(701, 800, 0, 1, 1).set_val(pops::Real(19)); });
  EXPECT_THROW(fixture.system->step(0.125), std::exception);
  EXPECT_EQ(visits(), 1);
  EXPECT_EQ(fixture.system->program_accepted_state(), accepted);
  EXPECT_EQ(fixture.system->macro_step(), 0);
  EXPECT_DOUBLE_EQ(fixture.system->time(), 0.0);
  // The artifact requests a new shape at the same solve/value/subslot on retry. Only the real
  // facade-owned context rollback clears that old contract; weakening the shape guard is invalid.
  EXPECT_NO_THROW(fixture.system->step(0.0625));
  EXPECT_EQ(visits(), 3);
  EXPECT_EQ(fixture.system->macro_step(), 1);
  EXPECT_DOUBLE_EQ(fixture.system->time(), 0.0625);
  context.for_each_program_resource_level([&](int level) {
    auto& scratch = context.hierarchy_field_scratch(701, 800, 0, 2, 0);
    EXPECT_EQ(scratch.ncomp(), 2);
    EXPECT_EQ(scratch.layout(), context.hierarchy_field_solution(701).layout());
    expect_value(scratch, pops::Real(0));
    expect_value(fixture.system->engine()->hierarchy().state(level), pops::Real(7));
  });
  pops::dynlib::close(handle);
  std::remove(source.c_str());
  std::remove(library.c_str());
}
