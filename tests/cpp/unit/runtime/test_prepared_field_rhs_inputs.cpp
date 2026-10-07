// FieldRhsV2 is exercised through real System/AMR issuers. No input proof is constructed here.
// These unit witnesses are not generated-package/C25 or whole-backend qualification.
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "explicit_amr_program.hpp"
#include "amr_tagging_test_authority.hpp"
#include "field_rhs_value_program_fixture.hpp"
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/dynamic/dynlib.hpp>
#include <pops/runtime/system/prepared_field_rhs_inputs.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/amr/field_solver_options.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>
#include <cstdint>

namespace {
using namespace pops;
using namespace pops::runtime::system;
constexpr int D = kNativeDimension;
constexpr int kCells = 8;
constexpr Real kPi = Real(3.141592653589793238462643383279502884L);
constexpr Real kParameter = Real(1.25);
using Field = MultiFab<D>;
using Inputs = PreparedFieldRhsInputs<D>;
static_assert(!std::is_default_constructible_v<Inputs>);
static_assert(!std::is_aggregate_v<Inputs>);
static_assert(std::is_copy_constructible_v<Inputs>);

int kernel_component(std::size_t component) {
  if (component > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("Field RHS fixture component exceeds the kernel index range");
  return static_cast<int>(component);
}

// Genuine Program-only storage via the production generated-block factory, not dummy closures.
struct StorageModel {
  using State = StateVec<1>;
  using Primitive = State;
  static constexpr int dimension = D, n_vars = 1, n_providers = 0;
  static constexpr bool program_only_storage = true;
  static constexpr int program_state_ghost_depth = 1;
  static PreparedProviderIdentity provider_identity() { return {"test.field-rhs-v2.state", 1}; }
  void serialize_exact_parameters(ExactContractBuilder& value) const { value.scalar(std::uint32_t{1}); }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  }
  POPS_HD nd::StateConversion<Primitive> recover(const State& u) const { return {u, {}}; }
  POPS_HD nd::StateConversion<State> make_conservative(const Primitive& u) const { return {u, {}}; }
  POPS_HD nd::StateConversionStatus admissibility(const State&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0] - Real(2); }
};

// The public facade adapter authenticates name, arity, variables, geometry and cadence.
// Its model-owned preparer delegates the complete request to the real Core storage factory.
PreparedSystemBlock<D> prepare_exact_system_block(
    CompiledSystemBlockPreparation<D, StorageModel> request) {
  return prepare_generated_system_block(std::move(request));
}
struct ProviderRhsModel {
  using State = StateVec<1>;
  static constexpr int dimension = D, n_vars = 1, n_providers = 2;
  bool nonfinite = false;
  POPS_HD Real elliptic_rhs(const State& u, const ProviderValues<2>& p) const {
    return nonfinite ? std::numeric_limits<Real>::quiet_NaN() : u[0] * u[0] + p[0] + p[1];
  }
};

void ready() {
  static Kokkos::ScopeGuard guard;
  comm_init();
}
SystemConfig<D> uniform_config() {
  ready();
  SystemConfig<D> c;
  for (int axis = 0; axis < D; ++axis) {
    c.shape[axis] = kCells; c.lower[axis] = 0; c.upper[axis] = 1;
    c.periodicity[axis] = false;
  }
  return c;
}
std::vector<double> initial(int cells = kCells) {
  std::size_t total = 1;
  for (int axis = 0; axis < D; ++axis) total *= cells;
  std::vector<double> values(total);
  for (std::size_t i = 0; i < total; ++i)
    values[i] = 2 + std::cos(2 * double(kPi) * (double(i % cells) + .5) / cells);
  return values;
}

// Device functors are named: the tests also remain usable in a CUDA-selected C++ build.
struct AnalyticAuxKernel {
  FieldView<Real, D> output;
  int component, cells;
  Real time;
  POPS_HD void operator()(const Index<D>& cell) const {
    const Real x = (Real(cell[0]) + Real(.5)) / Real(cells);
    output(cell, component) = Kokkos::sin(Real(2) * kPi * x) + Real(3) * time;
  }
};
struct DerivedAuxKernel {
  FieldView<const Real, D> input;
  FieldView<Real, D> output;
  int in_component, out_component;
  Real parameter;
  POPS_HD void operator()(const Index<D>& cell) const {
    output(cell, out_component) = Real(2) * input(cell, in_component) + parameter;
  }
};
struct RhsErrorKernel {
  FieldView<const Real, D> state, rhs;
  FieldView<Real, D> error;
  Geometry<D> frame;
  Real time;
  POPS_HD void operator()(const Index<D>& cell) const {
    const Real a = Kokkos::sin(Real(2) * kPi * frame.cell_coordinate(0, cell[0])) + Real(3) * time;
    const Real u = state(cell, 0);
    const Real expected = u * u + Real(3) * a + kParameter;
    error(cell, 0) = Kokkos::abs(rhs(cell, 0) - expected) / (Real(1) + Kokkos::abs(expected));
  }
};
struct ParityErrorKernel {
  FieldView<const Real, D> first, second;
  FieldView<Real, D> error;
  POPS_HD void operator()(const Index<D>& cell) const {
    error(cell, 0) = Kokkos::abs(first(cell, 0) - second(cell, 0));
  }
};

template<class Runtime>
AuxiliaryComponentKey install_auxiliaries(Runtime& system, int count) {
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "field", "scalar"};
  AuxiliaryStorageShape<D> shape;
  for (int axis = 0; axis < D; ++axis) shape.halo[axis] = 1;
  const AuxiliaryComponentKey a{"test.field-rhs-v2.owner", "aux", "analytic", "a"};
  const AuxiliaryComponentKey b{"test.field-rhs-v2.owner", "aux", "derived", "b"};
  const AuxiliaryComponentKey phi{"test.field-rhs-v2.owner", "field", "potential", "potential"};
  using Provider = PreparedAuxiliaryProvider<D>;
  if (count != 0) {
    system.install_prepared_auxiliary_provider(Provider{
      "test.field-rhs-v2.analytic", AuxiliaryProviderKind::derived,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{a, contract, shape}}, {}, Provider::launcher_type::trusted_extension(
        PreparedProviderIdentity{"test.field-rhs-v2.analytic", 1}, "space-and-stage-time",
        [](const AuxiliaryKernelLaunchContext<D>& launch) {
          const Real time = launch.point.require_physical_time("test.field-rhs-v2.clock");
          const auto address = launch.outputs.at(0).address;
          auto& output = *launch.storage.candidate->find(address.group);
          const int cells = kCells * (1 << launch.point.level);
          for (std::size_t local = 0; local < output.local_size(); ++local)
            for_each_cell(output.box(local), AnalyticAuxKernel{
              output.fab(local).view(), kernel_component(address.component), cells, time});
          Kokkos::fence();
        })});
    system.install_prepared_auxiliary_provider(Provider{
      "test.field-rhs-v2.derived", AuxiliaryProviderKind::derived,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{b, contract, shape}}, {{a, contract, shape}}, Provider::launcher_type::trusted_extension(
        PreparedProviderIdentity{"test.field-rhs-v2.derived", 1}, "twice-aux-plus-runtime-param",
        [&system](const AuxiliaryKernelLaunchContext<D>& launch) {
          // A real System/AMR-owned RuntimeParams value; no parameter is hidden in State/ProviderValues.
          const Real parameter = system.program_params(0).get(0);
          const auto source = launch.dependencies.at(0).address;
          const auto target = launch.outputs.at(0).address;
          const auto& input = *launch.storage.candidate->find(source.group);
          auto& output = *launch.storage.candidate->find(target.group);
          for (std::size_t local = 0; local < output.local_size(); ++local)
            for_each_cell(output.box(local), DerivedAuxKernel{
              input.fab(local).view(), output.fab(local).view(),
              kernel_component(source.component), kernel_component(target.component), parameter});
          Kokkos::fence();
        })});
    system.install_auxiliary_consumer_plan(AuxiliaryConsumerProviderPlan<D>{
      "test.field-rhs-v2.consumer", {{{a, contract, shape}, 0}, {{b, contract, shape}, 1}}});
  }
  system.install_prepared_auxiliary_provider(Provider{
    "test.field-rhs-v2.field", AuxiliaryProviderKind::field_output,
    {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
    {{phi, contract, shape}}, {}});
  system.seal_auxiliary_providers();
  return phi;
}

struct Observation {
  std::optional<Inputs> copied;
  std::set<int> levels;
  std::size_t cells = 0;
  bool inject_nonfinite = false;
};

void expect_expired(const Inputs& inputs) {
  EXPECT_THROW((void)inputs.state(), std::logic_error);
  EXPECT_THROW((void)inputs.frame(), std::logic_error);
  EXPECT_THROW((void)inputs.point(), std::logic_error);
  EXPECT_THROW((void)inputs.boundary_point(), std::logic_error);
  EXPECT_THROW((void)inputs.level(), std::logic_error);
  EXPECT_THROW((void)inputs.binding_identity(), std::logic_error);
  EXPECT_THROW((void)inputs.source_identity(), std::logic_error);
  EXPECT_THROW((void)inputs.consumer_qid(), std::logic_error);
  EXPECT_THROW((void)inputs.topology_epoch(), std::logic_error);
  EXPECT_THROW((void)inputs.materialization_generation(), std::logic_error);
  EXPECT_THROW((void)inputs.execution(), std::logic_error);
  EXPECT_THROW((void)inputs.lane(), std::logic_error);
  EXPECT_THROW(inputs.validate_provider_inputs<2>(), std::logic_error);
  EXPECT_THROW((void)inputs.provider_values_view<2>(0), std::logic_error);
  // Refusal convergence must remain available after all executable getters expire.
  EXPECT_NO_THROW((void)inputs.consensus_lane());
}
FieldRhsCallbackV2<D> observing_callback(Observation& observed) {
  return [&observed](const Inputs& inputs, Field& rhs) {
    const auto factory = make_poisson_rhs_v2(ProviderRhsModel{
      observed.inject_nonfinite && my_rank() == 0});
    inputs.validate_provider_inputs<2>();
    EXPECT_EQ(inputs.consumer_qid(), "test.field-rhs-v2.consumer");
    EXPECT_EQ(inputs.binding_identity(), "test.field-rhs-v2.binding");
    EXPECT_EQ(inputs.boundary_point().level, inputs.level());
    EXPECT_DOUBLE_EQ(inputs.point().time, inputs.boundary_point().physical_time);
    EXPECT_FALSE(inputs.source_identity().empty());
    observed.copied.emplace(inputs); // Actual issued input; this is not a new proof.
    observed.levels.insert(inputs.level());
    const auto& state = inputs.state();
    Field bad(rhs.layout(), rhs.distribution(), rhs.local_rank(), 2, rhs.ghosts());
    EXPECT_ANY_THROW(factory(inputs, bad)); // Collective convergence may wrap a serial argument error.
    EXPECT_THROW(inputs.validate_provider_inputs<1>(), std::invalid_argument);
    EXPECT_THROW((void)inputs.provider_values_view<2>(state.local_size()), std::invalid_argument);
    factory(inputs, rhs);
    const auto frame = inputs.frame();
    const Real time = inputs.point().time;
    Field error(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
    error.set_val(0);
    for (std::size_t local = 0; local < rhs.local_size(); ++local) {
      for_each_cell(inputs.execution(), rhs.box(local), RhsErrorKernel{
        state.fab(local).view(), std::as_const(rhs).fab(local).view(), error.fab(local).view(), frame, time});
      observed.cells += static_cast<std::size_t>(rhs.box(local).numPts());
    }
    inputs.execution().fence();
    EXPECT_LE(reduce_max_local(error), Real(256) * std::numeric_limits<Real>::epsilon());
  };
}

void install_uniform_field(System<D>& system, Observation& observed, int count = 2,
                           bool fault = false, int cells = kCells,
                           Real domain_lower = 0, Real domain_upper = 1) {
  Extent<D> extent; RealVector<D> lower{}, upper{};
  for (int axis = 0; axis < D; ++axis) {
    extent[axis] = cells; lower[axis] = domain_lower; upper[axis] = domain_upper;
  }
  const auto frame = Geometry<D>::from_bounds(Box<D>::from_extents(extent), lower, upper);
  const auto& actual_frame = system.prepared_block_geometry();
  if (actual_frame.domain() != frame.domain() || actual_frame.lower() != frame.lower() ||
      actual_frame.upper() != frame.upper())
    throw std::invalid_argument("Field RHS fixture geometry differs from its actual System");
  system.install_prepared_boundary_execution_lane(std::make_shared<ExecutionLane>(
    ExecutionLane::duplicate_world_collectively("test.field-rhs-v2.execution")));
  system.install_block_state_route("material", "test.field-rhs-v2.state");
  system.install_prepared_block(prepare_compiled_system_block<D>(
    system, "material", StorageModel{}, "state_storage", "unavailable", "conservative",
    "explicit", 1.4, 1, true, 1));
  system.register_configured_field_solver_provider("cartesian_cg", "test.field-rhs-v2.slot",
    {"pops.system.cartesian-cg-options@1",
     {{"abs_tol", 0.0}, {"max_iterations", std::int64_t{200}}, {"rel_tol", 1e-12}}});
  system.set_field_solver_plan("test.field-rhs-v2.slot", "test.field-rhs-v2.plan",
    "test.field-rhs-v2.provider", "test.field-rhs-v2.owner", "material", "potential",
    {"test.field-rhs-v2.binding"}, {"material"}, {"potential"}, {1.0}, "test.field-rhs-v2.slot");
  const auto faces = std::vector<double>(2 * D, 0.0);
  const auto dirichlet_alpha = std::vector<double>(2 * D, 1.0);
  system.set_field_topology_authority("test.field-rhs-v2.slot", "builtin_rectangular_cell_graph_v1",
    "test.field-rhs-v2.dirichlet", "test.field-rhs-v2.dirichlet.v1");
  // These added unit witnesses solve pure Poisson with rho=u*u+3*a+P. Its mean is
  // 5.75+9*time for the authored profile, so homogeneous Dirichlet faces are required.
  // No reaction, neutralizing background or silent RHS projection is introduced.
  system.set_field_boundary_plan("test.field-rhs-v2.slot", std::vector<std::string>(2 * D, "dirichlet"),
    dirichlet_alpha, faces, faces);
  system.set_field_nullspace("test.field-rhs-v2.slot", "pops.field-nullspace.operator-topology-derived",
    {"pops.field-nullspace.operator-topology-derived.options@1", {{"gauge.value", 0.0}}});
  const auto output = install_auxiliaries(system, count);
  system.register_elliptic_field("material", "potential", {output}, 1);
  FieldRhsCallbackV2<D> callback;
  if (count == 0) {
    const auto legacy = make_poisson_rhs(StorageModel{});
    const auto v2 = make_poisson_rhs_v2(StorageModel{});
    callback = [&observed, legacy, v2](const Inputs& inputs, Field& rhs) {
      inputs.validate_provider_inputs<0>();
      Field reference(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
      reference.set_val(0); legacy(inputs.state(), reference); v2(inputs, rhs);
      Field error(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
      error.set_val(0);
      for (std::size_t local = 0; local < rhs.local_size(); ++local)
        for_each_cell(inputs.execution(), rhs.box(local), ParityErrorKernel{
          std::as_const(reference).fab(local).view(), std::as_const(rhs).fab(local).view(),
          error.fab(local).view()});
      inputs.execution().fence();
      EXPECT_EQ(reduce_max_local(error), Real(0));
      observed.copied.emplace(inputs); observed.levels.insert(inputs.level());
    };
  } else callback = observing_callback(observed);
  observed.inject_nonfinite = fault;
  system.set_block_elliptic_field_v2("material", "potential", "test.field-rhs-v2.binding",
    "test.field-rhs-v2.consumer", count, std::move(callback));
  system.set_state("material", initial(cells)); system.set_program_block_map({0});
  system.seed_program_params(0, {double(kParameter)});
}

template<class Runtime>
void install_value_program(Runtime& system) {
  system.install_program(POPS_TEST_FIELD_RHS_VALUE_PROGRAM_SO);
}

template<class Context, class Runtime>
Context& started_value_context(Runtime& system, const char* symbol) {
  // Plan installation runs inside the real first Program entry after official hash publication.
  system.step(.03125);
  if (system.installed_program_hash() != test::field_rhs_value_program::identity)
    throw std::logic_error("Field RHS test Program failed to retain its exported identity");
  const auto library = dynlib::open(POPS_TEST_FIELD_RHS_VALUE_PROGRAM_SO);
  if (!dynlib::valid(library))
    throw std::runtime_error("cannot reopen installed Field RHS test Program");
  const auto getter = reinterpret_cast<void* (*)(Runtime*)>(dynlib::sym(library, symbol));
  Context* context = getter ? static_cast<Context*>(getter(&system)) : nullptr;
  dynlib::close(library); // The official Program installation retains the actual library.
  if (!context) throw std::logic_error("installed Field RHS test Program lost its owning Context");
  return *context;
}

runtime::program::ProgramContext<D>& installed_value_context(System<D>& system) {
  install_value_program(system);
  return started_value_context<runtime::program::ProgramContext<D>>(
    system, "pops_test_field_rhs_uniform_context");
}
runtime::program::AmrProgramContext<D>& installed_value_context(AmrSystem<D>& system) {
  // The AMR caller installed the real artifact before any hierarchy materialization,
  // then completed bootstrap. Starting it here must never reinstall a live runtime.
  return started_value_context<runtime::program::AmrProgramContext<D>>(
    system, "pops_test_field_rhs_amr_context");
}

template<class Context>
Field& produce_stage(Context& context, std::int64_t ssa = 1, Real scale = Real(1)) {
  auto root = context.capture_program_value(0, 0, context.state(0));
  auto& stage = context.scratch_state(ssa, 0, context.state(0));
  auto write = context.begin_program_value_write(ssa, 0, stage, {root});
  context.lincomb(stage, scale, context.state(0), Real(0), context.state(0));
  (void)context.complete_program_value_write(std::move(write));
  return stage;
}

TEST(PreparedFieldRhsInputs, RealSpaceTimeAuxDerivedAndParameterRemainSeparateFromState) {
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed);
  auto& context = installed_value_context(system);
  context.configure_primary_clock("test.field-rhs-v2.clock"); context.begin_step(.25);
  context.set_stage_time(1, 2);
  auto& stage = produce_stage(context);
  const auto before = system.state_global("material");
  auto result = context.solve_fields_from_program_values_at(context.boundary_evaluation_point(7),
    10, "test.field-rhs-v2.slot", {{0, &stage, 1}});
  ASSERT_TRUE(result.report().solved_value_available()) << result.report().reason;
  EXPECT_EQ(system.state_global("material"), before);
  (void)result.consume(SolveConsumption::kAccept);
  EXPECT_EQ(observed.levels, std::set<int>{0});
  ASSERT_TRUE(observed.copied.has_value());
  expect_expired(*observed.copied);
}

TEST(PreparedFieldRhsInputs, StageEvaluationScopeRestoresSourceStageAndNestedChildWindowOnUnwind) {
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed);
  runtime::program::ProgramContext<D> context(&system);
  context.configure_primary_clock("test.field-rhs-v2.clock"); context.begin_step(.4);
  context.set_stage_time(1, 3);
  const auto parent = context.boundary_evaluation_point(21);
  {
    auto second_child = context.logical_evaluation_scope(1, 2);
    context.set_stage_time(1, 2); // cI: the producer's source stage.
    const auto source = context.boundary_evaluation_point(21);
    EXPECT_DOUBLE_EQ(source.dt, .2);
    EXPECT_DOUBLE_EQ(source.physical_time, system.time() + .3);
    EXPECT_EQ(source.stage_fraction, (::pops::amr::Rational(3, 4)));
    {
      auto field_evaluation = context.stage_evaluation_scope(0, 1); // cE differs from cI.
      const auto field = context.boundary_evaluation_point(22);
      EXPECT_DOUBLE_EQ(field.dt, source.dt);
      EXPECT_DOUBLE_EQ(field.physical_time, system.time() + .2);
      EXPECT_EQ(field.stage_fraction, (::pops::amr::Rational(1, 2)));
    }
    EXPECT_EQ(context.boundary_evaluation_point(21), source);
    try {
      auto field_evaluation = context.stage_evaluation_scope(0, 1);
      auto nested_second_child = context.logical_evaluation_scope(1, 2);
      const auto nested = context.boundary_evaluation_point(22);
      EXPECT_DOUBLE_EQ(nested.dt, .1);
      EXPECT_DOUBLE_EQ(nested.physical_time, system.time() + .3);
      EXPECT_EQ(nested.stage_fraction, (::pops::amr::Rational(3, 4)));
      throw std::runtime_error("exercise actual nested Field evaluation unwind");
    } catch (const std::runtime_error&) {}
    EXPECT_EQ(context.boundary_evaluation_point(21), source);
  }
  EXPECT_EQ(context.boundary_evaluation_point(21), parent);
}

TEST(PreparedFieldRhsInputs, InstalledPlanRejectsForeignRepeatedSourceAndIsImmutable) {
  using namespace runtime::program;
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed);
  ProgramContext<D> context(&system);
  // This installation validates plan structure only. A plan alone cannot publish a native value;
  // all producer entry points separately require a genuinely installed compiled Program identity.
  ProgramValuePlan plan{"test.unverified-plan-structure",
    {{0, 0, ProgramValueStorage::State, -1, 0, {}, 0, {}},
     {1, 0, ProgramValueStorage::Alias, -1, 0, {}, 0, {0}}}, {{10, 0, 1}}};
  auto foreign = plan;
  foreign.fields.push_back({11, 1, 1}); // Same already-visited SSA, another owner's Field edge.
  EXPECT_ANY_THROW(context.install_program_value_plan(foreign));
  EXPECT_NO_THROW(context.install_program_value_plan(plan));
  EXPECT_NO_THROW(context.install_program_value_plan(plan));
  auto changed = plan;
  changed.program_identity += ".changed";
  EXPECT_ANY_THROW(context.install_program_value_plan(changed));
  EXPECT_ANY_THROW((void)context.capture_program_value(0, 0, context.state(0)));
}

TEST(PreparedFieldRhsInputs, PlanClosureRejectsMissingCyclicAndExtraneousProducers) {
  using namespace runtime::program;
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed);
  ProgramContext<D> context(&system);
  ProgramValuePlan missing{"test.unverified-plan-structure",
    {{1, 0, ProgramValueStorage::Alias, -1, 0, {}, 0, {0}}}, {{10, 0, 1}}};
  EXPECT_ANY_THROW(context.install_program_value_plan(missing));
  ProgramValuePlan cyclic{"test.unverified-plan-structure",
    {{1, 0, ProgramValueStorage::Alias, -1, 0, {}, 0, {2}},
     {2, 0, ProgramValueStorage::Alias, -1, 0, {}, 0, {1}}}, {{10, 0, 1}}};
  EXPECT_ANY_THROW(context.install_program_value_plan(cyclic));
  ProgramValuePlan extraneous{"test.unverified-plan-structure",
    {{0, 0, ProgramValueStorage::State, -1, 0, {}, 0, {}},
     {1, 0, ProgramValueStorage::State, -1, 0, {}, 0, {}}}, {{10, 0, 0}}};
  EXPECT_ANY_THROW(context.install_program_value_plan(extraneous));
}

TEST(PreparedFieldRhsInputs, CountZeroUsesTheActualLegacyFactory) {
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed, 0);
  runtime::program::ProgramContext<D> context(&system);
  context.configure_primary_clock("test.field-rhs-v2.clock"); context.begin_step(.25);
  auto result = context.solve_fields_from_state_at(context.boundary_evaluation_point(3),
    "test.field-rhs-v2.slot", 0, context.state(0));
  ASSERT_TRUE(result.report().solved_value_available()) << result.report().reason;
  (void)result.consume(SolveConsumption::kAccept);
  ASSERT_TRUE(observed.copied.has_value());
  EXPECT_THROW((void)observed.copied->state(), std::logic_error);
}

TEST(PreparedFieldRhsInputs, ExpiredIssuedCopyRetainsConsensusLaneAfterRuntimeDestruction) {
  Observation observed;
  std::string expected_lane_identity;
  {
    System<D> system(uniform_config());
    install_uniform_field(system, observed, 0);
    expected_lane_identity = std::string(system.prepared_boundary_execution_lane().identity());
    runtime::program::ProgramContext<D> context(&system);
    context.configure_primary_clock("test.field-rhs-v2.clock"); context.begin_step(.25);
    auto result = context.solve_fields_from_state_at(context.boundary_evaluation_point(3),
      "test.field-rhs-v2.slot", 0, context.state(0));
    ASSERT_TRUE(result.report().solved_value_available()) << result.report().reason;
    (void)result.consume(SolveConsumption::kAccept);
    ASSERT_TRUE(observed.copied.has_value());
  }
  expect_expired(*observed.copied);
  EXPECT_EQ(observed.copied->consensus_lane().identity(), expected_lane_identity);
  EXPECT_EQ(observed.copied->consensus_lane().size(), n_ranks());
  observed.copied.reset(); // Every rank releases the retained owning lane in the same order.
}

TEST(PreparedFieldRhsInputs, WrongLevelAndForeignShapeRefuseBeforeCallbackOrPublication) {
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed);
  auto& context = installed_value_context(system);
  context.configure_primary_clock("test.field-rhs-v2.clock"); context.begin_step(.25);
  auto& stage = produce_stage(context);
  const auto state = system.state_global("material");
  const auto auxiliary = system.capture_auxiliary_checkpoint_accepted_state();
  auto point = context.boundary_evaluation_point(11);
  point.level = 1; // A normal public request at a level that this Uniform runtime does not own.
  EXPECT_ANY_THROW((void)context.solve_fields_from_program_values_at(point,
    10, "test.field-rhs-v2.slot", {{0, &stage, 1}}));
  const auto actual_point = context.boundary_evaluation_point(11);
  std::array<runtime::multiblock::BoundaryEvaluationPoint, 5> altered;
  altered.fill(actual_point);
  altered[0].physical_time += .01;
  altered[1].dt *= .5;
  ++altered[2].tick;
  ++altered[3].substep;
  altered[4].stage_fraction = ::pops::amr::Rational(1, 2);
  for (const auto& fabricated_point : altered)
    EXPECT_ANY_THROW((void)context.solve_fields_from_program_values_at(fabricated_point,
      10, "test.field-rhs-v2.slot", {{0, &stage, 1}}));
  EXPECT_FALSE(observed.copied.has_value());
  auto foreign_config = uniform_config();
  for (int axis = 0; axis < D; ++axis) {
    foreign_config.shape[axis] = kCells + 1;
    foreign_config.lower[axis] = 2; foreign_config.upper[axis] = 3;
  }
  System<D> foreign(foreign_config); Observation foreign_observed;
  // The foreign runtime has real storage on another Frame/domain. Its field is not a proof token.
  install_uniform_field(foreign, foreign_observed, 2, false, kCells + 1, 2, 3);
  auto& foreign_context = installed_value_context(foreign);
  // The installed clock is immutable. Keep it while testing a foreign state owner/Frame.
  foreign_context.configure_primary_clock("test.field-rhs-v2.clock");
  EXPECT_ANY_THROW((void)context.solve_fields_from_program_values_at(context.boundary_evaluation_point(11),
    10, "test.field-rhs-v2.slot", {{0, &foreign_context.state(0), 1}}));
  EXPECT_FALSE(observed.copied.has_value());
  EXPECT_EQ(system.state_global("material"), state);
  EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), auxiliary);
}

TEST(PreparedFieldRhsInputs, OneRankNonfiniteCandidateLeavesAcceptedStorageAndAuxiliaryExact) {
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed);
  auto& context = installed_value_context(system);
  context.configure_primary_clock("test.field-rhs-v2.clock"); context.begin_step(.25);
  context.set_stage_time(1, 2);
  auto& stage = produce_stage(context);
  auto warm = context.solve_fields_from_program_values_at(context.boundary_evaluation_point(4),
    10, "test.field-rhs-v2.slot", {{0, &stage, 1}});
  ASSERT_TRUE(warm.report().solved_value_available()) << warm.report().reason;
  (void)warm.consume(SolveConsumption::kAccept);
  const auto state = system.state_global("material");
  const auto auxiliary = system.capture_auxiliary_checkpoint_accepted_state();
  const AuxiliaryComponentKey output{"test.field-rhs-v2.owner", "field", "potential", "potential"};
  const auto potential = system.auxiliary_component(output);
  auto produced = context.program_value(1, 0, stage);
  auto source_validator = context.require_program_field_value(produced, 10, 0, stage);
  EXPECT_NO_THROW(source_validator());
  system.begin_step_transaction();
  observed.inject_nonfinite = true;
  EXPECT_THROW((void)context.solve_fields_from_program_values_at(context.boundary_evaluation_point(5),
    10, "test.field-rhs-v2.slot", {{0, &stage, 1}}), std::runtime_error);
  system.rollback_step_transaction();
  EXPECT_EQ(system.state_global("material"), state);
  EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), auxiliary);
  EXPECT_EQ(system.auxiliary_component(output), potential);
  EXPECT_THROW(produced.validate(), std::logic_error);
  EXPECT_THROW(source_validator(), std::logic_error);
  ASSERT_TRUE(observed.copied.has_value());
  expect_expired(*observed.copied);
}

TEST(PreparedFieldRhsInputs, SameLayoutAndFrameFromForeignRuntimeDoNotAuthenticateSourceOwner) {
  System<D> system(uniform_config()), foreign(uniform_config());
  Observation observed, foreign_observed;
  install_uniform_field(system, observed);
  install_uniform_field(foreign, foreign_observed);
  auto& context = installed_value_context(system);
  auto& foreign_context = installed_value_context(foreign);
  context.configure_primary_clock("test.field-rhs-v2.clock"); context.begin_step(.25);
  foreign_context.configure_primary_clock("test.field-rhs-v2.clock");
  foreign_context.begin_step(.25);
  auto& own_stage = produce_stage(context);
  auto& foreign_stage = produce_stage(foreign_context);
  ASSERT_EQ(context.state(0).layout(), foreign_context.state(0).layout());
  ASSERT_EQ(context.state(0).distribution(), foreign_context.state(0).distribution());
  ASSERT_EQ(context.state(0).ncomp(), foreign_context.state(0).ncomp());
  ASSERT_EQ(context.state(0).ghosts(), foreign_context.state(0).ghosts());
  const auto state = system.state_global("material");
  const auto auxiliary = system.capture_auxiliary_checkpoint_accepted_state();
  // A rank-local owner refusal must converge before any peer enters its Field callback.
  auto& requested = my_rank() == 0 ? foreign_stage : own_stage;
  EXPECT_ANY_THROW((void)context.solve_fields_from_program_values_at(context.boundary_evaluation_point(11),
    10, "test.field-rhs-v2.slot", {{0, &requested, 1}}));
  EXPECT_FALSE(observed.copied.has_value());
  EXPECT_FALSE(foreign_observed.copied.has_value());
  EXPECT_EQ(system.state_global("material"), state);
  EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), auxiliary);
}

TEST(PreparedFieldRhsInputs, ActualProducedSsaAndInstalledAliasKeepSourceBirthAtDistinctFieldStage) {
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed);
  auto& context = installed_value_context(system);
  context.begin_step(.4);
  auto second_child = context.logical_evaluation_scope(1, 2);
  context.set_stage_time(1, 2);
  auto& stage = produce_stage(context);
  auto produced = context.program_value(1, 0, stage);
  const auto birth = produced.birth_point();
  auto alias = context.alias_program_value(2, produced);
  EXPECT_EQ(alias.birth_point(), birth);
  EXPECT_EQ(alias.source_ssa(), 2);
  EXPECT_THROW((void)context.alias_program_value(99, produced), std::invalid_argument);
  EXPECT_THROW((void)context.require_program_field_value(alias, 10, 0, stage), std::invalid_argument);
  context.require_program_field_value(alias, 12, 0, stage)();
  {
    auto field_evaluation = context.stage_evaluation_scope(0, 1);
    const auto field_point = context.boundary_evaluation_point(23);
    EXPECT_NE(field_point.physical_time, birth.physical_time);
    auto result = context.solve_fields_from_program_values_at(field_point,
      12, "test.field-rhs-v2.slot", {{0, &stage, 2}});
    ASSERT_TRUE(result.report().solved_value_available()) << result.report().reason;
    (void)result.consume(SolveConsumption::kAccept);
  }
  EXPECT_EQ(produced.birth_point(), birth);
  EXPECT_EQ(context.boundary_evaluation_point(0), birth);
  ASSERT_TRUE(observed.copied.has_value());
  expect_expired(*observed.copied);
}

TEST(PreparedFieldRhsInputs, DeclaredSourceAndExactOwnedBufferMustBothMatchInstalledFieldEdge) {
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed);
  auto& context = installed_value_context(system);
  context.begin_step(.25);
  auto& first = produce_stage(context, 1);
  auto& second = produce_stage(context, 3, Real(2));
  const auto state = system.state_global("material");
  const auto auxiliary = system.capture_auxiliary_checkpoint_accepted_state();
  // Source 3 has a successful real producer, but Field node 10 installed an edge from source 1.
  EXPECT_ANY_THROW((void)context.solve_fields_from_program_values_at(context.boundary_evaluation_point(24),
    10, "test.field-rhs-v2.slot", {{0, &second, 3}}));
  // A source label from the right edge cannot authenticate another producer's buffer.
  EXPECT_ANY_THROW((void)context.solve_fields_from_program_values_at(context.boundary_evaluation_point(24),
    10, "test.field-rhs-v2.slot", {{0, &second, 1}}));
  // Omitting a planned stage must not silently replace it with the runtime's accepted image.
  EXPECT_ANY_THROW((void)context.solve_fields_from_program_values_at(context.boundary_evaluation_point(24),
    10, "test.field-rhs-v2.slot", {}));
  Field copied = first;
  ASSERT_EQ(copied.layout(), first.layout());
  ASSERT_EQ(copied.distribution(), first.distribution());
  ASSERT_EQ(copied.local_size(), first.local_size());
  ASSERT_EQ(copied.ncomp(), first.ncomp());
  ASSERT_EQ(copied.ghosts(), first.ghosts());
  Field copy_error(first.layout(), first.distribution(), first.local_rank(), 1, first.ghosts());
  copy_error.set_val(0);
  for (std::size_t local = 0; local < first.local_size(); ++local) {
    // Field/Fab copy owns independent storage, even when every copied value is exact.
    ASSERT_NE(copied.fab(local).view().data, first.fab(local).view().data);
    for_each_cell(first.box(local), ParityErrorKernel{
      std::as_const(first).fab(local).view(), std::as_const(copied).fab(local).view(),
      copy_error.fab(local).view()});
  }
  Kokkos::fence();
  EXPECT_EQ(reduce_max_local(copy_error), Real(0));
  EXPECT_THROW((void)context.program_value(1, 0, copied), std::invalid_argument);
  EXPECT_FALSE(observed.copied.has_value());
  EXPECT_EQ(system.state_global("material"), state);
  EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), auxiliary);
}

TEST(PreparedFieldRhsInputs, ResetOfSameOwnedSlotAndNewAttemptExpireProducedValuesAndAliases) {
  System<D> system(uniform_config()); Observation observed;
  install_uniform_field(system, observed);
  auto& context = installed_value_context(system);
  context.begin_step(.25);
  auto& first = produce_stage(context);
  auto value = context.program_value(1, 0, first);
  auto alias = context.alias_program_value(2, value);
  auto validator = context.require_program_field_value(value, 10, 0, first);
  const auto layout = first.layout();
  auto& reused = context.scratch_state(1, 0, context.state(0));
  EXPECT_EQ(&reused, &first);
  EXPECT_EQ(reused.layout(), layout);
  EXPECT_THROW(value.validate(), std::logic_error);
  EXPECT_THROW(alias.validate(), std::logic_error);
  EXPECT_THROW(validator(), std::logic_error);
  auto root = context.capture_program_value(0, 0, context.state(0));
  auto pending = context.begin_program_value_write(1, 0, reused, {root});
  EXPECT_THROW((void)context.program_value(1, 0, reused), std::logic_error);
  context.lincomb(reused, Real(1), context.state(0), Real(0), context.state(0));
  auto replacement = context.complete_program_value_write(std::move(pending));
  auto replacement_validator = context.require_program_field_value(replacement, 10, 0, reused);
  EXPECT_NO_THROW(replacement_validator());
  context.begin_step(.25);
  EXPECT_THROW(replacement.validate(), std::logic_error);
  EXPECT_THROW(replacement_validator(), std::logic_error);
  EXPECT_FALSE(observed.copied.has_value());
}

TEST(PreparedFieldRhsInputs, ActualMultilevelAmrIssuesOneInputForEachLiveLevel) {
  ready();
  AmrSystemConfig<D> config;
  config.level_count = 2; config.regrid_every = 0; config.explicit_bootstrap = true;
  for (int axis = 0; axis < D; ++axis) {
    config.shape[axis] = kCells; config.lower[axis] = 0; config.upper[axis] = 1;
    config.periodicity[axis] = true; config.coarse_max_grid[axis] = kCells;
    config.transition_buffers.front()[axis] = 1;
    config.transition_lookaheads.front()[axis] = 1;
  }
  AmrSystem<D> system(config);
  test::install_amr_runtime_authority(system, "test.field-rhs-v2.amr-execution");
  system.install_block_state_route("material", "test.field-rhs-v2.state");
  // This is the same generated storage factory as the production program-only block route.
  system.install_prepared_amr_block(prepare_compiled_amr_system_block<D>(
    "material", StorageModel{}, "state_storage", "unavailable", "conservative", "imex",
    1.4, 1, 1, 0.0, double(kWenoEpsilon), false, "test.field-rhs-v2.state-model"));
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  const AuxiliaryComponentKey output{"test.field-rhs-v2.owner", "field", "potential", "potential"};
  const AmrFieldHierarchyPolicyAuthority hierarchy{
    "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan("test.field-rhs-v2.slot", "test.field-rhs-v2.plan",
    "test.field-rhs-v2.provider", "test.field-rhs-v2.owner", "material", "potential", {output}, 1,
    {"test.field-rhs-v2.binding"}, {"material"}, {"potential"}, {1.0}, "geometric_mg", hierarchy,
    geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{}));
  system.set_field_reaction("test.field-rhs-v2.slot", 1.0);
  install_auxiliaries(system, 2);
  system.register_elliptic_field("material", "potential", {output}, 1);
  Observation observed;
  system.set_block_elliptic_field_v2("material", "potential", "test.field-rhs-v2.executable-rhs",
    "test.field-rhs-v2.binding", "test.field-rhs-v2.consumer", 2, observing_callback(observed));
  test::install_prepared_threshold_union(system,
    {{"material", "u", 2.4, test::PreparedThresholdRelation::Above, "test.field-rhs-v2.state"}},
    "test.field-rhs-v2.tagging", "test.field-rhs-v2.clock");
  system.bind_bootstrap_subject("test.field-rhs-v2.state", "material", "bound_level_zero");
  system.stage_bootstrap_array("test.field-rhs-v2.state", "material", "cell", "cell", 1,
    config.shape, initial());
  Extent<D> prolongation{}, restriction{};
  for (int axis = 0; axis < D; ++axis) prolongation[axis] = 1;
  system.register_bootstrap_transfer_route("test.field-rhs-v2.prolongation",
    {"test.field-rhs-v2.state"}, "test.field-rhs-v2.conservative-linear", "cell", "cell",
    "conservative", "dense", "prolongation", "conservative_linear", 2, prolongation,
    config.transition_ratios.front());
  system.register_bootstrap_transfer_route("test.field-rhs-v2.restriction",
    {"test.field-rhs-v2.state"}, "test.field-rhs-v2.volume-average", "cell", "cell",
    "conservative", "dense", "restriction", "volume_average", 1, restriction,
    config.transition_ratios.front());
  install_value_program(system);
  system.begin_bootstrap_plan();
  system.seed_program_params(0, {double(kParameter)});
  (void)system.materialize_bootstrap_action("test.field-rhs-v2.state", "initialize_level_zero",
    "bound_level_zero", 0);
  ASSERT_TRUE(system.bootstrap_next_level()); // A real hierarchy action, not a declared level count.
  (void)system.materialize_bootstrap_action("test.field-rhs-v2.state", "prolong_from_parent",
    "conservative_linear", 1);
  system.commit_bootstrap_level();
  system.mark_bound();
  auto& context = installed_value_context(system);
  context.configure_primary_clock("test.field-rhs-v2.clock"); context.begin_step(.25);
  context.set_stage_time(1, 2);
  auto& stage = produce_stage(context);
  auto result = context.solve_fields_from_program_values_at(context.boundary_evaluation_point(13),
    10, "test.field-rhs-v2.slot", {{0, &stage, 1}});
  ASSERT_TRUE(result.report().solved_value_available()) << result.report().reason;
  (void)result.consume(SolveConsumption::kAccept);
  EXPECT_EQ(system.field_provider_levels("test.field-rhs-v2.slot"), 2);
  EXPECT_EQ(observed.levels, (std::set<int>{0, 1}));
  ASSERT_TRUE(observed.copied.has_value());
  EXPECT_THROW((void)observed.copied->frame(), std::logic_error);
}

} // namespace
