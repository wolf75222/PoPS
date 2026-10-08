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
#include <pops/runtime/nd_proof/field_rhs_aux_incidence_program_fixture.hpp>
#include <algorithm>
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

std::size_t local_valid_cell_count(const Field& field) {
  std::size_t cells = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local)
    cells += static_cast<std::size_t>(field.box(local).numPts());
  return cells;
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
      const auto local_cells = local_valid_cell_count(error);
      ::testing::Test::RecordProperty("field_rhs_parity_local_fabs", std::to_string(error.local_size()));
      ::testing::Test::RecordProperty("field_rhs_parity_local_valid_cells", std::to_string(local_cells));
      if (local_cells == 0)
        EXPECT_EQ(reduce_max_local(error), -std::numeric_limits<Real>::infinity());
      else
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
  const auto local_cells = local_valid_cell_count(copy_error);
  RecordProperty("field_copy_local_fabs", std::to_string(copy_error.local_size()));
  RecordProperty("field_copy_local_valid_cells", std::to_string(local_cells));
  if (local_cells == 0)
    EXPECT_EQ(reduce_max_local(copy_error), -std::numeric_limits<Real>::infinity());
  else
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

// A separate real two-slot composition, leaving the thirteen original witnesses intact.
// u is constant; tag is a passive mesh selector, not an elliptic unknown.
struct IncidenceStorageModel {
  using State = StateVec<2>;
  using Primitive = State;
  static constexpr int dimension = D, n_vars = 2, n_providers = 0;
  static constexpr bool program_only_storage = true;
  static constexpr int program_state_ghost_depth = 1;
  static PreparedProviderIdentity provider_identity() { return {"test.amr-aux.incidence-state", 1}; }
  void serialize_exact_parameters(ExactContractBuilder& exact) const { exact.scalar(std::uint32_t{1}); }
  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u", "tag"}, 2, {VariableRole::Scalar, VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u", "tag"}, 2, {VariableRole::Scalar, VariableRole::Scalar}};
  }
  POPS_HD nd::StateConversion<Primitive> recover(const State& value) const { return {value, {}}; }
  POPS_HD nd::StateConversion<State> make_conservative(const Primitive& value) const { return {value, {}}; }
  POPS_HD nd::StateConversionStatus admissibility(const State&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State& state) const { return state[0]; }
};
// Generic externally forced equation u_t=Q with zero transport; tag is passive.
// This is an actual provider-aware STATE operator, separate from storage-only witnesses.
struct IncidenceForcedStateModel : IncidenceStorageModel {
  struct Schema {
    static constexpr int dimension = D, nvars = 2;
    using Conservative = StateVec<nvars>;
    using Primitive = StateVec<nvars>;
  };
  static constexpr int n_providers = 1;
  static constexpr bool program_only_storage = false;
  static PreparedProviderIdentity provider_identity() {
    return {"test.amr-aux.incidence-forced-state", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& exact) const {
    exact.text("u_t=Q; tag_t=0; zero transport").scalar(std::uint32_t{1});
  }
  POPS_HD State flux(const State&, const auto&, int) const { return {}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return Real(0); }
  POPS_HD void wave_speeds(const State&, const auto&, int,
                          Real& lower, Real& upper) const { lower = upper = Real(0); }
  POPS_HD State source(const State&, const ProviderValues<1>& providers) const {
    return State{providers[0], Real(0)};
  }
};
static_assert(PhysicalModel<IncidenceForcedStateModel>);

struct IncidencePhiRhs {
  using State = StateVec<2>;
  static constexpr int dimension = D, n_vars = 2, n_providers = 0;
  POPS_HD Real elliptic_rhs(const State& state) const { return state[0]; }
};
struct IncidencePsiRhs {
  using State = StateVec<2>;
  static constexpr int dimension = D, n_vars = 2, n_providers = 1;
  POPS_HD Real elliptic_rhs(const State&, const ProviderValues<1>& values) const { return values[0]; }
};
struct IncidenceScaleKernel {
  FieldView<const Real, D> source;
  FieldView<Real, D> target;
  int source_component, target_component;
  POPS_HD void operator()(const Index<D>& cell) const {
    target(cell, target_component) = Real(2) * source(cell, source_component);
  }
};
struct IncidenceConstantErrorKernel {
  FieldView<const Real, D> values;
  FieldView<Real, D> error;
  int component;
  Real expected;
  POPS_HD void operator()(const Index<D>& cell) const {
    error(cell, 0) = Kokkos::abs(values(cell, component) - expected);
  }
};
struct IncidenceDependencyErrorKernel {
  FieldView<const Real, D> phi, Q;
  FieldView<Real, D> error;
  int phi_component, Q_component;
  POPS_HD void operator()(const Index<D>& cell) const {
    error(cell, 0) = Kokkos::abs(Q(cell, Q_component) - Real(2) * phi(cell, phi_component));
  }
};
struct IncidenceStateSourceErrorKernel {
  FieldView<const Real, D> source, Q;
  FieldView<Real, D> error;
  int Q_component;
  POPS_HD void operator()(const Index<D>& cell) const {
    error(cell, 0) = Kokkos::max(Kokkos::abs(source(cell, 0) - Q(cell, Q_component)),
                                Kokkos::abs(source(cell, 1)));
  }
};
using IncidenceCheckpoint = AuxiliaryCheckpointAcceptedState<D>;
constexpr const char* kIncidenceOwner = "test.amr-aux.incidence-owner";
constexpr const char* kIncidenceQProvider = "test.amr-aux.incidence-Q";
struct IncidenceCallbackSnapshot {
  int level;
  runtime::multiblock::BoundaryEvaluationPoint actual_point;
  std::vector<IncidenceCheckpoint> metadata;
};
struct IncidenceObservation {
  bool capture = false, fail_fine_Q = false;
  std::vector<IncidenceCallbackSnapshot> snapshots;
  std::vector<AuxiliaryEvaluationPoint> Q_launches;
  std::optional<AuxiliaryEvaluationPoint> fine_Q_fault_point;
  std::vector<std::optional<runtime::multiblock::BoundaryEvaluationPoint>> phi_points{2};
};
bool incidence_invalidated(const IncidenceCheckpoint& image, const std::string& identity) {
  return std::find(image.invalidated_providers.begin(), image.invalidated_providers.end(), identity) !=
         image.invalidated_providers.end();
}
std::optional<AuxiliaryEvaluationPoint> incidence_provenance(
    const IncidenceCheckpoint& image, const std::string& identity) {
  const auto provider = std::find_if(image.providers.begin(), image.providers.end(),
      [&](const auto& item) { return item.identity == identity; });
  if (provider == image.providers.end()) throw std::logic_error("incidence witness provider absent");
  return provider->accepted_point;
}
Real incidence_local_constant_error(AmrSystem<D>& system, const AuxiliaryComponentKey& key,
                                    int level, Real expected) {
  const auto address = system.auxiliary_address(key);
  const auto* groups = system.prepared_amr_provider_storage_groups(level);
  if (!groups) throw std::logic_error("incidence witness has no actual level carriers");
  const auto* values = groups->find(address.group);
  if (!values) throw std::logic_error("incidence witness group absent");
  Field error(values->layout(), values->distribution(), values->local_rank(), 1, values->ghosts());
  error.set_val(0);
  for (std::size_t local = 0; local < values->local_size(); ++local)
    for_each_cell(values->box(local), IncidenceConstantErrorKernel{
      values->fab(local).view(), error.fab(local).view(), kernel_component(address.component), expected});
  Kokkos::fence();
  return reduce_max_local(error);
}
Real incidence_local_dependency_error(AmrSystem<D>& system, const AuxiliaryComponentKey& phi,
                                      const AuxiliaryComponentKey& Q, int level) {
  const auto phi_address = system.auxiliary_address(phi);
  const auto Q_address = system.auxiliary_address(Q);
  const auto* groups = system.prepared_amr_provider_storage_groups(level);
  if (!groups) throw std::logic_error("incidence witness level carriers absent");
  const auto* phi_values = groups->find(phi_address.group);
  const auto* Q_values = groups->find(Q_address.group);
  if (!phi_values || !Q_values) throw std::logic_error("incidence witness dependency carriers absent");
  Field error(Q_values->layout(), Q_values->distribution(), Q_values->local_rank(), 1, Q_values->ghosts());
  error.set_val(0);
  for (std::size_t local = 0; local < Q_values->local_size(); ++local)
    for_each_cell(Q_values->box(local), IncidenceDependencyErrorKernel{
      phi_values->fab(local).view(), Q_values->fab(local).view(), error.fab(local).view(),
      kernel_component(phi_address.component), kernel_component(Q_address.component)});
  Kokkos::fence();
  return reduce_max_local(error);
}
runtime::program::AmrProgramContext<D>& incidence_context(AmrSystem<D>& system) {
  system.step(.03125); // Real installer entry, value plan and hierarchy advance.
  if (system.installed_program_hash() != test::aux_incidence_value_program::identity)
    throw std::logic_error("incidence Program identity differs from its real DSO");
  const auto library = dynlib::open(POPS_TEST_FIELD_RHS_AUX_INCIDENCE_PROGRAM_SO);
  if (!dynlib::valid(library)) throw std::runtime_error("cannot reopen incidence Program");
  const auto getter = reinterpret_cast<void* (*)(AmrSystem<D>*)>(
      dynlib::sym(library, "pops_test_aux_incidence_amr_context"));
  auto* context = getter ? static_cast<runtime::program::AmrProgramContext<D>*>(getter(&system)) : nullptr;
  dynlib::close(library); // Official installation retains the actual DSO.
  if (!context) throw std::logic_error("incidence Program lost its installed context");
  return *context;
}

enum class IncidenceRestoreWitness {
  original, repeated_publication, rank_local_refusal, accepted_state_consumer
};

void actual_field_incidence_lifecycle(IncidenceRestoreWitness extra) {
  ready();
  AmrSystemConfig<D> config;
  config.level_count = 2; config.regrid_every = 0; config.explicit_bootstrap = true;
  for (int axis = 0; axis < D; ++axis) {
    config.shape[axis] = kCells; config.lower[axis] = 0; config.upper[axis] = 1;
    config.periodicity[axis] = true; config.coarse_max_grid[axis] = kCells;
    config.transition_buffers.front()[axis] = 1;
    config.transition_lookaheads.front()[axis] = 1;
  }
  IncidenceObservation observed;
  AmrSystem<D> system(config);
  test::install_amr_runtime_authority(system, "test.amr-aux.incidence-execution");
  system.install_block_state_route("material", "test.amr-aux.incidence-state");
  if (extra == IncidenceRestoreWitness::accepted_state_consumer) {
    auto prepared = prepare_compiled_amr_system_block<D>(
      "material", IncidenceForcedStateModel{}, "none", "rusanov", "conservative", "imex",
      1.4, 1, 1, 0.0, double(kWenoEpsilon), false, "test.amr-aux.incidence-state-source");
    ASSERT_EQ(prepared.provider_components, 1);
    system.install_prepared_amr_block(std::move(prepared));
  } else {
  system.install_prepared_amr_block(prepare_compiled_amr_system_block<D>(
    "material", IncidenceStorageModel{}, "state_storage", "unavailable", "conservative", "imex",
    1.4, 1, 1, 0.0, double(kWenoEpsilon), false, "test.amr-aux.incidence-state-model"));
  }
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "field", "scalar"};
  AuxiliaryStorageShape<D> shape;
  for (int axis = 0; axis < D; ++axis) shape.halo[axis] = 1;
  const AuxiliaryComponentKey phi{kIncidenceOwner, "field", "phi", "phi"};
  const AuxiliaryComponentKey psi{kIncidenceOwner, "field", "psi", "psi"};
  const AuxiliaryComponentKey Q{kIncidenceOwner, "aux", "Q", "Q"};
  const AmrFieldHierarchyPolicyAuthority hierarchy{
    "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  const auto install_plan = [&](const std::string& slot, const std::string& field,
                               const std::string& binding, const AuxiliaryComponentKey& output) {
    system.set_field_solver_plan(slot, slot + "/plan", slot + "/provider", kIncidenceOwner,
      "material", field, {output}, 1, {binding}, {"material"}, {field}, {1.0},
      "geometric_mg", hierarchy,
      geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{}));
    system.set_field_reaction(slot, 1.0); // Screened equations have no periodic mean constraint.
  };
  install_plan("test.amr-aux.phi-slot", "phi", "test.amr-aux.phi-binding", phi);
  install_plan("test.amr-aux.psi-slot", "psi", "test.amr-aux.psi-binding", psi);
  using Provider = PreparedAuxiliaryProvider<D>;
  const AuxiliaryEvaluationPolicy numerical{
    AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation};
  const AuxiliaryEvaluationPolicy external_output{
    AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once};
  system.install_prepared_auxiliary_provider(Provider{
    "test.amr-aux.phi-output", AuxiliaryProviderKind::field_output, external_output, {{phi, contract, shape}}, {}});
  system.install_prepared_auxiliary_provider(Provider{
    "test.amr-aux.psi-output", AuxiliaryProviderKind::field_output, external_output, {{psi, contract, shape}}, {}});
  system.install_prepared_auxiliary_provider(Provider{
    kIncidenceQProvider, AuxiliaryProviderKind::derived, numerical, {{Q, contract, shape}},
    {{phi, contract, shape}}, Provider::launcher_type::trusted_extension(
      PreparedProviderIdentity{kIncidenceQProvider, 1}, "Q=2phi; controlled fine publication failure",
      [&observed](const AuxiliaryKernelLaunchContext<D>& launch) {
        observed.Q_launches.push_back(launch.point);
        if (observed.fail_fine_Q && launch.point.level == 1) {
          observed.fine_Q_fault_point = launch.point; // Record only the actual throwing branch.
          throw std::runtime_error("intentional fine Q publication failure");
        }
        const auto source = launch.dependencies.at(0).address;
        const auto target = launch.outputs.at(0).address;
        const auto* input = launch.storage.candidate->find(source.group);
        auto* output = launch.storage.candidate->find(target.group);
        if (!input || !output) throw std::logic_error("actual phi/Q carrier absent");
        for (std::size_t local = 0; local < output->local_size(); ++local)
          for_each_cell(output->box(local), IncidenceScaleKernel{
            input->fab(local).view(), output->fab(local).view(),
            kernel_component(source.component), kernel_component(target.component)});
        Kokkos::fence();
      })});
  system.install_auxiliary_consumer_plan({"test.amr-aux.psi-consumer", {{{Q, contract, shape}, 0}}});
  if (extra == IncidenceRestoreWitness::accepted_state_consumer)
    system.install_auxiliary_consumer_plan({"test.amr-aux.incidence-state-source", {{{Q, contract, shape}, 0}}});
  system.seal_auxiliary_providers();
  system.register_elliptic_field("material", "phi", {phi}, 1);
  system.register_elliptic_field("material", "psi", {psi}, 1);
  const auto phi_rhs = make_poisson_rhs_v2(IncidencePhiRhs{});
  const auto psi_rhs = make_poisson_rhs_v2(IncidencePsiRhs{});
  system.set_block_elliptic_field_v2("material", "phi", "test.amr-aux.phi-rhs",
    "test.amr-aux.phi-binding", "test.amr-aux.phi-consumer", 0,
    [&observed, phi_rhs](const Inputs& inputs, Field& rhs) {
      phi_rhs(inputs, rhs);
      if (inputs.boundary_point().dt > 0)
        observed.phi_points.at(static_cast<std::size_t>(inputs.level())) = inputs.boundary_point();
    });
  system.set_block_elliptic_field_v2("material", "psi", "test.amr-aux.psi-rhs",
    "test.amr-aux.psi-binding", "test.amr-aux.psi-consumer", 1,
    [&system, &observed, psi_rhs](const Inputs& inputs, Field& rhs) {
      psi_rhs(inputs, rhs);
      if (observed.capture)
        observed.snapshots.push_back({inputs.level(), inputs.boundary_point(),
          system.capture_auxiliary_checkpoint_accepted_state()});
    });
  test::install_prepared_threshold_union(system,
    {{"material", "tag", 2.4, test::PreparedThresholdRelation::Above, "test.amr-aux.incidence-state"}},
    "test.amr-aux.incidence-tagging", "test.field-rhs-v2.clock");
  system.bind_bootstrap_subject("test.amr-aux.incidence-state", "material", "bound_level_zero");
  const auto tagging = initial();
  std::vector<double> state(2 * tagging.size());
  for (std::size_t cell = 0; cell < tagging.size(); ++cell) {
    state[cell] = 3.; state[tagging.size() + cell] = tagging[cell]; // Public component-major array.
  }
  system.stage_bootstrap_array("test.amr-aux.incidence-state", "material", "cell", "cell", 2,
    config.shape, state);
  Extent<D> prolongation{}, restriction{};
  for (int axis = 0; axis < D; ++axis) prolongation[axis] = 1;
  system.register_bootstrap_transfer_route("test.amr-aux.incidence-prolongation",
    {"test.amr-aux.incidence-state"}, "test.amr-aux.conservative-linear", "cell", "cell",
    "conservative", "dense", "prolongation", "conservative_linear", 2, prolongation,
    config.transition_ratios.front());
  system.register_bootstrap_transfer_route("test.amr-aux.incidence-restriction",
    {"test.amr-aux.incidence-state"}, "test.amr-aux.volume-average", "cell", "cell",
    "conservative", "dense", "restriction", "volume_average", 1, restriction,
    config.transition_ratios.front());
  system.install_program(POPS_TEST_FIELD_RHS_AUX_INCIDENCE_PROGRAM_SO);
  system.begin_bootstrap_plan();
  system.seed_program_params(0, {double(kParameter)});
  (void)system.materialize_bootstrap_action("test.amr-aux.incidence-state", "initialize_level_zero",
    "bound_level_zero", 0);
  ASSERT_TRUE(system.bootstrap_next_level());
  (void)system.materialize_bootstrap_action("test.amr-aux.incidence-state", "prolong_from_parent",
    "conservative_linear", 1);
  system.commit_bootstrap_level();
  system.mark_bound();
  auto& context = incidence_context(system);
  const auto& actual_fine = system.prepared_amr_block_state(0, 1);
  Real local_fine_cells = 0;
  for (std::size_t local = 0; local < actual_fine.local_size(); ++local)
    local_fine_cells += static_cast<Real>(actual_fine.box(local).numPts());
  const Real actual_fine_cells = actual_fine.distribution().replicated()
      ? local_fine_cells : all_reduce_sum(local_fine_cells, context.prepared_execution_lane());
  const Real full_fine_cells = static_cast<Real>(system.prepared_amr_level_geometry(1).domain().numPts());
  ASSERT_GT(actual_fine_cells, Real(0));
  ASSERT_LT(actual_fine_cells, full_fine_cells) << "the real refined layout must be sparse";
  context.begin_step(.25); context.set_stage_time(1, 2);
  ASSERT_EQ(system.step_transaction_depth(), 0U);
  auto& baseline_stage = produce_stage(context);
  auto baseline = context.solve_fields_from_program_values_at(context.boundary_evaluation_point(13),
    13, "test.amr-aux.psi-slot", {{0, &baseline_stage, 1}});
  ASSERT_TRUE(baseline.report().solved_value_available()) << baseline.report().reason;
  (void)baseline.consume(SolveConsumption::kAccept);
  const auto baseline_image = system.capture_auxiliary_checkpoint_accepted_state();
  ASSERT_EQ(baseline_image.size(), 2U);
  for (const auto& level : baseline_image) {
    ASSERT_TRUE(incidence_provenance(level, kIncidenceQProvider).has_value());
    EXPECT_FALSE(incidence_invalidated(level, kIncidenceQProvider));
  }

  context.set_stage_time(3, 4);
  auto& stage = produce_stage(context);
  auto phi_solve = context.solve_fields_from_program_values_at(context.boundary_evaluation_point(14),
    10, "test.amr-aux.phi-slot", {{0, &stage, 1}});
  ASSERT_TRUE(phi_solve.report().solved_value_available()) << phi_solve.report().reason;
  (void)phi_solve.consume(SolveConsumption::kAccept);
  const auto dirty_after_phi = system.capture_auxiliary_checkpoint_accepted_state();
  ASSERT_EQ(dirty_after_phi.size(), 2U);
  for (const auto& level : dirty_after_phi) EXPECT_TRUE(incidence_invalidated(level, kIncidenceQProvider));

  for (const auto& actual_point : observed.phi_points) ASSERT_TRUE(actual_point.has_value());
  const auto collect_current_phi = [&] {
    std::vector<AmrSystem<D>::ProgramFieldLevel> publication;
    const auto address = system.auxiliary_address(phi);
    for (int level = 0; level < 2; ++level) {
      const auto* carriers = system.prepared_amr_provider_storage_groups(level);
      if (!carriers) throw std::logic_error("current phi observation has no actual level carriers");
      const auto* values = carriers->find(address.group);
      if (!values) throw std::logic_error("current phi observation has no actual provider group");
      publication.push_back({*observed.phi_points.at(static_cast<std::size_t>(level)),
        {{phi, "test.amr-aux.phi-output", values, kernel_component(address.component)}}});
    }
    return publication;
  };
  {
    const auto publication = collect_current_phi();
    auto fine_comparison = publication[1].point;
    fine_comparison.level = publication[0].point.level; // Comparison only; actual points are retained.
    ASSERT_EQ(fine_comparison, publication[0].point);
    // Real public publisher; this borrowed observation ends before any restore.
    system.publish_program_field_components("test.amr-aux.phi-republication-first", publication);
  }
  observed.capture = true; observed.fail_fine_Q = true; observed.snapshots.clear();
  observed.Q_launches.clear(); observed.fine_Q_fault_point.reset();
  EXPECT_THROW((void)context.solve_fields_from_program_values_at(context.boundary_evaluation_point(15),
    13, "test.amr-aux.psi-slot", {{0, &stage, 1}}), std::exception);
  ASSERT_TRUE(observed.fine_Q_fault_point.has_value());
  EXPECT_EQ(observed.fine_Q_fault_point->level, 1);
  EXPECT_EQ(observed.fine_Q_fault_point->event, AuxiliaryEvaluationEvent::before_field_solve);
  const auto fault_launches = std::count(observed.Q_launches.begin(), observed.Q_launches.end(),
                                       *observed.fine_Q_fault_point);
  EXPECT_EQ(fault_launches, 1);
  const auto coarse_launches = std::count_if(observed.Q_launches.begin(), observed.Q_launches.end(),
      [](const auto& point) { return point.level == 0; });
  EXPECT_GE(coarse_launches, 1);
  const auto partial = system.capture_auxiliary_checkpoint_accepted_state();
  ASSERT_EQ(partial.size(), 2U);
  EXPECT_FALSE(incidence_invalidated(partial[0], kIncidenceQProvider));
  EXPECT_TRUE(incidence_invalidated(partial[1], kIncidenceQProvider));
  EXPECT_EQ(incidence_provenance(partial[1], kIncidenceQProvider),
            incidence_provenance(baseline_image[1], kIncidenceQProvider));
  ASSERT_FALSE(observed.snapshots.empty());
  EXPECT_EQ(observed.snapshots.front().level, 0);
  EXPECT_EQ(observed.fine_Q_fault_point->clock, observed.snapshots.front().actual_point.clock);
  EXPECT_EQ(observed.fine_Q_fault_point->accepted_step,
            static_cast<std::uint64_t>(observed.snapshots.front().actual_point.tick));
  EXPECT_EQ(observed.fine_Q_fault_point->stage, observed.snapshots.front().actual_point.stage);
  EXPECT_FALSE(incidence_invalidated(observed.snapshots.front().metadata[0], kIncidenceQProvider));
  EXPECT_TRUE(incidence_invalidated(observed.snapshots.front().metadata[1], kIncidenceQProvider));
  if (extra == IncidenceRestoreWitness::rank_local_refusal) {
    const auto accepted_state_coarse = system.block_level_state_global("material", 0);
    const auto accepted_state_fine = system.block_level_state_global("material", 1);
    const auto accepted_dirty = system.dirty_auxiliary_provider_identities();
    const auto accepted_topology = system.checkpoint_topology_epoch();
    const auto accepted_step = system.macro_step();
    const auto launches_before_refusal = observed.Q_launches;
    auto refused_image = partial;
    ASSERT_FALSE(refused_image[1].groups.empty());
    // A real incompatible public checkpoint request on exactly one prepared rank.
    // Its halo differs from the actual carrier; the restore's admission collective
    // must reject before any candidate or accepted publication, on every rank.
    const auto& lane = context.prepared_execution_lane();
    if (lane.rank() == 0) ++refused_image[1].groups.front().shape.halo[0];
    EXPECT_THROW(system.restore_auxiliary_checkpoint_accepted_state(refused_image),
                 std::invalid_argument);
    EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), partial);
    EXPECT_EQ(system.block_level_state_global("material", 0), accepted_state_coarse);
    EXPECT_EQ(system.block_level_state_global("material", 1), accepted_state_fine);
    EXPECT_EQ(system.dirty_auxiliary_provider_identities(), accepted_dirty);
    EXPECT_EQ(system.checkpoint_topology_epoch(), accepted_topology);
    EXPECT_EQ(system.macro_step(), accepted_step);
    EXPECT_EQ(system.step_transaction_depth(), 0U);
    EXPECT_EQ(observed.Q_launches, launches_before_refusal);
  }
  // Public POPSAUX3 capture/restore keeps distinct level memberships; no Input or
  // never-published marker and no live physical payload is manufactured here.
  system.restore_auxiliary_checkpoint_accepted_state(partial);
  EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), partial);
  const auto dirty_union = system.dirty_auxiliary_provider_identities();
  EXPECT_TRUE(std::find(dirty_union.begin(), dirty_union.end(), kIncidenceQProvider) != dirty_union.end());

  if (extra == IncidenceRestoreWitness::repeated_publication) {
    // Reuse the same topology and checkpoint twice. Every publication reacquires
    // the current real carriers; no borrowed storage pointer crosses a restore.
    const auto restored_topology = system.checkpoint_topology_epoch();
    const auto restored_step = system.macro_step();
    for (int repetition = 0; repetition < 2; ++repetition) {
      system.restore_auxiliary_checkpoint_accepted_state(partial);
      EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), partial);
      EXPECT_EQ(system.checkpoint_topology_epoch(), restored_topology);
      EXPECT_EQ(system.macro_step(), restored_step);
      {
        const auto current_publication = collect_current_phi();
        system.publish_program_field_components(
          "test.amr-aux.phi-republication-after-repeated-restore", current_publication);
      }
      const auto republished = system.capture_auxiliary_checkpoint_accepted_state();
      ASSERT_EQ(republished.size(), 2U);
      EXPECT_TRUE(incidence_invalidated(republished[0], kIncidenceQProvider));
      EXPECT_TRUE(incidence_invalidated(republished[1], kIncidenceQProvider));
      // Field publication dirties Q but cannot itself advance Q's accepted point.
      for (int level = 0; level < 2; ++level)
        EXPECT_EQ(incidence_provenance(republished[level], kIncidenceQProvider),
                  incidence_provenance(partial[level], kIncidenceQProvider));
    }
    system.restore_auxiliary_checkpoint_accepted_state(partial);
    EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), partial);
  }
  {
    // Restore replaced the provider allocations. Reacquire every actual group/value
    // and reconstruct observations; only the genuinely issued point values are reused.
    const auto current_publication = collect_current_phi();
    system.publish_program_field_components("test.amr-aux.phi-republication-reopens-coarse",
                                           current_publication);
  }
  const auto reopened = system.capture_auxiliary_checkpoint_accepted_state();
  ASSERT_EQ(reopened.size(), 2U);
  EXPECT_TRUE(incidence_invalidated(reopened[0], kIncidenceQProvider));
  EXPECT_TRUE(incidence_invalidated(reopened[1], kIncidenceQProvider));
  observed.fail_fine_Q = false; observed.snapshots.clear();
  auto complete = context.solve_fields_from_program_values_at(context.boundary_evaluation_point(15),
    13, "test.amr-aux.psi-slot", {{0, &stage, 1}});
  ASSERT_TRUE(complete.report().solved_value_available()) << complete.report().reason;
  (void)complete.consume(SolveConsumption::kAccept);
  ASSERT_EQ(observed.snapshots.size(), 2U);
  const auto& coarse = observed.snapshots[0];
  const auto& fine = observed.snapshots[1];
  EXPECT_EQ(coarse.level, 0); EXPECT_EQ(fine.level, 1);
  EXPECT_FALSE(incidence_invalidated(coarse.metadata[0], kIncidenceQProvider));
  EXPECT_TRUE(incidence_invalidated(coarse.metadata[1], kIncidenceQProvider));
  EXPECT_FALSE(incidence_invalidated(fine.metadata[0], kIncidenceQProvider));
  EXPECT_FALSE(incidence_invalidated(fine.metadata[1], kIncidenceQProvider));
  EXPECT_EQ(coarse.metadata[0], fine.metadata[0]) << "a rejected fine ancestor is not an accepted publication";
  const auto clean = system.capture_auxiliary_checkpoint_accepted_state();
  for (const auto& level : clean) EXPECT_FALSE(incidence_invalidated(level, kIncidenceQProvider));
  const Real field_bound = Real(1e-9); // New constant screened-field witness; old 13 bounds unchanged.
  for (int level = 0; level < 2; ++level) {
    const auto* groups = system.prepared_amr_provider_storage_groups(level);
    ASSERT_NE(groups, nullptr);
    for (const auto& component : std::array<std::pair<AuxiliaryComponentKey, Real>, 3>{
           {{phi, Real(3)}, {Q, Real(6)}, {psi, Real(6)}}}) {
      const auto address = system.auxiliary_address(component.first);
      const auto* actual_values = groups->find(address.group);
      ASSERT_NE(actual_values, nullptr);
      const auto cells = local_valid_cell_count(*actual_values);
      const Real error = incidence_local_constant_error(system, component.first, level, component.second);
      if (cells == 0)
        EXPECT_EQ(error, -std::numeric_limits<Real>::infinity());
      else
        EXPECT_LE(error, field_bound);
    }
    const auto* actual_Q = groups->find(system.auxiliary_address(Q).group);
    ASSERT_NE(actual_Q, nullptr);
    const Real dependency_error = incidence_local_dependency_error(system, phi, Q, level);
    if (local_valid_cell_count(*actual_Q) == 0)
      EXPECT_EQ(dependency_error, -std::numeric_limits<Real>::infinity());
    else
      EXPECT_EQ(dependency_error, Real(0));
  }
  if (extra == IncidenceRestoreWitness::accepted_state_consumer) {
    // The source operator belongs to the accepted generated block, not the Field RHS adapter.
    const auto check_accepted_state_source = [&](Real expected) {
      for (int level = 0; level < 2; ++level) {
        ASSERT_TRUE(observed.phi_points[level].has_value());
        const auto& accepted = system.prepared_amr_block_state(0, level);
        Field accepted_input(accepted);
        Field source(accepted.layout(), accepted.distribution(), accepted.local_rank(),
                     accepted.ncomp(), accepted.ghosts());
        source.set_val(Real(0));
        system.prepared_amr_block_level_source_into_at(
          0, *observed.phi_points[level], accepted_input, source);
        Field error(source.layout(), source.distribution(), source.local_rank(), 1, source.ghosts());
        error.set_val(Real(0));
        const auto* groups = system.prepared_amr_provider_storage_groups(level);
        ASSERT_NE(groups, nullptr);
        const auto Q_address = system.auxiliary_address(Q);
        const auto* current_Q = groups->find(Q_address.group);
        ASSERT_NE(current_Q, nullptr);
        for (std::size_t local = 0; local < source.local_size(); ++local)
          for_each_cell(source.box(local), IncidenceStateSourceErrorKernel{
            std::as_const(source).fab(local).view(), current_Q->fab(local).view(), error.fab(local).view(),
            kernel_component(Q_address.component)});
        Kokkos::fence();
        const Real dependency_error = reduce_max_local(error);
        if (local_valid_cell_count(source) == 0)
          EXPECT_EQ(dependency_error, -std::numeric_limits<Real>::infinity());
        else
          EXPECT_EQ(dependency_error, Real(0));
        for (int component = 0; component < 2; ++component) {
          const Real reference = component == 0 ? expected : Real(0);
          for (std::size_t local = 0; local < source.local_size(); ++local)
            for_each_cell(source.box(local), IncidenceConstantErrorKernel{
              std::as_const(source).fab(local).view(), error.fab(local).view(), component, reference});
          Kokkos::fence();
          const Real actual_error = reduce_max_local(error);
          if (local_valid_cell_count(source) == 0)
            EXPECT_EQ(actual_error, -std::numeric_limits<Real>::infinity());
          else
            EXPECT_LE(actual_error, field_bound);
        }
        const Real Q_error = incidence_local_constant_error(system, Q, level, expected);
        if (local_valid_cell_count(source) == 0)
          EXPECT_EQ(Q_error, -std::numeric_limits<Real>::infinity());
        else
          EXPECT_LE(Q_error, field_bound);
      }
    };
    check_accepted_state_source(Real(6));
    const auto accepted_state_coarse = system.block_level_state_global("material", 0);
    const auto accepted_state_fine = system.block_level_state_global("material", 1);
    const auto original_image = system.capture_auxiliary_checkpoint_accepted_state();

    // An actually produced scaled SSA state yields a different real accepted forcing.
    context.set_stage_time(7, 8);
    auto& doubled_stage = produce_stage(context, 1, Real(2));
    auto doubled_phi = context.solve_fields_from_program_values_at(context.boundary_evaluation_point(16),
      10, "test.amr-aux.phi-slot", {{0, &doubled_stage, 1}});
    ASSERT_TRUE(doubled_phi.report().solved_value_available()) << doubled_phi.report().reason;
    (void)doubled_phi.consume(SolveConsumption::kAccept);
    auto doubled_psi = context.solve_fields_from_program_values_at(context.boundary_evaluation_point(17),
      13, "test.amr-aux.psi-slot", {{0, &doubled_stage, 1}});
    ASSERT_TRUE(doubled_psi.report().solved_value_available()) << doubled_psi.report().reason;
    (void)doubled_psi.consume(SolveConsumption::kAccept);
    const auto doubled_image = system.capture_auxiliary_checkpoint_accepted_state();
    ASSERT_FALSE(observed.Q_launches.empty());
    const auto actual_refresh_point = observed.Q_launches.back();
    check_accepted_state_source(Real(12));

    system.restore_auxiliary_checkpoint_accepted_state(original_image);
    EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), original_image);
    check_accepted_state_source(Real(6)); // Rejects a stale pre-restore Q=12 binding.

    // Public all-level refresh at a genuinely issued point. The real fine launcher
    // throws after coarse candidate preparation, before accepted publication;
    // refresh restores its registry snapshot by swap.
    observed.fail_fine_Q = true;
    observed.fine_Q_fault_point.reset();
    EXPECT_THROW(system.refresh_auxiliary(actual_refresh_point), std::exception);
    ASSERT_TRUE(observed.fine_Q_fault_point.has_value());
    EXPECT_EQ(observed.fine_Q_fault_point->level, 1);
    EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), original_image);
    observed.fail_fine_Q = false;
    check_accepted_state_source(Real(6)); // Accepted provider plan must survive registry rollback.
    system.refresh_auxiliary(actual_refresh_point);
    check_accepted_state_source(Real(6));

    system.restore_auxiliary_checkpoint_accepted_state(doubled_image);
    EXPECT_EQ(system.capture_auxiliary_checkpoint_accepted_state(), doubled_image);
    check_accepted_state_source(Real(12));
    EXPECT_EQ(system.block_level_state_global("material", 0), accepted_state_coarse);
    EXPECT_EQ(system.block_level_state_global("material", 1), accepted_state_fine);
  }
}

TEST(PreparedFieldRhsInputs, ActualFieldPublicationReopensCoarseAuxWhileFineRemainsDirty) {
  actual_field_incidence_lifecycle(IncidenceRestoreWitness::original);
}

TEST(PreparedFieldRhsInputs, RepeatedActualAuxiliaryRestoreRebindsEveryNewFieldPublication) {
  actual_field_incidence_lifecycle(IncidenceRestoreWitness::repeated_publication);
}

TEST(PreparedFieldRhsInputs, OneRankAuxiliaryRestoreRefusalLeavesAcceptedIncidenceExact) {
  actual_field_incidence_lifecycle(IncidenceRestoreWitness::rank_local_refusal);
}

TEST(PreparedFieldRhsInputs, AcceptedProviderStateSourceSurvivesRestoreAndRegistryRollback) {
  actual_field_incidence_lifecycle(IncidenceRestoreWitness::accepted_state_consumer);
}
} // namespace
