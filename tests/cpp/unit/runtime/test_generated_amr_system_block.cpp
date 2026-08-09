#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/core/foundation/native_dimension.hpp>

#include <cstdint>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <int Dim>
struct AdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  using Aux = pops::AuxState<Dim>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_aux = pops::aux_comps_for<Law, Dim>();

  Law law{};

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.generated-amr.scalar-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }

  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const Aux&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
AdvectionModel<Dim> advection_model() {
  pops::RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = pops::Real(axis + 1);
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

template <int Dim>
class TestFieldNullspaceProvider final : public pops::FieldNullspaceProvider<Dim> {
 public:
  std::string_view identity() const noexcept override {
    return "tests.field-nullspace.exact-ranked";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "tests.field-nullspace.exact-ranked@1";
  }
  pops::PreparedProviderOptions default_options() const override {
    return {"tests.field-nullspace.options@1", {}};
  }
  bool accepts_options(const pops::PreparedProviderOptions& options) const noexcept override {
    return options.schema_identity == "tests.field-nullspace.options@1" && options.values.empty();
  }
  pops::PreparedProviderSupport supports(
      const pops::FieldNullspaceProviderRequest<Dim>&) const noexcept override {
    return pops::PreparedProviderSupport::reject(1, "test provider is registry-only");
  }
  std::string expected_prepared_contract(
      const pops::FieldNullspaceProviderRequest<Dim>&) const override {
    return "tests.field-nullspace.prepared@1";
  }
  pops::PreparedFieldNullspace<Dim> prepare(
      const pops::FieldNullspaceProviderRequest<Dim>&) const override {
    throw std::logic_error("test provider is registry-only");
  }
};

template <int Dim>
struct RenamedAdvectionModel : AdvectionModel<Dim> {
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative,
            {"renamed_u"},
            1,
            {pops::VariableRole::Custom},
            {"transported_density"}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive,
            {"renamed_u"},
            1,
            {pops::VariableRole::Custom},
            {"transported_density"}};
  }
};

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
pops::amr::tagging::ClusterResult<Dim> centered_cluster(
    const pops::amr::hierarchy::LevelLayout<Dim>& parent) {
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = parent.domain().lo[axis] + 2;
    upper[axis] = parent.domain().hi[axis] - 2;
  }
  const pops::mesh::BoxArray<Dim> boxes(std::vector<pops::Box<Dim>>{pops::Box<Dim>{lower, upper}});
  pops::amr::tagging::ClusterOptions<Dim> options;
  options.min_efficiency = 0.7;
  for (int axis = 0; axis < Dim; ++axis) {
    options.min_box_size[static_cast<std::size_t>(axis)] = 1;
    options.max_box_size[static_cast<std::size_t>(axis)] = 16;
  }
  options.budget = {16, 256, 8192, 64, 1U << 20};
  pops::amr::tagging::ClusterResultIdentity<Dim> identity{
      "test.generated-amr.cluster", parent.exact_identity(), options, {}, boxes.boxes()};
  return {boxes, std::move(identity)};
}

template <int Dim>
void publish_centered_fine_level(pops::AmrSystem<Dim>& system) {
  auto* engine = system.engine();
  ASSERT_NE(engine, nullptr);
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  const pops::amr::RefinementRatio<Dim> ratio(ratio_components);
  const pops::amr::regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = {16, 120},
      .fine_layout = {16, 120},
      .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()},
  };
  auto prepared =
      engine->prepare_regrid(0, ratio, centered_cluster(engine->hierarchy().layout(0)), budget);
  ASSERT_FALSE(prepared.removes_fine_level());
  ASSERT_TRUE(prepared.fine_layout().has_value());
  pops::MultiFab<Dim> child(
      prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
      engine->hierarchy().state(0).local_rank(), engine->hierarchy().state(0).ncomp(),
      engine->hierarchy().state(0).ghosts());
  child.set_val(pops::Real(1));
  engine->publish_regrid(0, std::move(prepared), std::move(child));
}

template <int Dim>
pops::runtime::multiblock::BoundaryEvaluationPoint point(int level) {
  return {.clock = "test-clock",
          .tick = 0,
          .level = level,
          .substep = 0,
          .stage = 0,
          .stage_fraction = {0, 1},
          .dt = 0.01,
          .physical_time = 0.0};
}

template <int Dim>
struct RuntimeFieldBoundaryProbe {
  inline static int prepare_residual_calls = 0;
  inline static int prepare_jvp_calls = 0;
  inline static int residual_calls = 0;
  inline static int jvp_calls = 0;
  inline static std::array<bool, 3> levels{};
  inline static int stage = -1;
  inline static pops::Real time = pops::Real(-1);
  inline static std::array<pops::Real, 3> state_min_by_level{};
  inline static bool force_failure = false;

  static void reset() {
    prepare_residual_calls = 0;
    prepare_jvp_calls = 0;
    residual_calls = 0;
    jvp_calls = 0;
    levels.fill(false);
    stage = -1;
    time = pops::Real(-1);
    state_min_by_level.fill(pops::Real(-1));
    force_failure = false;
  }

  static void observe(int face, const pops::FieldBoundaryExecutionContext<Dim>& context) {
    if (face < 0 || face >= 2 * Dim || context.failure == nullptr || context.state_count != 1 ||
        context.states == nullptr || context.state_identities == nullptr ||
        context.state_identities[0].empty() || context.state_distributions == nullptr ||
        context.parameters == nullptr || context.parameter_count != 1) {
      if (context.failure != nullptr) {
        context.failure->code = 902;
        context.failure->face = face;
      }
      return;
    }
    if (context.point.level >= 0 &&
        context.point.level < static_cast<int>(state_min_by_level.size())) {
      levels[static_cast<std::size_t>(context.point.level)] = true;
      state_min_by_level[static_cast<std::size_t>(context.point.level)] =
          pops::reduce_min_local(*context.states[0]);
    }
    stage = context.point.stage_slot;
    time = context.point.time;
    if (force_failure) {
      context.failure->code = 903;
      context.failure->face = face;
    }
  }

  static void prepare_residual(int face, const pops::MultiFab<Dim>& iterate,
                               pops::MultiFab<Dim>& operator_view,
                               const pops::Geometry<Dim>& geometry,
                               const pops::FieldBoundaryExecutionContext<Dim>& context) {
    (void)iterate;
    (void)operator_view;
    (void)geometry;
    ++prepare_residual_calls;
    observe(face, context);
  }

  static void prepare_jvp(int face, const pops::MultiFab<Dim>& iterate,
                          const pops::MultiFab<Dim>& direction, pops::MultiFab<Dim>& direction_view,
                          const pops::Geometry<Dim>& geometry,
                          const pops::FieldBoundaryExecutionContext<Dim>& context) {
    (void)iterate;
    (void)direction;
    (void)direction_view;
    (void)geometry;
    ++prepare_jvp_calls;
    observe(face, context);
  }

  static void add_residual(int face, const pops::MultiFab<Dim>& iterate,
                           pops::MultiFab<Dim>& output, const pops::Geometry<Dim>& geometry,
                           const pops::FieldBoundaryExecutionContext<Dim>& context) {
    (void)iterate;
    (void)output;
    (void)geometry;
    ++residual_calls;
    observe(face, context);
  }

  static void apply_jvp(int face, const pops::MultiFab<Dim>& iterate,
                        const pops::MultiFab<Dim>& direction, pops::MultiFab<Dim>& output,
                        const pops::Geometry<Dim>& geometry,
                        const pops::FieldBoundaryExecutionContext<Dim>& context) {
    (void)iterate;
    (void)direction;
    (void)output;
    (void)geometry;
    ++jvp_calls;
    observe(face, context);
  }

  static pops::CompiledFieldBoundaryKernel<Dim> kernel() {
    return {"test.runtime-amr-boundary",
            "test.runtime-amr-boundary.residual",
            "test.runtime-amr-boundary.jvp",
            &prepare_residual,
            &prepare_jvp,
            &add_residual,
            &apply_jvp,
            true};
  }
};

static_assert(pops::PreparedAmrSystemBlock<1>::dimension == 1);
static_assert(pops::PreparedAmrSystemBlock<2>::dimension == 2);
static_assert(pops::PreparedAmrSystemBlock<3>::dimension == 3);
static_assert(!std::is_same_v<pops::PreparedAmrSystemBlock<1>, pops::PreparedAmrSystemBlock<2>>);

TEST(GeneratedAmrSystemBlock, PreparesOneExactNativePackageImage) {
  constexpr int Dim = pops::kNativeDimension;
  auto prepared = pops::prepare_compiled_amr_system_block<Dim>(
      "tracer", advection_model<Dim>(), "minmod", "rusanov", "conservative", "explicit", 1.4, 2, 3);
  constexpr int expected_aux_components = pops::aux_comps_for<AdvectionModel<Dim>, Dim>();

  EXPECT_EQ(prepared.name, "tracer");
  EXPECT_EQ(prepared.ncomp, 1);
  EXPECT_EQ(prepared.aux_components, expected_aux_components);
  EXPECT_EQ(prepared.substeps, 2);
  EXPECT_EQ(prepared.stride, 3);
  EXPECT_EQ(prepared.time_route, "explicit");
  EXPECT_TRUE(static_cast<bool>(prepared.materialize_level));
  EXPECT_FALSE(prepared.collective_contract.empty());
  EXPECT_NE(prepared.provider_identity.find(".nd/" + std::to_string(Dim) + "/"), std::string::npos);
  for (int axis = 0; axis < Dim; ++axis)
    EXPECT_EQ(prepared.ghosts[axis], 2);
}

TEST(GeneratedAmrSystemBlock, PackageContractAuthenticatesPhysicalModelParameters) {
  constexpr int Dim = pops::kNativeDimension;
  pops::RealVector<Dim> other_velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    other_velocity[axis] = pops::Real(axis + 2);
  const auto first = pops::prepare_compiled_amr_system_block<Dim>(
      "tracer", advection_model<Dim>(), "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1);
  const auto second = pops::prepare_compiled_amr_system_block<Dim>(
      "tracer", AdvectionModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(other_velocity)},
      "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1);
  EXPECT_NE(first.collective_contract, second.collective_contract);
}

TEST(GeneratedAmrSystemBlock, PackageContractAuthenticatesVariableNamesRolesAndUserRoles) {
  constexpr int Dim = pops::kNativeDimension;
  const AdvectionModel<Dim> model = advection_model<Dim>();
  RenamedAdvectionModel<Dim> renamed;
  renamed.law = model.law;
  const auto first = pops::prepare_compiled_amr_system_block<Dim>(
      "tracer", model, "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1);
  const auto second = pops::prepare_compiled_amr_system_block<Dim>(
      "tracer", renamed, "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1);
  EXPECT_NE(first.collective_contract, second.collective_contract);
}

TEST(GeneratedAmrSystemBlock, RejectsUnpreparedOptionalAuthorities) {
  constexpr int Dim = pops::kNativeDimension;
  EXPECT_THROW((void)pops::prepare_compiled_amr_system_block<Dim>(
                   "tracer", advection_model<Dim>(), "minmod", "rusanov", "conservative",
                   "explicit", 1.4, 1, 1, 0.0, static_cast<double>(pops::kWenoEpsilon), true),
               std::invalid_argument);
  EXPECT_THROW((void)pops::prepare_compiled_amr_system_block<Dim>(
                   "tracer", advection_model<Dim>(), "minmod", "rusanov", "conservative",
                   "explicit", 1.4, 1, 1, 0.0, 1.0e-8, false),
               std::invalid_argument);
}

TEST(GeneratedAmrSystemBlock, CellPrimitiveConversionConsumesPreparedRecoveryOutcome) {
  constexpr int Dim = pops::kNativeDimension;
  const auto prepared = pops::prepare_compiled_amr_system_block<Dim>(
      "tracer", advection_model<Dim>(), "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1);

  const std::array<double, 1> conservative{2.0};
  std::array<double, 1> primitive{-9.0};
  const pops::RecoveryReport accepted =
      prepared.conservative_to_primitive(conservative.data(), primitive.data());
  EXPECT_TRUE(accepted.publication_permitted());
  EXPECT_DOUBLE_EQ(primitive[0], 2.0);

  const std::array<double, 1> invalid{std::numeric_limits<double>::quiet_NaN()};
  primitive[0] = -9.0;
  const pops::RecoveryReport rejected =
      prepared.conservative_to_primitive(invalid.data(), primitive.data());
  EXPECT_FALSE(rejected.publication_permitted());
  EXPECT_DOUBLE_EQ(primitive[0], -9.0);
}

TEST(GeneratedAmrSystemBlock, PrimitiveToConservativePublicationRoundtripsBeforeCommit) {
  constexpr int Dim = pops::kNativeDimension;
  const auto prepared = pops::prepare_compiled_amr_system_block<Dim>(
      "tracer", advection_model<Dim>(), "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1);

  const std::array<double, 1> primitive{2.0};
  std::array<double, 1> published{-9.0};
  EXPECT_NO_THROW(prepared.primitive_to_conservative(primitive.data(), published.data()));
  EXPECT_DOUBLE_EQ(published[0], 2.0);

  const std::array<double, 1> invalid{std::numeric_limits<double>::quiet_NaN()};
  published[0] = -9.0;
  EXPECT_THROW(prepared.primitive_to_conservative(invalid.data(), published.data()),
               std::runtime_error);
  EXPECT_DOUBLE_EQ(published[0], -9.0);
}

TEST(GeneratedAmrSystemBlock, FacadeRetainsAndExecutesPreparedRootLevel) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  ASSERT_EQ(system.n_blocks(), 0);

  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>(), "minmod", "rusanov",
                                "conservative", "explicit", 1.4, 2, 1);
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));

  ASSERT_EQ(system.n_blocks(), 1);
  ASSERT_EQ(system.n_levels(), 1);
  pops::MultiFab<Dim>* const auxiliary = &system.prepared_amr_level_auxiliary(0);
  const auto& state = system.engine()->hierarchy().state(0);
  pops::MultiFab<Dim> poisson_rhs(state.layout(), state.distribution(), state.local_rank(), 1,
                                  state.ghosts());
  poisson_rhs.set_val(pops::Real(0));
  system.add_prepared_amr_poisson_rhs(0, poisson_rhs);
  EXPECT_EQ(pops::reduce_max_local(poisson_rhs), pops::Real(0));
  const auto& evaluation = system.evaluate_prepared_amr_level(point<Dim>(0));
  EXPECT_EQ(evaluation.point, point<Dim>(0));
  EXPECT_EQ(evaluation.spatial_contract, system.engine()->spatial_contract());
  EXPECT_EQ(evaluation.topology_epoch, system.engine()->topology_epoch());
  EXPECT_EQ(evaluation.materialization_generation, system.engine()->materialization_generation());
  EXPECT_EQ(evaluation.residual.ncomp(), 1);
  EXPECT_EQ(evaluation.residual.layout(), system.engine()->hierarchy().state(0).layout());
  EXPECT_EQ(evaluation.integrated_face_fluxes.size(),
            system.engine()->hierarchy().state(0).local_size());
  EXPECT_EQ(&system.prepared_amr_level_evaluation(0), &evaluation);
  const pops::AmrSystem<Dim>& const_system = system;
  EXPECT_EQ(&const_system.prepared_amr_level_auxiliary(0), auxiliary);
}

TEST(GeneratedAmrSystemBlock, RegridRebuildsExactFineGhostProvidersAndInvalidatesLedger) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.prepared_amr_level_auxiliary(0).set_val(pops::Real(3));
  (void)system.evaluate_prepared_amr_level(point<Dim>(0));

  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();

  ASSERT_EQ(system.n_levels(), 2);
  EXPECT_EQ(pops::reduce_max_local(system.prepared_amr_level_auxiliary(0)), pops::Real(3));
  EXPECT_THROW((void)system.prepared_amr_level_evaluation(0), std::logic_error);
  pops::MultiFab<Dim>* const fine_auxiliary = &system.prepared_amr_level_auxiliary(1);
  const auto& fine = system.evaluate_prepared_amr_level(point<Dim>(1));
  EXPECT_EQ(fine.point, point<Dim>(1));
  EXPECT_EQ(fine.spatial_contract, system.engine()->spatial_contract());
  EXPECT_EQ(fine.topology_epoch, system.engine()->topology_epoch());
  EXPECT_EQ(fine.materialization_generation, system.engine()->materialization_generation());
  EXPECT_EQ(fine.residual.layout(), system.engine()->hierarchy().state(1).layout());
  EXPECT_EQ(fine.integrated_face_fluxes.size(), system.engine()->hierarchy().state(1).local_size());
  const pops::AmrSystem<Dim>& const_system = system;
  EXPECT_EQ(&const_system.prepared_amr_level_auxiliary(1), fine_auxiliary);
}

TEST(GeneratedAmrSystemBlock, ProgramContextEvaluatesExactStageStateWithoutPublishingIt) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  (void)system.engine();
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  context->begin_step(0.01);
  pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
  stage.set_val(pops::Real(2));
  pops::MultiFab<Dim> residual = context->rhs_scratch_like(stage);
  context->rhs_into(0, stage, residual, 7);

  EXPECT_EQ(pops::reduce_min_local(stage), pops::Real(2));
  EXPECT_EQ(pops::reduce_max_local(context->state(0)), pops::Real(1));
  EXPECT_EQ(system.prepared_amr_level_evaluation(0).point.stage, 7);

  pops::MultiFab<Dim> foreign(stage.layout(), stage.distribution(), stage.local_rank(), 2,
                              stage.ghosts());
  foreign.set_val(pops::Real(9));
  pops::MultiFab<Dim> foreign_residual = context->rhs_scratch_like(foreign);
  EXPECT_THROW(context->rhs_into(0, foreign, foreign_residual, 8), std::invalid_argument);
  EXPECT_EQ(pops::reduce_max_local(context->state(0)), pops::Real(1));
  EXPECT_EQ(system.prepared_amr_level_evaluation(0).point.stage, 7);
}

TEST(GeneratedAmrSystemBlock, RegistersOnlyExactRankedNullspaceProviders) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.register_field_nullspace_provider(std::make_shared<TestFieldNullspaceProvider<Dim>>());
  EXPECT_THROW(
      system.register_field_nullspace_provider(std::make_shared<TestFieldNullspaceProvider<Dim>>()),
      std::invalid_argument);
}

TEST(GeneratedAmrSystemBlock, DefaultFieldPublishesOnlyAfterSolveOutcomeAcceptance) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.set_poisson();
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.set_program_block_map({0});
  system.prepared_amr_level_auxiliary(0).set_val(pops::Real(7));

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  context->begin_step(0.01);
  pops::SolveOutcome outcome = context->solve_default_field_on_coarse_level();
  ASSERT_TRUE(outcome.report().solved_value_available());
  EXPECT_EQ(pops::reduce_min(system.prepared_amr_level_auxiliary(0), 0), pops::Real(7));

  const pops::SolveReport accepted = outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
  EXPECT_NEAR(static_cast<double>(pops::reduce_max(system.prepared_amr_level_auxiliary(0), 0)), 0.0,
              1.0e-8);
  EXPECT_EQ(system.field_provider_levels("pops.amr.default-field"), 1);
  EXPECT_EQ(system.field_provider_slots(), std::vector<std::string>{"pops.amr.default-field"});
}

TEST(GeneratedAmrSystemBlock, NamedFieldConsumesExactStageWithoutPublishingState) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.level-local", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan("field/tracer", "test.named-field-plan", "test.named-field",
                               "test.aux-owner", "tracer", "phi", {"test.rhs"}, {"tracer"},
                               {"charge"}, {1.0}, "geometric_mg", hierarchy,
                               pops::geometric_mg_amr_field_solver_options(
                                   pops::GeometricMgOptions{}, pops::CompositeFacOptions{}));
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.register_elliptic_field("tracer", "phi", {0}, 1);
  system.set_block_elliptic_field(
      "tracer", "phi",
      [](const pops::MultiFab<Dim>&, pops::MultiFab<Dim>& rhs) { rhs.set_val(pops::Real(0)); });
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  context->begin_step(0.01);
  pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
  stage.set_val(pops::Real(3));
  pops::SolveOutcome outcome =
      context->solve_fields_from_state_at(point<Dim>(0), "field/tracer", 0, stage);
  ASSERT_TRUE(outcome.report().solved_value_available());
  EXPECT_EQ(pops::reduce_max(context->state(0), 0), pops::Real(1));
  (void)outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_EQ(pops::reduce_max(context->state(0), 0), pops::Real(1));
  EXPECT_EQ(system.field_provider_levels("field/tracer"), 1);
}

TEST(GeneratedAmrSystemBlock,
     DynamicFieldBoundaryConsumesExactStageAndPublishesOnlyAfterNewtonAcceptance) {
  constexpr int Dim = pops::kNativeDimension;
  RuntimeFieldBoundaryProbe<Dim>::reset();
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan("field/tracer", "test.dynamic-field-plan", "test.dynamic-field",
                               "test.aux-owner", "tracer", "phi", {"test.rhs"}, {"tracer"},
                               {"charge"}, {1.0}, "geometric_mg", hierarchy,
                               pops::geometric_mg_amr_field_solver_options(
                                   pops::GeometricMgOptions{}, pops::CompositeFacOptions{}));
  system.set_field_reaction("field/tracer", 50.0);
  system.set_field_boundary_dependencies("field/tracer", {"tracer"}, {0}, {}, {}, {});
  system.set_field_boundary_parameters("field/tracer", {0.25});
  system.set_field_boundary_kernel("field/tracer", RuntimeFieldBoundaryProbe<Dim>::kernel());
  system.set_field_newton_plan("field/tracer", 1.0e-9, 4, 1.0e-10, 80, 20, 1.0e-4, 1.0 / 1024.0);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.register_elliptic_field("tracer", "phi", {0}, 1);
  system.set_block_elliptic_field(
      "tracer", "phi",
      [](const pops::MultiFab<Dim>&, pops::MultiFab<Dim>& rhs) { rhs.set_val(pops::Real(1)); });
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();
  system.set_program_block_map({0});
  system.prepared_amr_level_auxiliary(0).set_val(pops::Real(7));
  system.prepared_amr_level_auxiliary(1).set_val(pops::Real(7));

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  context->begin_step(0.01);
  pops::MultiFab<Dim> stage_state = context->scratch_state_like(context->state(0));
  stage_state.set_val(pops::Real(3));
  auto evaluation = point<Dim>(0);
  evaluation.stage = 4;
  evaluation.physical_time = 0.125;
  RuntimeFieldBoundaryProbe<Dim>::force_failure = true;
  EXPECT_THROW(
      (void)context->solve_fields_from_state_at(evaluation, "field/tracer", 0, stage_state),
      std::runtime_error);
  EXPECT_EQ(pops::reduce_min(system.prepared_amr_level_auxiliary(0), 0), pops::Real(7));
  EXPECT_EQ(pops::reduce_min(system.prepared_amr_level_auxiliary(1), 0), pops::Real(7));

  RuntimeFieldBoundaryProbe<Dim>::force_failure = false;
  pops::SolveOutcome outcome =
      context->solve_fields_from_state_at(evaluation, "field/tracer", 0, stage_state);
  ASSERT_TRUE(outcome.report().solved_value_available()) << outcome.report().reason;
  EXPECT_EQ(pops::reduce_min(system.prepared_amr_level_auxiliary(0), 0), pops::Real(7));
  EXPECT_EQ(pops::reduce_min(system.prepared_amr_level_auxiliary(1), 0), pops::Real(7));
  EXPECT_GT(RuntimeFieldBoundaryProbe<Dim>::prepare_residual_calls, 0);
  EXPECT_GT(RuntimeFieldBoundaryProbe<Dim>::prepare_jvp_calls, 0);
  EXPECT_GT(RuntimeFieldBoundaryProbe<Dim>::residual_calls, 0);
  EXPECT_GT(RuntimeFieldBoundaryProbe<Dim>::jvp_calls, 0);
  EXPECT_TRUE(RuntimeFieldBoundaryProbe<Dim>::levels[0]);
  EXPECT_TRUE(RuntimeFieldBoundaryProbe<Dim>::levels[1]);
  EXPECT_EQ(RuntimeFieldBoundaryProbe<Dim>::stage, 4);
  EXPECT_EQ(RuntimeFieldBoundaryProbe<Dim>::time, pops::Real(0.125));
  EXPECT_EQ(RuntimeFieldBoundaryProbe<Dim>::state_min_by_level[0], pops::Real(3));
  EXPECT_EQ(RuntimeFieldBoundaryProbe<Dim>::state_min_by_level[1], pops::Real(1));

  const pops::SolveReport accepted = outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
  EXPECT_NE(pops::reduce_min(system.prepared_amr_level_auxiliary(0), 0), pops::Real(7));
  EXPECT_NE(pops::reduce_min(system.prepared_amr_level_auxiliary(1), 0), pops::Real(7));
}

TEST(GeneratedAmrSystemBlock, CompositeFieldInstallsCoverageAwareNullspaceOnEveryLiveLevel) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    config.periodicity[axis] = true;
  }
  pops::AmrSystem<Dim> system(config);
  system.set_poisson();
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  context->begin_step(0.01);
  pops::SolveOutcome outcome = context->solve_default_field_on_coarse_level();
  ASSERT_TRUE(outcome.report().solved_value_available());
  (void)outcome.consume(pops::SolveConsumption::kAccept);

  EXPECT_EQ(system.field_provider_levels("pops.amr.default-field"), 2);
  EXPECT_EQ(system.field_potential_level_global("pops.amr.default-field", 0).size(),
            cell_count(config.shape));
  pops::Extent<Dim> fine_shape{};
  for (int axis = 0; axis < Dim; ++axis)
    fine_shape[axis] = 16;
  EXPECT_EQ(system.field_potential_level_global("pops.amr.default-field", 1).size(),
            cell_count(fine_shape));
}

TEST(GeneratedAmrSystemBlock, ProgramContextRefusesUnsynchronizedHierarchyBeforeMutation) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  EXPECT_THROW(
      context->advance_hierarchy(0.01, [&](double) { context->state(0).set_val(pops::Real(9)); }),
      std::runtime_error);
  EXPECT_EQ(pops::reduce_max_local(system.engine()->hierarchy().state(0)), pops::Real(1));
  EXPECT_EQ(pops::reduce_max_local(system.engine()->hierarchy().state(1)), pops::Real(1));
}

TEST(GeneratedAmrSystemBlock, ProgramContextRetainsAndInterpolatesExactLevelHistory) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  (void)system.engine();
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("clock.macro");
  context->declare_clock_relation("clock.macro", "clock.fast", 2);
  context->register_history("tracer.rate", 2, 1, 0, "tracer.U", "cell.conservative", "clock.macro",
                            "dense.linear");

  pops::MultiFab<Dim> sample = context->scratch_state_like(context->state(0));
  context->begin_step(0.2);
  sample.set_val(pops::Real(10));
  context->store_history("tracer.rate", sample, 0);
  context->rotate_histories("clock.macro");

  context->begin_step(0.4);
  sample.set_val(pops::Real(20));
  context->store_history("tracer.rate", sample, 0);
  pops::MultiFab<Dim> interpolated = context->scratch_state_like(sample);
  interpolated.set_val(pops::Real(-1));
  context->interpolate_history_linear(interpolated, "tracer.rate", 2, 0, "clock.macro",
                                      "clock.fast", -1, pops::Real(0));

  EXPECT_EQ(pops::reduce_min_local(interpolated), pops::Real(15));
  EXPECT_EQ(pops::reduce_max_local(interpolated), pops::Real(15));
  EXPECT_THROW((void)context->schedule_decision(17, true, true), std::runtime_error);
}

TEST(GeneratedAmrSystemBlock, ProgramContextRefusesHistoryRegridBeforeTopologyMutation) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  auto* engine = system.engine();
  ASSERT_NE(engine, nullptr);
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("clock.macro");
  context->register_history("tracer.rate", 1, 1, 0, "tracer.U", "cell.conservative", "clock.macro",
                            "dense.linear");
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  const pops::amr::RefinementRatio<Dim> ratio(ratio_components);
  const pops::amr::regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = {16, 120},
      .fine_layout = {16, 120},
      .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()},
  };
  auto prepared =
      context->prepare_regrid(0, ratio, centered_cluster(engine->hierarchy().layout(0)), budget);
  ASSERT_TRUE(prepared.fine_layout().has_value());
  pops::MultiFab<Dim> child(
      prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
      engine->hierarchy().state(0).local_rank(), engine->hierarchy().state(0).ncomp(),
      engine->hierarchy().state(0).ghosts());
  child.set_val(pops::Real(1));

  EXPECT_THROW(context->publish_regrid(std::move(prepared), std::move(child)), std::runtime_error);
  EXPECT_EQ(engine->hierarchy().num_levels(), 1u);
}

TEST(GeneratedAmrSystemBlock, CflUsesFinestExactGeometryAndPreparedModelSpeed) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>(), "minmod", "rusanov",
                                "conservative", "explicit", 1.4, 2, 1);
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  publish_centered_fine_level(system);
  system.install_program_step([](double) {});

  constexpr double cfl = 0.4;
  const double dt = system.step_cfl(cfl, 1.0e-12);
  const double expected = cfl * (1.0 / 16.0) * 2.0 / static_cast<double>(Dim);
  EXPECT_NEAR(dt, expected, 1.0e-12);
  EXPECT_EQ(system.last_dt_bound(), "transport:tracer");
}

TEST(GeneratedAmrSystemBlock, CflAuthenticatesRequestsAndBoundOrderBeforeCallbacks) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.install_program_step([](double) {});

  int callback_count = 0;
  const std::string bound_label =
      pops::n_ranks() == 1 ? "shared" : (pops::my_rank() == 0 ? "rank-zero" : "rank-other");
  system.add_dt_bound(bound_label, [&callback_count] {
    ++callback_count;
    return 0.5;
  });

  const double locally_invalid_cfl = pops::n_ranks() > 1 && pops::my_rank() != 0 ? 0.4 : -0.4;
  EXPECT_ANY_THROW((void)system.step_cfl(locally_invalid_cfl));
  EXPECT_EQ(callback_count, 0);

  if (pops::n_ranks() == 1) {
    EXPECT_NO_THROW((void)system.step_cfl(0.4));
    EXPECT_EQ(callback_count, 1);
  } else {
    EXPECT_THROW((void)system.step_cfl(0.4), std::invalid_argument);
    EXPECT_EQ(callback_count, 0);
  }
}

}  // namespace
