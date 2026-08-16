#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/core/foundation/native_dimension.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {
template <int Dim, class Model>
void add_test_compiled_model(
    AmrSystem<Dim>& system, const std::string& name, Model model,
    const std::string& limiter = "minmod", const std::string& riemann = "rusanov",
    const std::string& reconstruction = "conservative", const std::string& time = "explicit",
    double gamma = static_cast<double>(kPhysicalDefaultGamma), int substeps = 1, int stride = 1,
    const std::vector<std::string>& implicit_vars = {},
    const std::vector<std::string>& implicit_roles = {}, double positivity_floor = 0.0,
    double weno_epsilon = static_cast<double>(kWenoEpsilon), bool wave_speed_cache = false) {
  add_compiled_model<Dim>(system, name, std::move(model), limiter, riemann, reconstruction, time,
                          gamma, substeps, stride, implicit_vars, implicit_roles, positivity_floor,
                          weno_epsilon, wave_speed_cache, "tests.tracer/physical_flux");
}

template <int Dim, class Model>
PreparedAmrSystemBlock<Dim> prepare_test_compiled_amr_system_block(
    const std::string& name, Model model, const std::string& limiter, const std::string& riemann,
    const std::string& reconstruction, const std::string& time, double gamma, int substeps,
    int stride, double positivity_floor = 0.0,
    double weno_epsilon = static_cast<double>(kWenoEpsilon), bool wave_speed_cache = false,
    const std::string& provider_consumer_qid = "tests.tracer/physical_flux") {
  return prepare_compiled_amr_system_block<Dim>(
      name, std::move(model), limiter, riemann, reconstruction, time, gamma, substeps, stride,
      positivity_floor, weno_epsilon, wave_speed_cache, provider_consumer_qid);
}
}  // namespace pops

#define add_compiled_model add_test_compiled_model
#define prepare_compiled_amr_system_block prepare_test_compiled_amr_system_block

namespace {

template <int Dim>
struct AdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;

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
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
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
struct DiffusiveAdvectionModel : AdvectionModel<Dim> {
  pops::Real diffusivity_value = pops::Real(0);

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.generated-amr.diffusive-scalar-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    AdvectionModel<Dim>::serialize_exact_parameters(contract);
    contract.scalar(diffusivity_value);
  }
  POPS_HD pops::Real diffusivity() const { return diffusivity_value; }
};

template <int Dim>
DiffusiveAdvectionModel<Dim> diffusive_advection_model(pops::Real diffusivity) {
  DiffusiveAdvectionModel<Dim> result;
  result.law = pops::nd::ScalarAdvection<Dim>::prepare({});
  result.diffusivity_value = diffusivity;
  return result;
}

template <int Dim>
pops::runtime::system::AuxiliaryComponentKey install_field_output(pops::AmrSystem<Dim>& system,
                                                                  const std::string& owner,
                                                                  const std::string& field) {
  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  AuxiliaryComponentKey key{owner, "field", field, "potential"};
  AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "amr-field", "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "test.field-output/" + owner + "/" + field,
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{key, contract, shape}},
      {}});
  system.seal_auxiliary_providers();
  return key;
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
  pops::PreparedFieldNullspace<Dim> prepare(const pops::FieldNullspaceProviderRequest<Dim>&,
                                            const pops::ExecutionLane&) const override {
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
      "tracer", advection_model<Dim>(), "minmod", "rusanov", "conservative", "explicit", 1.4, 2, 3,
      0.0, static_cast<double>(pops::kWenoEpsilon), false, "tests.tracer/physical_flux");

  EXPECT_EQ(prepared.name, "tracer");
  EXPECT_EQ(prepared.ncomp, 1);
  EXPECT_EQ(prepared.provider_components, 0);
  EXPECT_EQ(prepared.reconstruction_order, 2);
  EXPECT_EQ(prepared.substeps, 2);
  EXPECT_EQ(prepared.stride, 3);
  EXPECT_EQ(prepared.time_route, "explicit");
  EXPECT_TRUE(static_cast<bool>(prepared.materialize_level));
  EXPECT_FALSE(prepared.collective_contract.empty());
  EXPECT_NE(prepared.provider_identity.find(".nd/" + std::to_string(Dim) + "/"), std::string::npos);
  EXPECT_FALSE(prepared.staircase_provider_identity.empty());
  EXPECT_FALSE(prepared.cut_cell_provider_identity.empty());
  for (int axis = 0; axis < Dim; ++axis)
    EXPECT_EQ(prepared.ghosts[axis], 2);

  const auto weno = pops::prepare_compiled_amr_system_block<Dim>(
      "weno-tracer", advection_model<Dim>(), "weno5", "rusanov", "conservative", "explicit", 1.4, 1,
      1, 0.0, static_cast<double>(pops::kWenoEpsilon), false, "tests.tracer/physical_flux");
  EXPECT_EQ(weno.reconstruction_order, 5);
  for (int axis = 0; axis < Dim; ++axis)
    EXPECT_EQ(weno.ghosts[axis], 3);
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
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/facade-root-level");
  ASSERT_EQ(system.n_blocks(), 0);

  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>(), "minmod", "rusanov",
                                "conservative", "explicit", 1.4, 2, 1);
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));

  ASSERT_EQ(system.n_blocks(), 1);
  ASSERT_EQ(system.n_levels(), 1);
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
}

TEST(GeneratedAmrSystemBlock, VariableNamesReadAuthenticatedPreparedBlockMetadata) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/variable-names");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());

  EXPECT_EQ(system.variable_names("tracer", "conservative"), (std::vector<std::string>{"u"}));
  EXPECT_EQ(system.variable_names("tracer", "primitive"), (std::vector<std::string>{"u"}));
  EXPECT_THROW((void)system.variable_names("missing", "conservative"), std::runtime_error);
  EXPECT_THROW((void)system.variable_names("tracer", "invalid"), std::invalid_argument);
}

TEST(GeneratedAmrSystemBlock, RegridRebuildsExactFineGhostProvidersAndInvalidatesLedger) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/regrid-ghost-providers");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  (void)system.evaluate_prepared_amr_level(point<Dim>(0));

  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();

  ASSERT_EQ(system.n_levels(), 2);
  EXPECT_THROW((void)system.prepared_amr_level_evaluation(0), std::logic_error);
  const auto& fine = system.evaluate_prepared_amr_level(point<Dim>(1));
  EXPECT_EQ(fine.point, point<Dim>(1));
  EXPECT_EQ(fine.spatial_contract, system.engine()->spatial_contract());
  EXPECT_EQ(fine.topology_epoch, system.engine()->topology_epoch());
  EXPECT_EQ(fine.materialization_generation, system.engine()->materialization_generation());
  EXPECT_EQ(fine.residual.layout(), system.engine()->hierarchy().state(1).layout());
  EXPECT_EQ(fine.integrated_face_fluxes.size(), system.engine()->hierarchy().state(1).local_size());
}

TEST(GeneratedAmrSystemBlock,
     EmbeddedBoundaryRematerializesPerLevelAndWeightsCompositeMassAndSidecars) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/embedded-boundary-levels");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  EXPECT_THROW(
      system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0}, "staircase", -1.0),
      std::invalid_argument);
  system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0}, "staircase");
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));

  const auto initial_report = system.effective_options_report();
  EXPECT_TRUE(initial_report.eb.enabled);
  EXPECT_EQ(initial_report.eb.geometry_mode, "staircase");
  EXPECT_FALSE(initial_report.eb.semantic_digest.empty());
  EXPECT_FALSE(initial_report.eb.materialization_digest.empty());
  const double initial_mass = system.mass("tracer");
  EXPECT_GT(initial_mass, 0.0);
  EXPECT_LT(initial_mass, 1.0);
  const auto active = system.output_embedded_boundary_local_pieces("pops_active", 0);
  const auto phi = system.output_embedded_boundary_local_pieces("pops_phi", 0);
  const auto kappa = system.output_embedded_boundary_local_pieces("pops_kappa", 0);
  ASSERT_EQ(active.size(), phi.size());
  ASSERT_EQ(active.size(), kappa.size());
  ASSERT_FALSE(active.empty());
  EXPECT_EQ(active.front().box, phi.front().box);
  EXPECT_EQ(active.front().box, kappa.front().box);
  EXPECT_EQ(active.front().owner_rank, phi.front().owner_rank);
  auto observer = pops::ObserverMpiLane::duplicate_world_collectively(
      "test/generated-amr-system-block/embedded-boundary-output");
  const auto gathered = system.output_embedded_boundary_root_pieces(observer, "pops_kappa", 0);
  if (observer.rank() == 0)
    EXPECT_FALSE(gathered.empty());
  else
    EXPECT_TRUE(gathered.empty());
  observer.close_collectively();

  const auto& root_evaluation = system.evaluate_prepared_amr_level(point<Dim>(0));
  const pops::Real residual_magnitude =
      std::max(Kokkos::abs(pops::reduce_min_local(root_evaluation.residual, 0)),
               Kokkos::abs(pops::reduce_max_local(root_evaluation.residual, 0)));
  EXPECT_GT(static_cast<double>(residual_magnitude), 0.0);
  EXPECT_NEAR(static_cast<double>(pops::reduce_sum_local(root_evaluation.residual, 0)), 0.0,
              1.0e-12);
  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();

  ASSERT_EQ(system.n_levels(), 2);
  const auto fine_active = system.output_embedded_boundary_local_pieces("pops_active", 1);
  EXPECT_GT(pops::all_reduce_sum(static_cast<long>(fine_active.size())), 0L);
  const auto rematerialized_report = system.effective_options_report();
  EXPECT_EQ(rematerialized_report.eb.semantic_digest, initial_report.eb.semantic_digest);
  EXPECT_NE(rematerialized_report.eb.materialization_digest,
            initial_report.eb.materialization_digest);
  EXPECT_NEAR(system.mass("tracer"), initial_mass, 1.0e-12);
  EXPECT_NEAR(system.composite_reduce("tracer", "sum", 0, {0}), initial_mass, 1.0e-12);
  EXPECT_NEAR(system.composite_reduce("tracer", "sum", 0, {0, 1}), initial_mass, 1.0e-12);
}

TEST(GeneratedAmrSystemBlock, CutCellCapabilityExecutesAtExactRank) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/cut-cell-exact-rank");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  EXPECT_NO_THROW(
      system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0}, "cutcell"));
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  EXPECT_EQ(system.effective_options_report().eb.geometry_mode, "cutcell");
  EXPECT_NO_THROW((void)system.evaluate_prepared_amr_level(point<Dim>(0)));
}

TEST(GeneratedAmrSystemBlock, DiffusiveEmbeddedRoutesRefuseWithoutFaceGeometry) {
  constexpr int Dim = pops::kNativeDimension;
  for (const std::string mode : {"staircase", "cutcell"}) {
    pops::AmrSystemConfig<Dim> config;
    config.level_count = 1;
    config.transition_ratios.clear();
    config.transition_buffers.clear();
    config.transition_lookaheads.clear();
    for (int axis = 0; axis < Dim; ++axis)
      config.shape[axis] = 8;
    pops::AmrSystem<Dim> system(config);
    pops::test::install_amr_runtime_authority(
        system, "tests.generated-amr/diffusive-embedded-refusal/" + mode);
    system.install_block_state_route("diffusive", "state/diffusive");
    pops::add_compiled_model<Dim>(system, "diffusive", diffusive_advection_model<Dim>(0.125));
    system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0}, mode);
    system.set_conservative_state("diffusive", std::vector<double>(cell_count(config.shape), 1.0));
    EXPECT_THROW((void)system.evaluate_prepared_amr_level(point<Dim>(0)), std::invalid_argument);
  }
}

TEST(GeneratedAmrSystemBlock, EmbeddedBoundaryAuthoringRejectsDivergentMpiInputBeforeMutation) {
  if (pops::n_ranks() == 1)
    return;
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system,
                                            "tests.generated-amr/embedded-boundary-authoring");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());

  const std::string divergent_mode = pops::my_rank() == 0 ? "staircase" : "invalid-on-rank";
  EXPECT_ANY_THROW(
      system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0}, divergent_mode));
  EXPECT_NO_THROW(
      system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0}, "staircase"));
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  EXPECT_EQ(system.effective_options_report().eb.geometry_mode, "staircase");
}

TEST(GeneratedAmrSystemBlock, ProgramContextOwnsOneExactHierarchyTensorAuthority) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system,
                                            "tests.generated-amr/hierarchy-tensor-authority");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  const auto slots = pops::runtime::program::tensor_elliptic_detail::assembly_slots<Dim>();
  const auto options = pops::runtime::program::tensor_elliptic_detail::default_options();
  const auto configure = [&] {
    context->configure_hierarchy_tensor_solver(
        0, 1, std::string(pops::runtime::program::tensor_elliptic_detail::kCompositeTensorProvider),
        "test.generated-amr.tensor-plan",
        std::string(pops::runtime::program::tensor_elliptic_detail::kScalarTensorEllipticContract),
        slots, "pops.tensor-elliptic.solution", options);
  };

  ASSERT_NO_THROW(configure());
  EXPECT_FALSE(context->uses_prepared_krylov_fallback());
  pops::MultiFab<Dim>* first_solution = &context->hierarchy_solution();
  ASSERT_NO_THROW(configure());
  EXPECT_EQ(&context->hierarchy_solution(), first_solution);

  context->for_each_program_resource_level([&](int) {
    pops::MultiFab<Dim> fallback = context->rhs_scratch_like(context->state(0));
    for (int row = 0; row < Dim; ++row)
      for (int column = 0; column < Dim; ++column)
        context
            ->assembly_target(
                fallback,
                pops::runtime::program::tensor_elliptic_detail::coefficient_slot(row, column))
            .set_val(row == column ? pops::Real(1) : pops::Real(0));
    context->assembly_target(fallback, "pops.tensor-elliptic.rhs").set_val(pops::Real(0));
    context->assembly_target(fallback, "pops.tensor-elliptic.flux").set_val(pops::Real(0));
    context->stage_linear_initial_guess();
    EXPECT_EQ(&context->assembly_source(fallback, "pops.tensor-elliptic.solution"),
              &context->hierarchy_solution());
    EXPECT_THROW((void)context->assembly_target(fallback, "undeclared"), std::invalid_argument);
  });

  pops::SolveOutcome outcome =
      context->solve_hierarchy_tensor(0, 1, pops::Real(1.0e-8), pops::Real(0), 8);
  ASSERT_TRUE(outcome.report().solved_value_available()) << outcome.report().reason;
  EXPECT_TRUE(outcome.consume(pops::SolveConsumption::kAccept).solved());
}

TEST(GeneratedAmrSystemBlock, ProgramContextEvaluatesExactStageStateWithoutPublishingIt) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/exact-stage-state");
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

TEST(GeneratedAmrSystemBlock, RhsGroupPrevalidatesAndPublishesFullAndFluxRoundsAtomically) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/rhs-group-atomic");
  system.install_block_state_route("tracer", "state/tracer");
  system.install_block_state_route("peer", "state/peer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  pops::add_compiled_model<Dim>(system, "peer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.set_conservative_state("peer", std::vector<double>(cell_count(config.shape), 3.0));
  (void)system.engine();
  system.set_program_block_map({0, 1});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-rhs-group-clock");
  context->begin_step(0.01);
  pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
  stage.set_val(pops::Real(2));

  pops::MultiFab<Dim> full = context->rhs_scratch_like(stage);
  pops::MultiFab<Dim> grouped_full = context->rhs_scratch_like(stage);
  context->rhs_into(0, stage, full, 11);
  context->rhs_group(12, {{0, &stage, &grouped_full, 13, 0}});
  EXPECT_EQ(pops::difference_sum_sq_all_local(full, grouped_full), pops::Real(0));

  pops::MultiFab<Dim> flux = context->rhs_scratch_like(stage);
  pops::MultiFab<Dim> grouped_flux = context->rhs_scratch_like(stage);
  context->neg_div_flux_default_into(0, stage, flux, 14);
  context->rhs_group(15, {{0, &stage, &grouped_flux, 16, 1}});
  EXPECT_EQ(pops::difference_sum_sq_all_local(flux, grouped_flux), pops::Real(0));

  pops::MultiFab<Dim> peer_stage = context->scratch_state_like(context->state(1));
  peer_stage.set_val(pops::Real(4));
  pops::MultiFab<Dim> peer_flux = context->rhs_scratch_like(peer_stage);
  pops::MultiFab<Dim> grouped_first = context->rhs_scratch_like(stage);
  pops::MultiFab<Dim> grouped_second = context->rhs_scratch_like(peer_stage);
  context->neg_div_flux_default_into(1, peer_stage, peer_flux, 17);
  context->rhs_group(
      18, {{0, &stage, &grouped_first, 19, 0}, {1, &peer_stage, &grouped_second, 20, 1}});
  EXPECT_EQ(pops::difference_sum_sq_all_local(full, grouped_first), pops::Real(0));
  EXPECT_EQ(pops::difference_sum_sq_all_local(peer_flux, grouped_second), pops::Real(0));

  const auto published_point = system.prepared_amr_level_evaluation(0).point;
  const auto published_epoch = system.prepared_amr_level_evaluation(0).topology_epoch;
  const auto published_generation =
      system.prepared_amr_level_evaluation(0).materialization_generation;
  ASSERT_NE(system.prepared_amr_level_evaluation_if_present(0), nullptr);

  // The first request prepares successfully.  The second has a valid grouped request shape and a
  // distinct runtime block, but its state cannot be staged into that block's exact live storage.
  // It therefore fails after the first detached evaluation and must not publish either result.
  pops::MultiFab<Dim> malformed_peer(peer_stage.layout(), peer_stage.distribution(),
                                     peer_stage.local_rank(), 2, peer_stage.ghosts());
  malformed_peer.set_val(pops::Real(5));
  pops::MultiFab<Dim> malformed_output = context->rhs_scratch_like(malformed_peer);
  pops::MultiFab<Dim> first_late_output = context->rhs_scratch_like(stage);
  first_late_output.set_val(pops::Real(31));
  malformed_output.set_val(pops::Real(37));
  EXPECT_THROW(context->rhs_group(21, {{0, &stage, &first_late_output, 22, 0},
                                       {1, &malformed_peer, &malformed_output, 23, 0}}),
               std::exception);
  EXPECT_EQ(pops::reduce_min_local(first_late_output), pops::Real(31));
  EXPECT_EQ(pops::reduce_max_local(first_late_output), pops::Real(31));
  EXPECT_EQ(pops::reduce_min_local(malformed_output), pops::Real(37));
  EXPECT_EQ(pops::reduce_max_local(malformed_output), pops::Real(37));
  const auto& published_after_failure = system.prepared_amr_level_evaluation(0);
  EXPECT_EQ(published_after_failure.point.stage, published_point.stage);
  EXPECT_EQ(published_after_failure.point.tick, published_point.tick);
  EXPECT_EQ(published_after_failure.topology_epoch, published_epoch);
  EXPECT_EQ(published_after_failure.materialization_generation, published_generation);
  EXPECT_NE(system.prepared_amr_level_evaluation_if_present(0), nullptr);

  // Exercise the same late-failure path while a real hierarchy attempt owns the active flux
  // registry.  The successful two-block round makes reflux bases visible to the attempt; the
  // following later failure must leave its sentinels and the successful published evaluation intact.
  pops::MultiFab<Dim> active_first = context->rhs_scratch_like(stage);
  pops::MultiFab<Dim> active_second = context->rhs_scratch_like(peer_stage);
  pops::MultiFab<Dim> active_failure_first = context->rhs_scratch_like(stage);
  pops::MultiFab<Dim> active_failure_second = context->rhs_scratch_like(malformed_peer);
  context->install([](double) {}, context);
  system.set_program_block_map({0, 1});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.generated-amr/rhs-group-atomic@1", std::vector<FluxBudget>(2, FluxBudget{1, 1}), 0, 0);
  context->advance_hierarchy(0.01, [&](double) {
    context->rhs_group(
        30, {{0, &stage, &active_first, 31, 0}, {1, &peer_stage, &active_second, 32, 1}});
    active_failure_first.set_val(pops::Real(41));
    active_failure_second.set_val(pops::Real(43));
    EXPECT_THROW(context->rhs_group(33, {{0, &stage, &active_failure_first, 34, 0},
                                         {1, &malformed_peer, &active_failure_second, 35, 0}}),
                 std::exception);
    EXPECT_EQ(pops::reduce_min_local(active_failure_first), pops::Real(41));
    EXPECT_EQ(pops::reduce_max_local(active_failure_first), pops::Real(41));
    EXPECT_EQ(pops::reduce_min_local(active_failure_second), pops::Real(43));
    EXPECT_EQ(pops::reduce_max_local(active_failure_second), pops::Real(43));
  });
  EXPECT_EQ(system.prepared_amr_level_evaluation(0).point.stage, 31);
  EXPECT_NE(system.prepared_amr_level_evaluation_if_present(0), nullptr);

  pops::MultiFab<Dim> first_output = context->rhs_scratch_like(stage);
  pops::MultiFab<Dim> second_output = context->rhs_scratch_like(stage);
  first_output.set_val(pops::Real(11));
  second_output.set_val(pops::Real(13));
  // The second request maps to the already selected runtime block. The complete group must reject
  // this before evaluating or publishing the first request's residual/flux metadata.
  EXPECT_THROW(context->rhs_group(
                   24, {{0, &stage, &first_output, 25, 0}, {0, &stage, &second_output, 26, 1}}),
               std::exception);
  EXPECT_EQ(pops::reduce_min_local(first_output), pops::Real(11));
  EXPECT_EQ(pops::reduce_max_local(first_output), pops::Real(11));
  EXPECT_EQ(pops::reduce_min_local(second_output), pops::Real(13));
  EXPECT_EQ(pops::reduce_max_local(second_output), pops::Real(13));
  EXPECT_EQ(system.prepared_amr_level_evaluation(0).point.stage, 31);

  if (pops::n_ranks() > 1) {
    pops::MultiFab<Dim> divergent_output = context->rhs_scratch_like(stage);
    divergent_output.set_val(pops::Real(17));
    // A rank-local rate identity is otherwise valid, but the collective request contract must
    // reject the divergent group before either rank evaluates or publishes its output.
    const int divergent_rate = pops::my_rank() == 0 ? 28 : 29;
    EXPECT_THROW(context->rhs_group(27, {{0, &stage, &divergent_output, divergent_rate, 0}}),
                 std::exception);
    EXPECT_EQ(pops::reduce_min_local(divergent_output), pops::Real(17));
    EXPECT_EQ(pops::reduce_max_local(divergent_output), pops::Real(17));
    EXPECT_EQ(system.prepared_amr_level_evaluation(0).point.stage, 31);
  }

  // In a fresh active attempt, consume the peer block's one authenticated basis through the
  // singular path. Both grouped detached evaluations then succeed; block zero's grouped basis
  // succeeds, while block one's grouped basis exceeds its bound during flux-basis preparation.
  // The detached registry must not be published, so block zero keeps no published evaluation and
  // neither grouped output may leave its sentinel value.
  pops::MultiFab<Dim> seeded_peer_flux = context->rhs_scratch_like(peer_stage);
  pops::MultiFab<Dim> flux_failure_first = context->rhs_scratch_like(stage);
  pops::MultiFab<Dim> flux_failure_second = context->rhs_scratch_like(peer_stage);
  pops::MultiFab<Dim> flux_registry_recovery = context->rhs_scratch_like(stage);
  context->advance_hierarchy(0.01, [&](double) {
    context->rhs_into(1, peer_stage, seeded_peer_flux, 36);
    flux_failure_first.set_val(pops::Real(47));
    flux_failure_second.set_val(pops::Real(53));
    EXPECT_THROW(context->rhs_group(37, {{0, &stage, &flux_failure_first, 38, 0},
                                         {1, &peer_stage, &flux_failure_second, 39, 1}}),
                 std::exception);
    EXPECT_EQ(pops::reduce_min_local(flux_failure_first), pops::Real(47));
    EXPECT_EQ(pops::reduce_max_local(flux_failure_first), pops::Real(47));
    EXPECT_EQ(pops::reduce_min_local(flux_failure_second), pops::Real(53));
    EXPECT_EQ(pops::reduce_max_local(flux_failure_second), pops::Real(53));
    EXPECT_EQ(system.prepared_amr_level_evaluation_if_present(0), nullptr);
    // If the failed detached registry had escaped, block zero would already have consumed its
    // bound of one and this recovery basis would be refused before it could publish.
    context->rhs_group(40, {{0, &stage, &flux_registry_recovery, 41, 0}});
    EXPECT_EQ(pops::difference_sum_sq_all_local(full, flux_registry_recovery), pops::Real(0));
    EXPECT_EQ(system.prepared_amr_level_evaluation(0).point.stage, 41);
  });
  EXPECT_EQ(system.prepared_amr_level_evaluation(0).point.stage, 41);
}

TEST(GeneratedAmrSystemBlock, RegistersOnlyExactRankedNullspaceProviders) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/nullspace-registry");
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
  pops::test::install_amr_runtime_authority(system,
                                            "tests.generated-amr/default-field-publication");
  system.set_poisson();
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  context->begin_step(0.01);
  pops::SolveOutcome outcome = context->solve_default_field_on_coarse_level();
  if (!outcome.report().solved_value_available()) {
    const pops::SolveReport rejected = outcome.consume(pops::SolveConsumption::kFailRun);
    FAIL() << rejected.reason;
  }

  const pops::SolveReport accepted = outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
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
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/named-field-stage");
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.level-local", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan("field/tracer", "test.named-field-plan", "test.named-field",
                               "test.aux-owner", "tracer", "phi",
                               {{"test.aux-owner", "field", "phi", "potential"}}, 1, {"test.rhs"},
                               {"tracer"}, {"charge"}, {1.0}, "geometric_mg", hierarchy,
                               pops::geometric_mg_amr_field_solver_options(
                                   pops::GeometricMgOptions{}, pops::CompositeFacOptions{}));
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  const auto output_key = install_field_output(system, "test.aux-owner", "phi");
  system.register_elliptic_field("tracer", "phi", {output_key}, 1);
  system.set_block_elliptic_field(
      "tracer", "phi", "test.generated-amr.named-field.rhs.zero@1",
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
  EXPECT_EQ(pops::reduce_max_local(context->state(0), 0), pops::Real(1));
  (void)outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_EQ(pops::reduce_max_local(context->state(0), 0), pops::Real(1));
  EXPECT_EQ(system.auxiliary_component(output_key).size(), cell_count(config.shape));
  EXPECT_EQ(system.field_provider_levels("field/tracer"), 1);
}

TEST(GeneratedAmrSystemBlock,
     DynamicFieldBoundaryConsumesExactStageAndPublishesOnlyAfterNewtonAcceptance) {
  pops::comm_init();
  constexpr int Dim = pops::kNativeDimension;
  RuntimeFieldBoundaryProbe<Dim>::reset();
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/dynamic-field-boundary");
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan("field/tracer", "test.dynamic-field-plan", "test.dynamic-field",
                               "test.aux-owner", "tracer", "phi",
                               {{"test.aux-owner", "field", "phi", "potential"}}, 1, {"test.rhs"},
                               {"tracer"}, {"charge"}, {1.0}, "geometric_mg", hierarchy,
                               pops::geometric_mg_amr_field_solver_options(
                                   pops::GeometricMgOptions{}, pops::CompositeFacOptions{}));
  system.set_field_reaction("field/tracer", 50.0);
  system.set_field_boundary_dependencies("field/tracer", {"tracer"}, {0}, {}, {}, {});
  system.set_field_boundary_parameters("field/tracer", {0.25});
  system.set_field_boundary_kernel("field/tracer", RuntimeFieldBoundaryProbe<Dim>::kernel());
  system.set_field_newton_plan("field/tracer", 1.0e-9, 4, 1.0e-10, 80, 20, 1.0e-4, 1.0 / 1024.0);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  const auto output_key = install_field_output(system, "test.aux-owner", "phi");
  system.register_elliptic_field("tracer", "phi", {output_key}, 1);
  system.set_block_elliptic_field(
      "tracer", "phi", "test.generated-amr.dynamic-field.rhs.one@1",
      [](const pops::MultiFab<Dim>&, pops::MultiFab<Dim>& rhs) { rhs.set_val(pops::Real(1)); });
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();
  system.set_program_block_map({0});

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

  RuntimeFieldBoundaryProbe<Dim>::force_failure = false;
  pops::SolveOutcome outcome =
      context->solve_fields_from_state_at(evaluation, "field/tracer", 0, stage_state);
  ASSERT_TRUE(outcome.report().solved_value_available()) << outcome.report().reason;
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
}

TEST(GeneratedAmrSystemBlock, CompositeFieldInstallsCoverageAwareNullspaceOnEveryLiveLevel) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    config.periodicity[axis] = true;
  }
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system,
                                            "tests.generated-amr/composite-field-nullspace");
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
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/unsynchronized-hierarchy");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();
  auto context = pops::test::install_forward_euler_program_context(system, false);
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
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/level-history");
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
  // Match the generated LinearInterpolation lowering: an authenticated retained history slot
  // determines the owner of the persistent output scratch before interpolation mutates it.
  pops::MultiFab<Dim>& retained = context->history("tracer.rate", 1, 0);
  pops::MultiFab<Dim>& interpolated = context->scratch_state(940001, 0, retained);
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
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/history-regrid-refusal");
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

TEST(GeneratedAmrSystemBlock, PreparedHistoryRemapAcceptsPublishedReplacement) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  config.regrid_every = 1;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/history-remap-acceptance");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 0.5}},
                                               "tests.generated-amr/history-remap-tagging@1");
  ASSERT_NE(system.engine(), nullptr);
  ASSERT_EQ(system.engine()->hierarchy().num_levels(), 2u);

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("clock.macro");
  context->declare_clock_relation("clock.macro", "clock.level.1", 2);
  context->install([](double) {}, context);
  system.set_program_block_map({0});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.generated-amr/history-remap@1", std::vector<FluxBudget>(1, FluxBudget{1, 1}), 0, 0);
  context->for_each_program_resource_level([&](int) {
    context->register_history("tracer.rate", 1, 1, 0, "tracer.U", "cell.conservative",
                              "clock.macro", "dense.linear");
  });
  for (const double dt : {0.1, 0.2, 0.3}) {
    context->begin_step(dt);
    context->for_each_program_resource_level([&](int) {
      pops::MultiFab<Dim> sample = context->scratch_state_like(context->state(0));
      sample.set_val(pops::Real(dt));
      context->store_history("tracer.rate", sample, 0);
    });
    context->for_each_program_resource_level(
        [&](int) { context->rotate_histories("clock.macro"); });
  }
  ASSERT_TRUE(system.regrid_from_prepared_tagging(0));
  EXPECT_EQ(system.history_names(), (std::vector<std::string>{"tracer.rate"}));
  for (const int level : {0, 1}) {
    EXPECT_TRUE(system.history_initialized("tracer.rate", level));
    EXPECT_EQ(system.history_fill_count("tracer.rate", level), 2);
    EXPECT_GT(system.history_slot_dt("tracer.rate", level, 0), 0.0);
    EXPECT_GT(system.history_slot_dt("tracer.rate", level, 1), 0.0);
  }
}

TEST(GeneratedAmrSystemBlock, NoopPreparedRegridPreservesInitializedHistory) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/history-remap-noop");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 2.0}},
                                               "tests.generated-amr/history-remap-noop-tagging@1");
  ASSERT_NE(system.engine(), nullptr);
  ASSERT_EQ(system.engine()->hierarchy().num_levels(), 1u);

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("clock.macro");
  context->install([](double) {}, context);
  system.set_program_block_map({0});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.generated-amr/history-remap-noop@1", std::vector<FluxBudget>(1, FluxBudget{1, 1}), 0,
      0);
  context->register_history("tracer.rate", 1, 1, 0, "tracer.U", "cell.conservative", "clock.macro",
                            "dense.linear");
  for (const double dt : {0.1, 0.2, 0.3}) {
    context->begin_step(dt);
    pops::MultiFab<Dim> sample = context->scratch_state_like(context->state(0));
    sample.set_val(pops::Real(dt));
    context->store_history("tracer.rate", sample, 0);
    context->rotate_histories("clock.macro");
  }
  const std::uint64_t topology_before = system.engine()->topology_epoch();
  const std::uint64_t materialization_before = system.engine()->materialization_generation();
  const auto names_before = system.history_names();
  const int fill_before = system.history_fill_count("tracer.rate", 0);
  const double dt0_before = system.history_slot_dt("tracer.rate", 0, 0);
  const double dt1_before = system.history_slot_dt("tracer.rate", 0, 1);
  const auto slot0_before = system.history_global("tracer.rate", 0, 0);
  const auto slot1_before = system.history_global("tracer.rate", 0, 1);

  EXPECT_FALSE(system.regrid_from_prepared_tagging(0));
  EXPECT_EQ(system.engine()->hierarchy().num_levels(), 1u);
  EXPECT_EQ(system.engine()->topology_epoch(), topology_before);
  EXPECT_EQ(system.engine()->materialization_generation(), materialization_before);
  EXPECT_EQ(system.history_names(), names_before);
  EXPECT_TRUE(system.history_initialized("tracer.rate", 0));
  EXPECT_EQ(system.history_fill_count("tracer.rate", 0), fill_before);
  EXPECT_EQ(system.history_slot_dt("tracer.rate", 0, 0), dt0_before);
  EXPECT_EQ(system.history_slot_dt("tracer.rate", 0, 1), dt1_before);
  EXPECT_EQ(system.history_global("tracer.rate", 0, 0), slot0_before);
  EXPECT_EQ(system.history_global("tracer.rate", 0, 1), slot1_before);
}

TEST(GeneratedAmrSystemBlock, CflUsesFinestExactGeometryAndPreparedModelSpeed) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/cfl-finest-geometry");
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

TEST(GeneratedAmrSystemBlock, DiffusiveCflUsesFinestParabolicGeometryAndReportsFrequency) {
  constexpr int Dim = pops::kNativeDimension;
  constexpr pops::Real diffusivity = pops::Real(0.125);
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/diffusive-cfl");
  system.install_block_state_route("diffusive", "state/diffusive");
  pops::add_compiled_model<Dim>(system, "diffusive", diffusive_advection_model<Dim>(diffusivity),
                                "minmod", "rusanov", "conservative", "explicit", 1.4, 2, 3);
  system.set_conservative_state("diffusive", std::vector<double>(cell_count(config.shape), 1.0));
  publish_centered_fine_level(system);
  system.install_program_step([](double) {});

  pops::Real inverse_dt = pops::Real(0);
  for (int axis = 0; axis < Dim; ++axis)
    inverse_dt += pops::Real(2) * diffusivity * pops::Real(16 * 16);
  const double expected = 0.4 * 2.0 / (3.0 * static_cast<double>(inverse_dt));
  EXPECT_NEAR(system.step_cfl(0.4, 1.0e-12), expected, 1.0e-12);
  EXPECT_EQ(system.last_dt_bound(), "parabolic_frequency:diffusive");
}

TEST(GeneratedAmrSystemBlock, CflAuthenticatesRequestsAndBoundOrderBeforeCallbacks) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/cfl-request-consensus");
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
