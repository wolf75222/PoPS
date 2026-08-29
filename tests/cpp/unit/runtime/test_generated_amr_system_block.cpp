#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"
#include "program_v5_fixture.hpp"
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/core/foundation/native_dimension.hpp>

#include <algorithm>
#include <array>
#include <cmath>
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

namespace pops::runtime::program::detail {

template <int Dim>
struct AmrProgramHistoryRemapCollectiveTestAccess {
  using context_type = ProgramExecutionServices<Dim>;
  using backend_type = typename context_type::amr_backend;
  using snapshot_type = typename backend_type::AcceptedContextSnapshot;

  static std::pair<std::uint64_t, std::uint64_t> resource_authority(const context_type& context) {
    const auto& backend = context.amr_test_backend_();
    return {backend.resource_epoch_, backend.resource_generation_};
  }

  static std::string history_key(const context_type& context, std::string_view name, int level) {
    return context.amr_test_backend_().history_key_(std::string(name), level);
  }

  static const snapshot_type& concrete(const AcceptedProgramExecutionServicesSnapshot& snapshot) {
    const auto* typed = dynamic_cast<const snapshot_type*>(&snapshot);
    if (typed == nullptr)
      throw std::logic_error("history-remap test received a foreign accepted snapshot");
    return *typed;
  }

  static bool has_history(const AcceptedProgramExecutionServicesSnapshot& snapshot,
                          std::string_view key) {
    return concrete(snapshot).history_levels_.contains(std::string(key));
  }

  static bool has_pending(const AcceptedProgramExecutionServicesSnapshot& snapshot,
                          std::string_view key) {
    return concrete(snapshot).pending_history_remaps_.contains(std::string(key));
  }

  static bool live_has_history(const context_type& context, std::string_view key) {
    return context.amr_test_backend_().history_levels_.contains(std::string(key));
  }

  static bool live_has_pending(const context_type& context, std::string_view key) {
    return context.amr_test_backend_().pending_history_remaps_.contains(std::string(key));
  }

  static std::size_t flux_depth(const AcceptedProgramExecutionServicesSnapshot& snapshot,
                                std::string_view key) {
    return concrete(snapshot).history_flux_expressions_.at(std::string(key)).size();
  }

  static std::uint64_t revision(const AcceptedProgramExecutionServicesSnapshot& snapshot) {
    return concrete(snapshot).accepted_state_revision_;
  }

  static bool is_detached(const AcceptedProgramExecutionServicesSnapshot& snapshot) {
    return concrete(snapshot).owner_ == nullptr;
  }

  static bool has_pending_history(const context_type& context, std::string_view name, int level) {
    const auto& backend = context.amr_test_backend_();
    return backend.pending_history_remaps_.contains(backend.history_key_(std::string(name), level));
  }

  static const auto& active_expression(const context_type& context,
                                       const typename context_type::field_type& field) {
    return context.amr_test_backend_().active_flux_expressions_.at(&field);
  }

  static const typename context_type::field_type& history_slot(const context_type& context,
                                                               std::string_view name, int level,
                                                               int slot) {
    const auto key = context.amr_test_backend_().history_key_(std::string(name), level);
    return context.runtime_state().hist_.histories.at(key).at(static_cast<std::size_t>(slot));
  }
};

}  // namespace pops::runtime::program::detail

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

/// A non-null field source makes the detached regrid witness scientific: the forward solve must
/// retain/rematerialize a real potential and publish it through its FieldOutput provider, rather
/// than merely carry a warm-start through a homogeneous test solve.
template <int Dim>
struct ForcedPoissonAdvectionModel : AdvectionModel<Dim> {
  using State = typename AdvectionModel<Dim>::State;

  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(1); }
};

template <int Dim>
ForcedPoissonAdvectionModel<Dim> forced_poisson_advection_model() {
  const AdvectionModel<Dim> base = advection_model<Dim>();
  ForcedPoissonAdvectionModel<Dim> result;
  result.law = base.law;
  return result;
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
std::vector<pops::test::program_v5::CallbackProgramResource> dense_resources(
    const pops::AmrSystem<Dim>& system,
    const std::vector<pops::test::program_v5::CallbackProgramResource::Kind>& kinds) {
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto program_blocks = system.program_block_map();
  if (program_blocks.empty() || kinds.empty())
    throw std::logic_error("generated AMR fixture requires an exact Program resource table");
  const auto accepted_runtime = system.accepted_amr_runtime();
  if (!accepted_runtime)
    throw std::logic_error("generated AMR fixture resource has no accepted runtime");
  const int level_count = static_cast<int>(accepted_runtime->hierarchy().num_levels());
  std::vector<Resource> resources;
  resources.reserve(static_cast<std::size_t>(level_count) * program_blocks.size() * kinds.size());
  for (int level = 0; level < level_count; ++level) {
    for (std::size_t program_block = 0; program_block < program_blocks.size(); ++program_block) {
      auto state_view = system.prepared_amr_block_state(program_blocks[program_block], level);
      if (!state_view)
        throw std::logic_error("generated AMR fixture resource has no accepted block state");
      const auto& state = *state_view;
      for (const auto kind : kinds)
        resources.push_back({kind, resources.size(), 0, static_cast<int>(program_block), level,
                             static_cast<std::uint32_t>(state.ncomp()),
                             static_cast<std::uint32_t>(state.ghosts()[0])});
    }
  }
  return resources;
}

template <int Dim>
void expect_history_preserved_on_current_coverage(const pops::AmrSystem<Dim>& system,
                                                  const std::vector<double>& before,
                                                  const std::vector<double>& after, int level) {
  const auto accepted_runtime = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_runtime);
  const auto& layout = accepted_runtime->hierarchy().layout(static_cast<std::size_t>(level));
  const pops::Box<Dim>& domain = layout.domain();
  ASSERT_EQ(after.size(), before.size());
  std::vector<bool> covered(after.size(), false);
  for (const pops::Box<Dim>& patch : layout.patches().boxes()) {
    const std::size_t cells = static_cast<std::size_t>(patch.numPts());
    for (std::size_t linear = 0; linear < cells; ++linear) {
      pops::Index<Dim> index{};
      std::size_t remaining = linear;
      std::size_t offset = 0;
      std::size_t stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        const std::size_t extent = static_cast<std::size_t>(patch.length(axis));
        index[axis] = patch.lo[axis] + static_cast<int>(remaining % extent);
        remaining /= extent;
        offset += static_cast<std::size_t>(index[axis] - domain.lo[axis]) * stride;
        stride *= static_cast<std::size_t>(domain.length(axis));
      }
      ASSERT_LT(offset, covered.size());
      covered[offset] = true;
    }
  }
  ASSERT_NE(std::find(covered.begin(), covered.end(), true), covered.end());
  for (std::size_t cell = 0; cell < covered.size(); ++cell)
    if (covered[cell])
      EXPECT_EQ(after[cell], before[cell]) << "covered child history cell=" << cell;
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
  auto* engine = pops::test::AmrSystemTestAccess<Dim>::engine(system);
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
    config.shape[axis] = 16;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/facade-root-level");
  ASSERT_EQ(system.n_blocks(), 0);

  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>(), "minmod", "rusanov",
                                "conservative", "explicit", 1.4, 2, 1);
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));

  ASSERT_EQ(system.n_blocks(), 1);
  ASSERT_EQ(system.n_levels(), 1);
  const auto accepted_runtime = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_runtime);
  const auto& state = accepted_runtime->hierarchy().state(0);
  pops::MultiFab<Dim> poisson_rhs(state.layout(), state.distribution(), state.local_rank(), 1,
                                  state.ghosts());
  poisson_rhs.set_val(pops::Real(0));
  system.add_prepared_amr_poisson_rhs(0, poisson_rhs);
  EXPECT_EQ(pops::reduce_max_local(poisson_rhs), pops::Real(0));
  const auto evaluation = system.evaluate_prepared_amr_level(point<Dim>(0));
  ASSERT_TRUE(evaluation);
  EXPECT_EQ(evaluation->point, point<Dim>(0));
  EXPECT_EQ(evaluation->spatial_contract, accepted_runtime->spatial_contract());
  EXPECT_EQ(evaluation->topology_epoch, accepted_runtime->topology_epoch());
  EXPECT_EQ(evaluation->materialization_generation, accepted_runtime->materialization_generation());
  EXPECT_EQ(evaluation->residual.ncomp(), 1);
  EXPECT_EQ(evaluation->residual.layout(), accepted_runtime->hierarchy().state(0).layout());
  EXPECT_EQ(evaluation->integrated_face_fluxes.size(),
            accepted_runtime->hierarchy().state(0).local_size());
  const auto prepared = system.prepared_amr_level_evaluation(0);
  ASSERT_TRUE(prepared);
  EXPECT_EQ(prepared.get(), evaluation.get());
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
  const auto accepted_runtime = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_runtime);
  const auto fine = system.evaluate_prepared_amr_level(point<Dim>(1));
  ASSERT_TRUE(fine);
  EXPECT_EQ(fine->point, point<Dim>(1));
  EXPECT_EQ(fine->spatial_contract, accepted_runtime->spatial_contract());
  EXPECT_EQ(fine->topology_epoch, accepted_runtime->topology_epoch());
  EXPECT_EQ(fine->materialization_generation, accepted_runtime->materialization_generation());
  EXPECT_EQ(fine->residual.layout(), accepted_runtime->hierarchy().state(1).layout());
  EXPECT_EQ(fine->integrated_face_fluxes.size(),
            accepted_runtime->hierarchy().state(1).local_size());
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

  const auto root_evaluation = system.evaluate_prepared_amr_level(point<Dim>(0));
  ASSERT_TRUE(root_evaluation);
  const pops::Real residual_magnitude =
      std::max(Kokkos::abs(pops::reduce_min_local(root_evaluation->residual, 0)),
               Kokkos::abs(pops::reduce_max_local(root_evaluation->residual, 0)));
  EXPECT_GT(static_cast<double>(residual_magnitude), 0.0);
  EXPECT_NEAR(static_cast<double>(pops::reduce_sum_local(root_evaluation->residual, 0)), 0.0,
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

TEST(GeneratedAmrSystemBlock, ProgramExecutionServicesOwnsOneExactHierarchyTensorAuthority) {
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
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state, Resource::Kind::rhs});
  struct Result {
    bool direct = false;
    bool same_solution = false;
    bool source_is_solution = false;
    bool undeclared_rejected = false;
    bool solved = false;
  } result;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/hierarchy-tensor-authority@1", "test-clock", resources, {},
      [&result](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        const auto slots = pops::runtime::program::tensor_elliptic_detail::assembly_slots<Dim>();
        const auto options = pops::runtime::program::tensor_elliptic_detail::default_options();
        context.configure_hierarchy_tensor_solver(
            0, 1,
            std::string(pops::runtime::program::tensor_elliptic_detail::kCompositeTensorProvider),
            "test.generated-amr.tensor-plan",
            std::string(
                pops::runtime::program::tensor_elliptic_detail::kScalarTensorEllipticContract),
            slots, "pops.tensor-elliptic.solution", options);
        result.direct = !context.uses_prepared_krylov_fallback();
        pops::MultiFab<Dim>* first_solution = &context.hierarchy_solution();
        context.configure_hierarchy_tensor_solver(
            0, 1,
            std::string(pops::runtime::program::tensor_elliptic_detail::kCompositeTensorProvider),
            "test.generated-amr.tensor-plan",
            std::string(
                pops::runtime::program::tensor_elliptic_detail::kScalarTensorEllipticContract),
            slots, "pops.tensor-elliptic.solution", options);
        result.same_solution = &context.hierarchy_solution() == first_solution;

        context.begin_step(dt);
        context.for_each_program_resource_level([&](int level) {
          const auto state_slot = static_cast<pops::runtime::program::ProgramCacheSlot>(level * 2);
          const auto rhs_slot = state_slot + 1;
          auto& fallback = context.rhs_scratch(rhs_slot, 0, context.state(0));
          for (int row = 0; row < Dim; ++row)
            for (int column = 0; column < Dim; ++column)
              context
                  .assembly_target(
                      fallback,
                      pops::runtime::program::tensor_elliptic_detail::coefficient_slot(row, column))
                  .set_val(row == column ? pops::Real(1) : pops::Real(0));
          context.assembly_target(fallback, "pops.tensor-elliptic.rhs").set_val(pops::Real(0));
          context.assembly_target(fallback, "pops.tensor-elliptic.flux").set_val(pops::Real(0));
          context.stage_linear_initial_guess();
          result.source_is_solution =
              &context.assembly_source(fallback, "pops.tensor-elliptic.solution") ==
              &context.hierarchy_solution();
          try {
            (void)context.assembly_target(fallback, "undeclared");
          } catch (const std::invalid_argument&) {
            result.undeclared_rejected = true;
          }
        });
        auto outcome = context.solve_hierarchy_tensor(0, 1, pops::Real(1.0e-8), pops::Real(0), 8);
        if (outcome.report().solved_value_available())
          result.solved = outcome.consume(pops::SolveConsumption::kAccept).solved();
      });
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_TRUE(result.direct);
  EXPECT_TRUE(result.same_solution);
  EXPECT_TRUE(result.source_is_solution);
  EXPECT_TRUE(result.undeclared_rejected);
  EXPECT_TRUE(result.solved);
}

TEST(GeneratedAmrSystemBlock, ProgramExecutionServicesEvaluatesExactStageStateWithoutPublishingIt) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/exact-stage-state");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.set_program_block_map({0});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state, Resource::Kind::rhs});
  struct Result {
    pops::Real stage_min = pops::Real(-1);
    pops::Real accepted_min = pops::Real(-1);
    bool foreign_rejected = false;
  } result;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/exact-stage-state@1", "test-clock", resources, {},
      [&result](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
        context.advance_hierarchy(dt, [&context, &result](double) {
          auto& stage = context.scratch_state(0, 0, context.state(0));
          stage.set_val(pops::Real(2));
          auto& residual = context.rhs_scratch(1, 0, stage);
          context.rhs_into(0, stage, residual, 7);
          result.stage_min = pops::reduce_min_local(stage);
          result.accepted_min = pops::reduce_max_local(context.state(0));

          pops::MultiFab<Dim> foreign(stage.layout(), stage.distribution(), stage.local_rank(), 2,
                                      stage.ghosts());
          foreign.set_val(pops::Real(9));
          pops::MultiFab<Dim> foreign_residual = context.rhs_scratch_like(foreign);
          try {
            context.rhs_into(0, foreign, foreign_residual, 8);
          } catch (const std::invalid_argument&) {
            result.foreign_rejected = true;
          }
          result.accepted_min =
              std::min(result.accepted_min, pops::reduce_max_local(context.state(0)));
        });
      });
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_EQ(result.stage_min, pops::Real(2));
  EXPECT_EQ(result.accepted_min, pops::Real(1));
  EXPECT_TRUE(result.foreign_rejected);
  const auto evaluation = system.prepared_amr_level_evaluation(0);
  ASSERT_TRUE(evaluation);
  EXPECT_EQ(evaluation->point.stage, 7);
  const auto accepted_runtime = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_runtime);
  EXPECT_EQ(pops::reduce_max_local(accepted_runtime->hierarchy().state(0)), pops::Real(1));
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
  system.set_program_block_map({0, 1});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  std::vector<Resource> resources;
  for (int level = 0; level < system.n_levels(); ++level) {
    for (int block = 0; block < system.n_blocks(); ++block) {
      const auto state = system.prepared_amr_block_state(block, level);
      ASSERT_TRUE(state);
      const auto base = resources.size();
      resources.push_back({Resource::Kind::state, base, 0, block, level,
                           static_cast<std::uint32_t>(state->ncomp()),
                           static_cast<std::uint32_t>(state->ghosts()[0])});
      for (int slot = 0; slot < 7; ++slot)
        resources.push_back({Resource::Kind::rhs, resources.size(), 0, block, level,
                             static_cast<std::uint32_t>(state->ncomp()),
                             static_cast<std::uint32_t>(state->ghosts()[0])});
    }
  }
  struct Result {
    int dispatches = 0;
    bool full_equal = false;
    bool flux_equal = false;
    bool cross_equal = false;
    bool malformed_rejected = false;
    bool malformed_sentinels = false;
    bool active_rejected = false;
    bool active_sentinels = false;
    bool duplicate_rejected = false;
    bool duplicate_sentinels = false;
    bool divergent_rejected = false;
    bool divergent_sentinel = false;
    bool fresh_rejected = false;
    bool fresh_sentinels = false;
    bool recovery_equal = false;
  } result;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/rhs-group-atomic@1", "test-clock", resources, {},
      [&result](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        ++result.dispatches;
        context.begin_step(dt);
        const auto block_stride = 8;
        const auto block0 = static_cast<pops::runtime::program::ProgramCacheSlot>(
            (context.level() * context.n_blocks()) * block_stride);
        const auto block1 = block0 + block_stride;
        auto& stage = context.scratch_state(block0, 0, context.state(0));
        stage.set_val(pops::Real(2));
        auto& full = context.rhs_scratch(block0 + 1, 0, stage);
        auto& grouped_full = context.rhs_scratch(block0 + 2, 0, stage);
        context.rhs_into(0, stage, full, 11);
        context.rhs_group(12, {{0, &stage, &grouped_full, 13, 0}});
        result.full_equal = pops::difference_sum_sq_all_local(full, grouped_full) == pops::Real(0);

        auto& flux = context.rhs_scratch(block0 + 3, 0, stage);
        auto& grouped_flux = context.rhs_scratch(block0 + 4, 0, stage);
        context.neg_div_flux_default_into(0, stage, flux, 14);
        context.rhs_group(15, {{0, &stage, &grouped_flux, 16, 1}});
        result.flux_equal = pops::difference_sum_sq_all_local(flux, grouped_flux) == pops::Real(0);

        auto& peer_stage = context.scratch_state(block1, 0, context.state(1));
        peer_stage.set_val(pops::Real(4));
        auto& peer_flux = context.rhs_scratch(block1 + 1, 0, peer_stage);
        auto& grouped_first = context.rhs_scratch(block0 + 5, 0, stage);
        auto& grouped_second = context.rhs_scratch(block1 + 2, 0, peer_stage);
        context.neg_div_flux_default_into(1, peer_stage, peer_flux, 17);
        context.rhs_group(
            18, {{0, &stage, &grouped_first, 19, 0}, {1, &peer_stage, &grouped_second, 20, 1}});
        result.cross_equal =
            pops::difference_sum_sq_all_local(full, grouped_first) == pops::Real(0) &&
            pops::difference_sum_sq_all_local(peer_flux, grouped_second) == pops::Real(0);

        if (result.dispatches == 1) {
          pops::MultiFab<Dim> malformed_peer(peer_stage.layout(), peer_stage.distribution(),
                                             peer_stage.local_rank(), 2, peer_stage.ghosts());
          malformed_peer.set_val(pops::Real(5));
          auto malformed_output = context.rhs_scratch_like(malformed_peer);
          auto first_late_output = context.rhs_scratch_like(stage);
          first_late_output.set_val(pops::Real(31));
          malformed_output.set_val(pops::Real(37));
          try {
            context.rhs_group(21, {{0, &stage, &first_late_output, 22, 0},
                                   {1, &malformed_peer, &malformed_output, 23, 0}});
          } catch (const std::exception&) {
            result.malformed_rejected = true;
          }
          result.malformed_sentinels =
              pops::reduce_min_local(first_late_output) == pops::Real(31) &&
              pops::reduce_max_local(first_late_output) == pops::Real(31) &&
              pops::reduce_min_local(malformed_output) == pops::Real(37) &&
              pops::reduce_max_local(malformed_output) == pops::Real(37);
          return;
        }

        if (result.dispatches == 2) {
          auto& active_first = context.rhs_scratch(block0 + 6, 0, stage);
          auto& active_second = context.rhs_scratch(block1 + 3, 0, peer_stage);
          auto& active_failure_first = context.rhs_scratch(block0 + 7, 0, stage);
          auto& active_failure_second = context.rhs_scratch(block1 + 4, 0, peer_stage);
          context.advance_hierarchy(0.01, [&](double) {
            context.rhs_group(
                30, {{0, &stage, &active_first, 31, 0}, {1, &peer_stage, &active_second, 32, 1}});
            active_failure_first.set_val(pops::Real(41));
            active_failure_second.set_val(pops::Real(43));
            try {
              context.rhs_group(33, {{0, &stage, &active_failure_first, 34, 0},
                                     {1, &peer_stage, &active_failure_second, 35, 0}});
            } catch (const std::exception&) {
              result.active_rejected = true;
            }
            result.active_sentinels =
                pops::reduce_min_local(active_failure_first) == pops::Real(41) &&
                pops::reduce_max_local(active_failure_first) == pops::Real(41) &&
                pops::reduce_min_local(active_failure_second) == pops::Real(43) &&
                pops::reduce_max_local(active_failure_second) == pops::Real(43);
          });
          return;
        }

        if (result.dispatches == 3) {
          auto& first_output = context.rhs_scratch(block0 + 1, 0, stage);
          auto& second_output = context.rhs_scratch(block0 + 2, 0, stage);
          first_output.set_val(pops::Real(11));
          second_output.set_val(pops::Real(13));
          try {
            context.rhs_group(
                24, {{0, &stage, &first_output, 25, 0}, {0, &stage, &second_output, 26, 1}});
          } catch (const std::exception&) {
            result.duplicate_rejected = true;
          }
          result.duplicate_sentinels = pops::reduce_min_local(first_output) == pops::Real(11) &&
                                       pops::reduce_max_local(first_output) == pops::Real(11) &&
                                       pops::reduce_min_local(second_output) == pops::Real(13) &&
                                       pops::reduce_max_local(second_output) == pops::Real(13);
          if (pops::n_ranks() > 1) {
            auto& divergent_output = context.rhs_scratch(block0 + 3, 0, stage);
            divergent_output.set_val(pops::Real(17));
            const int divergent_rate = pops::my_rank() == 0 ? 28 : 29;
            try {
              context.rhs_group(27, {{0, &stage, &divergent_output, divergent_rate, 0}});
            } catch (const std::exception&) {
              result.divergent_rejected = true;
            }
            result.divergent_sentinel =
                pops::reduce_min_local(divergent_output) == pops::Real(17) &&
                pops::reduce_max_local(divergent_output) == pops::Real(17);
          }
          return;
        }

        auto& seeded_peer_flux = context.rhs_scratch(block1 + 5, 0, peer_stage);
        auto flux_failure_first = context.rhs_scratch_like(stage);
        auto flux_failure_second = context.rhs_scratch_like(peer_stage);
        auto& flux_registry_recovery = context.rhs_scratch(block0 + 7, 0, stage);
        context.advance_hierarchy(0.01, [&](double) {
          context.rhs_into(1, peer_stage, seeded_peer_flux, 36);
          flux_failure_first.set_val(pops::Real(47));
          flux_failure_second.set_val(pops::Real(53));
          try {
            context.rhs_group(37, {{0, &stage, &flux_failure_first, 38, 0},
                                   {1, &peer_stage, &flux_failure_second, 39, 1}});
          } catch (const std::exception&) {
            result.fresh_rejected = true;
          }
          result.fresh_sentinels = pops::reduce_min_local(flux_failure_first) == pops::Real(47) &&
                                   pops::reduce_max_local(flux_failure_first) == pops::Real(47) &&
                                   pops::reduce_min_local(flux_failure_second) == pops::Real(53) &&
                                   pops::reduce_max_local(flux_failure_second) == pops::Real(53);
          context.rhs_group(40, {{0, &stage, &flux_registry_recovery, 41, 0}});
          result.recovery_equal =
              pops::difference_sum_sq_all_local(full, flux_registry_recovery) == pops::Real(0);
        });
      });
  ASSERT_NO_THROW(system.step(0.01));
  {
    const auto evaluation = system.prepared_amr_level_evaluation(0);
    ASSERT_TRUE(evaluation);
    EXPECT_EQ(evaluation->point.stage, 18);
  }
  ASSERT_NO_THROW(system.step(0.01));
  {
    const auto evaluation = system.prepared_amr_level_evaluation(0);
    ASSERT_TRUE(evaluation);
    EXPECT_EQ(evaluation->point.stage, 31);
  }
  ASSERT_NO_THROW(system.step(0.01));
  {
    const auto evaluation = system.prepared_amr_level_evaluation(0);
    ASSERT_TRUE(evaluation);
    EXPECT_EQ(evaluation->point.stage, 31);
  }
  ASSERT_NO_THROW(system.step(0.01));
  {
    const auto evaluation = system.prepared_amr_level_evaluation(0);
    ASSERT_TRUE(evaluation);
    EXPECT_EQ(evaluation->point.stage, 41);
  }
  EXPECT_EQ(result.dispatches, 4);
  EXPECT_TRUE(result.full_equal);
  EXPECT_TRUE(result.flux_equal);
  EXPECT_TRUE(result.cross_equal);
  EXPECT_TRUE(result.malformed_rejected);
  EXPECT_TRUE(result.malformed_sentinels);
  EXPECT_TRUE(result.active_rejected);
  EXPECT_TRUE(result.active_sentinels);
  EXPECT_TRUE(result.duplicate_rejected);
  EXPECT_TRUE(result.duplicate_sentinels);
  if (pops::n_ranks() > 1) {
    EXPECT_TRUE(result.divergent_rejected);
    EXPECT_TRUE(result.divergent_sentinel);
  }
  EXPECT_TRUE(result.fresh_rejected);
  EXPECT_TRUE(result.fresh_sentinels);
  EXPECT_TRUE(result.recovery_equal);
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
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  struct Result {
    bool solved = false;
    std::string reason;
  } result;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/default-field-publication@1", "test-clock", resources, {},
      [&result](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
        auto outcome = context.solve_default_field_on_coarse_level();
        if (!outcome.report().solved_value_available()) {
          result.reason = outcome.report().reason;
          (void)outcome.consume(pops::SolveConsumption::kFailRun);
          return;
        }
        result.solved = outcome.consume(pops::SolveConsumption::kAccept).solved();
      });
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_TRUE(result.solved) << result.reason;
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
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  struct Result {
    bool solved = false;
    pops::Real state_before = pops::Real(-1);
    pops::Real state_after = pops::Real(-1);
  } result;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/named-field-stage@1", "test-clock", resources, {},
      [&result](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
        auto& stage = context.scratch_state(0, 0, context.state(0));
        stage.set_val(pops::Real(3));
        result.state_before = pops::reduce_max_local(context.state(0), 0);
        auto outcome = context.solve_fields_from_state_at(point<Dim>(0), "field/tracer", 0, stage);
        if (outcome.report().solved_value_available())
          result.solved = outcome.consume(pops::SolveConsumption::kAccept).solved();
        result.state_after = pops::reduce_max_local(context.state(0), 0);
      });
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_TRUE(result.solved);
  EXPECT_EQ(result.state_before, pops::Real(1));
  EXPECT_EQ(result.state_after, pops::Real(1));
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
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  struct Result {
    bool first_rejected = false;
    bool solved = false;
  } result;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/dynamic-field-boundary@1", "test-clock", resources, {},
      [&result](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
        auto& stage_state = context.scratch_state(0, 0, context.state(0));
        stage_state.set_val(pops::Real(3));
        auto evaluation = point<Dim>(0);
        evaluation.stage = 4;
        evaluation.physical_time = 0.125;
        RuntimeFieldBoundaryProbe<Dim>::force_failure = true;
        try {
          (void)context.solve_fields_from_state_at(evaluation, "field/tracer", 0, stage_state);
        } catch (const std::runtime_error&) {
          result.first_rejected = true;
        }
        RuntimeFieldBoundaryProbe<Dim>::force_failure = false;
        auto outcome =
            context.solve_fields_from_state_at(evaluation, "field/tracer", 0, stage_state);
        if (outcome.report().solved_value_available())
          result.solved = outcome.consume(pops::SolveConsumption::kAccept).solved();
      });
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_TRUE(result.first_rejected);
  EXPECT_TRUE(result.solved);
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
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  bool solved = false;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/composite-field-nullspace@1", "test-clock", resources, {},
      [&solved](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
        auto outcome = context.solve_default_field_on_coarse_level();
        if (outcome.report().solved_value_available())
          solved = outcome.consume(pops::SolveConsumption::kAccept).solved();
      });
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_TRUE(solved);

  EXPECT_EQ(system.field_provider_levels("pops.amr.default-field"), 2);
  EXPECT_EQ(system.field_potential_level_global("pops.amr.default-field", 0).size(),
            cell_count(config.shape));
  pops::Extent<Dim> fine_shape{};
  for (int axis = 0; axis < Dim; ++axis)
    fine_shape[axis] = 16;
  EXPECT_EQ(system.field_potential_level_global("pops.amr.default-field", 1).size(),
            cell_count(fine_shape));
}

TEST(GeneratedAmrSystemBlock, BootstrapRecomputeHelmholtzAfterCreateLevel) {
  constexpr int Dim = pops::kNativeDimension;
  constexpr const char* state_route = "tests.generated-amr/bootstrap-helmholtz/state";
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  config.regrid_every = 0;
  config.explicit_bootstrap = true;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 16;
    config.periodicity[axis] = true;
    config.coarse_max_grid[axis] = 8;
    config.transition_buffers.front()[axis] = 1;
    config.transition_lookaheads.front()[axis] = 1;
  }
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system,
                                            "tests.generated-amr/bootstrap-helmholtz-runtime");
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan("field/tracer", "test.bootstrap-helmholtz.plan",
                               "test.bootstrap-helmholtz", "test.aux-owner", "tracer", "phi",
                               {{"test.aux-owner", "field", "phi", "potential"}}, 1, {"test.rhs"},
                               {"tracer"}, {"charge"}, {1.0}, "geometric_mg", hierarchy,
                               pops::geometric_mg_amr_field_solver_options(
                                   pops::GeometricMgOptions{}, pops::CompositeFacOptions{}));
  system.set_field_reaction("field/tracer", 1.0);
  system.install_block_state_route("tracer", state_route);
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  const auto output_key = install_field_output(system, "test.aux-owner", "phi");
  system.register_elliptic_field("tracer", "phi", {output_key}, 1);
  system.set_block_elliptic_field(
      "tracer", "phi", "test.generated-amr.bootstrap-helmholtz.rhs.rho@1",
      [](const pops::MultiFab<Dim>& state, pops::MultiFab<Dim>& rhs) {
        for (std::size_t local = 0; local < state.local_size(); ++local) {
          const auto input = state.fab(local).view();
          const auto output = rhs.fab(local).view();
          pops::for_each_cell(state.box(local), [=] POPS_HD(const pops::Index<Dim>& cell) {
            output(cell, 0) = input(cell, 0);
          });
        }
        Kokkos::fence();
      });
  pops::test::install_prepared_threshold_union(
      system, {{"tracer", "u", 1.08, pops::test::PreparedThresholdRelation::Above, state_route}},
      "tests.generated-amr/bootstrap-helmholtz/tagging@1");
  system.bind_bootstrap_subject(state_route, "tracer", "bound_level_zero");
  const std::size_t cells = cell_count(config.shape);
  std::vector<double> initial(cells, 1.0);
  for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
    pops::Index<Dim> index{};
    std::size_t remaining = ordinal;
    for (int axis = 0; axis < Dim; ++axis) {
      index[axis] = static_cast<int>(remaining % 16);
      remaining /= 16;
    }
    const double x = (static_cast<double>(index[0]) + 0.5) / 16.0;
    const double y = (static_cast<double>(index[1]) + 0.5) / 16.0;
    const double r2 = (x - 0.25) * (x - 0.25) + (y - 0.5) * (y - 0.5);
    initial[ordinal] = 1.0 + 0.5 * std::exp(-140.0 * r2);
  }
  system.stage_bootstrap_array(state_route, "tracer", "cell", "cell", 1, config.shape, initial);
  pops::Extent<Dim> prolongation_ghosts{};
  pops::Extent<Dim> restriction_ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    prolongation_ghosts[axis] = 1;
  system.register_bootstrap_transfer_route(
      "tests.generated-amr/bootstrap-helmholtz/prolongation", {state_route},
      "tests.generated-amr/bootstrap-helmholtz/conservative-linear@1", "cell", "cell",
      "conservative", "dense", "prolongation", "conservative_linear", 2, prolongation_ghosts,
      config.transition_ratios.front());
  system.register_bootstrap_transfer_route(
      "tests.generated-amr/bootstrap-helmholtz/restriction", {state_route},
      "tests.generated-amr/bootstrap-helmholtz/volume-average@1", "cell", "cell", "conservative",
      "dense", "restriction", "volume_average", 1, restriction_ghosts,
      config.transition_ratios.front());

  system.begin_bootstrap_plan();
  system.set_program_block_map({0});
  (void)system.materialize_bootstrap_action(state_route, "initialize_level_zero",
                                            "bound_level_zero", 0);
  (void)system.recompute_bootstrap_field(state_route, "field/tracer");
  ASSERT_TRUE(system.bootstrap_next_level()) << "bootstrap did not create a refined level";
  (void)system.materialize_bootstrap_action(state_route, "prolong_from_parent",
                                            "conservative_linear", 1);
  EXPECT_NO_THROW((void)system.recompute_bootstrap_field(state_route, "field/tracer"));
  EXPECT_EQ(system.field_provider_levels("field/tracer"), 2);
  system.commit_bootstrap_level();
  EXPECT_EQ(system.field_provider_levels("field/tracer"), 2);
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  struct Result {
    bool solved = false;
    std::string reason;
  } result;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/bootstrap-helmholtz-runtime@1", "test-clock", resources, {},
      [&result](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
        auto& stage = context.scratch_state(0, 0, context.state(0));
        const auto& live = context.state(0);
        for (std::size_t local = 0; local < live.local_size(); ++local) {
          const auto input = live.fab(local).view();
          const auto output = stage.fab(local).view();
          pops::for_each_cell(live.box(local), [=] POPS_HD(const pops::Index<Dim>& cell) {
            output(cell, 0) = input(cell, 0);
          });
        }
        Kokkos::fence();
        auto evaluation = point<Dim>(0);
        evaluation.dt = dt;
        auto outcome = context.solve_fields_from_state_at(evaluation, "field/tracer", 0, stage);
        if (!outcome.report().solved_value_available()) {
          result.reason = outcome.report().reason;
          (void)outcome.consume(pops::SolveConsumption::kFailRun);
          return;
        }
        result.solved = outcome.consume(pops::SolveConsumption::kAccept).solved();
      });
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_TRUE(result.solved) << result.reason;
}

TEST(GeneratedAmrSystemBlock,
     ProgramExecutionServicesRefusesUnsynchronizedHierarchyBeforeMutation) {
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
  system.set_program_block_map({0});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  bool rejected = false;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/unsynchronized-hierarchy@1", "test-clock", resources, {},
      [&rejected](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        try {
          context.advance_hierarchy(
              dt, [&context](double) { context.state(0).set_val(pops::Real(9)); });
        } catch (const std::runtime_error&) {
          rejected = true;
        }
      });
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_TRUE(rejected);
  const auto accepted_runtime = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_runtime);
  EXPECT_EQ(pops::reduce_max_local(accepted_runtime->hierarchy().state(0)), pops::Real(1));
  EXPECT_EQ(pops::reduce_max_local(accepted_runtime->hierarchy().state(1)), pops::Real(1));
}

TEST(GeneratedAmrSystemBlock, ProgramExecutionServicesRetainsAndInterpolatesExactLevelHistory) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/level-history");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.set_program_block_map({0});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::rhs, Resource::Kind::state});
  struct Result {
    pops::Real retained = pops::Real(-1);
    pops::Real interpolated_min = pops::Real(-1);
    pops::Real interpolated_max = pops::Real(-1);
    pops::Real retained_after = pops::Real(-1);
    bool address_stable = false;
    bool schedule_rejected = false;
  } result;
  int dispatch = 0;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/level-history@1", "clock.macro", resources, {},
      [&dispatch, &result](pops::test::explicit_amr_program_detail::context_type& context,
                           double dt) {
        if (dispatch++ == 0) {
          context.declare_clock_relation("clock.macro", "clock.fast", 2);
          context.register_history("tracer.rate", 2, 1, 0, "tracer.U", "cell.conservative",
                                   "clock.macro", "dense.linear");
          context.begin_step(dt);
          auto& sample = context.rhs_scratch(0, 0, context.state(0));
          sample.set_val(pops::Real(10));
          context.store_history("tracer.rate", sample, 0);
          context.rotate_histories("clock.macro");
          return;
        }

        context.begin_step(dt);
        auto& sample = context.rhs_scratch(0, 0, context.state(0));
        sample.set_val(pops::Real(20));
        context.store_history("tracer.rate", sample, 0);
        auto& retained = context.history("tracer.rate", 1, 0);
        const auto* const retained_address = &retained;
        result.retained = pops::reduce_min_local(retained);
        auto& interpolated = context.scratch_state(1, 0, retained);
        interpolated.set_val(pops::Real(-1));
        context.interpolate_history_linear(interpolated, "tracer.rate", 2, 0, "clock.macro",
                                           "clock.fast", -1, pops::Real(0));
        result.interpolated_min = pops::reduce_min_local(interpolated);
        result.interpolated_max = pops::reduce_max_local(interpolated);
        sample.set_val(pops::Real(30));
        context.store_history("tracer.rate", sample, 0);
        result.address_stable = &context.history("tracer.rate", 1, 0) == retained_address;
        result.retained_after = pops::reduce_min_local(retained);
        try {
          (void)context.schedule_decision(17, true, true);
        } catch (const std::runtime_error&) {
          result.schedule_rejected = true;
        }
      });
  ASSERT_NO_THROW(system.step(0.2));
  ASSERT_NO_THROW(system.step(0.4));
  EXPECT_EQ(result.retained, pops::Real(10));
  EXPECT_EQ(result.interpolated_min, pops::Real(15));
  EXPECT_EQ(result.interpolated_max, pops::Real(15));
  EXPECT_EQ(result.retained_after, pops::Real(10));
  EXPECT_TRUE(result.address_stable);
  EXPECT_TRUE(result.schedule_rejected);
}

TEST(GeneratedAmrSystemBlock, ProgramExecutionServicesRefusesHistoryRegridBeforeTopologyMutation) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/history-regrid-refusal");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  auto* engine = pops::test::AmrSystemTestAccess<Dim>::engine(system);
  ASSERT_NE(engine, nullptr);
  system.set_program_block_map({0});
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  const pops::amr::RefinementRatio<Dim> ratio(ratio_components);
  const pops::amr::regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = {16, 120},
      .fine_layout = {16, 120},
      .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()},
  };
  auto cluster = std::make_shared<pops::amr::tagging::ClusterResult<Dim>>(
      centered_cluster(engine->hierarchy().layout(0)));
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  bool rejected = false;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/history-regrid-refusal@1", "clock.macro", resources, {},
      [&rejected, ratio, budget, cluster](
          pops::test::explicit_amr_program_detail::context_type& context, double dt) mutable {
        context.register_history("tracer.rate", 1, 1, 0, "tracer.U", "cell.conservative",
                                 "clock.macro", "dense.linear");
        context.begin_step(dt);
        auto prepared = context.prepare_regrid(0, ratio, *cluster, budget);
        if (!prepared.fine_layout().has_value())
          throw std::logic_error("history regrid test did not prepare a fine level");
        pops::MultiFab<pops::kNativeDimension> child(
            prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
            context.state(0).local_rank(), context.state(0).ncomp(), context.state(0).ghosts());
        child.set_val(pops::Real(1));
        try {
          context.publish_regrid(std::move(prepared), std::move(child));
        } catch (const std::runtime_error&) {
          rejected = true;
        }
      });
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_TRUE(rejected);
  const auto accepted_runtime = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_runtime);
  EXPECT_EQ(accepted_runtime->hierarchy().num_levels(), 1u);
}

TEST(GeneratedAmrSystemBlock,
     TransactionalRegridTransfersDetachedFieldPotentialAndRollsBackExactly) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  config.regrid_every = 1;
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/field-regrid-transaction");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.set_poisson();
  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> input_shape;
  for (int axis = 0; axis < Dim; ++axis)
    input_shape.halo[axis] = 1;
  const AuxiliaryComponentKey input_key{"tests.generated-amr/field-regrid-transaction", "input",
                                        "rho", "value"};
  const AuxiliaryComponentContract input_contract{"cell-average", "cell", "unitless", "amr-input",
                                                  "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "tests.generated-amr/field-regrid-transaction/input@1",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::after_regrid, AuxiliaryFreshness::evaluation},
      {{input_key, input_contract, input_shape}},
      {}});
  const auto output_key =
      install_field_output(system, "tests.generated-amr/field-regrid-transaction", "phi");
  system.register_default_elliptic_field_output({output_key}, 1);
  // The periodic Poisson nullspace neutralizes a uniform source.  A declared Helmholtz reaction
  // keeps this a real detached solve while making the nonzero field/output witness observable.
  system.set_field_reaction("pops.amr.default-field", 1.0);
  system.set_field_topology_authority("pops.amr.default-field", "builtin_rectangular_cell_graph_v1",
                                      "tests.generated-amr/field-regrid-transaction@1",
                                      "tests.generated-amr/field-regrid-transaction:v1");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", forced_poisson_advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 0.0));
  pops::test::install_prepared_refine_coarsen_threshold(
      system, {"tracer", "u", 0.5, pops::test::PreparedThresholdRelation::Above},
      {"tracer", "u", 0.5, pops::test::PreparedThresholdRelation::Below},
      "tests.generated-amr/field-regrid-transaction/tagging@1");
  const std::vector<double> accepted_root(cell_count(config.shape), 2.0);
  system.set_field_potential_level("pops.amr.default-field", 0, accepted_root);
  system.set_program_block_map({0});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/field-regrid-transaction@1", "clock.macro", resources, {},
      [](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
      });
  ASSERT_NO_THROW(system.mark_bound());

  ASSERT_NO_THROW(system.step(0.01));
  ASSERT_EQ(system.n_levels(), 1);
  ASSERT_NO_THROW(
      system.stage_auxiliary_input(input_key, std::vector<double>(cell_count(config.shape), 1.0)));
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));

  // The first real forward regrid reaches the detached FieldPlan aggregate, then is rejected by
  // the enclosing transaction.  The accepted potential and topology must remain byte-identical.
  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(0.01));
  ASSERT_NO_THROW(system.rollback_step_transaction());
  EXPECT_EQ(system.n_levels(), 1);
  EXPECT_EQ(system.field_potential_level_global("pops.amr.default-field", 0), accepted_root);

  // Retry the same prepared callback.  Its field image is sourced from the frozen accepted
  // potential, not a zeroed solver workspace, and the new fine level is consequently nonzero.
  ASSERT_NO_THROW(system.step(0.01));
  ASSERT_EQ(system.n_levels(), 2);
  const auto root = system.field_potential_level_global("pops.amr.default-field", 0);
  ASSERT_FALSE(root.empty());
  EXPECT_TRUE(std::any_of(root.begin(), root.end(), [](double value) { return value != 0.0; }));
  const auto fine = system.field_potential_level_global("pops.amr.default-field", 1);
  ASSERT_FALSE(fine.empty());
  EXPECT_GT(*std::max_element(fine.begin(), fine.end()), 0.0);
  const auto output = system.auxiliary_component(output_key, 1);
  ASSERT_FALSE(output.empty());
  EXPECT_TRUE(std::any_of(output.begin(), output.end(), [](double value) { return value != 0.0; }));
}

TEST(GeneratedAmrSystemBlock, PreparedHistoryRemapAcceptsPublishedReplacement) {
  const auto exercise_deferred_ratio = [](std::int64_t temporal_numerator) {
    constexpr int Dim = pops::kNativeDimension;
    pops::AmrSystemConfig<Dim> config;
    for (int axis = 0; axis < Dim; ++axis) {
      config.shape[axis] = 32;
      config.transition_buffers.front()[axis] = 0;
      config.transition_lookaheads.front()[axis] = 0;
    }
    config.regrid_every = 0;
    pops::AmrSystem<Dim> system(config);
    pops::test::install_amr_runtime_authority(system,
                                              "tests.generated-amr/history-remap-acceptance");
    system.set_temporal_relations({temporal_numerator}, {1}, {"integral_only"});
    system.install_block_state_route("tracer", "state/tracer");
    pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
    system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
    pops::test::install_prepared_refine_coarsen_threshold(
        system, {"tracer", "u", 0.5, pops::test::PreparedThresholdRelation::Above},
        {"tracer", "u", 0.5, pops::test::PreparedThresholdRelation::Below},
        "tests.generated-amr/history-remap-tagging@1");
    const auto accepted_runtime = system.accepted_amr_runtime();
    ASSERT_TRUE(accepted_runtime);
    ASSERT_EQ(accepted_runtime->hierarchy().num_levels(), 2u);
    system.set_program_block_map({0});
    using Resource = pops::test::program_v5::CallbackProgramResource;
    const auto resources =
        dense_resources<Dim>(system, {Resource::Kind::rhs, Resource::Kind::state});
    struct CallbackResult {
      int dispatch = 0;
      bool deferred_seen = false;
      double deferred_dt = 0.0;
      pops::Real effective_min = pops::Real(-1);
      pops::Real effective_max = pops::Real(-1);
      pops::Real expected_min = pops::Real(-1);
      pops::Real expected_max = pops::Real(-1);
      std::size_t expression_size = 0;
      bool expression_basis = false;
      bool repeated_same = false;
    } callback_result;
    pops::test::install_explicit_amr_callback_program<Dim>(
        system, "tests.generated-amr/history-remap@1", "clock.macro", resources, {},
        [&callback_result, temporal_numerator](
            pops::test::explicit_amr_program_detail::context_type& context, double dt) {
          const int dispatch = callback_result.dispatch++;
          if (dispatch < 3) {
            if (dispatch == 0) {
              context.declare_clock_relation("clock.macro", "clock.fast", temporal_numerator);
              context.register_history("tracer.rate", 2, 1, 0, "tracer.U", "cell.conservative",
                                       "clock.macro", "dense.linear");
            }
            context.advance_hierarchy(dt, [&](double) {
              const auto rhs_slot =
                  static_cast<pops::runtime::program::ProgramCacheSlot>(context.level() * 2);
              auto& stage = context.state(0);
              auto& sample = context.rhs_scratch(rhs_slot, 0, stage);
              context.rhs_into(0, stage, sample, 0);
              context.store_history("tracer.rate", sample, 0);
              context.rotate_histories("clock.macro");
            });
            return;
          }
          if (dispatch == 3) {
            context.advance_hierarchy(dt, [&](double) {
              const auto rhs_slot =
                  static_cast<pops::runtime::program::ProgramCacheSlot>(context.level() * 2);
              auto& stage = context.state(0);
              auto& sample = context.rhs_scratch(rhs_slot, 0, stage);
              context.rhs_into(0, stage, sample, 0);
              context.store_history("tracer.rate", sample, 0);
              context.rotate_histories("clock.macro");
            });
            return;
          }
          if (dispatch == 4) {
            context.advance_hierarchy(dt, [&](double local_dt) {
              if (context.level() != 1 ||
                  !pops::runtime::program::detail::AmrProgramHistoryRemapCollectiveTestAccess<
                      Dim>::has_pending_history(context, "tracer.rate", context.level())) {
                const auto rhs_slot =
                    static_cast<pops::runtime::program::ProgramCacheSlot>(context.level() * 2);
                auto& stage = context.state(0);
                auto& sample = context.rhs_scratch(rhs_slot, 0, stage);
                context.rhs_into(0, stage, sample, 0);
                context.store_history("tracer.rate", sample, 0);
                context.rotate_histories("clock.macro");
                return;
              }
              callback_result.deferred_seen = true;
              callback_result.deferred_dt = local_dt;
              auto& stage = context.state(0);
              const auto rhs_slot =
                  static_cast<pops::runtime::program::ProgramCacheSlot>(context.level() * 2);
              auto& sample = context.rhs_scratch(rhs_slot, 0, stage);
              context.rhs_into(0, stage, sample, 0);
              context.store_history("tracer.rate", sample, 0);
              auto& raw_lag = context.history("tracer.rate", 1, 1);
              auto expected = context.rhs_scratch_like(sample);
              if (temporal_numerator == 1)
                pops::lincomb(expected, pops::Real(1), raw_lag, pops::Real(0), raw_lag);
              else
                pops::lincomb(expected, pops::Real(0.5), sample, pops::Real(0.5), raw_lag);
              auto& effective = context.history("tracer.rate", 1, 0);
              callback_result.effective_min = pops::reduce_min_local(effective);
              callback_result.effective_max = pops::reduce_max_local(effective);
              callback_result.expected_min = pops::reduce_min_local(expected);
              callback_result.expected_max = pops::reduce_max_local(expected);
              const auto& expression =
                  pops::runtime::program::detail::AmrProgramHistoryRemapCollectiveTestAccess<
                      Dim>::active_expression(context, effective);
              callback_result.expression_size = expression.size();
              for (const auto& [identity, term] : expression) {
                (void)identity;
                if (term.coefficient.size() == 1 && term.coefficient.contains(0) &&
                    term.coefficient.at(0) == (temporal_numerator == 1
                                                   ? pops::amr::Rational(1, 1)
                                                   : pops::amr::Rational(1, 2)) &&
                    term.basis != nullptr && !term.basis->faces.empty())
                  callback_result.expression_basis = true;
              }
              auto& repeated = context.history("tracer.rate", 1, 0);
              callback_result.repeated_same =
                  pops::reduce_min_local(repeated) == callback_result.effective_min &&
                  pops::reduce_max_local(repeated) == callback_result.effective_max;
            });
            return;
          }
          if (dispatch == 5)
            context.begin_step(dt);
        });
    for (const double dt : {0.1, 0.2, 0.3})
      ASSERT_NO_THROW(system.step(dt));
    const auto retained_child_slot0 = system.history_global("tracer.rate", 1, 0);
    const auto retained_child_slot1 = system.history_global("tracer.rate", 1, 1);
    const double retained_child_dt0 = system.history_slot_dt("tracer.rate", 1, 0);
    const double retained_child_dt1 = system.history_slot_dt("tracer.rate", 1, 1);
    const auto full_boxes = system.patch_boxes();
    std::vector<double> contracted(cell_count(config.shape), 0.25);
    std::size_t center = 0;
    std::size_t stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      center += static_cast<std::size_t>(config.shape[axis] / 2) * stride;
      stride *= static_cast<std::size_t>(config.shape[axis]);
    }
    contracted[center] = 1.0;
    system.set_conservative_state("tracer", contracted);
    system.execute_prepared_tagging(0);
    ASSERT_TRUE(system.regrid_from_prepared_tagging(0));
    EXPECT_NE(system.patch_boxes(), full_boxes);
    const auto retained_after_contraction =
        pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
            system.program_accepted_state());
    EXPECT_TRUE(retained_after_contraction.pending_history_remaps.empty());
    expect_history_preserved_on_current_coverage(system, retained_child_slot0,
                                                 system.history_global("tracer.rate", 1, 0), 1);
    expect_history_preserved_on_current_coverage(system, retained_child_slot1,
                                                 system.history_global("tracer.rate", 1, 1), 1);
    EXPECT_EQ(system.history_slot_dt("tracer.rate", 1, 0), retained_child_dt0);
    EXPECT_EQ(system.history_slot_dt("tracer.rate", 1, 1), retained_child_dt1);

    // Capture nonempty interface provenance on the retained partial child before the genuine
    // coverage expansion makes its parent samples the deferred source.
    ASSERT_NO_THROW(system.step(0.3));

    std::vector<double> expanded(cell_count(config.shape), 0.25);
    expanded[center] = 1.0;
    std::size_t axis_stride = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      expanded[center - axis_stride] = 1.0;
      expanded[center + axis_stride] = 1.0;
      axis_stride *= static_cast<std::size_t>(config.shape[axis]);
    }
    system.set_conservative_state("tracer", expanded);
    system.execute_prepared_tagging(0);
    ASSERT_TRUE(system.regrid_from_prepared_tagging(0));
    const auto pending_after_regrid =
        pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
            system.program_accepted_state());
    ASSERT_EQ(pending_after_regrid.pending_history_remaps.size(), 1u);
    EXPECT_EQ(pending_after_regrid.pending_history_remaps.front().temporal_numerator,
              temporal_numerator);
    EXPECT_EQ(pending_after_regrid.pending_history_remaps.front().temporal_denominator, 1);
    EXPECT_EQ(pending_after_regrid.pending_history_remaps.front().target_dt,
              pending_after_regrid.pending_history_remaps.front().source_dt /
                  static_cast<double>(temporal_numerator));
    const auto pending_bytes = system.program_accepted_state();
    EXPECT_NO_THROW(system.restore_checkpoint_accepted_state(pending_bytes));
    EXPECT_EQ(system.program_accepted_state(), pending_bytes);
    // Cursor-walk POPSAND5 to its pending section.  The history key also occurs in earlier slot
    // payloads, so searching raw bytes would mutate the wrong record.
    const std::string& pending_key = pending_after_regrid.pending_history_remaps.front().key;
    std::size_t cursor = 8 + 8;  // magic, native dimension
    const auto read_word = [&](std::size_t& position) {
      std::uint64_t value = 0;
      for (std::size_t byte = 0; byte < 8; ++byte)
        value |= std::uint64_t{pending_bytes[position + byte]} << (8 * byte);
      position += 8;
      return value;
    };
    const auto skip_string = [&](std::size_t& position) { position += read_word(position); };
    skip_string(cursor);               // spatial contract
    cursor += 16;                      // topology, materialization generation
    cursor += read_word(cursor) * 40;  // level clocks
    for (std::size_t count = read_word(cursor); count > 0; --count) {
      skip_string(cursor);
      cursor += 8;
    }
    for (std::size_t count = read_word(cursor); count > 0; --count) {
      skip_string(cursor);
      cursor += 8;
      for (int identity = 0; identity < 4; ++identity)
        skip_string(cursor);
      cursor += 16;
    }
    for (std::size_t count = read_word(cursor); count > 0; --count) {
      skip_string(cursor);
      cursor += 40;
    }
    const std::size_t pending_count_offset = cursor;
    ASSERT_EQ(read_word(cursor), 1u);
    const std::size_t record_offset = cursor;
    const std::size_t pending_key_length = read_word(cursor);
    const std::size_t key_offset = cursor;
    ASSERT_EQ(pending_key_length, pending_key.size());
    ASSERT_TRUE(std::equal(pending_key.begin(), pending_key.end(),
                           pending_bytes.begin() + static_cast<std::ptrdiff_t>(key_offset)));
    cursor += pending_key_length;
    const std::size_t after_key = cursor;
    ASSERT_LE(record_offset + 8 + pending_key_length + 96, pending_bytes.size());
    const auto write_word = [](std::vector<std::uint8_t>& bytes, std::size_t offset,
                               std::uint64_t value) {
      ASSERT_LE(offset + 8, bytes.size());
      for (std::size_t byte = 0; byte < 8; ++byte)
        bytes[offset + byte] = static_cast<std::uint8_t>(value >> (8 * byte));
    };
    const auto refuse_and_retry = [&](std::string_view label, std::vector<std::uint8_t> corrupt,
                                      std::string_view diagnostic_class) {
      SCOPED_TRACE(label);
      try {
        system.restore_checkpoint_accepted_state(corrupt);
        ADD_FAILURE() << "corrupt POPSAND5 accepted state was accepted";
      } catch (const std::exception& exception) {
        EXPECT_NE(std::string_view(exception.what()).find(diagnostic_class), std::string_view::npos)
            << exception.what();
      }
      EXPECT_EQ(system.program_accepted_state(), pending_bytes);
      EXPECT_NO_THROW(system.restore_checkpoint_accepted_state(pending_bytes));
      EXPECT_EQ(system.program_accepted_state(), pending_bytes);
    };
    auto foreign_key = pending_bytes;
    foreign_key[key_offset] ^= std::uint8_t{1};
    refuse_and_retry("foreign-key", std::move(foreign_key), "pending history remap");
    auto wrong_level = pending_bytes;
    write_word(wrong_level, after_key + 8, 0);  // child level; parent remains zero
    refuse_and_retry("wrong-child-level", std::move(wrong_level), "pending history remap");
    auto int_max_parent = pending_bytes;
    write_word(int_max_parent, after_key, std::numeric_limits<int>::max());
    refuse_and_retry("int-max-parent", std::move(int_max_parent), "pending history remap");
    auto stale_prior_topology = pending_bytes;
    write_word(stale_prior_topology, after_key + 16, pending_after_regrid.topology_epoch);
    refuse_and_retry("stale-prior-topology", std::move(stale_prior_topology),
                     "pending history remap");
    auto stale_prior_materialization = pending_bytes;
    write_word(stale_prior_materialization, after_key + 24,
               pending_after_regrid.materialization_generation);
    refuse_and_retry("stale-prior-materialization", std::move(stale_prior_materialization),
                     "pending history remap");
    auto stale_generation = pending_bytes;
    write_word(stale_generation, after_key + 40,
               pending_after_regrid.materialization_generation + 1);
    refuse_and_retry("stale-published-materialization", std::move(stale_generation),
                     "pending history remap");
    auto wrong_macro = pending_bytes;
    write_word(wrong_macro, after_key + 48,
               static_cast<std::uint64_t>(
                   pending_after_regrid.pending_history_remaps.front().accepted_macro_step + 1));
    refuse_and_retry("wrong-child-macro", std::move(wrong_macro), "pending history remap");
    auto wrong_dt = pending_bytes;
    write_word(wrong_dt, after_key + 72, 0);  // source_dt bit pattern
    refuse_and_retry("wrong-source-dt", std::move(wrong_dt), "pending history remap");
    auto amplified_count = pending_bytes;
    write_word(amplified_count, pending_count_offset, std::numeric_limits<std::uint64_t>::max());
    refuse_and_retry("amplified-pending-count", std::move(amplified_count),
                     "invalid exact AMR Program checkpoint");
    auto truncated_pending = pending_bytes;
    truncated_pending.resize(after_key + 48);
    refuse_and_retry("truncated-pending", std::move(truncated_pending),
                     "invalid exact AMR Program checkpoint");
    EXPECT_EQ(system.history_names(), (std::vector<std::string>{"tracer.rate"}));
    for (const int level : {0, 1}) {
      EXPECT_TRUE(system.history_initialized("tracer.rate", level));
      EXPECT_EQ(system.history_fill_count("tracer.rate", level), 2);
      EXPECT_GT(system.history_slot_dt("tracer.rate", level, 0), 0.0);
      EXPECT_GT(system.history_slot_dt("tracer.rate", level, 1), 0.0);
    }

    // The next child store recycles slot 0 at a new current interval, distinct from source_dt/N.
    // N=1 returns the exact parent lag; N=2 returns half-current/half-parent-old.  Reading it
    // cannot consume the accepted marker.
    constexpr double current_dt = 0.12;
    ASSERT_NE(current_dt, pending_after_regrid.pending_history_remaps.front().target_dt);
    ASSERT_NO_THROW(system.step(current_dt));
    EXPECT_TRUE(callback_result.deferred_seen);
    EXPECT_EQ(callback_result.deferred_dt, current_dt / static_cast<double>(temporal_numerator));
    EXPECT_EQ(callback_result.effective_min, callback_result.expected_min);
    EXPECT_EQ(callback_result.effective_max, callback_result.expected_max);
    EXPECT_EQ(callback_result.expression_size, temporal_numerator == 1 ? 1u : 2u);
    EXPECT_TRUE(callback_result.expression_basis);
    EXPECT_TRUE(callback_result.repeated_same);
    EXPECT_EQ(pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
                  system.program_accepted_state())
                  .pending_history_remaps.size(),
              1u);
    // Rotation consumes the *live* marker.  The accepted wire image is still the prior publication
    // until the normal hierarchy-refresh capture below runs.
    const auto stale_after_rotation =
        pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
            system.program_accepted_state());
    EXPECT_EQ(stale_after_rotation.pending_history_remaps.size(), 1u);
    EXPECT_NO_THROW(system.step(0.15));
    const auto published_after_rotation =
        pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
            system.program_accepted_state());
    EXPECT_TRUE(published_after_rotation.pending_history_remaps.empty());
  };

  exercise_deferred_ratio(1);
  exercise_deferred_ratio(2);
}

TEST(GeneratedAmrSystemBlock, DetachedHistoryRemapStagesExactProvenanceWithoutLiveClobber) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 32;
    config.transition_buffers.front()[axis] = 0;
    config.transition_lookaheads.front()[axis] = 0;
  }
  config.regrid_every = 0;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.generated-amr/detached-history-remap");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  pops::test::install_prepared_refine_coarsen_threshold(
      system, {"tracer", "u", 0.5, pops::test::PreparedThresholdRelation::Above},
      {"tracer", "u", 0.5, pops::test::PreparedThresholdRelation::Below},
      "tests.generated-amr/detached-history-remap-tagging@1");
  const auto accepted_runtime = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_runtime);
  ASSERT_EQ(accepted_runtime->hierarchy().num_levels(), 2u);
  system.set_program_block_map({0});

  using Resource = pops::test::program_v5::CallbackProgramResource;
  using History = pops::test::program_v5::CallbackProgramHistory;
  using Context = pops::test::explicit_amr_program_detail::context_type;
  using Access = pops::runtime::program::detail::AmrProgramHistoryRemapCollectiveTestAccess<Dim>;
  using Descriptor = pops::runtime::program::AmrProgramHistoryRemapDescriptor;
  using Entry = pops::runtime::program::AmrProgramHistoryRemapEntry;
  using Source = pops::runtime::program::AmrProgramHistoryRemapSource;
  using Marker = pops::runtime::program::AmrProgramPendingHistoryRemap;

  struct Observation {
    bool retained = false;
    bool parent_deferred = false;
    bool deferred_marker = false;
    bool removed = false;
    bool refused_without_clobber = false;
  } observation;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::rhs});
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/detached-history-remap@1", "clock.macro", resources, {},
      [&observation](Context& context, double dt) {
        context.begin_step(dt);
        const auto retained = Access::history_key(context, "retained", 1);
        const auto deferred = Access::history_key(context, "deferred", 1);
        const auto parent_deferred = Access::history_key(context, "deferred", 0);
        const auto [epoch, generation] = Access::resource_authority(context);

        auto accepted = context.create_accepted_context_snapshot();
        ASSERT_NE(accepted, nullptr);
        void* token = nullptr;
        auto staged = accepted->detach_for_forward(epoch, generation, token);
        ASSERT_NE(staged, nullptr);
        ASSERT_NE(token, nullptr);
        Descriptor descriptor;
        descriptor.parent_level = 0;
        descriptor.child_level = 1;
        descriptor.child_published = true;
        descriptor.child_physical_layout_changed = true;
        descriptor.history_plan = {{retained, {}, Source::RetainedChild},
                                   {deferred, parent_deferred, Source::ParentDeferred}};
        descriptor.prior_topology_epoch = epoch;
        descriptor.prior_materialization_generation = generation;
        descriptor.published_topology_epoch = epoch;
        descriptor.published_materialization_generation = generation;
        descriptor.accepted_macro_step = 0;
        descriptor.temporal_numerator = 2;
        descriptor.temporal_denominator = 1;
        descriptor.integral_only = true;
        descriptor.operation_identity = "tests.detached-history-remap";
        descriptor.prepared_pending_history_remaps.push_back(Marker{
            deferred, 0, 1, epoch, generation, epoch, generation, 0, 2, 1, 0.25, 0.125, false});
        staged->prepare_forward_history_remap(descriptor);
        auto rollback_image = staged->prepare_restore();
        observation.retained = Access::is_detached(*staged) &&
                               Access::is_detached(*rollback_image) &&
                               Access::has_history(*staged, retained) &&
                               Access::has_history(*rollback_image, retained);
        observation.parent_deferred =
            Access::has_history(*staged, deferred) &&
            Access::flux_depth(*staged, deferred) == Access::flux_depth(*staged, parent_deferred);
        observation.deferred_marker = Access::has_pending(*staged, deferred) &&
                                      Access::has_pending(*rollback_image, deferred);

        auto removal = accepted->detach_for_forward(epoch, generation, token);
        Descriptor remove_descriptor = descriptor;
        remove_descriptor.child_published = false;
        remove_descriptor.child_physical_layout_changed = false;
        remove_descriptor.history_plan = {{retained, {}, Source::Removed},
                                          {deferred, {}, Source::Removed}};
        remove_descriptor.prepared_pending_history_remaps.clear();
        removal->prepare_forward_history_remap(remove_descriptor);
        observation.removed =
            !Access::has_history(*removal, retained) && !Access::has_history(*removal, deferred);

        auto refused = accepted->detach_for_forward(epoch, generation, token);
        Descriptor malformed = descriptor;
        malformed.history_plan[1].parent_key = "missing-parent-history";
        try {
          refused->prepare_forward_history_remap(malformed);
        } catch (const std::logic_error&) {
          observation.refused_without_clobber = Access::has_history(*accepted, retained) &&
                                                Access::has_history(*accepted, deferred) &&
                                                Access::live_has_history(context, retained) &&
                                                Access::live_has_history(context, deferred) &&
                                                !Access::live_has_pending(context, deferred);
        }
      },
      {History{"retained", 1, 1, 0, "tracer.U", "cell.conservative", "clock.macro", "dense.linear"},
       History{"deferred", 1, 1, 0, "tracer.U", "cell.conservative", "clock.macro",
               "dense.linear"}});
  ASSERT_NO_THROW(system.step(0.1));
  EXPECT_TRUE(observation.retained);
  EXPECT_TRUE(observation.parent_deferred);
  EXPECT_TRUE(observation.deferred_marker);
  EXPECT_TRUE(observation.removed);
  EXPECT_TRUE(observation.refused_without_clobber);
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
  const auto accepted_runtime = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_runtime);
  ASSERT_EQ(accepted_runtime->hierarchy().num_levels(), 1u);
  system.set_program_block_map({0});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::rhs});
  int dispatch = 0;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/history-remap-noop@1", "clock.macro", resources, {},
      [&dispatch](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        if (dispatch++ == 0)
          context.register_history("tracer.rate", 2, 1, 0, "tracer.U", "cell.conservative",
                                   "clock.macro", "dense.linear");
        context.begin_step(dt);
        auto& sample = context.rhs_scratch(0, 0, context.state(0));
        sample.set_val(pops::Real(dt));
        context.store_history("tracer.rate", sample, 0);
        context.rotate_histories("clock.macro");
      });
  for (const double dt : {0.1, 0.2, 0.3})
    ASSERT_NO_THROW(system.step(dt));
  const auto accepted_before = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_before);
  const std::uint64_t topology_before = accepted_before->topology_epoch();
  const std::uint64_t materialization_before = accepted_before->materialization_generation();
  const auto names_before = system.history_names();
  const int fill_before = system.history_fill_count("tracer.rate", 0);
  const double dt0_before = system.history_slot_dt("tracer.rate", 0, 0);
  const double dt1_before = system.history_slot_dt("tracer.rate", 0, 1);
  const auto slot0_before = system.history_global("tracer.rate", 0, 0);
  const auto slot1_before = system.history_global("tracer.rate", 0, 1);

  EXPECT_FALSE(system.regrid_from_prepared_tagging(0));
  const auto accepted_after = system.accepted_amr_runtime();
  ASSERT_TRUE(accepted_after);
  EXPECT_EQ(accepted_after->hierarchy().num_levels(), 1u);
  EXPECT_EQ(accepted_after->topology_epoch(), topology_before);
  EXPECT_EQ(accepted_after->materialization_generation(), materialization_before);
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
  system.set_program_block_map({0});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/cfl-finest-geometry@1", "test-clock", resources, {},
      [](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
      });

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
  system.set_program_block_map({0});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/diffusive-cfl@1", "test-clock", resources, {},
      [](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
      });

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
  system.set_program_block_map({0});
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto resources = dense_resources<Dim>(system, {Resource::Kind::state});
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.generated-amr/cfl-request-consensus@1", "test-clock", resources, {},
      [](pops::test::explicit_amr_program_detail::context_type& context, double dt) {
        context.begin_step(dt);
      });

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
