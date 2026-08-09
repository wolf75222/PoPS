#include <gtest/gtest.h>

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <string>
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
}  // namespace pops

#define add_compiled_model add_test_compiled_model

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
    return {"test.amr-named-field.scalar-advection", 1};
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
pops::runtime::system::AuxiliaryComponentKey install_field_output(
    pops::AmrSystem<Dim>& system, const std::string& owner, const std::string& field) {
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
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
pops::runtime::multiblock::BoundaryEvaluationPoint evaluation_point(int stage) {
  return {.clock = "test-clock",
          .tick = 0,
          .level = 0,
          .substep = 0,
          .stage = stage,
          .stage_fraction = {0, 1},
          .dt = 0.01,
          .physical_time = 0.0};
}

template <int Dim>
pops::AmrSystemConfig<Dim> single_level_config() {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  return config;
}

}  // namespace

TEST(test_amr_named_field, DefaultFieldPublishesOnlyWhenSolveOutcomeIsAccepted) {
  constexpr int Dim = pops::kNativeDimension;
  const pops::AmrSystemConfig<Dim> config = single_level_config<Dim>();
  pops::AmrSystem<Dim> system(config);
  system.set_poisson();
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>());
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  context->begin_step(0.01);
  pops::SolveOutcome outcome = context->solve_default_field_on_coarse_level();
  ASSERT_TRUE(outcome.report().solved_value_available());

  const pops::SolveReport accepted = outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
  EXPECT_EQ(system.field_provider_slots(), std::vector<std::string>{"pops.amr.default-field"});
  EXPECT_EQ(system.field_provider_levels("pops.amr.default-field"), 1);
}

TEST(test_amr_named_field, NamedPlanConsumesExactStageWithoutPublishingConservativeState) {
  constexpr int Dim = pops::kNativeDimension;
  const pops::AmrSystemConfig<Dim> config = single_level_config<Dim>();
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
  const auto output_key = install_field_output(system, "test.aux-owner", "phi");
  system.register_elliptic_field("tracer", "phi", {output_key}, 1);
  pops::Real observed_stage = pops::Real(-1);
  system.set_block_elliptic_field(
      "tracer", "phi",
      [&observed_stage](const pops::MultiFab<Dim>& state, pops::MultiFab<Dim>& rhs) {
        observed_stage = pops::reduce_min_local(state);
        rhs.set_val(pops::Real(0));
      });
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  context->begin_step(0.01);
  pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
  stage.set_val(pops::Real(3));
  pops::SolveOutcome outcome =
      context->solve_fields_from_state_at(evaluation_point<Dim>(4), "field/tracer", 0, stage);
  const pops::SolveReport accepted = pops::consume_solve_outcome(std::move(outcome));
  EXPECT_EQ(pops::reduce_min_local(context->state(0), 0), pops::Real(1));
  EXPECT_EQ(pops::reduce_max_local(context->state(0), 0), pops::Real(1));

  EXPECT_TRUE(accepted.solved());
  EXPECT_EQ(pops::reduce_min_local(context->state(0), 0), pops::Real(1));
  EXPECT_EQ(pops::reduce_max_local(context->state(0), 0), pops::Real(1));
  EXPECT_EQ(system.field_provider_levels("field/tracer"), 1);
  EXPECT_EQ(observed_stage, pops::Real(3));
  EXPECT_EQ(system.auxiliary_component(output_key).size(), cell_count(config.shape));
}
