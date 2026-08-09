#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <limits>
#include <vector>

namespace {

template <int Dim>
pops::Extent<Dim> uniform_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::MultiFab<Dim> one_patch_field(int width, int ncomp) {
  const pops::Box<Dim> box = pops::Box<Dim>::from_extents(uniform_extent<Dim>(width));
  const pops::mesh::BoxArray<Dim> layout(std::vector<pops::Box<Dim>>{box});
  const pops::mesh::RankSpace<Dim> ranks(pops::Index<Dim>{}, uniform_extent<Dim>(1));
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  return pops::MultiFab<Dim>(layout, distribution, pops::Index<Dim>{}, ncomp, pops::Extent<Dim>{});
}

struct StiffLinearSource {
  using State = pops::StateVec<1>;
  static constexpr int n_vars = 1;

  POPS_HD State source(const State& state, const pops::ProviderValues<0>&) const {
    return State{-state[0]};
  }
  POPS_HD void source_jacobian(const State&, const pops::ProviderValues<0>&,
                               pops::Real (&jacobian)[1][1]) const {
    jacobian[0][0] = pops::Real(-1);
  }
};

template <int Dim>
struct AdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;

  Law law{};

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr-imex.scalar-advection", 1};
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
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

}  // namespace

TEST(test_amr_multiblock_imex, RankedImplicitSolveIsStableAndPublishesOnlyOnAcceptance) {
  constexpr int Dim = pops::kNativeDimension;
  auto state = one_patch_field<Dim>(2, 1);
  state.set_val(pops::Real(2));

  pops::SolveOutcome outcome = pops::backward_euler_source(
      StiffLinearSource{}, [](std::size_t) { return pops::ProviderStorageView<Dim, 0>{}; }, state,
      pops::Real(0.25), pops::NewtonOptions{});
  ASSERT_TRUE(outcome.report().solved_value_available());
  EXPECT_EQ(pops::reduce_min_local(state), pops::Real(2));
  const pops::SolveReport accepted = outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
  EXPECT_NEAR(static_cast<double>(pops::reduce_min_local(state)), 1.6,
              64.0 * std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(static_cast<double>(pops::reduce_max_local(state)), 1.6,
              64.0 * std::numeric_limits<double>::epsilon());
}

TEST(test_amr_multiblock_imex, ExactFacadeRejectsSecondSpatialPackageBeforeStateMutation) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("first", "state/first");
  pops::add_compiled_model<Dim>(system, "first", advection_model<Dim>(), "minmod", "rusanov",
                                "conservative", "imex");
  system.set_conservative_state("first", std::vector<double>(cell_count(config.shape), 1.0));
  const std::vector<double> accepted = system.block_level_state_global("first", 0);
  ASSERT_EQ(system.n_blocks(), 1);

  EXPECT_THROW(system.install_block_state_route("second", "state/second"), std::logic_error);
  EXPECT_EQ(system.n_blocks(), 1);
  EXPECT_EQ(system.block_level_state_global("first", 0), accepted);
}

TEST(test_amr_multiblock_imex, ImexMetadataNeverCreatesAnImplicitTemporalFallback) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>(), "minmod", "rusanov",
                                "conservative", "imex");
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  const std::vector<double> accepted = system.block_level_state_global("tracer", 0);

  EXPECT_THROW(system.step(0.1), std::logic_error);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted);
}
