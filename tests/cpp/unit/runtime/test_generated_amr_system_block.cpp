#include <gtest/gtest.h>

#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/core/foundation/native_dimension.hpp>

#include <string>
#include <type_traits>

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

TEST(GeneratedAmrSystemBlock, MissingFacadeSeamFailsBeforeMutation) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  pops::AmrSystem<Dim> system(config);
  ASSERT_EQ(system.n_blocks(), 0);

  EXPECT_THROW(pops::add_compiled_model<Dim>(system, "tracer", advection_model<Dim>(), "minmod",
                                             "rusanov", "conservative", "explicit", 1.4),
               std::runtime_error);
  EXPECT_EQ(system.n_blocks(), 0);
  EXPECT_TRUE(system.block_names().empty());
}

}  // namespace
