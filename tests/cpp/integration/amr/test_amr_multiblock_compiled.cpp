#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <int Dim>
struct AdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  Law law;
  pops::Real charge = pops::Real(0);

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.amr.multiblock.compiled.advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
    contract.scalar(charge);
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
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& value) const {
    return law.make_conservative(value);
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
  POPS_HD pops::Real elliptic_rhs(const State& state) const { return charge * state[0]; }
};

template <int Dim>
AdvectionModel<Dim> model(pops::Real velocity, pops::Real charge) {
  pops::RealVector<Dim> values{};
  values[0] = velocity;
  return {pops::nd::ScalarAdvection<Dim>::prepare(values), charge};
}

template <int Dim>
pops::AmrSystemConfig<Dim> config() {
  pops::AmrSystemConfig<Dim> result;
  result.level_count = 1;
  result.transition_ratios.clear();
  result.transition_buffers.clear();
  result.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis) {
    result.shape[axis] = 4;
    result.periodicity[axis] = true;
  }
  return result;
}

template <int Dim>
std::size_t cells(const pops::AmrSystemConfig<Dim>& config) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(config.shape[axis]);
  return result;
}

template <int Dim>
void add_block(pops::AmrSystem<Dim>& system, const std::string& name, pops::Real velocity) {
  const pops::Real charge = name == "ion" ? pops::Real(2) : pops::Real(-1);
  pops::add_compiled_model<Dim>(system, name, model<Dim>(velocity, charge), "minmod", "rusanov",
                                "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.amr.multiblock.compiled/physical_flux");
}

}  // namespace

TEST(test_amr_multiblock_compiled, TwoCompiledBlocksUseOneTransactionalCarrier) {
#if defined(POPS_HAS_KOKKOS)
  int argc = 0;
  char** argv = nullptr;
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  constexpr int Dim = pops::kNativeDimension;
  const auto cfg = config<Dim>();
  pops::AmrSystem<Dim> system(cfg);
  pops::test::install_amr_runtime_authority(system, "tests.amr.multiblock.compiled/runtime@1");
  system.install_block_state_route("ion", "state/ion");
  system.install_block_state_route("neutral", "state/neutral");
  add_block(system, "ion", pops::Real(0.25));
  add_block(system, "neutral", pops::Real(-0.125));
  system.set_conservative_state("ion", std::vector<double>(cells(cfg), 1.0));
  system.set_conservative_state("neutral", std::vector<double>(cells(cfg), 3.0));

  system.install_prepared_amr_coupling_operator(
      "tests.amr.multiblock.compiled/exchange", pops::CouplingOperatorView{"exchange"},
      [](pops::Real dt, const std::vector<pops::MultiFab<Dim>*>& states) {
        for (std::size_t local = 0; local < states[0]->local_size(); ++local) {
          auto first = states[0]->fab(local).view();
          auto second = states[1]->fab(local).view();
          pops::for_each_cell(
              states[0]->box(local), KOKKOS_LAMBDA(const pops::Index<Dim>& cell) {
                const pops::Real amount = dt * pops::Real(0.25) * first(cell);
                first(cell) -= amount;
                second(cell) += amount;
              });
        }
        if (dt > pops::Real(0.5))
          throw std::runtime_error("injected compiled coupling failure");
      });

  ASSERT_EQ(system.n_blocks(), 2);
  system.set_program_block_map({1, 0});
  pops::MultiFab<Dim> poisson_rhs(system.prepared_amr_block_state(0, 0));
  poisson_rhs.set_val(pops::Real(0));
  system.add_prepared_amr_poisson_rhs(0, poisson_rhs);
  EXPECT_NEAR(pops::reduce_min_local(poisson_rhs), -1.0, 1e-14);
  EXPECT_NEAR(pops::reduce_max_local(poisson_rhs), -1.0, 1e-14);

  pops::MultiFab<Dim> neutral(system.prepared_amr_block_state(1, 0));
  pops::MultiFab<Dim> ion(system.prepared_amr_block_state(0, 0));
  const pops::Real total_before = pops::reduce_sum_local(ion) + pops::reduce_sum_local(neutral);
  std::vector<pops::MultiFab<Dim>*> candidates{&neutral, &ion};
  const pops::runtime::multiblock::BoundaryEvaluationPoint accepted_point{
      "test.amr.multiblock.compiled", 0, 0, 0, 0, {0, 1}, 0.4, 0.0};
  EXPECT_EQ(system.apply_prepared_amr_program_candidates(0, pops::Real(0.4), candidates,
                                                         accepted_point, nullptr),
            1U);
  EXPECT_NEAR(pops::reduce_sum_local(ion) + pops::reduce_sum_local(neutral), total_before, 1e-12);
  system.publish_prepared_amr_program_candidates(0, candidates);
  EXPECT_NEAR(pops::reduce_min_local(system.prepared_amr_block_state(0, 0)), 0.9, 1e-14);
  EXPECT_NEAR(pops::reduce_min_local(system.prepared_amr_block_state(1, 0)), 3.1, 1e-14);

  neutral = system.prepared_amr_block_state(1, 0);
  ion = system.prepared_amr_block_state(0, 0);
  std::vector<pops::MultiFab<Dim>*> malformed{&neutral, &ion};
  if (pops::my_rank() == 0)
    malformed.front() = nullptr;
  EXPECT_ANY_THROW(system.publish_prepared_amr_program_candidates(0, malformed));
  EXPECT_NEAR(pops::reduce_min_local(system.prepared_amr_block_state(0, 0)), 0.9, 1e-14);
  EXPECT_NEAR(pops::reduce_min_local(system.prepared_amr_block_state(1, 0)), 3.1, 1e-14);

  const pops::runtime::multiblock::BoundaryEvaluationPoint rejected_point{
      "test.amr.multiblock.compiled", 0, 0, 0, 0, {0, 1}, 0.8, 0.0};
  EXPECT_THROW(system.apply_prepared_amr_program_candidates(0, pops::Real(0.8), candidates,
                                                            rejected_point, nullptr),
               std::runtime_error);
  EXPECT_NEAR(pops::reduce_min_local(ion), 0.9, 1e-14);
  EXPECT_NEAR(pops::reduce_min_local(neutral), 3.1, 1e-14);
  EXPECT_NEAR(pops::reduce_min_local(system.prepared_amr_block_state(0, 0)), 0.9, 1e-14);
  EXPECT_NEAR(pops::reduce_min_local(system.prepared_amr_block_state(1, 0)), 3.1, 1e-14);
}
