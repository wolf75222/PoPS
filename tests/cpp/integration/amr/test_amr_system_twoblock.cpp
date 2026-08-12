#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/system/system_coupling_registry.hpp>

#include <string>
#include <vector>

namespace {

template <int Dim>
struct ConstantModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  Law law;
  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.amr.system-twoblock.constant", 1};
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
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

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
void install(pops::AmrSystem<Dim>& system, const std::string& name) {
  pops::RealVector<Dim> velocity{};
  ConstantModel<Dim> model{pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
  pops::add_compiled_model<Dim>(system, name, model, "minmod", "rusanov", "conservative",
                                "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
                                {}, {}, 0.0, static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.amr.system-twoblock/physical_flux");
}

template <int Dim>
std::size_t cells(const pops::AmrSystemConfig<Dim>& cfg) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(cfg.shape[axis]);
  return result;
}

}  // namespace

TEST(test_amr_system_twoblock, AcceptedStatesRollbackTogether) {
  constexpr int Dim = pops::kNativeDimension;
  const auto cfg = config<Dim>();
  pops::AmrSystem<Dim> system(cfg);
  system.install_block_state_route("a", "state/a");
  system.install_block_state_route("b", "state/b");
  install(system, "a");
  install(system, "b");
  system.set_conservative_state("a", std::vector<double>(cells(cfg), 2.0));
  system.set_conservative_state("b", std::vector<double>(cells(cfg), 5.0));
  system.install_program_step([](double) {});
  system.set_program_block_map({0, 1});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.amr.system-twoblock/manual-program", std::vector<FluxBudget>{{8, 16}, {0, 0}}, 0, 0);
  const auto& prepared_budget = system.prepared_amr_program_flux_expression_budget();
  ASSERT_EQ(prepared_budget.blocks.size(), 2U);
  EXPECT_EQ(prepared_budget.blocks[1].rhs_basis_bound, 0U);
  EXPECT_EQ(prepared_budget.program_block_map.canonical_indices, (std::vector<std::size_t>{0, 1}));

  pops::MultiFab<Dim> first(system.prepared_amr_block_state(0, 0));
  pops::MultiFab<Dim> second(system.prepared_amr_block_state(1, 0));
  first.set_val(pops::Real(7));
  second.set_val(pops::Real(11));
  std::vector<pops::MultiFab<Dim>*> candidates{&first, &second};

  system.begin_step_transaction();
  system.publish_prepared_amr_program_candidates(0, candidates);
  EXPECT_EQ(pops::reduce_min_local(system.prepared_amr_block_state(0, 0)), pops::Real(7));
  EXPECT_EQ(pops::reduce_min_local(system.prepared_amr_block_state(1, 0)), pops::Real(11));
  system.rollback_step_transaction();
  EXPECT_EQ(pops::reduce_min_local(system.prepared_amr_block_state(0, 0)), pops::Real(2));
  EXPECT_EQ(pops::reduce_min_local(system.prepared_amr_block_state(1, 0)), pops::Real(5));
}

TEST(test_amr_system_twoblock, SingleBlockIsTheNEqualsOneCarrierCase) {
  constexpr int Dim = pops::kNativeDimension;
  const auto cfg = config<Dim>();
  pops::AmrSystem<Dim> system(cfg);
  system.install_block_state_route("only", "state/only");
  install(system, "only");
  system.set_conservative_state("only", std::vector<double>(cells(cfg), 3.0));
  system.set_program_block_map({0});

  ASSERT_EQ(system.prepared_amr_program_block_map().canonical_indices.size(), 1U);
  pops::MultiFab<Dim> candidate(system.prepared_amr_block_state(0, 0));
  candidate.set_val(pops::Real(4));
  std::vector<pops::MultiFab<Dim>*> candidates{&candidate};
  system.publish_prepared_amr_program_candidates(0, candidates);
  EXPECT_EQ(pops::reduce_min_local(system.prepared_amr_block_state(0, 0)), pops::Real(4));
}

TEST(test_amr_system_twoblock, MalformedConservationOwnerCannotPublishMaterialization) {
  constexpr int Dim = pops::kNativeDimension;
  const auto cfg = config<Dim>();
  pops::AmrSystem<Dim> system(cfg);
  system.install_block_state_route("a", "state/a");
  system.install_block_state_route("b", "state/b");
  install(system, "a");
  install(system, "b");
  system.set_conservative_state("a", std::vector<double>(cells(cfg), 2.0));
  system.set_conservative_state("b", std::vector<double>(cells(cfg), 5.0));

  using Group = pops::runtime::system::PreparedCouplingConservationGroup;
  using Operator = typename pops::AmrSystem<Dim>::PreparedCouplingOperator;
  Operator malformed(
      [](pops::Real, const std::vector<pops::MultiFab<Dim>*>&) {},
      std::vector<Group>{{"scalar-total", {{"not-a", 0, 0, "scalar"}, {"b", 1, 0, "scalar"}}}});
  system.install_prepared_amr_coupling_operator("tests.amr.system-twoblock/malformed-owner",
                                                pops::CouplingOperatorView{"malformed-owner"},
                                                std::move(malformed));

  EXPECT_ANY_THROW((void)system.prepared_amr_block_state(0, 0));
  EXPECT_ANY_THROW((void)system.prepared_amr_block_state(1, 0));
}
