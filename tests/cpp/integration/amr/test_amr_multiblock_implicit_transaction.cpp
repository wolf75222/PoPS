#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kImplicitTransactionRetryReason = 0x49545259u;

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
struct RelaxingAdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;

  Law law{};
  pops::Real rate = pops::Real(0);
  pops::Real equilibrium = pops::Real(0);

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.implicit-transaction.relaxing-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
    contract.scalar(rate).scalar(equilibrium);
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
  POPS_HD State source(const State& state, const pops::ProviderValues<0>&) const {
    return State{rate * (equilibrium - state[0])};
  }
  POPS_HD void source_jacobian(const State&, const pops::ProviderValues<0>&,
                               pops::Real (&jacobian)[1][1]) const {
    jacobian[0][0] = -rate;
  }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
RelaxingAdvectionModel<Dim> relaxing_model(pops::Real rate, pops::Real equilibrium) {
  pops::RealVector<Dim> velocity{};
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity), rate, equilibrium};
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
bool byte_exact_equal(const pops::MultiFab<Dim>& left, const pops::MultiFab<Dim>& right) {
  bool same = left.layout() == right.layout() && left.distribution() == right.distribution() &&
              left.local_rank() == right.local_rank() && left.local_size() == right.local_size() &&
              left.ncomp() == right.ncomp() && left.ghosts() == right.ghosts();
  for (std::size_t local = 0; same && local < left.local_size(); ++local) {
    const auto& left_fab = left.fab(local);
    const auto& right_fab = right.fab(local);
    auto left_host = left_fab.create_host_mirror();
    auto right_host = right_fab.create_host_mirror();
    left_fab.copy_to_host(left_host);
    right_fab.copy_to_host(right_host);
    if (left_host.size() != right_host.size())
      same = false;
    for (std::size_t entry = 0; same && entry < left_host.size(); ++entry)
      same = std::bit_cast<std::array<std::byte, sizeof(pops::Real)>>(left_host(entry)) ==
             std::bit_cast<std::array<std::byte, sizeof(pops::Real)>>(right_host(entry));
  }
  return pops::all_reduce_max(same ? 0L : 1L) == 0;
}

}  // namespace

TEST(test_amr_multiblock_implicit_transaction,
     RankedImplicitSolveRequiresAnExplicitLaneAndDefersPublication) {
  constexpr int Dim = pops::kNativeDimension;
  auto state = one_patch_field<Dim>(2, 1);
  state.set_val(pops::Real(2));
  const pops::ExecutionLane lane = pops::ExecutionLane::world();

  pops::SolveOutcome outcome = pops::backward_euler_source(
      StiffLinearSource{}, [](std::size_t) { return pops::ProviderStorageView<Dim, 0>{}; }, state,
      pops::Real(0.25), pops::NewtonOptions{}, lane);
  ASSERT_TRUE(outcome.report().solved_value_available());
  EXPECT_EQ(pops::reduce_min_local(state), pops::Real(2));
  const pops::SolveReport accepted = outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
  EXPECT_NEAR(static_cast<double>(pops::reduce_min_local(state)), 1.6,
              64.0 * std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(static_cast<double>(pops::reduce_max_local(state)), 1.6,
              64.0 * std::numeric_limits<double>::epsilon());
}

TEST(test_amr_multiblock_implicit_transaction,
     TwoBlockImplicitSourceTransactionRollsBackAndRetriesExactly) {
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
                                            "tests.implicit-transaction/multiblock-runtime@1");
  system.install_block_state_route("slow", "state/slow");
  system.install_block_state_route("fast", "state/fast");
  pops::add_compiled_model<Dim>(system, "slow", relaxing_model<Dim>(pops::Real(8), pops::Real(2)),
                                "minmod", "rusanov", "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.implicit-transaction/slow/physical-flux");
  pops::add_compiled_model<Dim>(system, "fast", relaxing_model<Dim>(pops::Real(24), pops::Real(-1)),
                                "minmod", "rusanov", "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.implicit-transaction/fast/physical-flux");
  system.set_conservative_state("slow", std::vector<double>(cell_count(config.shape), 0.25));
  system.set_conservative_state("fast", std::vector<double>(cell_count(config.shape), 3.0));
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  auto inject_retry = std::make_shared<bool>(true);
  context->configure_primary_clock("tests.implicit-transaction.multiblock-clock");
  context->install(
      [context, inject_retry, &system](double macro_dt) {
        context->advance_hierarchy(macro_dt, [context, inject_retry, &system](double level_dt) {
          std::array<pops::MultiFab<Dim>*, 2> accepted{};
          std::array<pops::MultiFab<Dim>*, 2> candidates{};
          context->set_stage_time(0, 1);
          for (int program_block = 0; program_block < 2; ++program_block) {
            accepted[program_block] = &context->state(program_block);
            auto& candidate =
                context->scratch_state(1000 + program_block, 0, *accepted[program_block]);
            // This is intentionally a source-only, atomic two-block transaction.  Do not add a
            // numerically-zero transport stage: temporal composition belongs to authored Program
            // evidence, while this fixture proves the implicit solve/retry/commit contract.
            context->lincomb(candidate, pops::Real(1), *accepted[program_block], pops::Real(0),
                             *accepted[program_block]);
            pops::SolveOutcome implicit = context->solve_source_default(
                program_block, candidate, pops::Real(level_dt), pops::NewtonOptions{});
            const pops::SolveReport report = implicit.consume(pops::SolveConsumption::kAccept);
            if (!report.solved())
              throw std::runtime_error("two-block implicit source did not converge");
            candidates[program_block] = &candidate;
          }
          if (*inject_retry) {
            *inject_retry = false;
            throw pops::runtime::program::StepAttemptRejected(
                pops::SolveStatus::kIterationLimit,
                pops::runtime::program::StepAttemptDisposition::kRetry,
                kImplicitTransactionRetryReason, "implicit-source",
                "injected-implicit-transaction-retry");
          }
          context->commit_many({{accepted[0], candidates[0]}, {accepted[1], candidates[1]}});
          // This one-level fixture publishes within the actual accepted level interval.
          context->stage_exchange({"test.nested.amr", "accepted.source",
                                   std::to_string(system.macro_step()), "backward_euler", 1, 1.0,
                                   1.0, level_dt, 1});
        });
      },
      context);
  // Installing a whole-system Program resets its unverified binding image.  Bind this Program
  // order only after that installation so the rollback snapshot and the prepared budget carry
  // the same exact non-positional map.
  system.set_program_block_map({1, 0});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.implicit-transaction.multiblock-program-v1", std::vector<FluxBudget>{{1, 1}, {1, 1}},
      0, 0);

  const pops::MultiFab<Dim> slow_before = system.prepared_amr_block_state(0, 0);
  const pops::MultiFab<Dim> fast_before = system.prepared_amr_block_state(1, 0);
  constexpr double dt = 0.05;
  try {
    system.step(dt);
    FAIL() << "the full-pack retry was not surfaced";
  } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
    EXPECT_EQ(rejected.disposition(), pops::runtime::program::StepAttemptDisposition::kRetry);
    EXPECT_EQ(rejected.reason_code(), kImplicitTransactionRetryReason);
    EXPECT_EQ(rejected.phase(), "implicit-source");
  }
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_TRUE(byte_exact_equal(slow_before, system.prepared_amr_block_state(0, 0)));
  EXPECT_TRUE(byte_exact_equal(fast_before, system.prepared_amr_block_state(1, 0)));

  ASSERT_NO_THROW(system.step(dt));
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.time(), dt);
  const auto slow_after = system.block_level_state_global("slow", 0);
  const auto fast_after = system.block_level_state_global("fast", 0);
  ASSERT_FALSE(slow_after.empty());
  ASSERT_FALSE(fast_after.empty());
  EXPECT_NEAR(slow_after.front(), (0.25 + dt * 8.0 * 2.0) / (1.0 + dt * 8.0), 1.0e-12);
  EXPECT_NEAR(fast_after.front(), (3.0 + dt * 24.0 * -1.0) / (1.0 + dt * 24.0), 1.0e-12);
  EXPECT_NE(slow_after.front(), fast_after.front());

  // Inner successful substeps remain provisional to their outer acceptance boundary.
  const auto accepted_exchange = system.program_exchange_records();
  ASSERT_EQ(accepted_exchange.size(), 1u);
  auto nested_steps = [&] {
    for (double sub_dt : {0.025, 0.075}) {
      system.begin_nested_step_transaction();
      EXPECT_EQ(system.step_transaction_depth(), 2u);
      system.step(sub_dt);
      system.commit_step_transaction();
      system.finalize_step_transaction();
      EXPECT_EQ(system.step_transaction_depth(), 1u);
    }
  };
  system.begin_step_transaction();
  nested_steps();
  const auto attempted_slow = system.block_level_state_global("slow", 0);
  const auto attempted_fast = system.block_level_state_global("fast", 0);
  ASSERT_EQ(system.program_exchange_records().size(), 2u);
  system.rollback_step_transaction();
  EXPECT_EQ(system.step_transaction_depth(), 0u);
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.time(), dt);
  EXPECT_EQ(system.block_level_state_global("slow", 0), slow_after);
  EXPECT_EQ(system.block_level_state_global("fast", 0), fast_after);
  ASSERT_EQ(system.program_exchange_records().size(), 1u);
  EXPECT_EQ(system.program_exchange_records().front().evaluation_context,
            accepted_exchange.front().evaluation_context);

  system.begin_step_transaction();
  nested_steps();
  system.commit_step_transaction();
  system.finalize_step_transaction();
  EXPECT_EQ(system.step_transaction_depth(), 0u);
  EXPECT_EQ(system.macro_step(), 3);
  EXPECT_NEAR(system.time(), dt + 0.025 + 0.075, 1e-15);
  EXPECT_EQ(system.block_level_state_global("slow", 0), attempted_slow);
  EXPECT_EQ(system.block_level_state_global("fast", 0), attempted_fast);
  ASSERT_EQ(system.program_exchange_records().size(), 2u);
}

TEST(test_amr_multiblock_implicit_transaction,
     CoupledJacvecPreservesUnrelatedAttemptCarrierWithPermutedProgramPair) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  config.shape = uniform_extent<Dim>(8);
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.coupled-jacvec/complete-carrier@1");
  const std::array<std::string, 3> names{"aux", "left", "right"};
  for (const auto& name : names)
    system.install_block_state_route(name, "state/" + name);
  for (const auto& name : names)
    pops::add_compiled_model<Dim>(system, name, relaxing_model<Dim>(pops::Real(0), pops::Real(0)),
                                  "minmod", "rusanov", "conservative", "explicit",
                                  static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {},
                                  0.0, static_cast<double>(pops::kWenoEpsilon), false,
                                  "tests.coupled-jacvec/" + name + "/physical-flux");
  const std::array<double, 3> initial{7.0, 2.0, 3.0};
  for (std::size_t block = 0; block < names.size(); ++block)
    system.set_conservative_state(names[block],
                                  std::vector<double>(cell_count(config.shape), initial[block]));

  auto inject_failure = std::make_shared<bool>(true);
  auto coupling_calls = std::make_shared<int>(0);
  auto injected_locally = std::make_shared<bool>(false);
  system.install_prepared_amr_coupling_operator(
      "tests.coupled-jacvec/aux-qualified-exchange@1",
      pops::CouplingOperatorView{"aux-qualified-exchange"},
      [inject_failure, coupling_calls, injected_locally](
          pops::Real dt, const std::vector<pops::MultiFab<Dim>*>& states) {
        ++*coupling_calls;
        if (states.size() != 3)
          throw std::runtime_error("paired Jacobian lost its complete canonical carrier");
        for (std::size_t local = 0; local < states[0]->local_size(); ++local) {
          const auto aux = states[0]->fab(local).view();
          const auto left = states[1]->fab(local).view();
          const auto right = states[2]->fab(local).view();
          pops::for_each_cell(
              states[0]->box(local), KOKKOS_LAMBDA(const pops::Index<Dim>& cell) {
                const pops::Real amount = dt * aux(cell) * (right(cell) - left(cell));
                left(cell) -= amount;
                right(cell) += amount;
                // Even an unrelated output is detached from the live attempt carrier.
                aux(cell) += dt;
              });
        }
        Kokkos::fence();
        if (*inject_failure && pops::my_rank() == 0) {
          *injected_locally = true;
          throw std::runtime_error("injected rank-local paired coupling failure");
        }
      });

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("tests.coupled-jacvec.complete-carrier-clock");
  context->install(
      [context, inject_failure, coupling_calls, injected_locally, &system](double macro_dt) {
        context->advance_hierarchy(macro_dt, [context, inject_failure, coupling_calls,
                                              injected_locally, &system](double) {
          context->set_stage_time(0, 1);
          auto& right = context->state(0);
          auto& aux = context->state(1);
          auto& left = context->state(2);
          auto& first = context->scratch_state(1100, 0, right);
          auto& second = context->scratch_state(1101, 0, left);
          auto& first_result = context->rhs_scratch(1102, 0, right);
          auto& second_result = context->rhs_scratch(1103, 0, left);
          context->lincomb(first, pops::Real(1), right, pops::Real(0), right);
          context->lincomb(second, pops::Real(1), left, pops::Real(0), left);
          const pops::MultiFab<Dim> right_before(right);
          const pops::MultiFab<Dim> left_before(left);
          const pops::MultiFab<Dim> original_aux(aux);
          const auto point = context->boundary_evaluation_point(0);
          EXPECT_THROW(context->rhs_jacvec_pair_into_at(point, 0, first, first_result, true, 0,
                                                        second, second_result, true),
                       std::invalid_argument);
          EXPECT_THROW(context->rhs_jacvec_pair_into_at(point, 0, first, first_result, true, 3,
                                                        second, second_result, true),
                       std::out_of_range);
          EXPECT_THROW(context->rhs_jacvec_pair_into_at(point, 0, first, first_result, true, 1,
                                                        second, second_result, true),
                       std::invalid_argument);
          auto foreign_point = point;
          ++foreign_point.level;
          EXPECT_THROW(context->rhs_jacvec_pair_into_at(foreign_point, 0, first, first_result, true,
                                                        2, second, second_result, true),
                       std::invalid_argument);
          const auto apply_pair = [&] {
            context->rhs_jacvec_pair_into_at(point, 0, first, first_result, true, 2, second,
                                             second_result, true);
          };
          for (pops::Real value : {pops::Real(7), pops::Real(11)}) {
            aux.set_val(value);
            const pops::MultiFab<Dim> aux_before(aux);
            first.set_val(pops::Real(3));
            second.set_val(pops::Real(2));
            first_result.set_val(pops::Real(-99));
            second_result.set_val(pops::Real(-99));
            if (*inject_failure) {
              try {
                apply_pair();
                FAIL() << "the rank-local coupling failure was not surfaced";
              } catch (const std::runtime_error& error) {
                const std::string expected =
                    context->prepared_execution_lane().size() == 1
                        ? "injected rank-local paired coupling failure"
                        : "prepared AMR coupling failed and rolled back collectively";
                EXPECT_EQ(std::string(error.what()), expected);
              }
              EXPECT_EQ(*coupling_calls, 1);
              EXPECT_EQ(*injected_locally, pops::my_rank() == 0);
              *inject_failure = false;
              EXPECT_TRUE(byte_exact_equal(aux_before, aux));
              EXPECT_EQ(pops::reduce_min_local(first_result), pops::Real(-99));
              EXPECT_EQ(pops::reduce_max_local(second_result), pops::Real(-99));
            }
            ASSERT_NO_THROW(apply_pair());
            EXPECT_EQ(pops::reduce_min_local(first_result), value);
            EXPECT_EQ(pops::reduce_max_local(first_result), value);
            EXPECT_EQ(pops::reduce_min_local(second_result), -value);
            EXPECT_EQ(pops::reduce_max_local(second_result), -value);
            const pops::MultiFab<Dim> base_first(first_result);
            const pops::MultiFab<Dim> base_second(second_result);
            // The zero perturbation has exactly zero residual difference, including the lift
            // from the unrelated current attempt input. Reusing scratch does not accumulate it.
            ASSERT_NO_THROW(apply_pair());
            EXPECT_TRUE(byte_exact_equal(base_first, first_result));
            EXPECT_TRUE(byte_exact_equal(base_second, second_result));
            first.set_val(pops::Real(4));
            ASSERT_NO_THROW(apply_pair());
            EXPECT_EQ(pops::reduce_min_local(first_result), pops::Real(2) * value);
            EXPECT_EQ(pops::reduce_max_local(second_result), pops::Real(-2) * value);
            EXPECT_TRUE(byte_exact_equal(aux_before, aux));
            EXPECT_TRUE(byte_exact_equal(right_before, right));
            EXPECT_TRUE(byte_exact_equal(left_before, left));
            EXPECT_EQ(pops::reduce_min_local(system.prepared_amr_block_state(0, 0)), pops::Real(7));
          }
          for (std::size_t local = 0; local < aux.local_size(); ++local)
            Kokkos::deep_copy(aux.fab(local).storage(), original_aux.fab(local).storage());
          Kokkos::fence();
        });
      },
      context);
  // Program pair {0,2} is non-prefix and maps to runtime {right,left}; aux sits between it.
  system.set_program_block_map({2, 0, 1});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.coupled-jacvec.complete-carrier-program-v1",
      std::vector<FluxBudget>{{1, 1}, {0, 0}, {1, 1}}, 0, 0);
  const auto ledger_budget = system.prepared_amr_interface_flux_ledger_budget();
  EXPECT_EQ(ledger_budget.max_transaction_depth, 2u);
  pops::runtime::multiblock::InterfaceFluxFragmentLedger depth_probe(0, ledger_budget);
  depth_probe.begin();
  auto evaluation = depth_probe.prepare_evaluation_begin();
  depth_probe.publish_prepared_begin(evaluation);
  EXPECT_THROW(depth_probe.begin(), std::runtime_error);
  EXPECT_THROW(depth_probe.commit(), std::runtime_error);
  depth_probe.rollback();
  depth_probe.commit();
  EXPECT_EQ(depth_probe.published_size(), 0u);
  const pops::MultiFab<Dim> aux_before(system.prepared_amr_block_state(0, 0));
  const pops::MultiFab<Dim> left_before(system.prepared_amr_block_state(1, 0));
  const pops::MultiFab<Dim> right_before(system.prepared_amr_block_state(2, 0));
  ASSERT_NO_THROW(system.step(0.125));
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.time(), 0.125);
  EXPECT_TRUE(byte_exact_equal(aux_before, system.prepared_amr_block_state(0, 0)));
  EXPECT_TRUE(byte_exact_equal(left_before, system.prepared_amr_block_state(1, 0)));
  EXPECT_TRUE(byte_exact_equal(right_before, system.prepared_amr_block_state(2, 0)));
}

TEST(test_amr_multiblock_implicit_transaction,
     MultilevelPairBudgetReservesOneEvaluationWithoutAcceptedApplications) {
  constexpr int Dim = pops::kNativeDimension;
  using namespace pops::runtime::multiblock;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  config.transition_ratios = {uniform_extent<Dim>(2)};
  config.transition_buffers = {uniform_extent<Dim>(1)};
  config.transition_lookaheads = {uniform_extent<Dim>(1)};
  config.shape = uniform_extent<Dim>(8);
  // The interface requires unique ownership even when a peer owns no face cells.
  config.distribute_coarse = true;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.pair-budget/multilevel-runtime@1");
  const std::array<std::string, 2> names{"left", "right"};
  for (const auto& name : names)
    system.install_block_state_route(name, "state/" + name);
  for (const auto& name : names)
    pops::add_compiled_model<Dim>(system, name, relaxing_model<Dim>(pops::Real(0), pops::Real(0)),
                                  "minmod", "rusanov", "conservative", "explicit",
                                  static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {},
                                  0.0, static_cast<double>(pops::kWenoEpsilon), false,
                                  "tests.pair-budget/" + name + "/physical-flux");
  for (const auto& name : names)
    system.set_conservative_state(name, std::vector<double>(cell_count(config.shape), 1.0));
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  const auto fine_shape = uniform_extent<Dim>(16);
  system.rebuild_hierarchy({pops::AmrPatch<Dim>{1, pops::Box<Dim>::from_extents(fine_shape)}}, {0});
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("tests.pair-budget.clock");
  context->install(
      [](double) {
        throw std::logic_error("budget fixture does not execute an authored time step");
      },
      context);
  system.set_program_block_map({1, 0});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.pair-budget/multilevel-program@1", std::vector<FluxBudget>{{1, 1}, {1, 1}}, 0, 0);
  const auto raw = pops::component::test_support::host_execution_context();
  const pops::component::PreparedExecutionContextV1 parent(
      raw.execution_identity, raw.context_version, raw.memory_space, raw.backend_identity,
      raw.device_identity, raw.scalar_type, raw.storage_precision, raw.compute_precision,
      raw.accumulation_precision, raw.reduction_precision, raw.stream_handle, raw.stream_identity,
      raw.communicator_f_handle, raw.communicator_datatype_f_handle, raw.communicator_identity,
      raw.communicator_datatype_identity);
  const auto execution = parent.for_lane(context->prepared_execution_lane());
  const auto install = [&](const pops::component::PreparedExecutionContextV1& execution) {
    system.install_prepared_amr_interface_flux_provider(
        "tests.pair-budget/interface@1", [&](auto& scheduler) {
          for (int level = 0; level < 2; ++level) {
            AxisAlignedInterface<Dim> route;
            route.identity = "tests.pair-budget/interface";
            route.sampling_provider_identity = "tests.pair-budget/sampler";
            route.level = level;
            route.left_block = 0;
            route.right_block = 1;
            route.left_axis = route.right_axis = 0;
            route.left_side = InterfaceSide::High;
            route.right_side = InterfaceSide::Low;
            route.right_component_for_left = {0};
            route.affine_mapping_identity = "tests.pair-budget/translation";
            route.right_normal_translation = pops::Real(1);
            route.left_trace_projection_identity = "tests.pair-budget/left.trace";
            route.right_trace_projection_identity = "tests.pair-budget/right.trace";
            route.left_trace_provider_identity = "test.cell-average.left";
            route.right_trace_provider_identity = "test.cell-average.right";
            route.left_trace_operation = route.right_trace_operation =
                InterfaceTraceOperation::CellAverage;
            route.left_trace_required_depth = route.right_trace_required_depth = 1;
            const auto geometry = system.prepared_amr_level_geometry(level);
            scheduler.install(
                route, system.prepared_amr_block_state(0, level), geometry,
                system.prepared_amr_block_state(1, level), geometry, execution.view(),
                InterfaceFluxEvaluatorFactory([execution] {
                  return InterfaceFluxEvaluator([execution](const auto&, const auto& batch) {
                    // This exact ABI handle belongs to the original carrier, not the
                    // current hierarchy. Rematerialization must keep its owner alive.
                    pops::component::validate_execution_context(execution.view());
#ifdef POPS_HAS_MPI
                    const pops::CommunicatorView communicator{MPI_Comm_f2c(
                        static_cast<MPI_Fint>(execution.view().communicator_f_handle))};
                    EXPECT_EQ(pops::all_reduce_sum(1L, communicator), communicator.size());
#endif
                    for (int i = 0; i < batch.face_count * batch.component_count; ++i)
                      batch.shared_flux[i] = pops::Real(0);
                  });
                }));
          }
        });
  };
#ifdef POPS_HAS_MPI
  // Equal rank sets and text cannot authenticate another communicator as the retained owner.
  const auto impostor_lane = pops::ExecutionLane::duplicate_collectively(
      context->prepared_execution_lane(), context->prepared_execution_lane().identity());
  const auto impostor_execution = parent.for_lane(impostor_lane);
  const auto before_install = system.program_accepted_state();
  EXPECT_THROW(install(impostor_execution), std::exception);
  EXPECT_EQ(system.program_accepted_state(), before_install);
#endif
  ASSERT_NO_THROW(install(execution));
  ASSERT_EQ(system.n_levels(), 2);
  const auto budget = system.prepared_amr_interface_flux_ledger_budget();
  EXPECT_EQ(budget.max_fragments_per_window, 0u);
  EXPECT_EQ(budget.max_payload_terms_per_window, 0u);
  EXPECT_EQ(budget.max_transaction_depth, 2u);
  EXPECT_EQ(budget.max_evaluation_fragments, 1u);
  EXPECT_EQ(budget.max_evaluation_payload_terms,
            cell_count(fine_shape) / static_cast<std::size_t>(fine_shape[0]));
  const auto before = system.program_accepted_state();
  const auto epoch = system.engine()->topology_epoch();
  InterfaceFluxFragmentLedger ledger(epoch, budget);
  ledger.begin();
  auto begin = ledger.prepare_evaluation_begin();
  ledger.publish_prepared_begin(begin);
  const BoundaryEvaluationPoint point{
      "tests.pair-budget.clock", 0, 1, 1, 23, {0, 1}, 0.0625, 0.0625};
  const pops::amr::ClockWindow window{{1, 0, {1, 2}, 0.0625}, {1, 0, {1, 1}, 0.125}};
  const auto clock = clock_stamp_in_window(point, window);
  EXPECT_EQ(clock.phase, pops::amr::Rational(1, 2));
  EXPECT_DOUBLE_EQ(clock.physical_time, 0.0625);
  const pops::amr::InterfaceFluxFragmentKey probe{
      "tests.pair-budget/interface",
      epoch,
      0,
      1,
      clock,
      "program-jacvec-pair",
      "pops.amr-program.paired-evaluation/tests.pair-budget/multilevel-program@1",
      "rhs-coherence/23",
      "program-pair/0/1/runtime-pair/1/0",
      window,
      pops::amr::InterfaceFluxOrientation::FineOutward};
  const pops::amr::InterfaceFluxFragmentMeasure measure{{1, 1}, 1.0, 0.0625};
  const std::vector<pops::Real> payload(budget.max_evaluation_payload_terms, pops::Real(2));
  ASSERT_NO_THROW(ledger.accumulate(probe, measure, payload));
  EXPECT_THROW(ledger.commit(), std::runtime_error);
  auto extra = probe;
  extra.application_identity = "second-evaluation";
  EXPECT_THROW(ledger.accumulate(extra, measure, payload), std::length_error);
  ledger.rollback();
  EXPECT_EQ(ledger.pending_size(), 0u);
  ledger.commit();
  EXPECT_EQ(ledger.published_size(), 0u);
  ledger.begin();
  EXPECT_THROW(ledger.accumulate(probe, measure, payload), std::length_error);
  ledger.rollback();
  EXPECT_EQ(system.program_accepted_state(), before);

  // The same full-carrier rebuild used by restart must retain the installed interface recipe.
  // Give this Program an exact accepted-application bound so a lost provider cannot silently
  // change an authenticated nonzero checkpoint fragment capacity into an inactive zero budget.
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.pair-budget/multilevel-program@1", std::vector<FluxBudget>{{1, 1}, {1, 1}}, 2, 128);
  const auto before_rebuild = system.prepared_amr_interface_flux_ledger_budget();
  ASSERT_EQ(before_rebuild.max_fragments_per_window, 6u);
  ASSERT_GT(before_rebuild.max_payload_terms_per_window, 0u);
  ASSERT_NO_THROW(system.rebuild_hierarchy(
      {pops::AmrPatch<Dim>{1, pops::Box<Dim>::from_extents(fine_shape)}}, {0}));
  ASSERT_NO_THROW(system.rebuild_hierarchy(
      {pops::AmrPatch<Dim>{1, pops::Box<Dim>::from_extents(fine_shape)}}, {0}));
#ifdef POPS_HAS_MPI
  int communicator_relation = MPI_UNEQUAL;
  ASSERT_EQ(
      MPI_Comm_compare(MPI_Comm_f2c(static_cast<MPI_Fint>(execution.view().communicator_f_handle)),
                       context->prepared_execution_lane().native_handle(), &communicator_relation),
      MPI_SUCCESS);
  EXPECT_EQ(communicator_relation, MPI_CONGRUENT);
#endif
  const auto after_rebuild = system.prepared_amr_interface_flux_ledger_budget();
  EXPECT_EQ(after_rebuild.max_fragments_per_window, before_rebuild.max_fragments_per_window);
  EXPECT_EQ(after_rebuild.max_payload_terms_per_window,
            before_rebuild.max_payload_terms_per_window);
  EXPECT_EQ(after_rebuild.max_transaction_depth, before_rebuild.max_transaction_depth);
  EXPECT_EQ(after_rebuild.max_evaluation_fragments, before_rebuild.max_evaluation_fragments);
  EXPECT_EQ(after_rebuild.max_evaluation_payload_terms,
            before_rebuild.max_evaluation_payload_terms);
  for (int level = 0; level < 2; ++level)
    EXPECT_EQ(system.interface_evaluation_count("tests.pair-budget/interface", level), 0u);

  const auto capture = [&] {
    for (int level = 0; level < 2; ++level) {
      auto& left = system.prepared_amr_block_state(0, level);
      auto& right = system.prepared_amr_block_state(1, level);
      pops::MultiFab<Dim> left_rhs(left.layout(), left.distribution(), left.local_rank(),
                                   left.ncomp(), left.ghosts());
      pops::MultiFab<Dim> right_rhs(right.layout(), right.distribution(), right.local_rank(),
                                    right.ncomp(), right.ghosts());
      left_rhs.set_val(pops::Real(0));
      right_rhs.set_val(pops::Real(0));
      // This fixture deliberately installs the reverse Program-to-runtime block map.
      const std::array<pops::MultiFab<Dim>*, 2> states{&right, &left};
      const std::array<pops::MultiFab<Dim>*, 2> rhs{&right_rhs, &left_rhs};
      auto evaluation = point;
      evaluation.level = level;
      const auto samples = system.capture_prepared_amr_interface_residual(evaluation, states, rhs);
      ASSERT_EQ(samples.size(), 1u);
      for (const auto flux : samples.front().flux_density)
        EXPECT_EQ(flux, pops::Real(0));
      EXPECT_EQ(pops::reduce_sum_local(left_rhs, 0), pops::Real(0));
      EXPECT_EQ(pops::reduce_sum_local(right_rhs, 0), pops::Real(0));
    }
  };
  const auto before_capture = system.program_accepted_state();
  ASSERT_NO_THROW(capture());
  EXPECT_EQ(system.program_accepted_state(), before_capture);
  for (int level = 0; level < 2; ++level)
    EXPECT_EQ(system.interface_evaluation_count("tests.pair-budget/interface", level), 1u);

  // A candidate that drops the declared high face must fail before replacing the live provider.
  const auto rebuilt_image = system.program_accepted_state();
  const auto rebuilt_fine = system.block_level_state_global("left", 1);
  auto incomplete_shape = fine_shape;
  incomplete_shape[0] /= 2;
  EXPECT_THROW(system.rebuild_hierarchy(
                   {pops::AmrPatch<Dim>{1, pops::Box<Dim>::from_extents(incomplete_shape)}}, {0}),
               std::exception);
  EXPECT_EQ(system.prepared_amr_interface_flux_ledger_budget().exact_contract,
            after_rebuild.exact_contract);
  EXPECT_EQ(system.program_accepted_state(), rebuilt_image);
  EXPECT_EQ(system.block_level_state_global("left", 1), rebuilt_fine);
  for (int level = 0; level < 2; ++level)
    EXPECT_EQ(system.interface_evaluation_count("tests.pair-budget/interface", level), 1u);
  ASSERT_NO_THROW(capture());
  EXPECT_EQ(system.program_accepted_state(), rebuilt_image);
  for (int level = 0; level < 2; ++level)
    EXPECT_EQ(system.interface_evaluation_count("tests.pair-budget/interface", level), 2u);
}

TEST(test_amr_multiblock_implicit_transaction, MetadataNeverCreatesAnImplicitTemporalFallback) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(
      system, "tests.implicit-transaction/no-native-dispatch-runtime@1");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", relaxing_model<Dim>(pops::Real(1), pops::Real(0)),
                                "minmod", "rusanov", "conservative", "imex",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.implicit-transaction.no-native-dispatch/physical-flux");
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  const std::vector<double> accepted = system.block_level_state_global("tracer", 0);

  try {
    system.step(0.1);
    FAIL() << "metadata must not synthesize an implicit Program";
  } catch (const std::logic_error& error) {
    EXPECT_STREQ(error.what(), "AmrSystem::step requires an installed whole-system Program");
  }
  EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted);
}

TEST(test_amr_multiblock_implicit_transaction,
     InterfaceProviderPublicationRefreshesExactAcceptedProgramContracts) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 4;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.interface-publication.amr-runtime@1");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", relaxing_model<Dim>(pops::Real(0), pops::Real(0)),
                                "minmod", "rusanov", "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.interface-publication.amr/physical-flux");
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  const auto initial_state = system.block_level_state_global("tracer", 0);
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("clock.amr-interface-publication");
  int refreshes = 0;
  std::string observed_budget;
  context->install([](double) {}, context,
                   [&]() {
                     ++refreshes;
                     observed_budget =
                         system.prepared_amr_interface_flux_ledger_budget().exact_contract;
                   });
  system.set_program_block_map({0});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget(
      "tests.interface-publication.amr-program@1", std::vector<FluxBudget>{{1, 1}}, 0, 0);
  const auto initial_budget = system.prepared_amr_interface_flux_ledger_budget().exact_contract;

  // An empty authenticated prefix still changes the exact provider/production contracts. Exercise
  // their publication without advancing a clock or adding numerical fixture work to this test.
  system.install_prepared_amr_interface_flux_provider("tests.interface-prefix@1", [](auto&) {});
  EXPECT_EQ(refreshes, 1);
  EXPECT_NE(observed_budget, initial_budget);
  EXPECT_EQ(observed_budget, system.prepared_amr_interface_flux_ledger_budget().exact_contract);
  const auto first_bytes = system.program_accepted_state();
  ASSERT_FALSE(first_bytes.empty());
  auto interface_budget = system.prepared_amr_interface_flux_ledger_budget();
  const auto first = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
      first_bytes, &interface_budget);
  const auto first_revision = system.program_accepted_state_revision();

  EXPECT_THROW(system.install_prepared_amr_interface_flux_provider(
                   "tests.rejected-interface-prefix@1",
                   [](auto&) { throw std::runtime_error("rejected interface installer"); }),
               std::runtime_error);
  EXPECT_EQ(refreshes, 1);
  EXPECT_EQ(system.program_accepted_state_revision(), first_revision);
  EXPECT_EQ(system.program_accepted_state(), first_bytes);

  system.install_prepared_amr_interface_flux_provider("tests.interface-prefix@2", [](auto&) {});
  EXPECT_EQ(refreshes, 2);
  EXPECT_EQ(system.program_accepted_state_revision(), first_revision + 1);
  const auto second_bytes = system.program_accepted_state();
  interface_budget = system.prepared_amr_interface_flux_ledger_budget();
  const auto second = pops::runtime::program::deserialize_amr_program_accepted_state<Dim>(
      second_bytes, &interface_budget);
  EXPECT_EQ(second.flux_budget_contract, first.flux_budget_contract);
  EXPECT_NE(second.coupling_contract, first.coupling_contract);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), initial_state);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
}

TEST(test_amr_multiblock_implicit_transaction,
     BalanceMailboxResetsOnlyInsideOutermostTransactionSnapshot) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 4;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.balance-mailbox.amr-runtime@1");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", relaxing_model<Dim>(pops::Real(0), pops::Real(0)),
                                "minmod", "rusanov", "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.balance-mailbox.amr/physical-flux");
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("clock.amr-balance-mailbox");
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '3');
  const std::array<std::pair<const char*, pops::Real>, 5> terms{{
      {"storage_change", 11},
      {"outward_boundary_flux", 2},
      {"sources", 5},
      {"reflux", 3},
      {"projection", 1},
  }};
  auto record = [&](pops::Real weight) {
    for (const auto& [name, value] : terms)
      context->record_balance_term(route, name, weight * value);
  };
  auto expect_mailbox = [&](pops::Real weight) {
    // Inspect retained native storage without relaxing the active-transaction public consumer gate.
    const auto actual = context->runtime_state().accepted_balance_terms(route, "test");
    ASSERT_EQ(actual.size(), terms.size());
    for (const auto& [name, value] : terms)
      EXPECT_EQ(actual.at(name), weight * value) << name;
  };
  context->install(
      [&](double macro_dt) {
        context->advance_hierarchy(macro_dt, [&](double) { record(pops::Real(1)); });
      },
      context);
  system.set_program_block_map({0});
  using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
  system.install_prepared_amr_program_flux_expression_budget("tests.balance-mailbox.amr-program@1",
                                                             std::vector<FluxBudget>{{1, 1}}, 0, 0);
  const auto initial = system.block_level_state_global("tracer", 0);

  for (int step = 0; step < 2; ++step) {
    system.step(0.125);
    expect_mailbox(pops::Real(1));
  }
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_DOUBLE_EQ(system.time(), 0.25);
  EXPECT_THROW((void)system.accepted_balance_terms(route), std::runtime_error);

  system.begin_step_transaction();
  EXPECT_THROW((void)system.accepted_balance_terms(route), std::runtime_error);
  record(pops::Real(0.5));
  system.begin_nested_step_transaction();
  expect_mailbox(pops::Real(0.5));
  system.step(0.125);
  expect_mailbox(pops::Real(1.5));
  system.commit_step_transaction();
  system.finalize_step_transaction();
  expect_mailbox(pops::Real(1.5));
  EXPECT_EQ(system.accepted_balance_terms(route).size(), terms.size());
  system.begin_nested_step_transaction();
  system.step(0.125);
  expect_mailbox(pops::Real(2.5));
  system.rollback_step_transaction();
  expect_mailbox(pops::Real(1.5));
  system.rollback_step_transaction();
  expect_mailbox(pops::Real(1));
  EXPECT_EQ(system.step_transaction_depth(), 0u);
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_DOUBLE_EQ(system.time(), 0.25);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), initial);

  system.begin_step_transaction();
  record(pops::Real(0.25));
  system.begin_nested_step_transaction();
  system.step(0.125);
  system.commit_step_transaction();
  system.finalize_step_transaction();
  expect_mailbox(pops::Real(1.25));
  system.commit_step_transaction();
  system.finalize_step_transaction();
  expect_mailbox(pops::Real(1.25));
  system.step(0.125);
  expect_mailbox(pops::Real(1));
  EXPECT_EQ(system.macro_step(), 4);
  EXPECT_DOUBLE_EQ(system.time(), 0.5);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), initial);
}
