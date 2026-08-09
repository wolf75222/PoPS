#include <gtest/gtest.h>

#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>
#include <pops/physics/bricks/elliptic.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/runtime/recovery/uniform_recovery_consumer.hpp>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

using pops::Real;

template <int N>
struct AcceptPositive {
  POPS_HD bool operator()(const Real (&value)[N], int* component = nullptr) const {
    for (int index = 0; index < N; ++index)
      if (!(value[index] > Real(0))) {
        if (component != nullptr)
          *component = index;
        return false;
      }
    if (component != nullptr)
      *component = -1;
    return true;
  }
};

struct UnavailableClosedForm {
  static constexpr pops::RecoveryMethodKind kind = pops::RecoveryMethodKind::kClosedForm;

  POPS_HD pops::RecoveryMethodResult<1> operator()(const Real (&)[1], const Real (&)[1]) const {
    return pops::RecoveryMethodResult<1>::continue_chain(
        pops::RecoveryCause::kClosedFormUnavailable);
  }
};

struct SquareResidual {
  Real target = 0;

  POPS_HD pops::LocalNonlinearEvaluationResult operator()(const Real (&value)[1],
                                                          Real (&residual)[1]) const {
    residual[0] = value[0] * value[0] - target;
    return pops::LocalNonlinearEvaluationResult::ok();
  }
};

struct SquareProblemFactory {
  POPS_HD auto operator()(const Real (&conserved)[1]) const {
    pops::PreparedLocalNonlinearControls controls;
    controls.max_iterations = 16;
    controls.absolute_tolerance = Real(1e-13);
    return pops::prepare_local_nonlinear_problem<1>(SquareResidual{conserved[0]},
                                                    pops::FiniteDifferenceLocalJacobian<1>{},
                                                    AcceptPositive<1>{}, controls);
  }
};

struct NegativeCandidate {
  static constexpr pops::RecoveryMethodKind kind = pops::RecoveryMethodKind::kClosedForm;

  POPS_HD pops::RecoveryMethodResult<1> operator()(const Real (&)[1], const Real (&)[1]) const {
    const Real value[1] = {Real(-2)};
    return pops::RecoveryMethodResult<1>::candidate(value);
  }
};

struct ExplicitReject {
  static constexpr pops::RecoveryMethodKind kind = pops::RecoveryMethodKind::kCustom;

  POPS_HD pops::RecoveryMethodResult<1> operator()(const Real (&)[1], const Real (&)[1]) const {
    return pops::RecoveryMethodResult<1>::reject(pops::RecoveryCause::kExplicitRejection);
  }
};

struct InitialGuessOrReject {
  static constexpr pops::RecoveryMethodKind kind = pops::RecoveryMethodKind::kCustom;

  POPS_HD pops::RecoveryMethodResult<1> operator()(const Real (&conserved)[1],
                                                   const Real (&initial_guess)[1]) const {
    if (conserved[0] < Real(0))
      return pops::RecoveryMethodResult<1>::reject(pops::RecoveryCause::kExplicitRejection);
    return pops::RecoveryMethodResult<1>::candidate(initial_guess);
  }
};

struct NonFiniteCandidate {
  static constexpr pops::RecoveryMethodKind kind = pops::RecoveryMethodKind::kCustom;

  POPS_HD pops::RecoveryMethodResult<1> operator()(const Real (&)[1], const Real (&)[1]) const {
    const Real value[1] = {std::numeric_limits<Real>::quiet_NaN()};
    return pops::RecoveryMethodResult<1>::candidate(value);
  }
};

struct RepairCandidate {
  static constexpr pops::RecoveryMethodKind kind = pops::RecoveryMethodKind::kRepair;

  POPS_HD pops::RecoveryMethodResult<1> operator()(const Real (&)[1], const Real (&)[1]) const {
    const Real value[1] = {Real(1)};
    return pops::RecoveryMethodResult<1>::candidate(value);
  }
};

struct GuardedScalarHyperbolic {
  using State = pops::StateVec<1>;
  using Prim = pops::StateVec<1>;
using Providers = pops::ProviderValues<0>;
  static constexpr int n_vars = 1;

POPS_HD State flux(const State& value, const Providers&, int) const { return value; }
POPS_HD Real max_wave_speed(const State&, const Providers&, int) const { return Real(1); }
  POPS_HD Prim to_primitive(const State& value) const { return value; }
  POPS_HD State to_conservative(const Prim& value) const { return value; }
  POPS_HD bool recovery_admissible(const Prim& value, int* failing_component) const {
    if (!(value[0] > Real(0))) {
      if (failing_component != nullptr)
        *failing_component = 0;
      return false;
    }
    if (failing_component != nullptr)
      *failing_component = -1;
    return true;
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"q"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"q"}, 1, {pops::VariableRole::Scalar}};
  }
};

using GuardedScalarModel =
    pops::CompositeModel<GuardedScalarHyperbolic, pops::NoSource, pops::ChargeDensity>;

static_assert(pops::HyperbolicPhysicalModel<GuardedScalarHyperbolic>);
static_assert(pops::HasRecoveryAdmissibility<GuardedScalarModel>);

TEST(PreparedVariableRecovery, ordered_chain_uses_common_prepared_solver) {
  const auto methods = pops::recovery_methods(
      UnavailableClosedForm{}, pops::prepared_local_nonlinear_recovery<1>(SquareProblemFactory{}));
  const auto plan = pops::prepare_variable_recovery<1>(AcceptPositive<1>{}, methods);

  static_assert(decltype(methods)::size == 2);
  EXPECT_EQ(plan.method_kind(0), pops::RecoveryMethodKind::kClosedForm);
  EXPECT_EQ(plan.method_kind(1), pops::RecoveryMethodKind::kPreparedLocalNonlinear);
  EXPECT_EQ(plan.method_kind(2), pops::RecoveryMethodKind::kUnknown);

  const Real conserved[1] = {Real(4)};
  const Real initial_guess[1] = {Real(1)};
  const auto outcome = pops::recover_prepared_variable(plan, conserved, initial_guess);

  ASSERT_TRUE(outcome.recovered());
  EXPECT_TRUE(outcome.publication_permitted());
  EXPECT_EQ(outcome.attempted_methods, 2);
  EXPECT_EQ(outcome.selected_method, 1);
  EXPECT_EQ(outcome.last_method, 1);
  EXPECT_EQ(outcome.selected_method_kind, pops::RecoveryMethodKind::kPreparedLocalNonlinear);
  EXPECT_EQ(outcome.last_method_kind, pops::RecoveryMethodKind::kPreparedLocalNonlinear);
  EXPECT_GT(outcome.total_iterations, 0);
  EXPECT_NEAR(outcome.value[0], Real(2), Real(1e-10));
}

TEST(PreparedVariableRecovery, type_erased_report_preserves_selected_method_kind) {
  const auto methods = pops::recovery_methods(
      UnavailableClosedForm{}, pops::prepared_local_nonlinear_recovery<1>(SquareProblemFactory{}));
  const auto plan = pops::prepare_variable_recovery<1>(AcceptPositive<1>{}, methods);
  const Real conserved[1] = {Real(4)};
  const Real initial_guess[1] = {Real(1)};

  const auto report =
      pops::recovery_report(pops::recover_prepared_variable(plan, conserved, initial_guess));

  ASSERT_TRUE(report.publication_permitted());
  EXPECT_EQ(report.selected_method, 1);
  EXPECT_EQ(report.last_method, 1);
  EXPECT_EQ(report.selected_method_kind, pops::RecoveryMethodKind::kPreparedLocalNonlinear);
  EXPECT_EQ(report.last_method_kind, pops::RecoveryMethodKind::kPreparedLocalNonlinear);
  EXPECT_STREQ(pops::recovery_method_kind_name(report.selected_method_kind),
               "prepared_local_nonlinear");
}

TEST(PreparedVariableRecovery, rejected_chain_never_changes_solution_or_cache) {
  const auto plan = pops::prepare_variable_recovery<1>(
      AcceptPositive<1>{}, pops::recovery_methods(NegativeCandidate{}, ExplicitReject{}));
  const Real conserved[1] = {Real(4)};
  const Real initial_guess[1] = {Real(1)};
  const auto outcome = pops::recover_prepared_variable(plan, conserved, initial_guess);

  EXPECT_EQ(outcome.status, pops::RecoveryStatus::kRejected);
  EXPECT_EQ(outcome.cause, pops::RecoveryCause::kExplicitRejection);
  EXPECT_EQ(outcome.attempted_methods, 2);
  EXPECT_EQ(outcome.selected_method, -1);
  EXPECT_EQ(outcome.selected_method_kind, pops::RecoveryMethodKind::kUnknown);
  EXPECT_EQ(outcome.last_method, 1);
  EXPECT_EQ(outcome.last_method_kind, pops::RecoveryMethodKind::kCustom);
  EXPECT_FALSE(outcome.publication_permitted());

  Real accepted[1] = {Real(9)};
  pops::RecoveryWarmStartSlot<1> cache;
  const Real cached[1] = {Real(8)};
  cache.store(cached, 3, 7);
  pops::RecoveryPublicationTransaction<1> transaction(accepted, cache);

  EXPECT_FALSE(transaction.publish_tentative(outcome, 4, 8));
  EXPECT_EQ(accepted[0], Real(9));
  EXPECT_EQ(cache.value[0], Real(8));
  EXPECT_EQ(cache.topology_generation, std::uint64_t{3});
  EXPECT_EQ(cache.state_generation, std::uint64_t{7});
  EXPECT_TRUE(transaction.rollback());
  EXPECT_EQ(accepted[0], Real(9));
  EXPECT_EQ(cache.value[0], Real(8));
}

TEST(PreparedVariableRecovery, rejected_report_names_last_method_without_forging_selection) {
  const auto plan = pops::prepare_variable_recovery<1>(
      AcceptPositive<1>{}, pops::recovery_methods(NegativeCandidate{}, ExplicitReject{}));
  const Real conserved[1] = {Real(4)};
  const Real initial_guess[1] = {Real(1)};

  const auto report =
      pops::recovery_report(pops::recover_prepared_variable(plan, conserved, initial_guess));

  EXPECT_EQ(report.status, pops::RecoveryStatus::kRejected);
  EXPECT_EQ(report.selected_method, -1);
  EXPECT_EQ(report.selected_method_kind, pops::RecoveryMethodKind::kUnknown);
  EXPECT_EQ(report.last_method, 1);
  EXPECT_EQ(report.last_method_kind, pops::RecoveryMethodKind::kCustom);
  EXPECT_STREQ(pops::recovery_method_kind_name(report.last_method_kind), "custom");
}

TEST(PreparedVariableRecovery, tentative_publication_rolls_back_solution_and_warm_start) {
  const auto plan = pops::prepare_variable_recovery<1>(
      AcceptPositive<1>{},
      pops::recovery_methods(pops::prepared_local_nonlinear_recovery<1>(SquareProblemFactory{})));
  const Real conserved[1] = {Real(4)};
  const Real initial_guess[1] = {Real(1)};
  const auto outcome = pops::recover_prepared_variable(plan, conserved, initial_guess);
  ASSERT_TRUE(outcome.recovered());

  Real accepted[1] = {Real(9)};
  pops::RecoveryWarmStartSlot<1> cache;
  const Real cached[1] = {Real(8)};
  cache.store(cached, 3, 7);
  {
    pops::RecoveryPublicationTransaction<1> transaction(accepted, cache);
    ASSERT_TRUE(transaction.publish_tentative(outcome, 4, 8));
    EXPECT_NEAR(accepted[0], Real(2), Real(1e-10));
    EXPECT_NEAR(cache.value[0], Real(2), Real(1e-10));
    EXPECT_EQ(cache.topology_generation, std::uint64_t{4});
    EXPECT_EQ(cache.state_generation, std::uint64_t{8});
    ASSERT_TRUE(transaction.rollback());
  }
  EXPECT_EQ(accepted[0], Real(9));
  EXPECT_EQ(cache.value[0], Real(8));
  EXPECT_EQ(cache.topology_generation, std::uint64_t{3});
  EXPECT_EQ(cache.state_generation, std::uint64_t{7});

  pops::RecoveryPublicationTransaction<1> committed(accepted, cache);
  ASSERT_TRUE(committed.publish_tentative(outcome, 4, 8));
  ASSERT_TRUE(committed.commit());
  EXPECT_FALSE(committed.rollback());
  EXPECT_NEAR(accepted[0], Real(2), Real(1e-10));
  EXPECT_NEAR(cache.value[0], Real(2), Real(1e-10));
}

TEST(PreparedVariableRecovery, scope_exit_rolls_back_an_uncommitted_publication) {
  const auto plan = pops::prepare_variable_recovery<1>(
      AcceptPositive<1>{},
      pops::recovery_methods(pops::prepared_local_nonlinear_recovery<1>(SquareProblemFactory{})));
  const Real conserved[1] = {Real(4)};
  const Real initial_guess[1] = {Real(1)};
  const auto outcome = pops::recover_prepared_variable(plan, conserved, initial_guess);
  ASSERT_TRUE(outcome.recovered());

  Real accepted[1] = {Real(9)};
  pops::RecoveryWarmStartSlot<1> cache;
  const Real cached[1] = {Real(8)};
  cache.store(cached, 3, 7);
  {
    pops::RecoveryPublicationTransaction<1> transaction(accepted, cache);
    ASSERT_TRUE(transaction.publish_tentative(outcome, 4, 8));
    EXPECT_NEAR(accepted[0], Real(2), Real(1e-10));
  }
  EXPECT_EQ(accepted[0], Real(9));
  EXPECT_EQ(cache.value[0], Real(8));
  EXPECT_EQ(cache.topology_generation, std::uint64_t{3});
  EXPECT_EQ(cache.state_generation, std::uint64_t{7});
}

TEST(PreparedVariableRecovery, stale_warm_start_is_an_explicit_non_mutating_miss) {
  pops::RecoveryWarmStartSlot<2> cache;
  const Real cached[2] = {Real(3), Real(4)};
  cache.store(cached, 5, 9);
  Real destination[2] = {Real(11), Real(12)};

  EXPECT_FALSE(cache.load_if_current(6, 9, destination));
  EXPECT_EQ(destination[0], Real(11));
  EXPECT_EQ(destination[1], Real(12));
  EXPECT_EQ(cache.value[0], Real(3));
  EXPECT_EQ(cache.value[1], Real(4));
  EXPECT_TRUE(cache.valid);

  EXPECT_TRUE(cache.load_if_current(5, 9, destination));
  EXPECT_EQ(destination[0], Real(3));
  EXPECT_EQ(destination[1], Real(4));
}

TEST(PreparedVariableRecovery, uniform_consumer_reuses_only_exact_generation_qualified_cells) {
  const auto plan = pops::prepare_variable_recovery<1>(
      AcceptPositive<1>{},
      pops::recovery_methods(pops::prepared_local_nonlinear_recovery<1>(SquareProblemFactory{})));
  pops::PreparedUniformRecoveryConsumer<1, decltype(plan)> consumer(plan);

  const std::vector<double> first_conserved{4.0, 9.0};
  std::vector<double> primitive{77.0};
  const auto first = consumer.recover(first_conserved, primitive);
  ASSERT_TRUE(first.publication_permitted());
  EXPECT_EQ(first.cache_hits, std::size_t{0});
  EXPECT_EQ(first.topology_generation, std::uint64_t{1});
  EXPECT_EQ(first.state_generation, std::uint64_t{1});
  ASSERT_EQ(primitive.size(), std::size_t{2});
  EXPECT_NEAR(primitive[0], 2.0, 1e-10);
  EXPECT_NEAR(primitive[1], 3.0, 1e-10);

  const auto repeated = consumer.recover(first_conserved, primitive);
  ASSERT_TRUE(repeated.publication_permitted());
  EXPECT_EQ(repeated.cache_hits, std::size_t{2});
  EXPECT_EQ(repeated.topology_generation, std::uint64_t{1});
  EXPECT_EQ(repeated.state_generation, std::uint64_t{2});

  const std::vector<double> one_changed{16.0, 9.0};
  const auto changed = consumer.recover(one_changed, primitive);
  ASSERT_TRUE(changed.publication_permitted());
  EXPECT_EQ(changed.cache_hits, std::size_t{1});
  EXPECT_NEAR(primitive[0], 4.0, 1e-10);
  EXPECT_NEAR(primitive[1], 3.0, 1e-10);
}

TEST(PreparedVariableRecovery, uniform_consumer_failure_keeps_output_and_invalidates_all_slots) {
  const auto plan = pops::prepare_variable_recovery<1>(
      AcceptPositive<1>{}, pops::recovery_methods(InitialGuessOrReject{}));
  pops::PreparedUniformRecoveryConsumer<1, decltype(plan)> consumer(plan);

  const std::vector<double> accepted{4.0, 9.0};
  std::vector<double> primitive;
  ASSERT_TRUE(consumer.recover(accepted, primitive).publication_permitted());

  const std::vector<double> rejected{4.0, -1.0};
  const std::vector<double> sentinel{31.0, 41.0};
  primitive = sentinel;
  const auto failed = consumer.recover(rejected, primitive);
  EXPECT_FALSE(failed.publication_permitted());
  EXPECT_EQ(failed.failed_cell, std::size_t{1});
  EXPECT_EQ(failed.cache_hits, std::size_t{1});
  EXPECT_EQ(failed.recovery.status, pops::RecoveryStatus::kRejected);
  EXPECT_EQ(primitive, sentinel);

  const auto retry = consumer.recover(accepted, primitive);
  ASSERT_TRUE(retry.publication_permitted());
  EXPECT_EQ(retry.cache_hits, std::size_t{0})
      << "a failed batch must invalidate slots committed earlier in that batch";
}

TEST(PreparedVariableRecovery, malformed_and_repair_candidates_fail_closed) {
  const Real conserved[1] = {Real(4)};
  const Real initial_guess[1] = {Real(1)};

  const auto malformed_plan = pops::prepare_variable_recovery<1>(
      AcceptPositive<1>{}, pops::recovery_methods(NonFiniteCandidate{}, ExplicitReject{}));
  const auto malformed = pops::recover_prepared_variable(malformed_plan, conserved, initial_guess);
  EXPECT_EQ(malformed.status, pops::RecoveryStatus::kInvalidContract);
  EXPECT_EQ(malformed.cause, pops::RecoveryCause::kNonFiniteCandidate);
  EXPECT_EQ(malformed.attempted_methods, 1);
  EXPECT_FALSE(malformed.publication_permitted());

  const auto repair_plan = pops::prepare_variable_recovery<1>(
      AcceptPositive<1>{}, pops::recovery_methods(RepairCandidate{}));
  const auto repair = pops::recover_prepared_variable(repair_plan, conserved, initial_guess);
  EXPECT_EQ(repair.status, pops::RecoveryStatus::kInvalidContract);
  EXPECT_EQ(repair.cause, pops::RecoveryCause::kRepairPublicationForbidden);
  EXPECT_FALSE(repair.publication_permitted());
}

TEST(PreparedVariableRecovery, model_declared_admissibility_blocks_publication) {
  const GuardedScalarModel model{};
  const auto plan = pops::prepare_model_variable_recovery(model);
  EXPECT_EQ(plan.method_kind(0), pops::RecoveryMethodKind::kClosedForm);

  const Real negative[1] = {Real(-1)};
  const Real negative_guess[1] = {Real(2)};
  const auto rejected = pops::recover_prepared_variable(plan, negative, negative_guess);
  EXPECT_EQ(rejected.status, pops::RecoveryStatus::kExhausted);
  EXPECT_EQ(rejected.cause, pops::RecoveryCause::kInadmissibleCandidate);
  EXPECT_EQ(rejected.failing_component, 0);
  EXPECT_FALSE(rejected.publication_permitted());
}

TEST(PreparedVariableRecovery, model_declared_admissibility_permits_valid_candidate) {
  const GuardedScalarModel model{};
  const auto plan = pops::prepare_model_variable_recovery(model);
  const Real positive[1] = {Real(3)};
  const Real positive_guess[1] = {Real(1)};
  const auto recovered = pops::recover_prepared_variable(plan, positive, positive_guess);
  ASSERT_TRUE(recovered.publication_permitted());
  EXPECT_EQ(recovered.failing_component, -1);
  EXPECT_EQ(recovered.value[0], Real(3));
}

}  // namespace
