#include <gtest/gtest.h>

#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>

#include <cstdint>
#include <limits>

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
  EXPECT_GT(outcome.total_iterations, 0);
  EXPECT_NEAR(outcome.value[0], Real(2), Real(1e-10));
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

}  // namespace
