#include <pops/core/state/state.hpp>
#include <pops/physics/admissibility/admissibility.hpp>
#include <pops/physics/inversion/inversion.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void ensure_kokkos() {
#if defined(POPS_HAS_KOKKOS)
  static Kokkos::ScopeGuard guard;
  (void)guard;
#endif
}

enum class RecoveryFailure : std::uint8_t {
  kNone = 0,
  kInvalidState = 1,
  kNoConvergence = 2,
};

template <int Dim>
using Problem = pops::VariableInversionProblem<Dim, pops::StateVec<2>, pops::ProviderValues<1>,
                                               pops::StateVec<2>, RecoveryFailure>;

struct ClosedFormSource {
  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.closed-form-inversion", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(std::uint32_t{1});
  }
  pops::InversionResult<pops::StateVec<2>, RecoveryFailure> operator()(
      const pops::StateVec<2>& state, const pops::ProviderValues<1>& inputs,
      pops::InversionWorkspaceView workspace) const {
    if (workspace.as<pops::Real>() == nullptr)
      return decltype(operator()(state, inputs, workspace))::fail(RecoveryFailure::kNoConvergence);
    if (!(state[0] > 0.0))
      return decltype(operator()(state, inputs, workspace))::fail(RecoveryFailure::kInvalidState);
    return decltype(operator()(state, inputs, workspace))::success(
        pops::StateVec<2>{{state[0], state[1] / state[0] + inputs[0]}});
  }
};

struct IterativeSource {
  int iterations = 6;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.iterative-inversion", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(std::int32_t{iterations});
  }
  pops::InversionResult<pops::StateVec<2>, RecoveryFailure> operator()(
      const pops::StateVec<2>& state, const pops::ProviderValues<1>&,
      pops::InversionWorkspaceView workspace) const {
    auto* iterate = workspace.as<pops::Real>();
    if (iterate == nullptr || state[0] < 0.0)
      return decltype(operator()(state, {}, workspace))::fail(RecoveryFailure::kInvalidState);
    *iterate = state[0] > 1.0 ? state[0] : 1.0;
    for (int step = 0; step < iterations; ++step)
      *iterate = 0.5 * (*iterate + state[0] / *iterate);
    if (!std::isfinite(*iterate))
      return decltype(operator()(state, {}, workspace))::fail(RecoveryFailure::kNoConvergence);
    return decltype(operator()(state, {}, workspace))::success(
        pops::StateVec<2>{{*iterate, state[1]}});
  }
};

struct ConePredicate {
  pops::Real scale = 1.0;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.realizable-cone", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(scale);
  }
  bool operator()(const pops::StateVec<2>& candidate) const {
    return std::abs(candidate[1]) <= scale * candidate[0];
  }
};

struct SumPredicate {
  pops::Real upper = 4.0;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.custom-sum-inequality", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(upper);
  }
  bool operator()(const pops::StateVec<2>& candidate) const {
    return candidate[0] + candidate[1] < upper;
  }
};

struct ClampProjection {
  pops::Real lower_bound = 0.0;

  static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.explicit-projection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    contract.scalar(lower_bound);
  }
  pops::ProjectionResult<pops::StateVec<2>> operator()(const pops::StateVec<2>& candidate,
                                                       const pops::ProviderValues<1>&) const {
    auto projected = candidate;
    const bool changed = projected[0] <= lower_bound;
    if (changed)
      projected[0] = lower_bound + 1.0e-6;
    return {projected, changed};
  }
};

template <int Dim>
void exercise_dimension() {
  ensure_kokkos();
  Problem<Dim> problem("conservative:pair", "provider-pack:one", "primitive:pair",
                       "recovery-failure:v1", {sizeof(pops::Real), alignof(pops::Real)});
  pops::PreparedVariableInversion closed(problem, ClosedFormSource{});
  const void* allocation = closed.workspace_allocation_identity();
  const pops::ProviderValues<1> inputs{{0.25}};

  auto first = closed.attempt(pops::StateVec<2>{{2.0, 4.0}}, inputs);
  ASSERT_TRUE(first.succeeded());
  auto consumed = first.consume();
  ASSERT_TRUE(consumed.succeeded());
  EXPECT_DOUBLE_EQ(consumed.candidate()[1], 2.25);
  EXPECT_THROW(first.consume(), std::logic_error);

  pops::StateVec<2> published{{7.0, 9.0}};
  auto invalid = closed.attempt(pops::StateVec<2>{{-1.0, 4.0}}, inputs);
  ASSERT_FALSE(invalid.succeeded());
  EXPECT_EQ(invalid.failure(), RecoveryFailure::kInvalidState);
  auto rejected = invalid.consume();
  EXPECT_FALSE(rejected.succeeded());
  EXPECT_DOUBLE_EQ(published[0], 7.0);
  EXPECT_DOUBLE_EQ(published[1], 9.0);

  auto retry = closed.attempt(pops::StateVec<2>{{4.0, 8.0}}, inputs);
  EXPECT_EQ(closed.workspace_allocation_identity(), allocation);
  EXPECT_TRUE(retry.consume().succeeded());

  pops::PreparedVariableInversion iterative(problem, IterativeSource{});
  auto iterated = iterative.attempt(pops::StateVec<2>{{9.0, 3.0}}, inputs).consume();
  ASSERT_TRUE(iterated.succeeded());
  EXPECT_NEAR(iterated.candidate()[0], 3.0, 1.0e-12);

  pops::AdmissibleSet admissible(
      pops::FiniteComponents<0, 2>{10}, pops::PositiveComponent<0>{0.0, 11},
      pops::RealizabilityConstraint<ConePredicate>{ConePredicate{2.0}, 12},
      pops::CustomInequality<SumPredicate>{SumPredicate{4.0}, 13});
  EXPECT_TRUE(admissible.evaluate(pops::StateVec<2>{{1.0, 1.5}}).accepted);
  const auto nonfinite =
      admissible.evaluate(pops::StateVec<2>{{std::numeric_limits<pops::Real>::infinity(), 0.0}});
  EXPECT_FALSE(nonfinite.accepted);
  EXPECT_EQ(nonfinite.kind, pops::AdmissibilityConstraintKind::kFinite);
  const auto nonpositive = admissible.evaluate(pops::StateVec<2>{{0.0, 0.0}});
  EXPECT_EQ(nonpositive.diagnostic_code, 11u);
  const auto unrealizable = admissible.evaluate(pops::StateVec<2>{{1.0, 3.0}});
  EXPECT_EQ(unrealizable.diagnostic_code, 12u);
  const auto custom = admissible.evaluate(pops::StateVec<2>{{2.0, 2.5}});
  EXPECT_EQ(custom.diagnostic_code, 13u);

  pops::ProjectionProvider<Dim, pops::StateVec<2>, pops::ProviderValues<1>, ClampProjection>
      projection("primitive:pair", "provider-pack:one", ClampProjection{0.0});
  auto explicit_projection = projection.project(pops::StateVec<2>{{-1.0, 3.0}}, inputs);
  EXPECT_TRUE(explicit_projection.changed());
  const auto projected = std::move(explicit_projection).consume();
  EXPECT_GT(projected[0], 0.0);

  EXPECT_FALSE(problem.exact_contract().empty());
  EXPECT_FALSE(closed.collective_contract().empty());
  EXPECT_FALSE(admissible.exact_contract().empty());
  EXPECT_FALSE(projection.collective_contract().empty());
}

TEST(GenericInversionAdmissibility, ExactRankedClosedFormAndIterativeProviders) {
  exercise_dimension<1>();
  exercise_dimension<2>();
  exercise_dimension<3>();

  Problem<1> dim1("state", "inputs", "candidate", "failure", {8, 8});
  Problem<2> dim2("state", "inputs", "candidate", "failure", {8, 8});
  pops::PreparedVariableInversion first(dim1, ClosedFormSource{});
  pops::PreparedVariableInversion same(dim1, ClosedFormSource{});
  EXPECT_EQ(first.collective_contract(), same.collective_contract());
  EXPECT_NE(dim1.exact_contract(), dim2.exact_contract());
}

TEST(GenericInversionAdmissibility, DiagnosticAndScheduleCollisionsFailClosed) {
  EXPECT_THROW(
      (pops::AdmissibleSet(pops::FiniteComponents<0, 1>{7}, pops::PositiveComponent<0>{0.0, 7})),
      std::invalid_argument);

  const pops::EnforcementSchedule schedule(std::array<pops::EnforcementRule, 5>{
      pops::EnforcementRule{true, true}, pops::EnforcementRule{true, false},
      pops::EnforcementRule{true, true}, pops::EnforcementRule{true, false},
      pops::EnforcementRule{true, false}});
  EXPECT_TRUE(schedule.at(pops::EnforcementPhase::kInitialization).project_if_invalid);
  EXPECT_FALSE(schedule.at(pops::EnforcementPhase::kAcceptance).project_if_invalid);
  EXPECT_FALSE(schedule.exact_contract().empty());

  EXPECT_THROW((pops::EnforcementSchedule(std::array<pops::EnforcementRule, 5>{
                   pops::EnforcementRule{false, true}, {}, {}, {}, {}})),
               std::invalid_argument);
}

}  // namespace
