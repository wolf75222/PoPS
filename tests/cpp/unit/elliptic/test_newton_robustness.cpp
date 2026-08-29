// Newton de la source implicite GENERALISE : options (budget, tolerances, damping), SolveOutcome
// consomme explicitement et diagnostics (cellule fautive / composante) -- preuves :
//  (1) NON-EULER MULTI-VARIABLES : un systeme de relaxation NON LINEAIRE 3 variables (aucun layout
//      rho/m/E, aucune pression) converge sous tolerance -- le solveur n'est pas hardcode Euler.
//      La solution verifie l'equation BE W = Un + dt*S(W) au residu pres.
//  (2) DAMPING : newton amorti (damping < 1) converge vers la MEME racine (plus d'iterations).
//  (3) PATHOLOGIES : budget, singularite et NaN ne publient aucun candidat.
//  (4) OBSERVATEUR PUR : les diagnostics ne changent pas le candidat converge.
#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/state/state.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using pops::Real;

namespace {

constexpr int kDim = pops::kNativeDimension;
using Field = pops::MultiFab<kDim>;
using Layout = pops::mesh::BoxArray<kDim>;
using Distribution = pops::mesh::Distribution<kDim>;
using Providers = pops::ProviderValues<0>;

struct NoProviders {
  pops::ProviderStorageView<kDim, 0> operator()(std::size_t) const { return {}; }
};

inline constexpr NoProviders no_providers{};

struct SolveOutcomeHookTrace {
  int accepted = 0;
  int rejected = 0;
  int released = 0;
  pops::SolveConsumption failure = pops::SolveConsumption::kAccept;
};

void record_solve_accept(void* context) noexcept {
  ++static_cast<SolveOutcomeHookTrace*>(context)->accepted;
}

void record_solve_reject(void* context, pops::SolveConsumption action) {
  auto& trace = *static_cast<SolveOutcomeHookTrace*>(context);
  ++trace.rejected;
  trace.failure = action;
}

void record_solve_release(void* context) noexcept {
  ++static_cast<SolveOutcomeHookTrace*>(context)->released;
}

pops::ExecutionLane test_execution_lane() {
  return pops::ExecutionLane::world("pops.test.newton-robustness");
}

template <class Ranked, class Value>
Ranked filled_ranked(Value value) {
  Ranked result{};
  for (int axis = 0; axis < kDim; ++axis)
    result[axis] = value;
  return result;
}

Field make_mf(const Layout& layout, const Distribution& distribution, int components,
              pops::Extent<kDim> ghosts = {}) {
  Field field(layout, distribution,
              distribution.rank_space().coordinate(static_cast<std::size_t>(pops::my_rank())),
              components, ghosts);
  field.set_val(Real(0));
  return field;
}

struct CopyThree {
  pops::FieldView<const Real, kDim> source;
  pops::FieldView<Real, kDim> destination;

  POPS_HD void operator()(const pops::Index<kDim>& index) const {
    for (int component = 0; component < 3; ++component)
      destination(index, component) = source(index, component);
  }
};

void copy3(const Field& source, Field& destination) {
  for (std::size_t local = 0; local < destination.local_size(); ++local)
    pops::for_each_cell(destination.box(local),
                        CopyThree{source.fab(local).view(), destination.fab(local).view()});
}

struct DifferenceThree {
  pops::FieldView<const Real, kDim> left;
  pops::FieldView<const Real, kDim> right;

  POPS_HD Real operator()(const pops::Index<kDim>& index) const {
    Real maximum = Real(0);
    for (int component = 0; component < 3; ++component) {
      const Real difference = Kokkos::abs(left(index, component) - right(index, component));
      if (!Kokkos::isfinite(difference))
        return std::numeric_limits<Real>::infinity();
      maximum = difference > maximum ? difference : maximum;
    }
    return maximum;
  }
};

double max_difference3(const Field& left, const Field& right) {
  Real difference = Real(0);
  for (std::size_t local = 0; local < left.local_size(); ++local)
    difference = std::max(
        difference,
        pops::for_each_cell_reduce_max(
            left.box(local), DifferenceThree{left.fab(local).view(), right.fab(local).view()}));
  return static_cast<double>(pops::all_reduce_max(difference));
}

struct FillInitialState {
  pops::FieldView<Real, kDim> state;

  POPS_HD void operator()(const pops::Index<kDim>& index) const {
    Real coordinate_sum = Real(0);
    for (int axis = 0; axis < kDim; ++axis)
      coordinate_sum += static_cast<Real>(index[axis]);
    state(index, 0) = Real(1) + Real(0.1) * static_cast<Real>(index[0]);
    state(index, 1) = Real(-0.5) + Real(0.05) * coordinate_sum;
    state(index, 2) = Real(0.3);
  }
};

struct FillThreeState {
  pops::FieldView<Real, kDim> state;
  Real first;
  Real second;
  Real third;

  POPS_HD void operator()(const pops::Index<kDim>& index) const {
    state(index, 0) = first;
    state(index, 1) = second;
    state(index, 2) = third;
  }
};

struct ReplaceAtIndex {
  pops::FieldView<Real, kDim> state;
  pops::Index<kDim> target;
  int component;
  Real value;

  POPS_HD void operator()(const pops::Index<kDim>& index) const {
    bool matches = true;
    for (int axis = 0; axis < kDim; ++axis)
      matches = matches && index[axis] == target[axis];
    if (matches)
      state(index, component) = value;
  }
};

struct FillFailureStatistics {
  pops::FieldView<Real, kDim> statistics;
  int recoverable_priority;
  int fatal_priority;
  bool promote_negative;

  POPS_HD void operator()(const pops::Index<kDim>& index) const {
    const bool negative = index[0] < 0;
    statistics(index, 8) = negative ? Real(7) : Real(3);
    statistics(index, 9) = Real(1);
    statistics(index, 10) =
        static_cast<Real>(negative && !promote_negative ? recoverable_priority : fatal_priority);
  }
};

template <class Model>
struct BackwardEulerResidual {
  pops::FieldView<const Real, kDim> candidate;
  pops::FieldView<const Real, kDim> initial;
  Model model;
  Real dt;

  POPS_HD Real operator()(const pops::Index<kDim>& index) const {
    typename Model::State state{};
    for (int component = 0; component < Model::n_vars; ++component)
      state[component] = candidate(index, component);
    const typename Model::State source = model.source(state, Providers{});
    Real maximum = Real(0);
    for (int component = 0; component < Model::n_vars; ++component) {
      const Real residual = Kokkos::abs(candidate(index, component) - initial(index, component) -
                                        dt * source[component]);
      maximum = residual > maximum ? residual : maximum;
    }
    return maximum;
  }
};

}  // namespace

// Relaxation NON LINEAIRE 3 variables, sans aucun layout fluide (ni densite, ni pression) :
//   S0 = -k (u0 - u1 u2) ; S1 = -k (u1 - u0/2) ; S2 = -k u2^3.
struct StiffModel {
  using State = pops::StateVec<3>;
  static constexpr int n_vars = 3;
  static constexpr int n_providers = 0;
  Real k = 200.0;
  POPS_HD State flux(const State&, const auto&, int) const { return State{}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return 0; }
  POPS_HD State source(const State& u, const Providers&) const {
    State s{};
    s[0] = -k * (u[0] - u[1] * u[2]);
    s[1] = -k * (u[1] - Real(0.5) * u[0]);
    s[2] = -k * u[2] * u[2] * u[2];
    return s;
  }
  POPS_HD Real elliptic_rhs(const State&) const { return 0; }
};

// StiffModel + JACOBIEN ANALYTIQUE exact (trait HasSourceJacobian, vague 3) : le Newton doit
// converger vers la MEME racine que les differences finies (l'equation BE est identique).
struct JacStiffModel : StiffModel {
  POPS_HD void source_jacobian(const State& u, const Providers&, Real (&J)[3][3]) const {
    J[0][0] = -k;
    J[0][1] = k * u[2];
    J[0][2] = k * u[1];
    J[1][0] = k * Real(0.5);
    J[1][1] = -k;
    J[1][2] = 0;
    J[2][0] = 0;
    J[2][1] = 0;
    J[2][2] = -Real(3) * k * u[2] * u[2];
  }
};

// Source PATHOLOGIQUE : sqrt(u0 - 10) -> NaN des que u0 < 10 (toutes nos cellules), sur la
// composante 1 SEULEMENT quand u0 < seuil bas (pour viser UNE cellule fautive).
struct NanModel {
  using State = pops::StateVec<3>;
  static constexpr int n_vars = 3;
  static constexpr int n_providers = 0;
  POPS_HD State flux(const State&, const auto&, int) const { return State{}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return 0; }
  POPS_HD State source(const State& u, const Providers&) const {
    State s{};
    s[0] = -u[0];
    s[1] = u[0] < Real(0) ? std::sqrt(u[0]) : -u[1];  // u0 < 0 -> NaN sur la composante 1
    s[2] = -u[2];
    return s;
  }
  POPS_HD Real elliptic_rhs(const State&) const { return 0; }
};

// Jacobien local exactement singulier pour dt=0.125 : dS0/du0 = 8, donc
// J00 = 1 - dt*dS0/du0 = 0 alors que le residu reste fini et non nul.
struct SingularModel {
  using State = pops::StateVec<3>;
  static constexpr int n_vars = 3;
  static constexpr int n_providers = 0;
  POPS_HD State flux(const State&, const auto&, int) const { return State{}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return 0; }
  POPS_HD State source(const State& u, const Providers&) const {
    State s{};
    s[0] = Real(8) * u[0] + Real(1);
    s[1] = -u[1];
    s[2] = -u[2];
    return s;
  }
  POPS_HD void source_jacobian(const State&, const Providers&, Real (&J)[3][3]) const {
    J[0][0] = Real(8);
    J[0][1] = J[0][2] = Real(0);
    J[1][0] = J[1][2] = Real(0);
    J[1][1] = Real(-1);
    J[2][0] = J[2][1] = Real(0);
    J[2][2] = Real(-1);
  }
  POPS_HD Real elliptic_rhs(const State&) const { return 0; }
};

POPS_HD static pops::ImplicitEvaluationResult configured_evaluation(
    pops::ImplicitEvaluationStatus status, std::uint32_t reason) {
  switch (status) {
    case pops::ImplicitEvaluationStatus::kOk:
      return pops::ImplicitEvaluationResult::ok();
    case pops::ImplicitEvaluationStatus::kRetry:
      return pops::ImplicitEvaluationResult::retry(reason);
    case pops::ImplicitEvaluationStatus::kReject:
      return pops::ImplicitEvaluationResult::reject(reason);
    case pops::ImplicitEvaluationStatus::kFailed:
      return pops::ImplicitEvaluationResult::failed(reason);
    case pops::ImplicitEvaluationStatus::kInvalid:
      return pops::ImplicitEvaluationResult::invalid(reason);
  }
  return pops::ImplicitEvaluationResult::invalid(reason);
}

// The fallible source contract is a device POD plus an output parameter: no exception, optional,
// string, allocation, or host callback can escape into a Kokkos kernel. The historical source()
// remains present to prove that evaluate_source() takes precedence when a model opts in.
struct FallibleSourceModel {
  using State = pops::StateVec<3>;
  static constexpr int n_vars = 3;
  static constexpr int n_providers = 0;
  pops::ImplicitEvaluationStatus evaluation = pops::ImplicitEvaluationStatus::kOk;
  std::uint32_t reason = 0;

  POPS_HD State flux(const State&, const auto&, int) const { return State{}; }
  POPS_HD Real max_wave_speed(const State&, const auto&, int) const { return 0; }
  POPS_HD State source(const State&, const Providers&) const {
    return State{Real(1e6), Real(1e6), Real(1e6)};
  }
  POPS_HD pops::ImplicitEvaluationResult evaluate_source(const State& u, const Providers&,
                                                         State& output) const {
    output = State{-u[0], -u[1], -u[2]};
    return configured_evaluation(evaluation, reason);
  }
  POPS_HD Real elliptic_rhs(const State&) const { return 0; }
};

struct FallibleJacobianModel : StiffModel {
  pops::ImplicitEvaluationStatus evaluation = pops::ImplicitEvaluationStatus::kOk;
  std::uint32_t reason = 0;

  POPS_HD pops::ImplicitEvaluationResult evaluate_source_jacobian(const State& u, const Providers&,
                                                                  Real (&J)[3][3]) const {
    J[0][0] = -k;
    J[0][1] = k * u[2];
    J[0][2] = k * u[1];
    J[1][0] = k * Real(0.5);
    J[1][1] = -k;
    J[1][2] = 0;
    J[2][0] = 0;
    J[2][1] = 0;
    J[2][2] = -Real(3) * k * u[2] * u[2];
    return configured_evaluation(evaluation, reason);
  }
};

// pas de temps commun : k*dt = 10 (raide, un point-fixe explicite divergerait).
static constexpr Real kDt = 0.05;

// Fixture partageant la grille 4x4 mono-boite et l'etat initial U0 (meme grille/etat pour toutes
// les preuves de robustesse du Newton generalise). SetUpTestSuite : construit une fois par suite.
class NewtonRobustnessTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    dom_ = new pops::Box<kDim>{pops::Index<kDim>{}, filled_ranked<pops::Index<kDim>>(3)};
    ba_ = new Layout(std::vector<pops::Box<kDim>>{*dom_});
    pops::Extent<kDim> rank_extent = filled_ranked<pops::Extent<kDim>>(1);
    rank_extent[0] = pops::n_ranks();
    dm_ = new Distribution(Distribution::replicated(
        *ba_, pops::mesh::RankSpace<kDim>{pops::Index<kDim>{}, rank_extent}));
    U0_ = new Field(make_mf(*ba_, *dm_, 3));
    for (std::size_t local = 0; local < U0_->local_size(); ++local)
      pops::for_each_cell(U0_->box(local), FillInitialState{U0_->fab(local).view()});
  }
  static void TearDownTestSuite() {
    delete dom_;
    delete ba_;
    delete dm_;
    delete U0_;
    dom_ = nullptr;
    ba_ = nullptr;
    dm_ = nullptr;
    U0_ = nullptr;
  }

  static pops::Box<kDim>* dom_;
  static Layout* ba_;
  static Distribution* dm_;
  static Field* U0_;  // etat initial commun (verification BE, damping, jacobien, observateur)
};
pops::Box<kDim>* NewtonRobustnessTest::dom_ = nullptr;
Layout* NewtonRobustnessTest::ba_ = nullptr;
Distribution* NewtonRobustnessTest::dm_ = nullptr;
Field* NewtonRobustnessTest::U0_ = nullptr;

TEST(SolveOutcomeContract, unconsumed_outcome_fails_loud) {
  EXPECT_DEATH(
      {
        pops::SolveReport solve;
        solve.mark_solved("test_candidate");
        auto outcome = pops::SolveOutcome::serial(std::move(solve));
      },
      "PoPS contract violation: SolveOutcome destroyed before explicit consumption");
}

TEST(SolveOutcomeContract, incompatible_action_does_not_consume_a_valid_outcome) {
  pops::SolveReport solve;
  solve.mark_solved("test_candidate");
  auto outcome = pops::SolveOutcome::serial(std::move(solve));

  EXPECT_THROW(outcome.consume(pops::SolveConsumption::kFailRun), std::logic_error);
  EXPECT_TRUE(outcome.consume(pops::SolveConsumption::kAccept).solved_value_available());
  EXPECT_THROW(outcome.consume(pops::SolveConsumption::kAccept), std::logic_error);
}

TEST(SolveOutcomeContract, fail_run_cannot_be_downgraded_to_reject_attempt) {
  pops::SolveReport solve;
  solve.mark_failed(pops::SolveStatus::kBreakdown, pops::SolveAction::kFailRun,
                    "injected breakdown");
  auto outcome = pops::SolveOutcome::serial(std::move(solve));

  EXPECT_THROW(outcome.consume(pops::SolveConsumption::kRejectAttempt), std::logic_error);
  const pops::SolveReport failed = outcome.consume(pops::SolveConsumption::kFailRun);
  EXPECT_EQ(failed.status, pops::SolveStatus::kBreakdown);
  EXPECT_EQ(failed.action, pops::SolveAction::kFailRun);
}

TEST(SolveOutcomeContract, reject_attempt_may_be_escalated_to_fail_run) {
  pops::SolveReport solve;
  solve.mark_failed(pops::SolveStatus::kIterationLimit, pops::SolveAction::kRejectAttempt,
                    "retry budget exhausted");
  auto outcome = pops::SolveOutcome::serial(std::move(solve));

  const pops::SolveReport failed = outcome.consume(pops::SolveConsumption::kFailRun);
  EXPECT_EQ(failed.status, pops::SolveStatus::kIterationLimit);
  EXPECT_EQ(failed.action, pops::SolveAction::kFailRun);
}

TEST(SolveOutcomeContract, direct_consumer_cannot_continue_after_failed_solve) {
  pops::SolveReport solve;
  solve.mark_failed(pops::SolveStatus::kIterationLimit, pops::SolveAction::kRejectAttempt,
                    "direct consumer has no retry policy");

  EXPECT_THROW((void)pops::consume_solve_outcome(pops::SolveOutcome::serial(std::move(solve))),
               std::runtime_error);
}

TEST(SolveOutcomeContract, exhaustive_statuses_are_move_only_exact_once_and_never_publish_early) {
  static_assert(!std::is_copy_constructible_v<pops::SolveOutcome>);
  static_assert(!std::is_copy_assignable_v<pops::SolveOutcome>);
  static_assert(std::is_move_constructible_v<pops::SolveOutcome>);
  static_assert(!std::is_move_assignable_v<pops::SolveOutcome>);

  struct FailureCase {
    pops::SolveStatus status;
    pops::SolveAction authored_action;
    pops::SolveConsumption consumed_action;
  };
  // This table intentionally enumerates every non-success SolveStatus.  The action is authored
  // by the solver, while the consumer may only escalate RejectAttempt to FailRun.
  const std::array<FailureCase, 9> failures{{
      {pops::SolveStatus::kSingular, pops::SolveAction::kFailRun,
       pops::SolveConsumption::kFailRun},
      {pops::SolveStatus::kBreakdown, pops::SolveAction::kFailRun,
       pops::SolveConsumption::kFailRun},
      {pops::SolveStatus::kIterationLimit, pops::SolveAction::kRejectAttempt,
       pops::SolveConsumption::kRejectAttempt},
      {pops::SolveStatus::kInvalidEvaluation, pops::SolveAction::kFailRun,
       pops::SolveConsumption::kFailRun},
      {pops::SolveStatus::kCapabilityFailure, pops::SolveAction::kFailRun,
       pops::SolveConsumption::kFailRun},
      {pops::SolveStatus::kInvalidInput, pops::SolveAction::kFailRun,
       pops::SolveConsumption::kFailRun},
      {pops::SolveStatus::kIncompatibleRhs, pops::SolveAction::kRejectAttempt,
       pops::SolveConsumption::kRejectAttempt},
      {pops::SolveStatus::kInadmissibleCandidate, pops::SolveAction::kRejectAttempt,
       pops::SolveConsumption::kRejectAttempt},
      {pops::SolveStatus::kSafeguardFailure, pops::SolveAction::kRejectAttempt,
       pops::SolveConsumption::kFailRun},
  }};

  for (const FailureCase& test : failures) {
    SolveOutcomeHookTrace trace;
    pops::SolveReport report;
    report.mark_failed(test.status, test.authored_action, pops::solve_status_name(test.status));
    auto outcome = pops::SolveOutcome::serial(
        std::move(report), {&trace, record_solve_accept, nullptr, record_solve_release, {},
                            nullptr, record_solve_reject});
    EXPECT_EQ(trace.accepted, 0) << pops::solve_status_name(test.status);
    EXPECT_EQ(trace.rejected, 0) << pops::solve_status_name(test.status);
    EXPECT_EQ(trace.released, 0) << pops::solve_status_name(test.status);

    const pops::SolveReport consumed = outcome.consume(test.consumed_action);
    EXPECT_EQ(consumed.status, test.status);
    EXPECT_EQ(consumed.action,
              test.consumed_action == pops::SolveConsumption::kRejectAttempt
                  ? pops::SolveAction::kRejectAttempt
                  : pops::SolveAction::kFailRun);
    EXPECT_EQ(trace.accepted, 0);
    EXPECT_EQ(trace.rejected, 1);
    EXPECT_EQ(trace.released, 1);
    EXPECT_EQ(trace.failure, test.consumed_action);
    EXPECT_THROW(outcome.consume(test.consumed_action), std::logic_error);
  }

  SolveOutcomeHookTrace trace;
  pops::SolveReport report;
  report.mark_solved("candidate is private until consume");
  auto source = pops::SolveOutcome::serial(
      std::move(report), {&trace, record_solve_accept, nullptr, record_solve_release, {}, nullptr,
                          record_solve_reject});
  EXPECT_EQ(trace.accepted, 0);
  auto moved = std::move(source);
  EXPECT_EQ(trace.accepted, 0);
  EXPECT_TRUE(moved.consume(pops::SolveConsumption::kAccept).solved_value_available());
  EXPECT_EQ(trace.accepted, 1);
  EXPECT_EQ(trace.rejected, 0);
  EXPECT_EQ(trace.released, 1);
  EXPECT_THROW(moved.consume(pops::SolveConsumption::kAccept), std::logic_error);
}

// (1) NON-EULER MULTI-VARIABLES : converge sous tolerance ; W verifie l'equation BE au residu pres.
TEST_F(NewtonRobustnessTest, stiff_multivariable_relaxation_converges_to_backward_euler_root) {
  StiffModel m;
  Field U = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, U);

  pops::NewtonOptions opts;
  opts.max_iters = 25;
  opts.rel_tol = 1e-12;
  opts.abs_tol = 1e-13;
  pops::NewtonReport rep;
  auto outcome =
      pops::backward_euler_source(m, no_providers, U, kDt, opts, test_execution_lane(), {}, &rep);
  ASSERT_TRUE(outcome.report().solved_value_available());
  (void)outcome.consume(pops::SolveConsumption::kAccept);
  ASSERT_TRUE(rep.converged && rep.n_failed == 0)
      << "non converge (n_failed=" << rep.n_failed
      << ", res=" << static_cast<double>(rep.max_residual) << ")";

  // verification BE : W - Un - dt S(W) ~ 0 sur chaque cellule.
  Real worst = Real(0);
  for (std::size_t local = 0; local < U.local_size(); ++local)
    worst = std::max(
        worst, pops::for_each_cell_reduce_max(
                   U.box(local), BackwardEulerResidual<StiffModel>{
                                     static_cast<const Field&>(U).fab(local).view(),
                                     static_cast<const Field&>(*U0_).fab(local).view(), m, kDt}));
  worst = pops::all_reduce_max(worst);
  EXPECT_TRUE(worst <= 1e-10) << "residu BE " << worst << " > 1e-10";
  std::printf(
      "OK  (1) relaxation non lineaire 3-var NON Euler : converge (res BE %.1e, iters max "
      "%.0f/25)\n",
      worst, static_cast<double>(rep.max_iters_used));
}

// (2) DAMPING : Newton amorti (damping < 1) converge vers la MEME racine (plus d'iterations).
TEST_F(NewtonRobustnessTest, damped_newton_converges_to_same_root_as_undamped) {
  StiffModel m;
  Field U = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, U);
  pops::NewtonOptions opts;
  opts.max_iters = 25;
  opts.rel_tol = 1e-12;
  opts.abs_tol = 1e-13;
  pops::NewtonReport rep;
  auto reference =
      pops::backward_euler_source(m, no_providers, U, kDt, opts, test_execution_lane(), {}, &rep);
  ASSERT_TRUE(reference.report().solved_value_available());
  (void)reference.consume(pops::SolveConsumption::kAccept);
  ASSERT_TRUE(rep.converged && rep.n_failed == 0) << "racine de reference non convergee";

  Field Ud = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, Ud);
  pops::NewtonOptions od = opts;
  od.damping = 0.5;
  od.max_iters = 80;
  pops::NewtonReport repd;
  auto damped =
      pops::backward_euler_source(m, no_providers, Ud, kDt, od, test_execution_lane(), {}, &repd);
  ASSERT_TRUE(damped.report().solved_value_available());
  (void)damped.consume(pops::SolveConsumption::kAccept);

  const double dmax = max_difference3(U, Ud);
  EXPECT_TRUE(repd.converged) << "damping : non converge";
  EXPECT_TRUE(dmax <= 1e-8) << "damping : ecart racine " << dmax << " > 1e-8";
  std::printf("OK  (2) Newton amorti (damping=0.5) : meme racine (ecart %.1e), iters %.0f\n", dmax,
              static_cast<double>(repd.max_iters_used));
}

// (3) Une evaluation non finie produit un outcome invalide qui reste prive et est consomme FailRun.
TEST_F(NewtonRobustnessTest, invalid_evaluation_reports_cell_and_does_not_publish) {
  NanModel nm;
  Field Un2 = make_mf(*ba_, *dm_, 3);
  for (std::size_t local = 0; local < Un2.local_size(); ++local)
    pops::for_each_cell(Un2.box(local),
                        FillThreeState{Un2.fab(local).view(), Real(1), Real(0.2), Real(0.1)});
  pops::Index<kDim> failing_index{};
  for (int axis = 0; axis < kDim; ++axis)
    failing_index[axis] = 2 + axis % 2;
  for (std::size_t local = 0; local < Un2.local_size(); ++local)
    pops::for_each_cell(Un2.box(local),
                        ReplaceAtIndex{Un2.fab(local).view(), failing_index, 0, Real(-4)});
  Field accepted = make_mf(*ba_, *dm_, 3);
  copy3(Un2, accepted);
  pops::NewtonOptions options;
  pops::NewtonReport repf;
  auto outcome = pops::backward_euler_source(nm, no_providers, Un2, 0.1, options,
                                             test_execution_lane(), {}, &repf);
  ASSERT_EQ(outcome.report().status, pops::SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(max_difference3(Un2, accepted), 0.0)
      << "un candidat invalide a ete publie dans l'etat accepte";
  EXPECT_FALSE(repf.enabled) << "les diagnostics persistants ont ete modifies avant consommation";
  ASSERT_TRUE(outcome.report().failure.found);
  EXPECT_EQ(outcome.report().failure.rank, kDim);
  for (int axis = 0; axis < kDim; ++axis)
    EXPECT_EQ(outcome.report().failure.index[static_cast<std::size_t>(axis)], failing_index[axis]);
  for (int axis = kDim; axis < pops::SolveFailureLocation::maximum_rank; ++axis)
    EXPECT_EQ(outcome.report().failure.index[static_cast<std::size_t>(axis)], 0);
  EXPECT_EQ(outcome.report().failure.component, 1)
      << "la composante NaN initiale doit rester l'origine de l'echec";
  const pops::SolveReport failed = outcome.consume(pops::SolveConsumption::kFailRun);
  EXPECT_EQ(failed.action, pops::SolveAction::kFailRun);
  EXPECT_FALSE(repf.enabled);
  EXPECT_EQ(repf.n_failed, 0);
  EXPECT_EQ(repf.diagnostics.count("newton.outcome.fail_run"), 0u)
      << "un echec ne doit pas publier de diagnostics persistants";
}

// (3b) Une tolerance non satisfaite a l'epuisement du budget ne publie jamais le dernier itere.
TEST_F(NewtonRobustnessTest, iteration_limit_fails_closed_without_publishing) {
  StiffModel model;
  Field accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  Field state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);
  pops::NewtonOptions options;
  options.max_iters = 1;
  options.rel_tol = 1e-15;
  options.abs_tol = 1e-15;
  pops::NewtonReport report;

  auto outcome = pops::backward_euler_source(model, no_providers, state, kDt, options,
                                             test_execution_lane(), {}, &report);
  EXPECT_EQ(outcome.report().status, pops::SolveStatus::kIterationLimit);
  EXPECT_EQ(max_difference3(state, accepted), 0.0) << "le dernier itere non converge a ete publie";
  const pops::SolveReport rejected = outcome.consume(pops::SolveConsumption::kRejectAttempt);
  EXPECT_EQ(rejected.action, pops::SolveAction::kRejectAttempt);
}

// (3c) Un Jacobien singulier a son propre statut et ne publie pas l'itere partiel.
TEST_F(NewtonRobustnessTest, singular_jacobian_fails_closed_without_publishing) {
  SingularModel model;
  Field accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  Field state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);
  pops::NewtonOptions options;
  pops::NewtonReport report;

  auto outcome = pops::backward_euler_source(model, no_providers, state, 0.125, options,
                                             test_execution_lane(), {}, &report);
  EXPECT_EQ(outcome.report().status, pops::SolveStatus::kSingular);
  EXPECT_EQ(max_difference3(state, accepted), 0.0);
  (void)outcome.consume(pops::SolveConsumption::kFailRun);
}

// (3d) Une evaluation non finie est fatalement invalide : elle ne peut pas etre retrogradee en
// RejectAttempt, mais son candidat reste prive pendant la consommation FailRun.
TEST_F(NewtonRobustnessTest, prepared_invalid_failure_is_consumed_once_as_fail_run) {
  NanModel model;
  Field accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  pops::Index<kDim> failing_index{};
  for (int axis = 0; axis < kDim; ++axis)
    failing_index[axis] = 2 + axis % 2;
  for (std::size_t local = 0; local < accepted.local_size(); ++local)
    pops::for_each_cell(accepted.box(local),
                        ReplaceAtIndex{accepted.fab(local).view(), failing_index, 0, Real(-4)});
  Field state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);
  pops::NewtonOptions options;
  pops::NewtonReport diagnostics;

  auto outcome = pops::backward_euler_source(model, no_providers, state, 0.1, options,
                                             test_execution_lane(), {}, &diagnostics);
  EXPECT_EQ(outcome.report().status, pops::SolveStatus::kInvalidEvaluation);
  EXPECT_THROW(outcome.consume(static_cast<pops::SolveConsumption>(255)), std::logic_error);
  EXPECT_EQ(max_difference3(state, accepted), 0.0);
  EXPECT_THROW(outcome.consume(pops::SolveConsumption::kRejectAttempt), std::logic_error);
  const pops::SolveReport failed = outcome.consume(pops::SolveConsumption::kFailRun);
  EXPECT_EQ(failed.action, pops::SolveAction::kFailRun);
  EXPECT_EQ(max_difference3(state, accepted), 0.0);
  EXPECT_FALSE(diagnostics.enabled);
  EXPECT_EQ(diagnostics.diagnostics.count("newton.outcome.fail_run"), 0u);
  EXPECT_THROW(outcome.consume(pops::SolveConsumption::kFailRun), std::logic_error);
}

TEST_F(NewtonRobustnessTest,
       fallible_source_propagates_retry_reject_fail_and_invalid_without_publication) {
  static_assert(pops::HasFallibleSourceEvaluationFor<FallibleSourceModel, Providers>);
  constexpr std::uint32_t reason = 0xfedcba98u;
  struct FailureCase {
    pops::ImplicitEvaluationStatus status;
    pops::SolveAction action;
    const char* reason_fragment;
  };
  const std::array<FailureCase, 4> cases{{
      {pops::ImplicitEvaluationStatus::kRetry, pops::SolveAction::kRejectAttempt,
       "evaluation_retry"},
      {pops::ImplicitEvaluationStatus::kReject, pops::SolveAction::kRejectAttempt,
       "evaluation_reject"},
      {pops::ImplicitEvaluationStatus::kFailed, pops::SolveAction::kFailRun, "evaluation_failed"},
      {pops::ImplicitEvaluationStatus::kInvalid, pops::SolveAction::kFailRun, "invalid_evaluation"},
  }};

  for (const FailureCase& failure : cases) {
    SCOPED_TRACE(failure.reason_fragment);
    Field accepted = make_mf(*ba_, *dm_, 3);
    copy3(*U0_, accepted);
    Field state = make_mf(*ba_, *dm_, 3);
    copy3(accepted, state);
    FallibleSourceModel model{failure.status, reason};

    auto outcome = pops::backward_euler_source(model, no_providers, state, kDt,
                                               pops::NewtonOptions{}, test_execution_lane());
    EXPECT_EQ(outcome.report().status, pops::SolveStatus::kInvalidEvaluation);
    EXPECT_EQ(outcome.report().action, failure.action);
    EXPECT_NE(outcome.report().reason.find(failure.reason_fragment), std::string::npos);
    EXPECT_NE(outcome.report().reason.find(std::to_string(reason)), std::string::npos);
    EXPECT_EQ(max_difference3(state, accepted), 0.0);
    const auto consumption = failure.action == pops::SolveAction::kRejectAttempt
                                 ? pops::SolveConsumption::kRejectAttempt
                                 : pops::SolveConsumption::kFailRun;
    EXPECT_EQ(outcome.consume(consumption).action, failure.action);
    EXPECT_EQ(max_difference3(state, accepted), 0.0);
  }
}

TEST_F(NewtonRobustnessTest,
       fallible_analytic_jacobian_rejects_privately_and_success_keeps_legacy_result) {
  static_assert(pops::HasFallibleSourceJacobianEvaluationFor<FallibleJacobianModel, Providers>);
  constexpr std::uint32_t reason = 0x1234abcdu;
  Field accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  Field rejected_state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, rejected_state);
  FallibleJacobianModel rejected_model;
  rejected_model.evaluation = pops::ImplicitEvaluationStatus::kReject;
  rejected_model.reason = reason;

  auto rejected = pops::backward_euler_source(rejected_model, no_providers, rejected_state, kDt,
                                              pops::NewtonOptions{}, test_execution_lane());
  EXPECT_EQ(rejected.report().status, pops::SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(rejected.report().action, pops::SolveAction::kRejectAttempt);
  EXPECT_NE(rejected.report().reason.find("evaluation_reject"), std::string::npos);
  EXPECT_NE(rejected.report().reason.find(std::to_string(reason)), std::string::npos);
  EXPECT_EQ(max_difference3(rejected_state, accepted), 0.0);
  (void)rejected.consume(pops::SolveConsumption::kRejectAttempt);
  EXPECT_EQ(max_difference3(rejected_state, accepted), 0.0);

  pops::NewtonOptions options;
  options.max_iters = 25;
  options.rel_tol = 1e-12;
  options.abs_tol = 1e-13;
  Field fallible_state = make_mf(*ba_, *dm_, 3);
  Field legacy_state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, fallible_state);
  copy3(accepted, legacy_state);
  FallibleJacobianModel fallible_model;
  JacStiffModel legacy_model;
  auto fallible = pops::backward_euler_source(fallible_model, no_providers, fallible_state, kDt,
                                              options, test_execution_lane());
  auto legacy = pops::backward_euler_source(legacy_model, no_providers, legacy_state, kDt, options,
                                            test_execution_lane());
  ASSERT_TRUE(fallible.report().solved_value_available());
  ASSERT_TRUE(legacy.report().solved_value_available());
  (void)fallible.consume(pops::SolveConsumption::kAccept);
  (void)legacy.consume(pops::SolveConsumption::kAccept);
  EXPECT_EQ(max_difference3(fallible_state, legacy_state), 0.0);
}

// (3e) Meme un succes reste prive jusqu'a sa consommation explicite, puis ne peut etre consomme deux
// fois.
TEST_F(NewtonRobustnessTest, prepared_success_publishes_only_on_single_accept) {
  StiffModel model;
  Field accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  Field state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);
  pops::NewtonOptions options;
  options.max_iters = 25;
  options.rel_tol = 1e-12;
  options.abs_tol = 1e-13;

  auto outcome =
      pops::backward_euler_source(model, no_providers, state, kDt, options, test_execution_lane());
  ASSERT_TRUE(outcome.report().solved_value_available());
  EXPECT_EQ(max_difference3(state, accepted), 0.0) << "le candidat est visible avant consommation";
  EXPECT_THROW(outcome.consume(pops::SolveConsumption::kFailRun), std::logic_error);
  EXPECT_EQ(max_difference3(state, accepted), 0.0)
      << "une action incompatible a consomme ou publie le candidat";
  const pops::SolveReport solved = outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(solved.solved_value_available());
  EXPECT_GT(max_difference3(state, accepted), 0.0);
  EXPECT_THROW(outcome.consume(pops::SolveConsumption::kAccept), std::logic_error);
}

TEST_F(NewtonRobustnessTest, publication_layout_failure_does_not_consume_the_outcome) {
  StiffModel model;
  Field accepted = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, accepted);
  Field state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);

  auto outcome = pops::backward_euler_source(model, no_providers, state, kDt, pops::NewtonOptions{},
                                             test_execution_lane());
  ASSERT_TRUE(outcome.report().solved_value_available());

  // A caller changing the destination structure between prepare and consume used to let lincomb
  // enter an incompatible layout after the outcome had already been marked consumed. The complete
  // layout guard now rejects before publication and leaves the valid outcome available.
  state = make_mf(*ba_, *dm_, 3, filled_ranked<pops::Extent<kDim>>(1));
  EXPECT_THROW(outcome.consume(pops::SolveConsumption::kAccept), std::logic_error);

  state = make_mf(*ba_, *dm_, 3);
  copy3(accepted, state);
  const pops::SolveReport solved = outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(solved.solved_value_available());
  EXPECT_GT(max_difference3(state, accepted), 0.0);
}

// (4) Le contrat par defaut converge et le rapport optionnel ne change pas le candidat publie.
TEST_F(NewtonRobustnessTest, default_contract_converges_with_or_without_diagnostics) {
  StiffModel m;
  Field Ua = make_mf(*ba_, *dm_, 3), Ub = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, Ua);
  copy3(*U0_, Ub);

  pops::NewtonOptions odef;
  auto without_report =
      pops::backward_euler_source(m, no_providers, Ua, kDt, odef, test_execution_lane());
  ASSERT_TRUE(without_report.report().solved_value_available());
  (void)without_report.consume(pops::SolveConsumption::kAccept);
  pops::NewtonReport repo;
  auto with_report =
      pops::backward_euler_source(m, no_providers, Ub, kDt, odef, test_execution_lane(), {}, &repo);
  ASSERT_TRUE(with_report.report().solved_value_available());
  (void)with_report.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(repo.converged);

  EXPECT_EQ(max_difference3(Ua, Ub), 0.0)
      << "les diagnostics optionnels ont modifie le candidat sain";
  std::printf("OK  (4) diagnostics optionnels : meme etat accepte\n");
}

// (5) JACOBIEN ANALYTIQUE (vague 3) : meme racine que les differences finies.
TEST_F(NewtonRobustnessTest, analytic_jacobian_matches_finite_difference_root) {
  static_assert(!pops::HasSourceJacobianFor<StiffModel, Providers>,
                "StiffModel sans jacobien : FD historiques");
  static_assert(pops::HasSourceJacobianFor<JacStiffModel, Providers>,
                "JacStiffModel doit declarer le trait");

  StiffModel m;
  Field U = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, U);
  pops::NewtonOptions opts;
  opts.max_iters = 25;
  opts.rel_tol = 1e-12;
  opts.abs_tol = 1e-13;
  pops::NewtonReport rep;
  auto finite_difference =
      pops::backward_euler_source(m, no_providers, U, kDt, opts, test_execution_lane(), {}, &rep);
  ASSERT_TRUE(finite_difference.report().solved_value_available());
  (void)finite_difference.consume(pops::SolveConsumption::kAccept);
  ASSERT_TRUE(rep.converged && rep.n_failed == 0) << "racine FD de reference non convergee";

  JacStiffModel jm;
  Field Uj = make_mf(*ba_, *dm_, 3);
  copy3(*U0_, Uj);
  pops::NewtonReport repj;
  auto analytic = pops::backward_euler_source(jm, no_providers, Uj, kDt, opts,
                                              test_execution_lane(), {}, &repj);
  ASSERT_TRUE(analytic.report().solved_value_available());
  (void)analytic.consume(pops::SolveConsumption::kAccept);

  const double jdiff = max_difference3(U, Uj);
  EXPECT_TRUE(repj.converged) << "jacobien analytique : non converge";
  EXPECT_TRUE(jdiff <= 1e-9) << "jacobien analytique : ecart racine " << jdiff << " > 1e-9";
  std::printf("OK  (5) jacobien analytique : meme racine que les FD (ecart %.1e), iters %.0f\n",
              jdiff, static_cast<double>(repj.max_iters_used));
}

namespace {

struct SquareRootResidual {
  POPS_HD void operator()(const Real (&x)[1], Real (&residual)[1]) const {
    residual[0] = x[0] * x[0] - Real(2);
  }
};

struct SquareRootJacobian {
  POPS_HD bool operator()(const Real (&x)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(2) * x[0];
    return true;
  }
};

struct ConstantResidual {
  POPS_HD void operator()(const Real (&)[1], Real (&residual)[1]) const { residual[0] = Real(1); }
};

struct ZeroJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(0);
    return true;
  }
};

struct InvalidResidual {
  POPS_HD void operator()(const Real (&)[1], Real (&residual)[1]) const {
    residual[0] = std::numeric_limits<Real>::quiet_NaN();
  }
};

struct PositiveDomain {
  POPS_HD bool operator()(const Real (&x)[1], int* component) const {
    if (component != nullptr)
      *component = x[0] >= Real(0) ? -1 : 0;
    return x[0] >= Real(0);
  }
};

struct LinearResidual {
  POPS_HD void operator()(const Real (&x)[1], Real (&residual)[1]) const {
    residual[0] = x[0] - Real(1);
  }
};

struct WrongDirectionJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(-1);
    return true;
  }
};

struct TinyScaledResidual {
  POPS_HD void operator()(const Real (&x)[1], Real (&residual)[1]) const {
    residual[0] = Real(1e-20) * (x[0] - Real(1));
  }
};

struct TinyScaledJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = Real(1e-20);
    return true;
  }
};

struct HugeJacobian {
  POPS_HD bool operator()(const Real (&)[1], Real (&jacobian)[1][1]) const {
    jacobian[0][0] = std::numeric_limits<Real>::max();
    return true;
  }
};

struct FallibleScalarResidual {
  pops::LocalNonlinearEvaluationStatus status = pops::LocalNonlinearEvaluationStatus::kOk;
  std::uint32_t reason = 0;

  POPS_HD pops::LocalNonlinearEvaluationResult operator()(const Real (&x)[1],
                                                          Real (&residual)[1]) const {
    residual[0] = x[0] - Real(1);
    switch (status) {
      case pops::LocalNonlinearEvaluationStatus::kOk:
        return pops::LocalNonlinearEvaluationResult::ok();
      case pops::LocalNonlinearEvaluationStatus::kRetry:
        return pops::LocalNonlinearEvaluationResult::retry(reason);
      case pops::LocalNonlinearEvaluationStatus::kReject:
        return pops::LocalNonlinearEvaluationResult::reject(reason);
      case pops::LocalNonlinearEvaluationStatus::kFailed:
        return pops::LocalNonlinearEvaluationResult::failed(reason);
      case pops::LocalNonlinearEvaluationStatus::kInvalid:
        return pops::LocalNonlinearEvaluationResult::invalid(reason);
    }
    return pops::LocalNonlinearEvaluationResult::invalid(reason);
  }
};

pops::PreparedLocalNonlinearControls scalar_controls() {
  pops::PreparedLocalNonlinearControls controls;
  controls.max_iterations = 20;
  controls.absolute_tolerance = Real(1e-13);
  return controls;
}

}  // namespace

TEST(PreparedLocalNonlinear, FallibleEvaluationStatusAndReasonRemainDistinct) {
  constexpr std::uint32_t reason = 0x1234abcdu;
  const Real initial[1] = {Real(2)};
  struct Expected {
    pops::LocalNonlinearEvaluationStatus evaluation;
    pops::LocalNonlinearStatus solve;
  };
  const std::array<Expected, 4> cases{{
      {pops::LocalNonlinearEvaluationStatus::kRetry, pops::LocalNonlinearStatus::kEvaluationRetry},
      {pops::LocalNonlinearEvaluationStatus::kReject,
       pops::LocalNonlinearStatus::kEvaluationReject},
      {pops::LocalNonlinearEvaluationStatus::kFailed,
       pops::LocalNonlinearStatus::kEvaluationFailed},
      {pops::LocalNonlinearEvaluationStatus::kInvalid,
       pops::LocalNonlinearStatus::kInvalidEvaluation},
  }};
  for (const Expected& expected : cases) {
    const auto problem = pops::prepare_local_nonlinear_problem<1>(
        FallibleScalarResidual{expected.evaluation, reason},
        pops::FiniteDifferenceLocalJacobian<1>{}, pops::AcceptAllLocalCandidates<1>{},
        scalar_controls());
    const auto result = pops::solve_prepared_local_nonlinear(problem, initial);
    EXPECT_EQ(result.status, expected.solve);
    EXPECT_EQ(result.reason_code, reason);
    EXPECT_EQ(result.value[0], initial[0]);
  }
}

TEST(PreparedLocalNonlinear, FatalImplicitFailureDominatesRecoverableCollectiveStatus) {
  using pops::LocalNonlinearStatus;
  EXPECT_GT(pops::local_nonlinear_status_priority(LocalNonlinearStatus::kSingularJacobian),
            pops::local_nonlinear_status_priority(LocalNonlinearStatus::kEvaluationReject));
  EXPECT_GT(pops::local_nonlinear_status_priority(LocalNonlinearStatus::kInvalidEvaluation),
            pops::local_nonlinear_status_priority(LocalNonlinearStatus::kEvaluationRetry));
  for (int priority = 0; priority <= 9; ++priority) {
    const LocalNonlinearStatus status = pops::local_nonlinear_status_from_priority(priority);
    EXPECT_EQ(pops::local_nonlinear_status_priority(status), priority);
  }
}

TEST(PreparedLocalNonlinear, FiniteDifferenceAnalyticAndAdUseOneOutcomeContract) {
  const Real initial[1] = {Real(2)};
  const auto controls = scalar_controls();
  const auto finite_difference = pops::prepare_local_nonlinear_problem<1>(
      SquareRootResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, controls);
  const auto analytic = pops::prepare_local_nonlinear_problem<1>(
      SquareRootResidual{},
      pops::AnalyticLocalJacobian<1, SquareRootJacobian>{SquareRootJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, controls);
  const auto automatic_differentiation = pops::prepare_local_nonlinear_problem<1>(
      SquareRootResidual{},
      pops::AutomaticDifferentiationLocalJacobian<1, SquareRootJacobian>{SquareRootJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, controls);

  const auto fd = pops::solve_prepared_local_nonlinear(finite_difference, initial);
  const auto exact = pops::solve_prepared_local_nonlinear(analytic, initial);
  const auto ad = pops::solve_prepared_local_nonlinear(automatic_differentiation, initial);
  ASSERT_TRUE(fd.solved());
  ASSERT_TRUE(exact.solved());
  ASSERT_TRUE(ad.solved());
  EXPECT_NEAR(fd.value[0], std::sqrt(Real(2)), 1e-11);
  EXPECT_NEAR(exact.value[0], fd.value[0], 1e-11);
  EXPECT_NEAR(ad.value[0], exact.value[0], 1e-13);
  EXPECT_EQ(initial[0], Real(2));
}

TEST(PreparedLocalNonlinear, PivotThresholdIsRelativeToTheScaledEquation) {
  auto controls = scalar_controls();
  controls.absolute_tolerance = Real(1e-30);
  const Real initial[1] = {Real(2)};
  const auto problem = pops::prepare_local_nonlinear_problem<1>(
      TinyScaledResidual{},
      pops::AnalyticLocalJacobian<1, TinyScaledJacobian>{TinyScaledJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, controls);
  const auto result = pops::solve_prepared_local_nonlinear(problem, initial);
  ASSERT_TRUE(result.solved());
  EXPECT_NEAR(result.value[0], Real(1), 1e-13);
}

TEST(PreparedLocalNonlinear, EveryFailureClassIsExplicitAndLeavesTheGuessUntouched) {
  Real initial[1] = {Real(10)};
  auto budget_controls = scalar_controls();
  budget_controls.max_iterations = 1;
  const auto budget_problem = pops::prepare_local_nonlinear_problem<1>(
      SquareRootResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, budget_controls);
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(budget_problem, initial).status,
            pops::LocalNonlinearStatus::kIterationLimit);

  const auto singular_problem = pops::prepare_local_nonlinear_problem<1>(
      ConstantResidual{}, pops::AnalyticLocalJacobian<1, ZeroJacobian>{ZeroJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, scalar_controls());
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(singular_problem, initial).status,
            pops::LocalNonlinearStatus::kSingularJacobian);

  const auto invalid_problem = pops::prepare_local_nonlinear_problem<1>(
      InvalidResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, scalar_controls());
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(invalid_problem, initial).status,
            pops::LocalNonlinearStatus::kInvalidEvaluation);

  const auto overflow_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::AnalyticLocalJacobian<1, HugeJacobian>{HugeJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, scalar_controls(), Real(2), Real(1));
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(overflow_problem, initial).status,
            pops::LocalNonlinearStatus::kInvalidEvaluation);

  Real inadmissible_initial[1] = {Real(-1)};
  const auto inadmissible_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{}, PositiveDomain{},
      scalar_controls());
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(inadmissible_problem, inadmissible_initial).status,
            pops::LocalNonlinearStatus::kInadmissibleCandidate);

  auto safeguard_controls = scalar_controls();
  safeguard_controls.safeguard = pops::LocalSafeguardKind::kBacktrackingLineSearch;
  safeguard_controls.max_backtracks = 3;
  safeguard_controls.minimum_step = Real(0.01);
  Real safeguard_initial[1] = {Real(0)};
  const auto safeguard_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{},
      pops::AnalyticLocalJacobian<1, WrongDirectionJacobian>{WrongDirectionJacobian{}},
      pops::AcceptAllLocalCandidates<1>{}, safeguard_controls);
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(safeguard_problem, safeguard_initial).status,
            pops::LocalNonlinearStatus::kSafeguardFailure);

  const auto unsupported_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::UnsupportedLocalJacobian<1>{}, pops::AcceptAllLocalCandidates<1>{},
      scalar_controls());
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(unsupported_problem, safeguard_initial).status,
            pops::LocalNonlinearStatus::kUnsupportedCapability);

  auto invalid_controls = scalar_controls();
  invalid_controls.absolute_tolerance = Real(0);
  invalid_controls.relative_tolerance = Real(0);
  invalid_controls.step_tolerance = Real(0);
  const auto invalid_controls_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, invalid_controls);
  EXPECT_EQ(
      pops::solve_prepared_local_nonlinear(invalid_controls_problem, safeguard_initial).status,
      pops::LocalNonlinearStatus::kUnsupportedCapability);

  auto ignored_damping_controls = scalar_controls();
  ignored_damping_controls.initial_step = Real(0.5);
  const auto ignored_damping_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, ignored_damping_controls);
  EXPECT_EQ(pops::solve_prepared_local_nonlinear(ignored_damping_problem, safeguard_initial).status,
            pops::LocalNonlinearStatus::kUnsupportedCapability);

  auto unknown_safeguard_controls = scalar_controls();
  unknown_safeguard_controls.safeguard = static_cast<pops::LocalSafeguardKind>(99);
  const auto unknown_safeguard_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, unknown_safeguard_controls);
  EXPECT_EQ(
      pops::solve_prepared_local_nonlinear(unknown_safeguard_problem, safeguard_initial).status,
      pops::LocalNonlinearStatus::kUnsupportedCapability);

  auto maximum_backtracks_controls = scalar_controls();
  maximum_backtracks_controls.max_backtracks = std::numeric_limits<int>::max();
  const auto maximum_backtracks_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, maximum_backtracks_controls);
  const auto maximum_backtracks_result =
      pops::solve_prepared_local_nonlinear(maximum_backtracks_problem, safeguard_initial);
  EXPECT_TRUE(maximum_backtracks_result.solved());
  EXPECT_NEAR(maximum_backtracks_result.value[0], Real(1), 1e-12);

  auto maximum_attempts_controls = maximum_backtracks_controls;
  maximum_attempts_controls.max_evaluations = 16;
  maximum_attempts_controls.safeguard = pops::LocalSafeguardKind::kBacktrackingLineSearch;
  const auto maximum_attempts_problem = pops::prepare_local_nonlinear_problem<1>(
      LinearResidual{}, pops::FiniteDifferenceLocalJacobian<1>{},
      pops::AcceptAllLocalCandidates<1>{}, maximum_attempts_controls);
  const auto maximum_attempts_result =
      pops::solve_prepared_local_nonlinear(maximum_attempts_problem, safeguard_initial);
  EXPECT_TRUE(maximum_attempts_result.solved());
  EXPECT_NEAR(maximum_attempts_result.value[0], Real(1), 1e-12);

  EXPECT_EQ(initial[0], Real(10));
  EXPECT_EQ(inadmissible_initial[0], Real(-1));
  EXPECT_EQ(safeguard_initial[0], Real(0));
}

TEST(LocalNonlinearCollective, SignedLargeIndicesPreservePriorityAndLexicographicOrder) {
  pops::Index<kDim> negative{};
  pops::Index<kDim> positive{};
  for (int axis = 0; axis < kDim; ++axis) {
    negative[axis] = -1000000000 + 300000000 * axis;
    positive[axis] = 1000000000 - 300000000 * axis;
  }
  const Layout boxes(std::vector<pops::Box<kDim>>{pops::Box<kDim>{negative, negative},
                                                  pops::Box<kDim>{positive, positive}});
  pops::Extent<kDim> rank_extent = filled_ranked<pops::Extent<kDim>>(1);
  rank_extent[0] = pops::n_ranks();
  const Distribution distribution = Distribution::replicated(
      boxes, pops::mesh::RankSpace<kDim>{pops::Index<kDim>{}, rank_extent});
  Field statistics = make_mf(boxes, distribution, 11);
  const int recoverable =
      pops::local_nonlinear_status_priority(pops::LocalNonlinearStatus::kEvaluationReject);
  const int fatal =
      pops::local_nonlinear_status_priority(pops::LocalNonlinearStatus::kInvalidEvaluation);

  for (std::size_t local = 0; local < statistics.local_size(); ++local)
    pops::for_each_cell(statistics.box(local), FillFailureStatistics{statistics.fab(local).view(),
                                                                     recoverable, fatal, false});

  auto location = pops::collective_first_local_nonlinear_failure(statistics, fatal, 10, 8,
                                                                 test_execution_lane());
  ASSERT_TRUE(location.found);
  EXPECT_EQ(location.priority, fatal);
  EXPECT_EQ(location.index, positive);
  EXPECT_EQ(location.component, 3);

  for (std::size_t local = 0; local < statistics.local_size(); ++local)
    pops::for_each_cell(statistics.box(local), FillFailureStatistics{statistics.fab(local).view(),
                                                                     recoverable, fatal, true});
  location = pops::collective_first_local_nonlinear_failure(statistics, fatal, 10, 8,
                                                            test_execution_lane());
  ASSERT_TRUE(location.found);
  EXPECT_EQ(location.index, negative);
  EXPECT_EQ(location.component, 7);
  EXPECT_THROW((void)pops::collective_first_local_nonlinear_failure(statistics, fatal + 1, 10, 8,
                                                                    test_execution_lane()),
               std::runtime_error);
}
