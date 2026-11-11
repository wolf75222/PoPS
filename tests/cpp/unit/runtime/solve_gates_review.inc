// Spec 0.3 SV8 (N9, ROM-90, lot 2): the original-residual gate of a mesh solve, decided outside its
// realization. Included after test_program_context_contract.cpp helpers.

namespace sv8_gate_test {

using pops::runtime::program::OriginalResidualGate;
using pops::runtime::program::apply_original_residual_gate;
using pops::runtime::program::prepared_affine_original_residual;

OriginalResidualGate gate(Real norm, Real atol, Real rtol, Real scale) {
  OriginalResidualGate value;
  value.norm = norm;
  value.atol = atol;
  value.rtol = rtol;
  value.scale = scale;
  return value;
}

SolveReport solved_report() {
  SolveReport report;
  report.iters = 3;
  report.residual_norm = Real(1e-12);
  report.reference_residual_norm = Real(1);
  report.rel_residual = Real(1e-12);
  report.mark_solved("realization_converged");
  return report;
}

// A problem whose residual norms are scripted: the gate must ask for R(0) first, then R(u), from
// the problem itself, and never read anything from the realization.
struct ScriptedProblem {
  struct Field {
    Real tag = Real(0);
  };
  mutable std::vector<Real> iterates;
  Real zero_norm = Real(0);
  Real candidate_norm = Real(0);

  void true_residual(Field& out, const Field& rhs, const Field& iterate) const {
    iterates.push_back(iterate.tag);
    out.tag = rhs.tag - iterate.tag;
  }
  Real residual_norm(const Field& value) const {
    return value.tag == Real(10) ? zero_norm : candidate_norm;
  }
};

}  // namespace sv8_gate_test

TEST(SolveGates, BoundIsAtolPlusRtolTimesScale) {
  using namespace sv8_gate_test;
  EXPECT_EQ(gate(Real(0), Real(1e-13), Real(1e-10), Real(4)).bound(),
            Real(1e-13) + Real(1e-10) * Real(4));
  // At the bound the gate passes; one step above, it refuses.
  const Real bound = gate(Real(0), Real(0.5), Real(0.25), Real(2)).bound();
  EXPECT_TRUE(gate(bound, Real(0.5), Real(0.25), Real(2)).refusal().empty());
  const std::string above =
      gate(std::nextafter(bound, Real(2)), Real(0.5), Real(0.25), Real(2)).refusal();
  EXPECT_NE(above.find("SV8 gate original_residual"), std::string::npos) << above;
}

TEST(SolveGates, RefusalsFollowTheOrderOfRelationsSolve) {
  using namespace sv8_gate_test;
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const Real inf = std::numeric_limits<Real>::infinity();
  // An invalid norm comes first, even with an invalid accuracy.
  EXPECT_NE(gate(nan, nan, Real(1), Real(1)).refusal().find("invalid_norm"), std::string::npos);
  EXPECT_NE(gate(Real(-1), Real(1), Real(1), Real(1)).refusal().find("invalid_norm"),
            std::string::npos);
  EXPECT_NE(gate(inf, Real(1), Real(1), Real(1)).refusal().find("invalid_norm"),
            std::string::npos);
  // Then an accuracy whose bound is not finite or is negative.
  EXPECT_NE(gate(Real(1), Real(1), inf, Real(1)).refusal().find("invalid_accuracy"),
            std::string::npos);
  EXPECT_NE(gate(Real(1), nan, Real(1), Real(1)).refusal().find("invalid_accuracy"),
            std::string::npos);
  EXPECT_NE(gate(Real(0), Real(-1), Real(0), Real(0)).refusal().find("invalid_accuracy"),
            std::string::npos);
  // Then the residual above its bound.
  EXPECT_NE(gate(Real(3), Real(1), Real(1), Real(1)).refusal().find("original_residual"),
            std::string::npos);
  EXPECT_TRUE(gate(Real(0), Real(0), Real(0), Real(0)).refusal().empty());
}

TEST(SolveGates, ARefusedReportBecomesTheGateFailureWithTheConsumerAction) {
  using namespace sv8_gate_test;
  SolveReport report = solved_report();
  EXPECT_FALSE(apply_original_residual_gate(report, gate(Real(1), Real(0), Real(0.5), Real(1)),
                                            SolveAction::kRejectAttempt, "spatial_implicit"));
  EXPECT_EQ(report.status, SolveStatus::kSafeguardFailure);
  EXPECT_EQ(report.action, SolveAction::kRejectAttempt);
  EXPECT_FALSE(report.solved_value_available());
  EXPECT_EQ(report.reason.rfind("spatial_implicit SV8 gate original_residual", 0), 0u)
      << report.reason;
  // The realization's own report stays as evidence.
  EXPECT_NE(report.reason.find("realization reported realization_converged"), std::string::npos)
      << report.reason;
  EXPECT_EQ(report.iters, 3);
  EXPECT_TRUE(solve_report_is_publishable(report, 3));

  SolveReport passing = solved_report();
  EXPECT_TRUE(apply_original_residual_gate(passing, gate(Real(0.5), Real(0), Real(0.5), Real(1)),
                                           SolveAction::kFailRun, "spatial_implicit"));
  EXPECT_TRUE(passing.solved_value_available());
  EXPECT_EQ(passing.reason, "realization_converged");

  // A failed report is never touched: the gates run on a solved candidate only.
  SolveReport failed;
  failed.mark_failed(SolveStatus::kIterationLimit, SolveAction::kFailRun, "realization_budget");
  EXPECT_FALSE(apply_original_residual_gate(failed, gate(Real(9), Real(0), Real(0), Real(0)),
                                            SolveAction::kRejectAttempt, "spatial_implicit"));
  EXPECT_EQ(failed.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(failed.reason, "realization_budget");
}

TEST(SolveGates, TheResidualAndItsScaleComeFromTheProblem) {
  using namespace sv8_gate_test;
  ScriptedProblem problem;
  problem.zero_norm = Real(4);
  problem.candidate_norm = Real(3e-10);
  ScriptedProblem::Field residual, rhs{Real(10)}, iterate{Real(7)}, zero{Real(0)};
  const OriginalResidualGate measured = prepared_affine_original_residual(
      problem, residual, rhs, iterate, zero, Real(1e-13), Real(1e-10));
  ASSERT_EQ(problem.iterates.size(), 2u);
  EXPECT_EQ(problem.iterates[0], Real(0));  // R(0) first: the request scale
  EXPECT_EQ(problem.iterates[1], Real(7));  // then R(u)
  EXPECT_EQ(measured.scale, Real(4));
  EXPECT_EQ(measured.norm, Real(3e-10));
  EXPECT_EQ(measured.atol, Real(1e-13));
  EXPECT_EQ(measured.rtol, Real(1e-10));
  EXPECT_TRUE(measured.refusal().empty());
  problem.candidate_norm = Real(5e-10);
  EXPECT_NE(prepared_affine_original_residual(problem, residual, rhs, iterate, zero, Real(1e-13),
                                              Real(1e-10))
                .refusal()
                .find("original_residual"),
            std::string::npos);
}

TEST(SolveGates, ARefusedNativeCandidateRollsBackLikeAFailedSolve) {
  using namespace sv8_gate_test;
  ensure_kokkos();
  comm_init();
  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.review.sv8-gate-refusal");
  add_gas(sim);
  sim.set_state("gas", ic(4));
  sim.install_program_step([](double) {});
  sim.begin_step_transaction();
  const auto state = sim.get_state("gas");
  const auto time = sim.time();
  const auto step = sim.macro_step();
  auto pending = sim.solve_fields();
  ASSERT_TRUE(pending.report().solved_value_available()) << pending.report().reason;
  const int iterations = pending.report().iters;
  EXPECT_FALSE(apply_original_residual_gate(
      pending, gate(Real(1), Real(0), Real(0), Real(0)), SolveAction::kFailRun, "solve_linear"));
  EXPECT_EQ(pending.report().status, SolveStatus::kSafeguardFailure);
  EXPECT_EQ(pending.report().iters, iterations);
  // A refused candidate is a failure: it cannot be accepted, refused again or discarded.
  EXPECT_THROW((void)pending.consume(SolveConsumption::kAccept), std::logic_error);
  EXPECT_THROW(pending.refuse_candidate(SolveStatus::kSafeguardFailure, SolveAction::kFailRun, "x"),
               std::logic_error);
  const SolveReport consumed = pending.consume(SolveConsumption::kFailRun);
  EXPECT_EQ(consumed.status, SolveStatus::kSafeguardFailure);
  EXPECT_EQ(consumed.action, SolveAction::kFailRun);
  EXPECT_THROW(pending.refuse_candidate(SolveStatus::kSafeguardFailure, SolveAction::kFailRun, "x"),
               std::logic_error);
  EXPECT_EQ(sim.get_state("gas"), state);
  EXPECT_EQ(sim.time(), time);
  EXPECT_EQ(sim.macro_step(), step);
  EXPECT_NO_THROW(sim.rollback_step_transaction());
  EXPECT_NO_THROW(sim.step(.01));
}

TEST(SolveGates, AGateThatCannotBeMeasuredRefusesTheCandidate) {
  using namespace sv8_gate_test;
  using pops::runtime::program::apply_measured_original_residual_gate;
  SolveOutcome refused = SolveOutcome::serial(solved_report());
  EXPECT_FALSE(apply_measured_original_residual_gate(
      refused, []() -> OriginalResidualGate { throw std::logic_error("session busy"); },
      SolveAction::kRejectAttempt, "solve_linear"));
  EXPECT_EQ(refused.report().status, SolveStatus::kInvalidEvaluation);
  EXPECT_EQ(refused.report().action, SolveAction::kRejectAttempt);
  EXPECT_NE(refused.report().reason.find("SV8 gate measurement failed: session busy"),
            std::string::npos)
      << refused.report().reason;
  EXPECT_EQ(refused.consume(SolveConsumption::kRejectAttempt).status,
            SolveStatus::kInvalidEvaluation);

  SolveOutcome passing = SolveOutcome::serial(solved_report());
  EXPECT_TRUE(apply_measured_original_residual_gate(
      passing, [] { return gate(Real(0.5), Real(0), Real(0.5), Real(1)); },
      SolveAction::kFailRun, "solve_linear"));
  EXPECT_TRUE(passing.consume(SolveConsumption::kAccept).solved_value_available());
}
