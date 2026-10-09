// Independent T4 acceptance cases. Included after test_program_context_contract.cpp helpers.
// Tests use actual System field solves and PreparedLinear workspace publication, not a mock lease.
TEST(ProgramContextContract, DiscardSolvedNativeCandidateClosesPendingAttempt) {
  ensure_kokkos();
  comm_init();
  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.review.discard-native-candidate");
  add_gas(sim);
  sim.set_state("gas", ic(4));
  sim.install_program_step([](double) {});
  sim.begin_step_transaction();
  const auto state = sim.get_state("gas");
  const auto time = sim.time();
  const auto step = sim.macro_step();
  auto pending = sim.solve_fields();
  ASSERT_TRUE(pending.report().solved_value_available()) << pending.report().reason;
  const auto numerical = pending.report();
  EXPECT_THROW(sim.step(.01), std::exception);
  EXPECT_NO_THROW(pending.discard_candidate());
  EXPECT_EQ(pending.report().status, numerical.status);
  EXPECT_EQ(pending.report().action, numerical.action);
  EXPECT_EQ(pending.report().reason, numerical.reason);
  EXPECT_EQ(sim.get_state("gas"), state);
  EXPECT_EQ(sim.time(), time);
  EXPECT_EQ(sim.macro_step(), step);
  EXPECT_THROW((void)pending.consume(SolveConsumption::kAccept), std::logic_error);
  EXPECT_NO_THROW(sim.rollback_step_transaction());
  EXPECT_NO_THROW(sim.step(.01));
}

TEST(ProgramContextContract, PendingSolveBlocksEveryNativeStepEntryBeforeBody) {
  ensure_kokkos();
  comm_init();
  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.review.pending-step-entry");
  add_gas(sim);
  sim.set_state("gas", ic(4));
  int calls = 0;
  sim.install_program_step([&](double) { ++calls; });
  sim.begin_step_transaction();
  auto pending = sim.solve_fields();
  ASSERT_TRUE(pending.report().solved_value_available()) << pending.report().reason;
  const auto state = sim.get_state("gas");
  const auto time = sim.time();
  const auto step = sim.macro_step();
  EXPECT_THROW(sim.step(.01), std::exception);
  EXPECT_THROW((void)sim.step_cfl(.2, 1e-12, .01, 0.0), std::exception);
  EXPECT_THROW((void)sim.advance_program_region(.01), std::exception);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(sim.time(), time);
  EXPECT_EQ(sim.macro_step(), step);
  EXPECT_EQ(sim.get_state("gas"), state);
  EXPECT_NO_THROW((void)pending.consume(SolveConsumption::kAccept));
  EXPECT_NO_THROW(sim.rollback_step_transaction());
}

TEST(ProgramContextContract, RevokedPreparedLinearOutcomeReleasesSurvivingWorkspace) {
  ensure_kokkos();
  comm_init();
  auto sim = std::make_unique<NativeSystem>(native_config(4));
  install_execution_lane(*sim, "pops.review.linear-orphan");
  add_gas(*sim);
  sim->set_state("gas", ic(4));
  sim->set_program_block_map({0});
  auto context = std::make_unique<NativeProgramContext>(sim.get());
  context->begin_step(Real(.001));
  context->set_stage_time(0, 1);
  const auto& state = context->state(0);
  NativeField prototype = native_field_like(state, state.ncomp(), state.ghosts());
  const KrylovFootprint<kTestDimension> footprint{prototype.ncomp(), prototype.ghosts(), false};
  const auto method = cg_krylov_method<kTestDimension>();
  const OperatorFingerprint authority{501, 502, 503, 504};
  const OperatorFingerprint resources{505, 506, 507, 508};
  const auto snapshot = context->operator_evaluation_snapshot(authority, state, resources);
  PreparedAffineLinearProblem<kTestDimension> problem(
      prototype,
      PreparedAffineOperatorProvider<kTestDimension>::trusted_reentrant(
          [](NativeField& out, const NativeField& in) {
            detail::PreparedFieldAlgebra::copy(out, in);
          }, [] { return std::size_t{0}; }),
      PreparedLinearPreconditioner<kTestDimension>::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      PreparedNullspacePolicy<kTestDimension>::nonsingular(), [snapshot] { return snapshot; });
  KrylovWorkspace<kTestDimension> workspace(
      context->prepared_execution_communicator(), "pops.review.linear.workspace",
      "pops.review.linear.materialization", prototype, method, footprint);
  NativeField solution = native_field_like(prototype, prototype.ncomp(), prototype.ghosts());
  NativeField rhs = native_field_like(prototype, prototype.ncomp(), prototype.ghosts());
  solution.set_val(Real(0));
  rhs.set_val(Real(1));
  problem.prepare(snapshot);
  workspace.bind(problem);
  const KrylovControls<kTestDimension> controls{method, Real(1e-12), Real(0), 4};
  // The checkpoint protocol has already published its outer envelope at commit consensus.
  // Prevent new candidates in that sealed interval; finalization itself must remain noexcept.
  static_assert(noexcept(std::declval<NativeSystem&>().finalize_restart_transaction()));
  sim->begin_restart_transaction();
  sim->commit_restart_transaction();
  const auto expect_sealed_refusal = [&](auto operation) {
    std::string refusal;
    try {
      operation();
    } catch (const std::exception& error) {
      refusal = error.what();
    }
    EXPECT_NE(refusal.find("after the parent transaction was committed"), std::string::npos)
        << refusal;
  };
  expect_sealed_refusal([&] { (void)consume_solve_outcome(sim->solve_fields()); });
  expect_sealed_refusal([&] {
    (void)consume_solve_outcome(sim->solve_block_source(0, context->state(0), Real(.001),
                                                       NewtonOptions{}));
  });
  expect_sealed_refusal([&] {
    (void)consume_solve_outcome(
        context->solve_prepared_linear(problem, workspace, solution, rhs, controls));
  });
  sim->finalize_restart_transaction();
  EXPECT_EQ(sim->step_transaction_depth(), 0u);
  auto outcome = context->solve_prepared_linear(problem, workspace, solution, rhs, controls);
  ASSERT_TRUE(outcome.report().solved_value_available()) << outcome.report().reason;
  // Only System/Program ownership ends. Every numerical problem/workspace/field stays alive.
  context.reset();
  sim.reset();
  EXPECT_THROW((void)outcome.consume(SolveConsumption::kAccept), std::logic_error);
  // Refusal must release workspace+problem reservation, not strand their real native candidate.
  // A direct provider solve is valid because all its exact owners and inputs still exist.
  EXPECT_NO_THROW({
    auto next = solve_prepared_affine_outcome(problem, workspace, solution, rhs, controls);
    (void)consume_solve_outcome(std::move(next));
  });
}

TEST(ProgramContextContract, NativeStepRevokesEscapedSolveBeforeAcceptedSnapshotRestore) {
  ensure_kokkos();
  comm_init();
  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.review.escaped-step-outcome");
  add_gas(sim);
  sim.set_state("gas", ic(4));
  const auto accepted = sim.get_state("gas");
  std::optional<SolveOutcome> escaped;
  sim.install_program_step([&](double) { escaped.emplace(sim.solve_fields()); });
  EXPECT_THROW(sim.step(.01), std::exception);
  ASSERT_TRUE(escaped.has_value());
  ASSERT_TRUE(escaped->report().solved_value_available()) << escaped->report().reason;
  EXPECT_EQ(sim.time(), 0.0);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_EQ(sim.get_state("gas"), accepted);
  EXPECT_THROW((void)escaped->consume(SolveConsumption::kAccept), std::logic_error);
  // The stale refusal clears only its own outstanding count; the restored field owner can retry.
  sim.install_program_step([&](double) {
    auto next = sim.solve_fields();
    (void)consume_solve_outcome(std::move(next));
  });
  EXPECT_NO_THROW(sim.step(.01));
  EXPECT_DOUBLE_EQ(sim.time(), .01);
  EXPECT_EQ(sim.macro_step(), 1);
}
