// Runtime mechanics only. The public Python fixture uses the true FV exterior trace.
TEST(ProgramRuntime, IntegralCaptureRejectsRedigestedUntypedOrNoncanonicalUnits) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  const std::string canonical = R"({"kind":"physical_dimension","powers":[]})";
  for (const std::string bad : {
           std::string("not-json"),
           std::string(R"({"powers":[],"kind":"physical_dimension"})"),
           std::string(R"({"kind":"physical_dimension","powers":[["mass",2,4]]})"),
           std::string(R"({"kind":"physical_dimension","powers":[["mass",0,1]]})"),
           std::string(R"({"kind":"physical_dimension","powers":[["mass",1,1],["mass",2,1]]})")}) {
    for (const bool only_rank_zero : {false, true}) {
      SCOPED_TRACE(bad);
      SCOPED_TRACE(only_rank_zero);
      System<kNativeDimension> sim(unit_domain_config<kNativeDimension>(4));
      install_execution_lane(sim, "pops.test.integral-candidate.canonical-units");
      add_scalar(sim);
      sim.set_state("tracer", std::vector<double>(exact_cell_count<kNativeDimension>(4), 2.));
      auto ctx = runtime::program::make_program_execution_provider(&sim);
      ctx->configure_primary_clock("canonical-unit-clock");
      const auto key = [](const std::string& units) {
        return "pops.integral.v2/native-canonical/" + identity::sha256_hex(
            std::vector<std::uint8_t>(units.begin(), units.end())) + "/q";
      };
      const auto good_identity = key(canonical), bad_identity = key(bad);
      // Both declarations are identical on every rank, so the counterprobe
      // cannot pass merely because ledger declarations differ across ranks.
      ctx->declare_integral_state(good_identity, .7);
      ctx->declare_integral_state(bad_identity, .7);
      const auto ledger = sim.checkpoint_program_exchanges();
      const auto state = sim.get_state("tracer");
      bool consumed = false;
      ctx->install([&](double dt) {
        ctx->begin_step(dt);
        const bool inject = !only_rank_zero || ctx->prepared_execution_lane().rank() == 0;
        const auto& units = inject ? bad : canonical;
        const auto& identity = inject ? bad_identity : good_identity;
        const auto capture = ctx->capture_integral_candidate(identity, units);
        EXPECT_DOUBLE_EQ(ctx->integral_candidate_value(capture, identity, units), .7);
        consumed = true;
      });
      sim.set_program_block_map({0});
      if (ctx->prepared_execution_lane().size() == 1)
        EXPECT_THROW(sim.step(.01), std::invalid_argument);
      else
        EXPECT_THROW(sim.step(.01), std::runtime_error);
      EXPECT_FALSE(consumed);
      EXPECT_EQ(sim.get_state("tracer"), state);
      EXPECT_EQ(sim.checkpoint_program_exchanges(), ledger);
      EXPECT_DOUBLE_EQ(sim.time(), 0.);
      EXPECT_EQ(sim.macro_step(), 0);
    }
  }
}

TEST(ProgramRuntime, IntegralCapturePreservesCanonicalUnicodeAndLargeRationals) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  for (const std::string units : {
           std::string(R"({"kind":"physical_dimension","powers":[]})"),
           std::string(R"({"kind":"physical_dimension","powers":[["mass",-7,3],["\u6642\u9593",1,2]]})"),
           std::string(R"({"kind":"physical_dimension","powers":[["huge",18446744073709551617,18446744073709551616],["\ue000",1,1],["\ud800\udc00",1,1]]})")}) {
    SCOPED_TRACE(units);
    System<kNativeDimension> sim(unit_domain_config<kNativeDimension>(4));
    install_execution_lane(sim, "pops.test.integral-candidate.unicode-rational");
    add_scalar(sim);
    sim.set_state("tracer", std::vector<double>(exact_cell_count<kNativeDimension>(4), 2.));
    auto ctx = runtime::program::make_program_execution_provider(&sim);
    ctx->configure_primary_clock("unicode-unit-clock");
    const auto identity = "pops.integral.v2/native-unicode/" + identity::sha256_hex(
        std::vector<std::uint8_t>(units.begin(), units.end())) + "/q";
    ctx->declare_integral_state(identity, .7);
    const auto ledger = sim.checkpoint_program_exchanges();
    bool consumed = false;
    ctx->install([&](double dt) {
      ctx->begin_step(dt);
      const auto capture = ctx->capture_integral_candidate(identity, units);
      EXPECT_DOUBLE_EQ(ctx->integral_candidate_value(capture, identity, units), .7);
      consumed = true;
    });
    sim.set_program_block_map({0});
    EXPECT_NO_THROW(sim.step(.01));
    EXPECT_TRUE(consumed);
    EXPECT_EQ(sim.checkpoint_program_exchanges(), ledger);
    EXPECT_DOUBLE_EQ(sim.time(), .01);
    EXPECT_EQ(sim.macro_step(), 1);
  }
}

TEST(ProgramRuntime, IntegralCandidateCaptureAuthenticatesAndRevokesCollectively) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  System<kNativeDimension> sim(unit_domain_config<kNativeDimension>(n));
  install_execution_lane(sim, "pops.test.integral-candidate.capture");
  add_scalar(sim);
  std::vector<double> initial(exact_cell_count<kNativeDimension>(n), 2.);
  sim.set_state("tracer",initial);
  auto ctx = runtime::program::make_program_execution_provider(&sim);
  auto foreign_ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->configure_primary_clock("integral-clock");
  foreign_ctx->configure_primary_clock("integral-clock");
  const std::string units = R"({"kind":"physical_dimension","powers":[]})";
  const auto digest = identity::sha256_hex(std::vector<std::uint8_t>(units.begin(),units.end()));
  const std::string identity = "pops.integral.v2/native-fixture/"+digest+"/q";
  ctx->declare_integral_state(identity,Real(.7));
  const auto original_state = sim.get_state("tracer");
  const auto original_ledger = sim.checkpoint_program_exchanges();
  std::optional<runtime::program::PreparedIntegralCapture> stale;
  int mutation = 0;
  bool reject_after_kernel = true;
  bool kernel_reached = false;
  ctx->install([&](double dt) {
    ctx->begin_step(dt);
    if (mutation == 7) {
      (void)ctx->integral_candidate_value(*stale,identity,units);
      throw std::logic_error("revoked capture unexpectedly reached producer");
    }
    auto capture = ctx->capture_integral_candidate(identity,units);
    const bool injected_rank = ctx->prepared_execution_lane().rank() == 0;
    if (mutation == 1 && injected_rank) ctx->set_stage_time(1,2);
    if (mutation == 2 && injected_rank) ctx->resource_attempt().reject();
    if (mutation == 3 && injected_rank) {
      runtime::program::AcceptedExchangeLedger changed;
      changed.declare_integral(identity,.8);
      ctx->runtime_state().accepted_exchanges_ = std::move(changed);
    }
    if (mutation == 8 && injected_rank)
      ctx->runtime_state().accepted_exchanges_.declare_integral("injected-provenance",0.);
    const auto wanted_units = mutation == 4 && injected_rank ? units+" " : units;
    const auto wanted_identity = mutation == 5 && injected_rank ? identity+"different" : identity;
    if (mutation == 6) {
      foreign_ctx->begin_step(dt);
      (void)foreign_ctx->integral_candidate_value(capture,identity,units);
      throw std::logic_error("foreign owner unexpectedly authorized capture");
    }
    const double q = ctx->integral_candidate_value(capture,wanted_identity,wanted_units);
    stale.emplace(std::move(capture));
    kernel_reached = true;
    auto& state = ctx->state(0);
    for (std::size_t patch=0;patch<state.local_size();++patch) {
      const auto values = state.fab(patch).view();
      for_each_cell(state.box(patch),[=] POPS_HD(const Index<kNativeDimension>& cell) {
        values(cell,0) *= Real(1)-Real(dt*.3*q);
      });
    }
    if (reject_after_kernel)
      throw runtime::program::StepAttemptRejected(SolveStatus::kInvalidEvaluation,
          "feedback-after-kernel","reject after genuine scalar capture consumer");
  });
  sim.set_program_block_map({0});
  EXPECT_THROW(sim.step(.01),runtime::program::StepAttemptRejected);
  EXPECT_TRUE(kernel_reached);
  EXPECT_EQ(sim.get_state("tracer"),original_state);
  EXPECT_EQ(sim.checkpoint_program_exchanges(),original_ledger);
  reject_after_kernel = false;
  mutation = 7;
  kernel_reached = false;
  EXPECT_ANY_THROW(sim.step(.01));
  EXPECT_FALSE(kernel_reached);
  EXPECT_EQ(sim.get_state("tracer"),original_state);
  for (const int failure : {1,2,3,4,5,6,8}) {
    mutation = failure;
    kernel_reached = false;
    EXPECT_ANY_THROW(sim.step(.01));
    EXPECT_FALSE(kernel_reached);
    EXPECT_EQ(sim.get_state("tracer"),original_state);
    EXPECT_EQ(sim.checkpoint_program_exchanges(),original_ledger);
    EXPECT_DOUBLE_EQ(sim.time(),0.);
    EXPECT_EQ(sim.macro_step(),0);
  }
  mutation = 0;
  EXPECT_NO_THROW(sim.step(.01));
  EXPECT_TRUE(kernel_reached);
  for (const double value : sim.get_state("tracer"))
    EXPECT_NEAR(value,2.*(1.-.01*.3*.7),1e-14);
  EXPECT_EQ(sim.checkpoint_program_exchanges(),original_ledger);
}

TEST(ProgramRuntime, IntegralFeedbackRealExteriorConsumptionThenParentRollbackAndRetry) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  auto config = distributed_boundary_domain_config<kNativeDimension>(n);
  config.periodicity[0] = false;
  System<kNativeDimension> sim(config);
  install_execution_lane(sim,"pops.test.integral-feedback.real-exterior");
  // Prepare the physical Outflow authority before committing the scalar block.
  // foextrap is the native Extrapolate law; transverse faces remain periodic.
  const std::string state_identity = "test.program-runtime.integral-feedback.tracer.state@1";
  sim.install_block_state_route("tracer",state_identity);
  std::vector<std::string> face_types(static_cast<std::size_t>(2*kNativeDimension),"periodic");
  face_types[0] = face_types[1] = "foextrap";
  std::vector<std::string> face_identities;
  for (int face=0;face<2*kNativeDimension;++face)
    face_identities.push_back("test:integral-feedback/outflow-face:"+std::to_string(face));
  sim.install_hyperbolic_boundary("tracer","test:integral-feedback/outflow@1",1,
      face_types,std::vector<double>(static_cast<std::size_t>(2*kNativeDimension),0.),
      face_identities,{"scalar"},state_identity);
  sim.seal_auxiliary_providers();
  RealVector<kNativeDimension> velocity{};
  velocity[0] = Real(1);
  add_compiled_model(sim,"tracer",nd::ScalarAdvection<kNativeDimension>::prepare(velocity),"none","rusanov",
      "conservative","explicit");
  std::vector<double> initial(exact_cell_count<kNativeDimension>(n),2.);
  sim.set_state("tracer",initial);
  auto ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->configure_primary_clock("feedback-clock");
  const std::string units = R"({"kind":"physical_dimension","powers":[]})";
  const std::string identity = "pops.integral.v2/native-feedback/"+
      identity::sha256_hex(std::vector<std::uint8_t>(units.begin(),units.end()))+"/q";
  ctx->declare_integral_state(identity,Real(.7));
  const auto original_state = sim.get_state("tracer");
  const auto original_ledger = sim.checkpoint_program_exchanges();
  bool reject_after_delivery = true, delivered = false;
  ctx->install([&](double dt) {
    ctx->begin_step(dt);
    const auto capture = ctx->capture_integral_candidate(identity,units);
    const double q = ctx->integral_candidate_value(capture,identity,units);
    auto& state = ctx->state(0);
    for (std::size_t patch=0;patch<state.local_size();++patch) {
      const auto values = state.fab(patch).view();
      for_each_cell(state.box(patch),[=] POPS_HD(const Index<kNativeDimension>& cell) {
        values(cell,0) *= Real(1)-Real(dt*.3*q);
      });
    }
    auto residual = ctx->rhs_scratch_like(state);
    std::vector<nd::FaceField<kNativeDimension>> faces;
    ctx->neg_div_flux_default_with_faces_into(0,state,residual,0,faces);
    ctx->axpy(state,Real(dt),residual);
    const auto geometry = ctx->geometry();
    ctx->stage_exchange_batch([&](auto&& stage) {
      sync_host();
      // Same physical ownership rule as accepted_exchange_contributes in the
      // integrated SDK; collective staging is retained on noncontributors.
      if (state.distribution().replicated() && ctx->prepared_execution_lane().rank()!=0) return;
      for (const auto& patch : faces) {
        const auto box = patch.cell_box();
        const auto flux = patch.template field<0>().view();
        Real measure = 1;
        for (int axis=1;axis<kNativeDimension;++axis) measure *= geometry.spacing(axis);
        const auto extent = box.extent();
        for (std::int64_t ordinal=0;ordinal<box.numPts();++ordinal) {
          auto remainder = ordinal;
          auto cell = box.lo;
          for (int axis=0;axis<kNativeDimension;++axis) {
            cell[axis] += static_cast<int>(remainder%extent[axis]);
            remainder /= extent[axis];
          }
          if (!ctx->is_external_trace_face(0,1,cell)) continue;
          auto face = cell; ++face[0];
          std::string key = "physical-right";
          for (int axis=0;axis<kNativeDimension;++axis) key += ":"+std::to_string(cell[axis]);
          stage(runtime::program::ExchangeRecord{"real-native-transport","right-flux",
              "feedback-evaluation",key,-1,measure,flux(face,0)/measure,dt,1,0,1,0,true});
        }
      }
    });
    const runtime::program::AcceptedExchangeLedger::TraceSelection selection{
        "real-native-transport","right-flux",0,1,0,"feedback-evaluation"};
    (void)ctx->consume_external_trace(identity,selection,Real(-1));
    delivered = true;
    // The ledger's exactly-once rule is exercised after a genuine numerical face.
    EXPECT_ANY_THROW((void)ctx->consume_external_trace(identity,selection,Real(-1)));
    if (reject_after_delivery)
      throw runtime::program::StepAttemptRejected(SolveStatus::kInvalidEvaluation,
          "wall-current-after-consumption","parent rejects after real wall-current publication");
  });
  sim.set_program_block_map({0});
  EXPECT_THROW(sim.step(.01),runtime::program::StepAttemptRejected);
  EXPECT_TRUE(delivered);
  EXPECT_EQ(sim.get_state("tracer"),original_state);
  EXPECT_EQ(sim.checkpoint_program_exchanges(),original_ledger);
  EXPECT_DOUBLE_EQ(ctx->integral_state(identity),.7);
  EXPECT_DOUBLE_EQ(sim.time(),0.);
  EXPECT_EQ(sim.macro_step(),0);
  reject_after_delivery = false; delivered = false;
  EXPECT_NO_THROW(sim.step(.01));
  EXPECT_TRUE(delivered);
  EXPECT_NEAR(ctx->integral_state(identity),.7+.01*2.*(1.-.01*.3*.7),2e-13);
  for (const double value : sim.get_state("tracer"))
    EXPECT_NEAR(value,2.*(1.-.01*.3*.7),2e-13);
  const auto accepted_state = sim.get_state("tracer");
  const auto checkpoint = sim.checkpoint_program_exchanges();
  EXPECT_NO_THROW(sim.validate_checkpoint_program_exchanges(checkpoint));
  ASSERT_NO_THROW(sim.begin_restart_transaction());
  ASSERT_NO_THROW(sim.restore_checkpoint_program_exchanges(checkpoint));
  ASSERT_NO_THROW(sim.commit_restart_transaction());
  sim.finalize_restart_transaction();
  EXPECT_EQ(sim.checkpoint_program_exchanges(),checkpoint);
  EXPECT_EQ(sim.get_state("tracer"),accepted_state);
  EXPECT_EQ(sim.step_transaction_depth(),0u);
  EXPECT_DOUBLE_EQ(sim.time(),.01);
  EXPECT_EQ(sim.macro_step(),1);
}
