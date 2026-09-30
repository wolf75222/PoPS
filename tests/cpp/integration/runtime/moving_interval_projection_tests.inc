// Included by the existing authentic System/Program runtime integration suite.
TEST(ProgramRuntime, MovingIntervalProjectionEvaluatesBodiesWithinIssuedDuration) {
  if constexpr (kNativeDimension != 1) {
    GTEST_SKIP() << "Moving face/cell projection requires native Dim1 geometry";
  }
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  for (int n : {8, 32}) for (Real velocity : {Real(.7), Real(-.3)}) {
    System<kNativeDimension> sim(distributed_boundary_domain_config<kNativeDimension>(n));
    install_execution_lane(sim, "pops.test.moving-interval.projection");
    add_gas(sim, 1.4);
    const std::array<Real, kGasComponents> constants = velocity > 0 ?
        std::array<Real, kGasComponents>{Real(2.3), Real(0), Real(6)} :
        std::array<Real, kGasComponents>{Real(6), Real(0), Real(2.3)};
    std::vector<double> initial(static_cast<std::size_t>(n)*kGasComponents);
    for (int component = 0; component < kGasComponents; ++component)
      for (int cell = 0; cell < n; ++cell) initial[component*n+cell] = constants[component];
    sim.set_state("gas", initial);
    sim.set_program_block_map({0});
    auto ctx = runtime::program::make_program_execution_provider(&sim);
    ctx->configure_primary_clock("project-clock");
    ctx->initialize_moving_interval_geometry("mesh", 0, "unit-interval",Real(1e-13));
    // The authored PDE is componentwise U_t + div(velocity*U) = 0.
    // The projection policy is a centered trace, integrated over the actual
    // duration, with a separately authored zero source. Runtime has no PDE case.
    ctx->install([&](double dt) {
      ctx->begin_step(dt);
      auto evaluation = ctx->template project_moving_interval<kGasComponents>(
          "mesh", 0, "unit-interval", "centered-linear-path@1",
          [] POPS_HD(Real x, Real time) {
            return x+(x == Real(0) || x == Real(1) ? Real(0) :
                Real(.08)*Kokkos::sin(Real(6.2831853071795864769)*x)*time);
          },
          [=] POPS_HD(const StateVec<kGasComponents>& lower, const StateVec<kGasComponents>& upper,
              const runtime::program::MovingFaceGeometry&,
              const runtime::program::MovingIntervalTime& time) {
            runtime::program::MovingFaceEvaluation<kGasComponents> result;
            for (int component = 0; component < kGasComponents; ++component) {
              result.density[component] = (lower[component]+upper[component])/Real(2);
              result.physical_amount[component] = velocity*result.density[component]*time.duration;
            }
            return result;
          },
          [] POPS_HD(const StateVec<kGasComponents>&,
              const runtime::program::MovingCellGeometry&,
              const runtime::program::MovingIntervalTime&) { return StateVec<kGasComponents>{}; },
          Real(1e-13));
      EXPECT_DOUBLE_EQ(evaluation.interval_begin(), sim.time());
      EXPECT_DOUBLE_EQ(evaluation.interval_end(), sim.time()+dt);
      auto candidate = ctx->prepare_moving_interval_update(evaluation, Real(1e-13));
      ctx->commit_moving_interval(candidate);
    });
    sim.set_program_block_map({0});
    sim.begin_step_transaction();
    sim.step(.2);
    sim.step(.3);
    EXPECT_EQ(ctx->moving_interval_geometry("mesh").generation, 2u);
    const auto accepted = sim.get_state("gas");
    if (!accepted.empty())
      for (int component = 0; component < kGasComponents; ++component)
        for (int cell = 0; cell < n; ++cell)
          EXPECT_NEAR(accepted[component*n+cell], constants[component], 4e-13);
    const auto& geometry = ctx->moving_interval_geometry("mesh");
    ASSERT_TRUE(geometry.last_receipt.has_value());
    EXPECT_DOUBLE_EQ(geometry.last_receipt->point.physical_time, .2);
    EXPECT_DOUBLE_EQ(geometry.last_receipt->point.dt, .3);
    EXPECT_EQ(geometry.last_receipt->quadrature_identity, "centered-linear-path@1");
    EXPECT_NEAR(all_reduce_sum(pops::reduce_sum_local(geometry.measures, 0),
                              sim.prepared_boundary_execution_lane()), 1., 4e-13);
    sim.rollback_step_transaction();
    EXPECT_EQ(sim.get_state("gas"), initial);
    EXPECT_EQ(ctx->moving_interval_geometry("mesh").generation, 0u);
    EXPECT_FALSE(ctx->moving_interval_geometry("mesh").last_receipt.has_value());
    EXPECT_TRUE(sim.program_exchange_records().empty());
  }
}
