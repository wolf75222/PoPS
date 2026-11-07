// Independent native counterprobe. Include after test_program_runtime.cpp's helpers.
// No private access and no fake execution provider. Pending central native execution.
TEST(ProgramRuntime, IndependentTypedIntegralCaptureRejectsRehashedMalformedUnits) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  for (const std::string units : {std::string("not-json"),
                                 std::string(R"({"powers":[],"kind":"physical_dimension"})")}) {
    SCOPED_TRACE(units);
    System<kNativeDimension> sim(unit_domain_config<kNativeDimension>(4));
    install_execution_lane(sim, "sol61.t5.public-units");
    add_scalar(sim);
    sim.set_state("tracer", std::vector<double>(exact_cell_count<kNativeDimension>(4), 2.));
    auto ctx = runtime::program::make_program_execution_provider(&sim);
    ctx->configure_primary_clock("sol61-public-capture");
    const auto digest = identity::sha256_hex(
        std::vector<std::uint8_t>(units.begin(), units.end()));
    const auto key = "pops.integral.v2/independent-public/" + digest + "/q";
    ctx->declare_integral_state(key, .7);
    const auto ledger = sim.checkpoint_program_exchanges();
    const auto state = sim.get_state("tracer");
    bool consumed = false;
    ctx->install([&](double dt) {
      ctx->begin_step(dt);
      const auto capture = ctx->capture_integral_candidate(key, units);
      const double q = ctx->integral_candidate_value(capture, key, units);
      EXPECT_DOUBLE_EQ(q, .7);
      consumed = true;
    });
    sim.set_program_block_map({0});
    if (ctx->prepared_execution_lane().size() == 1) {
      EXPECT_THROW(sim.step(.01), std::invalid_argument);
    } else {
      EXPECT_THROW(sim.step(.01), std::runtime_error);
    }
    EXPECT_FALSE(consumed);
    EXPECT_EQ(sim.get_state("tracer"), state);
    EXPECT_EQ(sim.checkpoint_program_exchanges(), ledger);
    EXPECT_DOUBLE_EQ(sim.time(), 0.);
    EXPECT_EQ(sim.macro_step(), 0);
  }
}
