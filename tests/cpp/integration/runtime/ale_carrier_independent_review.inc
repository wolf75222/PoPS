// Independent review fixture. Include at the end of test_program_runtime.cpp
// after integrating the native moving-interval carrier. No alternative runtime.
TEST(ProgramRuntime, MovingIntervalsRefuseFiniteUnrecoverablePhysicalStateBeforePublication) {
  if constexpr (kNativeDimension != 1)
    GTEST_SKIP() << "Moving intervals require the Dim1 provider";
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int cells = 8;
  System<kNativeDimension> sim(distributed_boundary_domain_config<kNativeDimension>(cells));
  install_execution_lane(sim, "pops.review.moving-interval.recovery");
  add_gas(sim, 1.4);
  std::vector<double> initial(static_cast<std::size_t>(cells) * kGasComponents, 0.);
  for (int cell = 0; cell < cells; ++cell) {
    initial[cell] = 2.3;
    initial[(kGasComponents - 1) * cells + cell] = 6.;
  }
  sim.set_state("gas", initial);
  sim.set_program_block_map({0});
  auto ctx = runtime::program::make_program_execution_provider(&sim);
  ctx->configure_primary_clock("ale-review-clock");
  ctx->initialize_moving_interval_geometry("mesh", 0);
  const auto state_before = sim.get_state("gas");
  const auto geometry_before = ctx->moving_interval_geometry("mesh");
  constexpr int cache_node = 682301;
  ctx->cache_store_scratch(cache_node, ctx->state(0));
  const auto cache_before = sim.program_cache_global(cache_node);
  ctx->install([&](double dt) {
    ctx->begin_step(dt);
    const auto& accepted = ctx->moving_interval_geometry("mesh");
    auto proposed = accepted.coordinates;
    auto sweep = accepted.swept_volumes;
    std::vector<nd::FaceField<kNativeDimension>> physical, density;
    auto source = ctx->scratch_state_like(ctx->state(0));
    source.set_val(Real(0));
    const bool inject_invalid_state = sim.prepared_boundary_execution_lane().rank() == 0;
    for (std::size_t patch = 0; patch < ctx->state(0).local_size(); ++patch) {
      const auto box = ctx->state(0).box(patch);
      sweep[patch].set_val(Real(0));
      physical.emplace_back(box, kGasComponents);
      density.emplace_back(box, kGasComponents);
      physical.back().set_val(Real(0));
      density.back().set_val(Real(0));
      const auto input = ctx->state(0).fab(patch).view();
      const auto volume = accepted.measures.fab(patch).view();
      const auto source_view = source.fab(patch).view();
      for_each_cell(box, [=] POPS_HD(const Index<kNativeDimension>& cell) {
        // Unchanged positive volumes, exact GCL, zero relative flux, finite
        // source: the sole fault is rho_new=-1, rejected by prepared recovery.
        if (inject_invalid_state)
          source_view(cell, 0) = -(input(cell, 0) + Real(1)) * volume(cell);
      });
    }
    ctx->advance_moving_intervals("mesh", 0, proposed, sweep, physical, density, source, Real(0));
  });
  sim.begin_step_transaction();
  EXPECT_ANY_THROW(sim.step(.1));
  EXPECT_EQ(sim.time(), 0.);
  EXPECT_EQ(sim.macro_step(), 0);
  EXPECT_EQ(sim.get_state("gas"), state_before);
  EXPECT_EQ(sim.program_cache_global(cache_node), cache_before);
  EXPECT_TRUE(sim.program_exchange_records().empty());
  const auto& geometry_after = ctx->moving_interval_geometry("mesh");
  EXPECT_EQ(geometry_after.generation, geometry_before.generation);
  EXPECT_EQ(geometry_after.last_interval, geometry_before.last_interval);
  for (std::size_t patch = 0; patch < geometry_after.coordinates.size(); ++patch) {
    for (int role = 0; role < 3; ++role) {
      const auto& before = role == 0 ? geometry_before.measures.fab(patch) :
                           role == 1 ? geometry_before.coordinates[patch].template field<0>() :
                                       geometry_before.swept_volumes[patch].template field<0>();
      const auto& after = role == 0 ? geometry_after.measures.fab(patch) :
                          role == 1 ? geometry_after.coordinates[patch].template field<0>() :
                                      geometry_after.swept_volumes[patch].template field<0>();
      const auto a = before.create_host_mirror(), b = after.create_host_mirror();
      before.copy_to_host(a);
      after.copy_to_host(b);
      ASSERT_EQ(a.size(), b.size());
      for (std::size_t i = 0; i < a.size(); ++i) EXPECT_DOUBLE_EQ(a(i), b(i));
    }
  }
  sim.rollback_step_transaction();
}
