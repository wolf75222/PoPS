// Include in test_program_runtime.cpp; exercises the real owner-aware provider.
TEST(ProgramRuntime, ExplicitVectorPairingIncludesEveryComponentAndKeepsLegacyDotZero) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  const std::size_t cells = exact_cell_count<kNativeDimension>(n);
  System<kNativeDimension> sim(unit_domain_config<kNativeDimension>(n));
  install_execution_lane(sim, "pops.test.program.dot-all.v1");
  add_gas(sim, 1.4, "none");
  std::vector<double> initial(static_cast<std::size_t>(kGasComponents) * cells, 0.);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    initial[cell] = 1.;
    initial[(kGasComponents - 1) * cells + cell] = 2.5;
  }
  sim.set_state("gas", initial);
  sim.set_program_block_map({0});
  runtime::program::ProgramContext context(&sim);
  auto& accepted = context.state(0);
  auto increment = context.scratch_state_like(accepted);
  increment.set_val(Real(0));
  for (std::size_t patch = 0; patch < increment.local_size(); ++patch) {
    const auto values = increment.fab(patch).view();
    for_each_cell(increment.box(patch), [=] POPS_HD(const Index<kNativeDimension>& cell) {
      values(cell, 0) = Real(-.5);
      values(cell, 1) = Real(1);
    });
  }
  EXPECT_DOUBLE_EQ(context.dot(0, accepted, increment), -.5 * cells);
  EXPECT_DOUBLE_EQ(context.dot(0, increment, increment), .25 * cells);
  EXPECT_DOUBLE_EQ(context.dot_all(0, accepted, increment), -.5 * cells);
  EXPECT_DOUBLE_EQ(context.dot_all(0, increment, increment), 1.25 * cells);
  const double gamma = -2. * context.dot_all(0, accepted, increment) /
      context.dot_all(0, increment, increment);
  EXPECT_DOUBLE_EQ(gamma, .8);
  EXPECT_DOUBLE_EQ(-2. * context.dot(0, accepted, increment) /
      context.dot(0, increment, increment), 4.);
  auto& wrong_width = context.scalar_scratch(873201, 0, accepted, 1, 0);
  EXPECT_ANY_THROW(context.dot_all(0, accepted, wrong_width));
}
