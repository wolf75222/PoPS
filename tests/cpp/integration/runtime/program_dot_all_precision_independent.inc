// Independent cancellation probes. Receive serial and MPI via ExplicitVectorPairing*.
TEST(ProgramRuntime, ExplicitVectorPairingPreservesCancellationAcrossComponentsAndOwners) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  constexpr int n = 8;
  System<kNativeDimension> sim(distributed_boundary_domain_config<kNativeDimension>(n));
  install_execution_lane(sim, "pops.test.independent.dot-all.cancellation");
  add_gas(sim, 1.4, "none");
  const auto cells = exact_cell_count<kNativeDimension>(n);
  std::vector<double> initial(kGasComponents*cells, 0.);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    initial[cell] = 1.;
    initial[(kGasComponents-1)*cells+cell] = 2.5;
  }
  sim.set_state("gas", initial);
  sim.set_program_block_map({0});
  runtime::program::ProgramContext context(&sim);
  const auto before = sim.get_state("gas");
  auto left = context.scratch_state_like(context.state(0));
  auto right = context.scratch_state_like(context.state(0));
  right.set_val(Real(1));
  for (int across_owners = 0; across_owners < 2; ++across_owners)
    for (int shift = 0; shift < kGasComponents; ++shift) {
      left.set_val(Real(0));
      for (std::size_t patch = 0; patch < left.local_size(); ++patch) {
        const auto values = left.fab(patch).view();
        for_each_cell(left.box(patch), [=] POPS_HD(const Index<kNativeDimension>& cell) {
          for (int axis = 1; axis < kNativeDimension; ++axis)
            if (cell[axis] != 0) return;
          if (across_owners) {
            // MPI2 splits x at four: rank0 holds high+low, rank1 the negative high.
            if (cell[0] == 0) values(cell, shift) = Real(1e16);
            if (cell[0] == 1) values(cell, shift) = Real(1);
            if (cell[0] == 4) values(cell, shift) = Real(-1e16);
          } else if (cell[0] == 0) {
            // Low information must also survive the native component merge.
            values(cell, shift) = Real(1e16);
            values(cell, (shift+1)%kGasComponents) = Real(1);
            values(cell, (shift+2)%kGasComponents) = Real(-1e16);
          }
        });
      }
      EXPECT_DOUBLE_EQ(context.dot_all(0, left, right), 1.)
          << "across_owners=" << across_owners << ", component shift=" << shift;
    }
  EXPECT_EQ(sim.get_state("gas"), before);
}
