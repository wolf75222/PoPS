#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <memory>
#include <string>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

int run_collective_refusal() {
  AmrSystemConfig config;
  config.n = 4;
  config.L = 1.0;
  config.level_count = 1;
  config.regrid_every = 0;
  config.periodicity = {true, true};

  ModelSpec model;
  model.transport = "exb";
  model.source = "none";
  model.elliptic = "charge";

  AmrSystem system(config);
  system.add_block("tracer", model, "none", "rusanov", "conservative", "euler");
  system.install_program_step([](double) {});
  if (!system.uses_runtime_engine() || system.engine() == nullptr)
    return 1;
  auto context = std::make_shared<runtime::program::AmrProgramContext>(system.engine(), &system);
  context->configure_primary_clock("test.clock.cell-local-mpi-refusal");

  bool refused = false;
  try {
    context->prepare_same_level_cell_temporal_execution("test.clock.cell-local-mpi-refusal", 100,
                                                        0);
  } catch (const std::runtime_error& error) {
    refused = std::string(error.what()).find("no MPI-safe multi-box") != std::string::npos;
  }
  const long refusing_ranks = all_reduce_sum(refused ? 1L : 0L);
  const bool unchanged = system.program_accepted_state().empty();
  return refusing_ranks == n_ranks() && unchanged ? 0 : 1;
}

int pops_run_test_mpi_cell_temporal_program_refusal(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#else
  (void)argc;
  (void)argv;
#endif
  const int result = run_collective_refusal();
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_cell_temporal_program_refusal, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_cell_temporal_program_refusal,
                                    "test_mpi_cell_temporal_program_refusal"),
            0);
}
