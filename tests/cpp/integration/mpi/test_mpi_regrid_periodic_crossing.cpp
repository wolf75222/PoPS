// Native distributed proof that an oriented periodic feature can cross each domain seam while
// repeated AMR regrids preserve every requested tag and the composite conserved quantity.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "periodic_regrid_crossing_evidence.hpp"

#include <pops/parallel/comm.hpp>

#include <array>
#include <cstdio>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

int run_mpi_regrid_periodic_crossing(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  const int ranks = n_ranks();
  long local_failures = 0;
  const std::array<test::PeriodicRegridCrossingEvidence, 2> crossings{
      test::run_periodic_regrid_crossing_evidence(/*axis=*/0),
      test::run_periodic_regrid_crossing_evidence(/*axis=*/1)};

  for (const test::PeriodicRegridCrossingEvidence& crossing : crossings) {
    if (!crossing.passed())
      ++local_failures;
    if (rank == 0)
      std::printf(
          "AMR_PERIODIC_CROSSING np=%d axis=%c regrids=%d multipatch=%d seam=%d "
          "transverse=%d layout_consensus=%d composite_dm=%.3e injection_error=%.3e\n",
          ranks, crossing.axis == 0 ? 'x' : 'y', crossing.regrid_count,
          crossing.multi_patch_parent ? 1 : 0, crossing.periodic_low_and_high_covered ? 1 : 0,
          crossing.transverse_boundary_covered ? 1 : 0, crossing.layout_consensus ? 1 : 0,
          crossing.maximum_composite_mass_error, crossing.maximum_injection_error);
  }
  const long failures = all_reduce_sum(local_failures);
  if (rank == 0)
    std::printf("%s test_mpi_regrid_periodic_crossing (np=%d)\n", failures == 0 ? "OK" : "FAIL",
                ranks);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_regrid_periodic_crossing, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_regrid_periodic_crossing,
                                    "test_mpi_regrid_periodic_crossing"),
            0);
}
