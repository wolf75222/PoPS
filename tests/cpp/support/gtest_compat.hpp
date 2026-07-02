#pragma once

// Legacy-body adapter for the GoogleTest suite (campagne de portage natif).
//
// Most tests are NATIVE GoogleTest (TEST/TEST_F + EXPECT_*); RunTestBody below only remains for the
// deliberate exceptions. MPI/Kokkos LIFECYCLE RULE (campaign-wide precedent): MPI can be initialized
// at most once per process, so a test touching comm_init()/comm_finalize() uses exactly ONE of the
// two sanctioned idioms, chosen by its structure:
//   1. ONE single TEST wrapping the whole ordered pipeline, comm_init()/comm_finalize() inline --
//      for bodies that are one stateful scenario, especially those ending in a collective
//      fail-reduction (all_reduce_max/sum) replayed under mpirun at np=1/2/4;
//   2. a fixture with SetUpTestSuite()/TearDownTestSuite() holding comm_init()/comm_finalize() --
//      for files split into several TEST_F where the suite runs once per binary.
// Splitting such a body into multiple plain TESTs would re-init MPI per process (UB) and break the
// collective reduction. Explicit Kokkos::initialize/ScopeGuard is unnecessary in most tests:
// pops::detail::ensure_kokkos_initialized() self-initializes lazily on the first allocation
// (idempotent, atexit-finalized); a multi-TEST file that still needs a guard uses a
// ::testing::Environment. The rare files KEPT WRAPPED via RunTestBody take (argc, argv) for a real
// comm_init(&argc, &argv) under mpirun, or compile .so at runtime (native_loader).
#include <string>
#include <type_traits>

namespace pops::test {

template <class F>
int RunTestBody(F fn, const char* test_name) {
  if constexpr (std::is_invocable_r_v<int, F, int, char**>) {
    int argc = 1;
    std::string arg0(test_name);
    char* argv[] = {arg0.data(), nullptr};
    return fn(argc, argv);
  } else {
    return fn();
  }
}

}  // namespace pops::test
