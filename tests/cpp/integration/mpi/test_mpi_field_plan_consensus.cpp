// Exact collective consensus for resolved field-plan registries. Each scenario uses a fresh facade:
// setters are intentionally local/non-collective, then mark_bound compares one canonical std::map-
// ordered sequence of (provider_slot, plan_identity) before field-plan materialization.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/system.hpp>

#include <limits>
#include <stdexcept>
#include <string>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

void install(System& system, const std::string& slot, const std::string& plan_identity) {
  system.set_field_solver_plan(slot, plan_identity, "provider:" + slot, "output-owner", "plasma",
                               "potential", {"rhs-provider"}, {"plasma"}, {"potential"}, {1.0},
                               "geometric_mg", 0.0, 1.0e-8, 50, 2, 2, 2, 50, 0);
}

void install(AmrSystem& system, const std::string& slot, const std::string& plan_identity) {
  system.set_field_solver_plan(slot, plan_identity, "provider:" + slot, "output-owner", "plasma",
                               "potential", {"rhs-provider"}, {"plasma"}, {"potential"}, {1.0},
                               "geometric_mg", "composite", 0.0, 1.0e-8, 50, 2, 2, 2, 50, 0,
                               CompositeFacOptions{});
}

template <class SystemType>
bool bind_rejected(SystemType& system) {
  try {
    system.mark_bound();
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

template <class SystemType>
bool duplicate_rejected(SystemType& system) {
  try {
    install(system, "field-slot", "shared-plan-identity");
    install(system, "field-slot", "shared-plan-identity");
  } catch (const std::runtime_error&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

int run_field_plan_consensus(int argc, char** argv) {
  comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = my_rank();
  const int ranks = n_ranks();
  long failures = ranks == 2 ? 0 : 1;
  const auto require = [&failures](bool condition) {
    if (!condition)
      ++failures;
  };

  // Same registry shape and token lengths, but different bytes: both facades reject uniformly.
  {
    const std::string token = rank == 0 ? "plan-rank-0" : "plan-rank-1";
    System system(SystemConfig{16, 1.0, true});
    install(system, "field-slot", token);
    require(bind_rejected(system));
  }
  {
    const std::string token = rank == 0 ? "plan-rank-0" : "plan-rank-1";
    AmrSystem system(AmrSystemConfig{16});
    install(system, "field-slot", token);
    require(bind_rejected(system));
  }

  // The slot participates independently in the pair; an equal plan token cannot hide slot drift.
  {
    const std::string slot = rank == 0 ? "field-rank-0" : "field-rank-1";
    System system(SystemConfig{16, 1.0, true});
    install(system, slot, "shared-plan");
    require(bind_rejected(system));
  }
  {
    const std::string slot = rank == 0 ? "field-rank-0" : "field-rank-1";
    AmrSystem system(AmrSystemConfig{16});
    install(system, slot, "shared-plan");
    require(bind_rejected(system));
  }

  // Component length disagreement returns before the byte collective.
  {
    const std::string token = rank == 0 ? "x" : "plan-with-another-length";
    System system(SystemConfig{16, 1.0, true});
    install(system, "field-slot", token);
    require(bind_rejected(system));
  }
  {
    const std::string token = rank == 0 ? "x" : "plan-with-another-length";
    AmrSystem system(AmrSystemConfig{16});
    install(system, "field-slot", token);
    require(bind_rejected(system));
  }

  // A missing/extra plan agrees the pair count first. This is the case that deadlocked when the
  // setter itself was collective: rank 1 executes one more local setter than rank 0.
  {
    System system(SystemConfig{16, 1.0, true});
    install(system, "field-a", "plan-a");
    if (rank == 1)
      install(system, "field-b", "plan-b");
    require(bind_rejected(system));
  }
  {
    AmrSystem system(AmrSystemConfig{16});
    install(system, "field-a", "plan-a");
    if (rank == 1)
      install(system, "field-b", "plan-b");
    require(bind_rejected(system));
  }

  // Setter order is not semantic: std::map canonicalization produces the same two pairs.
  {
    System system(SystemConfig{16, 1.0, true});
    if (rank == 0) {
      install(system, "field-b", "plan-b");
      install(system, "field-a", "plan-a");
    } else {
      install(system, "field-a", "plan-a");
      install(system, "field-b", "plan-b");
    }
    require(!bind_rejected(system));
  }
  {
    AmrSystem system(AmrSystemConfig{16});
    if (rank == 0) {
      install(system, "field-b", "plan-b");
      install(system, "field-a", "plan-a");
    } else {
      install(system, "field-a", "plan-a");
      install(system, "field-b", "plan-b");
    }
    require(!bind_rejected(system));
  }

  // Duplicate slots are a local structural error, including byte-identical repeats; no collective
  // is entered and no partially overwritten plan survives.
  {
    System system(SystemConfig{16, 1.0, true});
    require(duplicate_rejected(system));
  }
  {
    AmrSystem system(AmrSystemConfig{16});
    require(duplicate_rejected(system));
  }

  // Native finite/domain guards remain authoritative even if a caller bypasses Python schemas.
  {
    System system(SystemConfig{16, 1.0, true});
    bool rejected = false;
    try {
      system.set_field_solver_plan("field-slot", "plan", "provider", "output-owner", "plasma",
                                   "potential", {"rhs-provider"}, {"plasma"}, {"potential"}, {1.0},
                                   "geometric_mg", 0.0, std::numeric_limits<double>::infinity(), 50,
                                   2, 2, 2, 50, 0);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected);
  }
  {
    AmrSystem system(AmrSystemConfig{16});
    CompositeFacOptions invalid;
    invalid.coarse_abs_tol = std::numeric_limits<Real>::quiet_NaN();
    bool rejected = false;
    try {
      system.set_field_solver_plan("field-slot", "plan", "provider", "output-owner", "plasma",
                                   "potential", {"rhs-provider"}, {"plasma"}, {"potential"}, {1.0},
                                   "geometric_mg", "composite", 0.0, 1.0e-8, 50, 2, 2, 2, 50, 0,
                                   invalid);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected);
  }

  failures = all_reduce_sum(failures);
  comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_field_plan_consensus, CanonicalRegistryRefusesDivergenceWithoutDeadlock) {
  EXPECT_EQ(pops::test::RunTestBody(&run_field_plan_consensus, "test_mpi_field_plan_consensus"), 0);
}
