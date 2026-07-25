#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/fv/flux_failure.hpp>
#include <pops/parallel/comm.hpp>

#include <cstdio>

namespace {

struct RecordOneFailure {
  pops::FluxEvaluationRecorder recorder;
  pops::EvaluationStatus status;
  std::uint32_t reason;

  POPS_HD void operator()(int, int, std::uint64_t& failure) const {
    using Evaluation = pops::FluxEvaluation<pops::StateVec<1>>;
    switch (status) {
      case pops::EvaluationStatus::kOk:
        return;
      case pops::EvaluationStatus::kRetry:
        recorder.record(Evaluation::retry(reason), failure);
        return;
      case pops::EvaluationStatus::kReject:
        recorder.record(Evaluation::reject(reason), failure);
        return;
      case pops::EvaluationStatus::kFailed:
        recorder.record(Evaluation::failed(reason), failure);
        return;
    }
  }
};

int run_mpi_flux_failure_collective(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  const int rank = pops::my_rank();
  const int ranks = pops::n_ranks();
  long failures = ranks < 2 ? 1 : 0;

  {
    pops::FluxEvaluationTracker tracker{pops::process_world_flux_collective};
    const auto status = rank == 0 ? pops::EvaluationStatus::kRetry
                                  : pops::EvaluationStatus::kReject;
    const std::uint32_t reason = rank == 0 ? 0xffffu : 0x20u;
    tracker.merge(pops::reduce_max_uint64_cell(
        pops::Box2D{{0, 0}, {0, 0}},
        RecordOneFailure{tracker.recorder(), status, reason}));
    const pops::FluxFailureReport report = tracker.collective_report();
    if (report.status != pops::EvaluationStatus::kReject || report.reason_code != 0x20u)
      ++failures;
  }

  {
    pops::FluxEvaluationTracker tracker{pops::process_world_flux_collective};
    const auto status = rank == 0 ? pops::EvaluationStatus::kFailed
                                  : pops::EvaluationStatus::kOk;
    tracker.merge(pops::reduce_max_uint64_cell(
        pops::Box2D{{0, 0}, {0, 0}},
        RecordOneFailure{tracker.recorder(), status, 0x55u}));
    try {
      tracker.throw_if_failed("mpi_flux_collective");
      ++failures;
    } catch (const pops::FluxEvaluationFailure& failure) {
      if (failure.status() != pops::EvaluationStatus::kFailed ||
          failure.reason_code() != 0x55u || failure.phase() != "mpi_flux_collective")
        ++failures;
    }
  }

  const long global_failures = pops::all_reduce_sum(failures);
  if (rank == 0)
    std::printf("%s test_mpi_flux_failure_collective np=%d\n",
                global_failures == 0 ? "OK" : "FAIL", ranks);
  pops::comm_finalize();
  return global_failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_flux_failure_collective, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&run_mpi_flux_failure_collective,
                                    "test_mpi_flux_failure_collective"),
            0);
}
