#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/fv/flux_failure.hpp>
#include <pops/parallel/comm.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>

namespace {

template <int Dim>
struct RecordOneFailure {
  pops::FluxEvaluationRecorder recorder;
  pops::EvaluationStatus status;
  std::uint32_t reason;

  POPS_HD void operator()(const pops::Index<Dim>&, std::uint64_t& failure) const {
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

template <int Dim>
struct RecordOneRecovery {
  pops::FluxEvaluationRecorder recorder;
  pops::RecoveryReport report;

  POPS_HD void operator()(const pops::Index<Dim>&, std::uint64_t& failure) const {
    recorder.record_recovery(report, failure);
  }
};

template <int Dim, class Recorder>
std::uint64_t reduce_max_failure(const pops::Box<Dim>& box, Recorder recorder) {
  pops::detail::ensure_kokkos_initialized();
  std::uint64_t result = 0;
  const pops::Index<Dim> lower = box.lo;
  const pops::Extent<Dim> extent = box.extent();
  Kokkos::parallel_reduce(
      "pops_mpi_flux_failure", Kokkos::RangePolicy<std::int64_t>(0, box.numPts()),
      KOKKOS_LAMBDA(const std::int64_t ordinal, std::uint64_t& accumulator) {
        pops::Index<Dim> index = lower;
        std::int64_t remainder = ordinal;
        for (int axis = 0; axis < Dim; ++axis) {
          index[axis] += static_cast<int>(remainder % extent[axis]);
          remainder /= extent[axis];
        }
        recorder(index, accumulator);
      },
      Kokkos::Max<std::uint64_t>(result));
  return result;
}

template <int Dim>
pops::Box<Dim> one_cell_box() {
  return {pops::Index<Dim>{}, pops::Index<Dim>{}};
}

int run_mpi_flux_failure_collective(int argc, char** argv) {
  constexpr int Dim = pops::kNativeDimension;
  pops::comm_init(&argc, &argv);
  const int rank = pops::my_rank();
  const int ranks = pops::n_ranks();
  long failures = ranks < 2 ? 1 : 0;

  {
    pops::FluxEvaluationTracker tracker{pops::process_world_flux_collective};
    const auto status =
        rank == 0 ? pops::EvaluationStatus::kRetry : pops::EvaluationStatus::kReject;
    const std::uint32_t reason = rank == 0 ? 0xffffu : 0x20u;
    tracker.merge(reduce_max_failure<Dim>(
        one_cell_box<Dim>(), RecordOneFailure<Dim>{tracker.recorder(), status, reason}));
    const pops::FluxFailureReport report = tracker.collective_report();
    if (report.status != pops::EvaluationStatus::kReject || report.reason_code != 0x20u)
      ++failures;
  }

  {
    pops::FluxEvaluationTracker tracker{pops::process_world_flux_collective};
    pops::RecoveryReport recovery;
    recovery.status =
        rank == 0 ? pops::RecoveryStatus::kRecovered : pops::RecoveryStatus::kRejected;
    recovery.cause =
        rank == 0 ? pops::RecoveryCause::kNone : pops::RecoveryCause::kExplicitRejection;
    recovery.reason_code = rank == 0 ? 0u : 0x755u;
    tracker.merge(reduce_max_failure<Dim>(one_cell_box<Dim>(),
                                          RecordOneRecovery<Dim>{tracker.recorder(), recovery}));
    const pops::FluxFailureReport report = tracker.collective_report();
    if (report.status != pops::EvaluationStatus::kReject || report.reason_code != 0x755u)
      ++failures;
  }

  {
    pops::FluxEvaluationTracker tracker{pops::process_world_flux_collective};
    const auto status = rank == 0 ? pops::EvaluationStatus::kFailed : pops::EvaluationStatus::kOk;
    tracker.merge(reduce_max_failure<Dim>(
        one_cell_box<Dim>(), RecordOneFailure<Dim>{tracker.recorder(), status, 0x55u}));
    try {
      tracker.throw_if_failed("mpi_flux_collective");
      ++failures;
    } catch (const pops::FluxEvaluationFailure& failure) {
      if (failure.status() != pops::EvaluationStatus::kFailed || failure.reason_code() != 0x55u ||
          failure.phase() != "mpi_flux_collective")
        ++failures;
    }
  }

  const long global_failures = pops::all_reduce_sum(failures);
  if (rank == 0)
    std::printf("%s test_mpi_flux_failure_collective np=%d\n", global_failures == 0 ? "OK" : "FAIL",
                ranks);
  pops::comm_finalize();
  return global_failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_flux_failure_collective, Runs) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_mpi_flux_failure_collective, "test_mpi_flux_failure_collective"),
      0);
}
