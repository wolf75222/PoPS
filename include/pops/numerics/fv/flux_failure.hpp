#pragma once

/// @file
/// @brief Device-to-host failure channel for pointwise numerical-flux evaluations.
///
/// Flux providers execute inside Kokkos kernels and therefore cannot throw.  Every spatial
/// entrypoint owns one tracker, passes its trivially-copyable recorder to all of its kernels, and
/// consumes the collective report before publishing the computed field.  The packed reduction is
/// ordered first by status severity (Ok < Retry < Reject < Failed), then by the unsigned reason
/// code.  Consequently concurrent failures produce one deterministic result on every backend and,
/// after the world reduction, on every MPI rank.

#include <pops/numerics/fv/flux_interfaces.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/export.hpp>

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace pops {

POPS_HD constexpr std::uint32_t flux_evaluation_severity(EvaluationStatus status) {
  switch (status) {
    case EvaluationStatus::kOk:
      return 0;
    case EvaluationStatus::kRetry:
      return 1;
    case EvaluationStatus::kReject:
      return 2;
    case EvaluationStatus::kFailed:
      return 3;
  }
  // An invalid external enum value is a fatal provider failure, never a successful evaluation.
  return 3;
}

inline const char* evaluation_status_name(EvaluationStatus status) noexcept {
  switch (status) {
    case EvaluationStatus::kOk:
      return "ok";
    case EvaluationStatus::kRetry:
      return "retry";
    case EvaluationStatus::kReject:
      return "reject";
    case EvaluationStatus::kFailed:
      return "failed";
  }
  return "failed";
}

struct FluxFailureReport {
  EvaluationStatus status = EvaluationStatus::kOk;
  std::uint32_t reason_code = 0;

  [[nodiscard]] bool failed() const noexcept { return status != EvaluationStatus::kOk; }
  [[nodiscard]] TransactionFailureAction action() const noexcept {
    return transaction_action(status);
  }
};

/// Fatal or attempt-rejecting numerical-flux result after the device and MPI reductions have
/// completed.  Direct spatial-operator callers receive this type.  Runtime step transactions map
/// Retry/Reject to StepAttemptRejected only after restoring their accepted snapshot; Failed is
/// deliberately rethrown as this exact fatal type.
#if defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
#define POPS_FLUX_EXCEPTION_ABI POPS_EXPORT
#else
#define POPS_FLUX_EXCEPTION_ABI
#endif

class POPS_FLUX_EXCEPTION_ABI FluxEvaluationFailure final : public std::runtime_error {
 public:
  FluxEvaluationFailure(EvaluationStatus status, std::uint32_t reason_code, std::string phase)
      : std::runtime_error(message(status, reason_code, phase)),
        status_(status),
        reason_code_(reason_code),
        phase_(std::move(phase)) {}
  ~FluxEvaluationFailure() noexcept override;

  [[nodiscard]] EvaluationStatus status() const noexcept { return status_; }
  [[nodiscard]] TransactionFailureAction action() const noexcept {
    return transaction_action(status_);
  }
  [[nodiscard]] std::uint32_t reason_code() const noexcept { return reason_code_; }
  [[nodiscard]] const std::string& phase() const noexcept { return phase_; }

 private:
  static std::string message(EvaluationStatus status, std::uint32_t reason_code,
                             const std::string& phase) {
    std::ostringstream out;
    out << "numerical flux evaluation " << evaluation_status_name(status) << " during " << phase
        << ": reason_code=0x" << std::hex << std::setw(8) << std::setfill('0') << reason_code;
    return out.str();
  }

  EvaluationStatus status_;
  std::uint32_t reason_code_;
  std::string phase_;
};

#if !defined(POPS_RUNTIME_SHARED_EXCEPTION_ABI)
inline FluxEvaluationFailure::~FluxEvaluationFailure() noexcept = default;
#endif

#undef POPS_FLUX_EXCEPTION_ABI

namespace detail {

inline constexpr std::uint64_t kFluxReasonMask = UINT64_C(0xffffffff);
inline constexpr int kFluxSeverityShift = 32;
inline constexpr std::uint32_t kNonFiniteFiniteVolumeReason = UINT32_C(0x4e46494e);  // "NFIN"

POPS_HD constexpr std::uint64_t pack_flux_failure(EvaluationStatus status,
                                                  std::uint32_t reason_code) {
  return (static_cast<std::uint64_t>(flux_evaluation_severity(status))
          << kFluxSeverityShift) |
         static_cast<std::uint64_t>(reason_code);
}

inline FluxFailureReport unpack_flux_failure(std::uint64_t packed) {
  const auto severity = static_cast<std::uint32_t>(packed >> kFluxSeverityShift);
  EvaluationStatus status = EvaluationStatus::kFailed;
  switch (severity) {
    case 0:
      status = EvaluationStatus::kOk;
      break;
    case 1:
      status = EvaluationStatus::kRetry;
      break;
    case 2:
      status = EvaluationStatus::kReject;
      break;
    case 3:
      status = EvaluationStatus::kFailed;
      break;
    default:
      status = EvaluationStatus::kFailed;
      break;
  }
  return {status, static_cast<std::uint32_t>(packed & kFluxReasonMask)};
}

}  // namespace detail

/// Stateless device helper used inside a Kokkos::Max<uint64_t> reduction.  The spatial kernel writes
/// its ordinary output as a side effect and joins any failure into the reduction accumulator; no
/// device allocation, atomic, pool or mutable global state is involved.
struct FluxEvaluationRecorder {
  template <class State>
  POPS_HD void record(const FluxEvaluation<State>& evaluation, std::uint64_t& aggregate) const {
    if (evaluation.succeeded())
      return;
    const std::uint64_t candidate =
        detail::pack_flux_failure(evaluation.status, evaluation.reason_code);
    if (candidate > aggregate)
      aggregate = candidate;
  }

  POPS_HD void record_nonfinite(Real value, std::uint64_t& aggregate) const {
    if (Kokkos::isfinite(value))
      return;
    const std::uint64_t candidate = detail::pack_flux_failure(
        EvaluationStatus::kFailed, detail::kNonFiniteFiniteVolumeReason);
    if (candidate > aggregate)
      aggregate = candidate;
  }
};

/// Explicit authority for the current transport scheduler's process-world collective order.
/// System/AmrSystem execute transport blocks synchronously on every world rank; their prepared
/// boundary lanes are named non-owning views of MPI_COMM_WORLD and complete halo/provider work
/// before the spatial kernel enters this reduction.  A future independently duplicated/concurrent
/// RHS lane must add and thread an exact CommunicatorView overload instead of reusing this token.
struct ProcessWorldFluxCollective final {};
inline constexpr ProcessWorldFluxCollective process_world_flux_collective{};

/// Allocation-free host accumulator of per-box device reductions. `collective_report()` is an
/// unconditional process-world collective: callers must invoke it at the same spatial-operator
/// boundary on every world rank, even when the local MultiFab has no boxes or no local evaluation
/// failed.  Construction requires the explicit scheduler authority above, preventing an unnoticed
/// default-WORLD collective from appearing in a future lane-aware call site.
class FluxEvaluationTracker {
 public:
  FluxEvaluationTracker() = delete;
  explicit constexpr FluxEvaluationTracker(ProcessWorldFluxCollective) noexcept {}

  [[nodiscard]] FluxEvaluationRecorder recorder() const noexcept { return {}; }

  void merge(std::uint64_t packed) noexcept {
    if (packed > packed_)
      packed_ = packed;
  }

  [[nodiscard]] bool local_failed() const noexcept { return packed_ != 0; }

  [[nodiscard]] FluxFailureReport collective_report() const {
    const auto global = all_reduce_max(packed_);
    return detail::unpack_flux_failure(global);
  }

  void throw_if_failed(const char* phase) const {
    const FluxFailureReport report = collective_report();
    if (report.failed())
      throw FluxEvaluationFailure(report.status, report.reason_code, phase);
  }

 private:
  std::uint64_t packed_ = 0;
};

static_assert(std::is_trivially_copyable_v<FluxEvaluationTracker>);

}  // namespace pops
