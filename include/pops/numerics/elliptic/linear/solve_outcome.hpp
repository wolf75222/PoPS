#pragma once

/// @file
/// @brief One-shot publication contract shared by global, field, and hierarchy solves.

#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>

namespace pops {

/// Exact disposition selected by the consumer of one prepared solve.
enum class SolveConsumption {
  kAccept,
  kRejectAttempt,
  kFailRun,
};

/// A solve report whose value/publication boundary must be consumed exactly once.
///
/// The report remains inspectable so the runtime can select the authored failure action, but the
/// solve result does not become an accepted graph value until consume(kAccept) succeeds. Optional
/// hooks keep an owning transaction open until that boundary. Distributed outcomes authenticate the
/// exact same action on every rank before any hook runs.
class [[nodiscard]] SolveOutcome final {
 public:
  using Hook = void (*)(void*);
  using ReleaseHook = void (*)(void*) noexcept;
  using FailureHook = void (*)(void*, SolveConsumption);

  struct PublicationHooks {
    void* context;
    Hook accept;
    Hook reject;
    ReleaseHook release;
    /// Optional shared owner for @ref context. Runtime-owned transactions normally outlive their
    /// outcomes and leave this empty; context-owned publication workspaces retain their exact owner
    /// here so returning an outcome cannot create a dangling callback.
    std::shared_ptr<void> lifetime{};
    /// Optional read-only validation performed before an Accept becomes irreversible. This permits
    /// a publication owner to reject layout drift without consuming an otherwise valid outcome.
    Hook validate_accept = nullptr;
    /// Optional action-aware failure callback. When present it replaces @ref reject so diagnostics
    /// can distinguish RejectAttempt from FailRun without creating a second outcome type.
    FailureHook consume_failure = nullptr;
  };

  SolveOutcome(const SolveOutcome&) = delete;
  SolveOutcome& operator=(const SolveOutcome&) = delete;
  SolveOutcome& operator=(SolveOutcome&&) = delete;

  SolveOutcome(SolveOutcome&& other) noexcept
      : report_(std::move(other.report_)),
        hooks_(other.hooks_),
        lane_(other.lane_),
        collective_(other.collective_),
        consumed_(std::exchange(other.consumed_, true)) {
    other.hooks_ = empty_hooks_();
    other.lane_ = nullptr;
    other.collective_ = Collective::kSerial;
  }

  ~SolveOutcome() {
    if (!consumed_) {
      std::fputs("PoPS contract violation: SolveOutcome destroyed before explicit consumption\n",
                 stderr);
      std::fflush(stderr);
      std::abort();
    }
  }

  static SolveOutcome serial(SolveReport report) {
    return serial(std::move(report), empty_hooks_());
  }

  static SolveOutcome serial(SolveReport report, PublicationHooks hooks) {
    return SolveOutcome(std::move(report), hooks, Collective::kSerial, nullptr);
  }

  static SolveOutcome collective_world(SolveReport report) {
    return collective_world(std::move(report), empty_hooks_());
  }

  static SolveOutcome collective_world(SolveReport report, PublicationHooks hooks) {
    return SolveOutcome(std::move(report), hooks, Collective::kWorld, nullptr);
  }

  static SolveOutcome collective_lane(SolveReport report, const ExecutionLane& lane) {
    return collective_lane(std::move(report), lane, empty_hooks_());
  }

  static SolveOutcome collective_lane(SolveReport report, const ExecutionLane& lane,
                                      PublicationHooks hooks) {
    return SolveOutcome(std::move(report), hooks, Collective::kExecutionLane, &lane);
  }

  [[nodiscard]] const SolveReport& report() const noexcept { return report_; }

  SolveReport consume(SolveConsumption action) {
    if (consumed_)
      throw std::logic_error("SolveOutcome has already been consumed");

    require_collective_action_consensus_(action);
    if (action == SolveConsumption::kAccept) {
      if (!report_.solved_value_available())
        throw std::logic_error("cannot accept a failed SolveOutcome");
      if (hooks_.validate_accept != nullptr)
        hooks_.validate_accept(hooks_.context);
    } else if (action == SolveConsumption::kRejectAttempt ||
               action == SolveConsumption::kFailRun) {
      if (report_.solved_value_available())
        throw std::logic_error("cannot reject a solved SolveOutcome");
    } else {
      throw std::logic_error("invalid SolveConsumption action");
    }

    // Validation and collective agreement are complete. From here onward a callback may mutate the
    // published state, so retrying the same outcome is forbidden even if that callback throws.
    consumed_ = true;
    struct Release {
      PublicationHooks* hooks;
      ~Release() {
        if (hooks->release != nullptr)
          hooks->release(hooks->context);
        *hooks = PublicationHooks{nullptr, nullptr, nullptr, nullptr};
      }
    } release{&hooks_};

    if (action == SolveConsumption::kAccept) {
      if (hooks_.accept != nullptr)
        hooks_.accept(hooks_.context);
    } else {
      report_.action = action == SolveConsumption::kRejectAttempt
                           ? SolveAction::kRejectAttempt
                           : SolveAction::kFailRun;
      if (hooks_.consume_failure != nullptr)
        hooks_.consume_failure(hooks_.context, action);
      else if (hooks_.reject != nullptr)
        hooks_.reject(hooks_.context);
    }
    return report_;
  }

 private:
  enum class Collective {
    kSerial,
    kWorld,
    kExecutionLane,
  };

  SolveOutcome(SolveReport report, PublicationHooks hooks, Collective collective,
               const ExecutionLane* lane)
      : report_(std::move(report)), hooks_(hooks), lane_(lane), collective_(collective) {}

  static PublicationHooks empty_hooks_() noexcept {
    return {nullptr, nullptr, nullptr, nullptr};
  }

  void require_collective_action_consensus_(SolveConsumption action) const {
    const long code = static_cast<long>(action);
    long minimum = code;
    long maximum = code;
    if (collective_ == Collective::kWorld) {
      minimum = all_reduce_min(code);
      maximum = all_reduce_max(code);
    } else if (collective_ == Collective::kExecutionLane) {
      if (lane_ == nullptr)
        throw std::logic_error("SolveOutcome has no execution lane");
      minimum = all_reduce_min(code, *lane_);
      maximum = all_reduce_max(code, *lane_);
    }
    if (minimum != maximum)
      throw std::logic_error("SolveOutcome consumption action differs between MPI ranks");
  }

  SolveReport report_;
  PublicationHooks hooks_ = empty_hooks_();
  const ExecutionLane* lane_ = nullptr;
  Collective collective_ = Collective::kSerial;
  bool consumed_ = false;
};

/// Consume a prepared outcome using the solver-authored success/failure action. This is the
/// explicit convenience boundary for direct native and Python-facade callers that do not implement
/// an adaptive retry policy of their own.
inline SolveReport consume_solve_outcome(SolveOutcome outcome) {
  return outcome.consume(
      outcome.report().solved_value_available()
          ? SolveConsumption::kAccept
          : (outcome.report().action == SolveAction::kRejectAttempt
                 ? SolveConsumption::kRejectAttempt
                 : SolveConsumption::kFailRun));
}

}  // namespace pops
