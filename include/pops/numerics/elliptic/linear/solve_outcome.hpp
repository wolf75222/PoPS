#pragma once

/// @file
/// @brief One-shot publication contract shared by global, field, and hierarchy solves.

#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace pops {

template <int Dim> class System;

/// Runtime-owned identity for the exact native attempt that produced a solve result.
/// Outcomes retain this small control object, never the System or its buffers.
struct SolveOutcomeAttemptAuthority {
  void* owner = nullptr;
  std::uint64_t incarnation = 1;
  std::uint64_t attempt = 1;
  std::atomic<std::size_t> pending{0};
};

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
  using AcceptHook = void (*)(void*) noexcept;
  using ReleaseHook = void (*)(void*) noexcept;
  using FailureHook = void (*)(void*, SolveConsumption);

  struct PublicationHooks {
    void* context;
    /// Publication is irreversible after collective validation. It must therefore be fail-stop:
    /// an implementation may terminate on an impossible post-validation copy failure, but it may
    /// never unwind and let callers continue with a partially published value.
    AcceptHook accept;
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
    /// Prepared external workspaces use these weak owner identities before any raw lane/callback.
    /// They do not extend the lifetime of caller-owned problem or workspace objects.
    std::weak_ptr<void> workspace_lifetime{};
    std::weak_ptr<void> problem_lifetime{};
    bool require_external_lifetimes = false;
  };

  SolveOutcome(const SolveOutcome&) = delete;
  SolveOutcome& operator=(const SolveOutcome&) = delete;
  SolveOutcome& operator=(SolveOutcome&&) = delete;

  SolveOutcome(SolveOutcome&& other) noexcept
      : report_(std::move(other.report_)),
        hooks_(other.hooks_),
        lane_(other.lane_),
        collective_(other.collective_),
        attempt_authority_(std::move(other.attempt_authority_)),
        attempt_lane_(std::move(other.attempt_lane_)),
        attempt_incarnation_(other.attempt_incarnation_),
        attempt_ordinal_(other.attempt_ordinal_),
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

    if (attempt_authority_) {
      const bool external_alive = !hooks_.require_external_lifetimes ||
                                  (!hooks_.workspace_lifetime.expired() &&
                                   !hooks_.problem_lifetime.expired());
      const bool current = external_alive && attempt_authority_->owner != nullptr &&
                           attempt_authority_->incarnation == attempt_incarnation_ &&
                           attempt_authority_->attempt == attempt_ordinal_;
      const long stale = current ? 0L : 1L;
      const long stale_anywhere = attempt_lane_ ? all_reduce_max(stale, *attempt_lane_) : stale;
      if (stale_anywhere != 0) {
        // The owner can have been destroyed or replaced. No raw lane or publication callback may
        // be used on such a rank. A live peer still releases its candidate before everyone fails.
        consumed_ = true;
        if (external_alive && hooks_.require_external_lifetimes && hooks_.release != nullptr)
          hooks_.release(hooks_.context);
        else if (current && hooks_.release != nullptr)
          hooks_.release(hooks_.context);
        hooks_ = empty_hooks_();
        release_attempt_();
        throw std::logic_error("SolveOutcome belongs to a revoked native solve attempt");
      }
    }

    if (!collective_report_disposition_agrees_()) {
      // An inconsistent distributed report cannot ever be consumed safely. Roll its candidate back
      // uniformly and close the transaction before surfacing the contract error; unlike a
      // caller-action mismatch, retrying this same outcome cannot repair its authored disposition.
      consumed_ = true;
      struct AttemptRelease {
        SolveOutcome* outcome;
        ~AttemptRelease() { outcome->release_attempt_(); }
      } attempt_release{this};
      struct Release {
        PublicationHooks* hooks;
        ~Release() {
          if (hooks->release != nullptr)
            hooks->release(hooks->context);
          *hooks = PublicationHooks{nullptr, nullptr, nullptr, nullptr};
        }
      } release{&hooks_};
      report_.action = SolveAction::kFailRun;
      if (hooks_.consume_failure != nullptr)
        hooks_.consume_failure(hooks_.context, SolveConsumption::kFailRun);
      else if (hooks_.reject != nullptr)
        hooks_.reject(hooks_.context);
      throw std::logic_error("SolveOutcome report disposition differs between MPI ranks");
    }
    require_collective_action_consensus_(action);
    if (action == SolveConsumption::kAccept) {
      if (!report_.solved_value_available())
        throw std::logic_error("cannot accept a failed SolveOutcome");
      if (hooks_.validate_accept != nullptr)
        require_collective_accept_validation_();
    } else if (action == SolveConsumption::kRejectAttempt || action == SolveConsumption::kFailRun) {
      if (report_.solved_value_available())
        throw std::logic_error("cannot reject a solved SolveOutcome");
      if (action == SolveConsumption::kRejectAttempt && report_.action == SolveAction::kFailRun)
        throw std::logic_error("cannot downgrade a FailRun SolveOutcome to RejectAttempt");
    } else {
      throw std::logic_error("invalid SolveConsumption action");
    }

    // Validation and collective agreement are complete. From here onward a callback may mutate the
    // published state, so retrying the same outcome is forbidden even if that callback throws.
    consumed_ = true;
    struct AttemptRelease {
      SolveOutcome* outcome;
      ~AttemptRelease() { outcome->release_attempt_(); }
    } attempt_release{this};
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
      report_.action = action == SolveConsumption::kRejectAttempt ? SolveAction::kRejectAttempt
                                                                  : SolveAction::kFailRun;
      if (hooks_.consume_failure != nullptr)
        hooks_.consume_failure(hooks_.context, action);
      else if (hooks_.reject != nullptr)
        hooks_.reject(hooks_.context);
    }
    return report_;
  }

 private:
  template <int Dim> friend class System;

  void bind_native_attempt_(std::shared_ptr<SolveOutcomeAttemptAuthority> authority,
                            std::shared_ptr<ExecutionLane> lane) noexcept {
    attempt_incarnation_ = authority->incarnation;
    attempt_ordinal_ = authority->attempt;
    authority->pending.fetch_add(1, std::memory_order_acq_rel);
    attempt_authority_ = std::move(authority);
    attempt_lane_ = std::move(lane);
  }

  void release_attempt_() noexcept {
    if (attempt_authority_) {
      if (attempt_authority_->pending.fetch_sub(1, std::memory_order_acq_rel) == 0)
        std::terminate();
      attempt_authority_.reset();
      attempt_lane_.reset();
    }
  }

  enum class Collective {
    kSerial,
    kWorld,
    kExecutionLane,
  };

  SolveOutcome(SolveReport report, PublicationHooks hooks, Collective collective,
               const ExecutionLane* lane)
      : report_(std::move(report)), hooks_(hooks), lane_(lane), collective_(collective) {}

  static PublicationHooks empty_hooks_() noexcept { return {nullptr, nullptr, nullptr, nullptr}; }

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

  [[nodiscard]] bool collective_report_disposition_agrees_() const {
    long action_code = 3;
    if (report_.action == SolveAction::kNone)
      action_code = 0;
    else if (report_.action == SolveAction::kRejectAttempt)
      action_code = 1;
    else if (report_.action == SolveAction::kFailRun)
      action_code = 2;
    const long code = (report_.solved_value_available() ? 4L : 0L) + action_code;
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
    return minimum == maximum;
  }

  void require_collective_accept_validation_() const {
    std::exception_ptr local_error;
    long failed = 0;
    try {
      hooks_.validate_accept(hooks_.context);
    } catch (...) {
      local_error = std::current_exception();
      failed = 1;
    }

    long failed_anywhere = failed;
    if (collective_ == Collective::kWorld) {
      failed_anywhere = all_reduce_max(failed);
    } else if (collective_ == Collective::kExecutionLane) {
      if (lane_ == nullptr)
        throw std::logic_error("SolveOutcome has no execution lane");
      failed_anywhere = all_reduce_max(failed, *lane_);
    }
    if (failed_anywhere == 0)
      return;
    if (collective_ == Collective::kSerial && local_error != nullptr)
      std::rethrow_exception(local_error);
    throw std::logic_error("SolveOutcome accept validation failed on at least one MPI rank");
  }

  SolveReport report_;
  PublicationHooks hooks_ = empty_hooks_();
  const ExecutionLane* lane_ = nullptr;
  Collective collective_ = Collective::kSerial;
  std::shared_ptr<SolveOutcomeAttemptAuthority> attempt_authority_;
  std::shared_ptr<ExecutionLane> attempt_lane_;
  std::uint64_t attempt_incarnation_ = 0;
  std::uint64_t attempt_ordinal_ = 0;
  bool consumed_ = false;
};

/// Consume a prepared outcome using the solver-authored success/failure action. This is the
/// explicit convenience boundary for direct native and Python-facade callers that do not implement
/// an adaptive retry policy of their own. Such callers may only continue with a solved value:
/// recoverable rejection is still consumed (and therefore rolls back its candidate), then promoted
/// to a fail-run exception instead of letting execution proceed on stale accepted fields.
inline SolveReport consume_solve_outcome(SolveOutcome outcome) {
  SolveReport report = outcome.consume(outcome.report().solved_value_available()
                                           ? SolveConsumption::kAccept
                                           : (outcome.report().action == SolveAction::kRejectAttempt
                                                  ? SolveConsumption::kRejectAttempt
                                                  : SolveConsumption::kFailRun));
  if (!report.solved_value_available())
    throw std::runtime_error(std::string("prepared solve failed: status=") + report.status_name() +
                             " action=" + report.action_name() + " reason=" + report.reason);
  return report;
}

}  // namespace pops
