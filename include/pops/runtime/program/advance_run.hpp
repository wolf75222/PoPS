#pragma once

/// @file
/// @brief The compiled run of an elaborated Advance (spec 0.3 section 5.2, CC14).
///
/// A run repeats attempts of the installed Program until the accepted time reaches its frontier.
/// One attempt is one System step transaction: the generated trial computes the candidate, the
/// generated assessment reads it before the commit, and a rejected candidate restores the
/// accepted state. The next attempt repeats the proposal of the rejected one (T28, next=None)
/// until the attempt budget of the Advance is spent (T38). The run lands on its frontier as
/// master's fixed step does (spec 0.3 sections 2.6.2 and 14.1): it keeps the proposal until the
/// frontier is within it, within four units in the last place of it, or the rounded endpoint of
/// its uninterrupted fixed-proposal segment, and the attempt that lands publishes the frontier
/// itself.
///
/// The Outputs of the Advance (CC14) are evaluated once at the start of the run, on the state from
/// which it starts, and after each publication whose macro step is a multiple of their step
/// cadence (when=every(k)) or whose time reaches one or more of their times (at) that no earlier
/// publication reached. Each of those times is a frontier: the attempt before it is capped at its
/// distance and lands on it exactly. An Output is evaluated after publication, outside every step:
/// its scalars by the generated entry of the artifact, its arrays by the gather of the accepted
/// state, a copy of core machinery (CF3).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pops::runtime::program {

/// Phase of the StepAttemptRejected raised by a generated assessment that rejects its candidate.
inline constexpr std::string_view kAssessmentRejectionPhase = "assessment";
/// Phase of the StepAttemptRejected raised by a retryable failure site of the trial (spec 0.3
/// T38): a field request whose dependency set reads the proposal. It rejects the attempt.
inline constexpr std::string_view kTrialRejectionPhase = "trial";
/// Phase of the StepAttemptRejected raised by a fixed failure site of the trial (T38): a field
/// request whose dependency set excludes the proposal, which a retry cannot change. It ends the
/// step and consumes no attempt. Any other rejection leaves the run as the error that raised it.
inline constexpr std::string_view kTrialFailurePhase = "trial_failure";

/// Why a compiled run returned.
enum class AdvanceRunStop : std::uint8_t {
  kReached = 0,        ///< the accepted time reached the frontier
  kStepBudget = 1,     ///< max_steps accepted steps before the frontier
  kAttemptBudget = 2,  ///< one step spent every attempt of the Advance
  kTrialFailure = 3,   ///< a fixed failure site of the trial ended the step (T38)
};

/// The uninterrupted segment of accepted steps of one proposal that a compiled run continues:
/// its origin and its step count, left at the accepted cursor (time, macro_step). A run at
/// another cursor or of another proposal starts a new segment at its accepted time.
///
/// The segment is controller memory, held natively only. Master keeps its fixed-step grid in the
/// temporal resume state (controller_state["fixed_dt_grid"], _step_strategy.py@bb9): the
/// continuation codec must carry the segment, since a run resumed without it can land otherwise
/// than an uninterrupted one. From 0 to 0.25 in steps of 0.05/16, a segment lost after k steps
/// gives 81 steps instead of 80 for 53 of the 79 values of k.
struct AdvanceSegment {
  double proposal = 0.0;
  double origin = 0.0;
  std::int64_t steps = 0;
  double time = 0.0;
  std::int64_t macro_step = -1;  ///< -1 before the first compiled run
};

/// The step rule of an elaborated Advance (N8 lot « Règle de pas compilée »): the proposal of each
/// step is b6's selection of System::step_cfl without its execution (the wave speeds of the
/// accepted state, floored at @p speed_floor, the coupled frequencies and the Program dt bound),
/// at @p fraction and capped at @p max_dt, taken once per step before its attempts and then cut at
/// the frontier by advance_attempt_duration. A rule has no fixed-proposal segment: each step starts
/// its own. The library word that elaborates onto it names the rule; the core does not.
struct AdvanceStepRule {
  double fraction = 0.0;
  double max_dt = std::numeric_limits<double>::infinity();  ///< +infinity when the rule has none
  double speed_floor = 0.0;  ///< the floor of step_cfl's default, set by the binding
};

/// One Output of the elaborated Advance as a compiled run schedules it (CC14).
struct AdvanceOutputSchedule {
  /// when=every(k): due after each publication whose macro step is a multiple of k; 0 for none.
  std::int64_t every_steps = 0;
  /// The times of at= that lie ahead of the run, strictly increasing; each one is a frontier.
  std::vector<double> at;
  /// The blocks whose accepted state the record publishes as arrays, gathered from every rank.
  std::vector<std::string> arrays;
};

/// One evaluation of an Output after a publication, or at the start of the run.
struct AdvanceOutputSample {
  int output = 0;               ///< index of the Output in the schedule of the run
  double time = 0.0;            ///< the achieved time of the publication
  std::int64_t macro_step = 0;  ///< its accepted macro step
  std::vector<double> scalars;  ///< the scalars of the record, in the order of the artifact
  std::vector<std::vector<double>> arrays;  ///< the gathered states, in the order of `arrays`
};

/// Native counters of one compiled run. Nothing of the run goes through the host interpreter.
struct AdvanceRunReport {
  std::int64_t steps = 0;       ///< accepted steps of the run
  std::int64_t rejections = 0;  ///< rejected attempts of the run
  AdvanceRunStop stop = AdvanceRunStop::kReached;
  std::int64_t attempts = 0;         ///< attempts of the step that spent its budget or failed
  std::string rejection;             ///< diagnostic of the last rejected attempt, empty when none
  std::string rejection_phase;       ///< phase of that rejection: assessment, trial, trial_failure
  std::string rejection_detail;      ///< its detail; a trial site starts with the site name
  std::string rejection_status;      ///< its solve status name (solve_status_name), empty when none
  std::int64_t prepared_solves = 0;  ///< prepared linear solves that ran, every attempt included
  /// The iteration count and final residual norm of the last of them; -1 and NaN without one.
  std::int64_t last_prepared_solve_iterations = -1;
  double last_prepared_solve_residual = std::numeric_limits<double>::quiet_NaN();
  AdvanceSegment segment;                    ///< the segment that the next run continues
  std::vector<AdvanceOutputSample> outputs;  ///< the evaluations of the Outputs, in run order
};

/// Duration of the next attempt, whether it publishes the frontier, and whether it keeps the
/// proposal, so that its segment continues.
struct AdvanceAttemptDuration {
  double dt = 0.0;
  bool lands = false;
  bool continuous = false;
};

namespace advance_run_detail {

/// The unit in the last place of @p value, as Python's math.ulp: the gap above a finite
/// magnitude, the gap below the largest one.
inline double ulp(double value) {
  const double magnitude = std::abs(value);
  const double above = std::nextafter(magnitude, std::numeric_limits<double>::infinity());
  return std::isfinite(above) ? above - magnitude : magnitude - std::nextafter(magnitude, 0.0);
}

/// The rounded endpoint origin + (steps + 1) * proposal of a fixed-proposal segment, with the
/// two roundings of master, whose Python binary64 rounds the product and then the sum. A target
/// with a fused multiply-add contracts the expression into one rounding by default (GCC
/// -ffp-contract=fast, Clang -ffp-contract=on; AArch64 has it in its base instruction set), and
/// the native module is not compiled with -ffp-contract=off as generated code is. The product
/// is stored and read back as a volatile, so that no compiler fuses it with the sum.
inline double segment_endpoint(double origin, std::int64_t steps, double proposal) {
  const volatile double product = static_cast<double>(steps + 1) * proposal;
  return origin + product;
}

}  // namespace advance_run_detail

/// The frontier of the next attempt from @p time: @p until, or the first time of an Output ahead
/// of it when that time comes first. @p cursors holds, per Output, the index of its first time
/// that no publication has reached.
inline double advance_next_frontier(double time, double until,
                                    const std::vector<AdvanceOutputSchedule>& schedule,
                                    const std::vector<std::size_t>& cursors) {
  double frontier = until;
  for (std::size_t output = 0; output < schedule.size(); ++output) {
    const auto& at = schedule[output].at;
    if (cursors[output] < at.size()) {
      const double next = at[cursors[output]];
      if (!(next > time))
        throw std::logic_error("Advance output time was reached without its evaluation");
      frontier = std::min(frontier, next);
    }
  }
  return frontier;
}

/// Whether Output @p output is due after the publication at (@p time, @p macro_step), and moves
/// its cursor past every time of it that the publication reached or passed: an Output whose
/// times a publication passes is evaluated once for that publication (CC14).
inline bool advance_output_due(std::size_t output, double time, std::int64_t macro_step,
                               const std::vector<AdvanceOutputSchedule>& schedule,
                               std::vector<std::size_t>& cursors) {
  const auto& entry = schedule[output];
  bool due = entry.every_steps > 0 && macro_step % entry.every_steps == 0;
  while (cursors[output] < entry.at.size() && time >= entry.at[cursors[output]]) {
    ++cursors[output];
    due = true;
  }
  return due;
}

/// Checks a schedule against the accepted time @p time from which a run starts.
inline void validate_advance_output_schedule(double time,
                                             const std::vector<AdvanceOutputSchedule>& schedule) {
  for (const auto& entry : schedule) {
    if (entry.every_steps < 0)
      throw std::invalid_argument("Advance output step cadence must be non-negative");
    double previous = time;
    for (const double at : entry.at) {
      if (!std::isfinite(at) || !(at > previous))
        throw std::invalid_argument(
            "Advance output times must be finite, strictly increasing and after the accepted "
            "time from which the run starts");
      previous = at;
    }
  }
}

/// Duration of the attempt from @p time toward @p frontier, in the segment of @p proposal that
/// started at @p origin and holds @p steps accepted steps.
///
/// The proposal is kept while its rounded endpoint precedes the frontier by more than four units
/// in the last place, and while the segment endpoint origin + (steps + 1) * proposal, its product
/// and its sum each rounded, is not the frontier; a proposal whose endpoint is the frontier lands
/// with it. These are the rules of master's FixedDt (python/pops/runtime/_step_strategy.py@bb9),
/// which absorb the rounding that repeated additions accumulate. A landing attempt takes the
/// binary64 distance whose rounded sum with @p time is the frontier: the nearest rounding first,
/// then the other faithful one. When neither publishes the frontier, which happens only for an
/// odd frontier above a time with a half-unit bit in a lower binade, the attempt ends at the last
/// representable time below the frontier; the next attempt then starts within a factor two of
/// the frontier, where the distance is exact (Sterbenz) and lands.
inline AdvanceAttemptDuration advance_attempt_duration(double time, double proposal,
                                                       double frontier, double origin,
                                                       std::int64_t steps) {
  if (!std::isfinite(time) || !std::isfinite(proposal) || !std::isfinite(frontier) ||
      !std::isfinite(origin) || !(proposal > 0.0) || !(frontier > time) || !(origin <= time) ||
      steps < 0 || steps == std::numeric_limits<std::int64_t>::max())
    throw std::invalid_argument(
        "Advance attempt duration requires a finite time before a finite frontier, a finite "
        "positive proposal and a segment that starts at or before the time");
  const double end = time + proposal;
  if (end == frontier)
    return {proposal, true, true};
  const double remaining = frontier - time;
  const double reached = time + std::min(proposal, remaining);
  const bool absorbs =
      end > frontier || !(proposal < remaining) ||
      advance_run_detail::segment_endpoint(origin, steps, proposal) == frontier ||
      std::abs(reached - frontier) <=
          4.0 * std::max(advance_run_detail::ulp(reached), advance_run_detail::ulp(frontier));
  if (!absorbs) {
    if (!(end > time))
      throw std::overflow_error("Advance proposal does not advance the binary64 clock");
    return {proposal, false, true};
  }
  const double landed = time + remaining;
  if (remaining > 0.0 && landed == frontier)
    return {remaining, true, remaining == proposal};
  // The other faithful rounding lies on the side of the exact distance: above the nearest one when
  // the nearest sum falls short of the frontier, below it when the sum passes the frontier.
  const double other =
      std::nextafter(remaining, landed < frontier ? std::numeric_limits<double>::infinity() : 0.0);
  if (other > 0.0 && time + other == frontier)
    return {other, true, other == proposal};
  const double below = landed < frontier ? remaining : other;
  if (!(below > 0.0) || !(time + below > time) || !(time + below < frontier))
    throw std::overflow_error("Advance attempt cannot approach its frontier in binary64");
  return {below, false, below == proposal};
}

}  // namespace pops::runtime::program
