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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pops::runtime::program {

/// Phase of the StepAttemptRejected raised by a generated assessment that rejects its candidate.
/// Any other rejection is a failure of the trial and leaves the run.
inline constexpr std::string_view kAssessmentRejectionPhase = "assessment";

/// Why a compiled run returned.
enum class AdvanceRunStop : std::uint8_t {
  kReached = 0,        ///< the accepted time reached the frontier
  kStepBudget = 1,     ///< max_steps accepted steps before the frontier
  kAttemptBudget = 2,  ///< one step spent every attempt of the Advance
};

/// The uninterrupted segment of accepted steps of one proposal that a compiled run continues:
/// its origin and its step count, left at the accepted cursor (time, macro_step). A run at
/// another cursor or of another proposal starts a new segment at its accepted time.
struct AdvanceSegment {
  double proposal = 0.0;
  double origin = 0.0;
  std::int64_t steps = 0;
  double time = 0.0;
  std::int64_t macro_step = -1;  ///< -1 before the first compiled run
};

/// Native counters of one compiled run. Nothing of the run goes through the host interpreter.
struct AdvanceRunReport {
  std::int64_t steps = 0;       ///< accepted steps of the run
  std::int64_t rejections = 0;  ///< rejected attempts of the run
  AdvanceRunStop stop = AdvanceRunStop::kReached;
  std::int64_t attempts = 0;  ///< attempts of the step that spent the attempt budget
  std::string rejection;      ///< diagnostic of the last rejected attempt, empty when none
  AdvanceSegment segment;     ///< the segment that the next run continues
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

}  // namespace advance_run_detail

/// Duration of the attempt from @p time toward @p frontier, in the segment of @p proposal that
/// started at @p origin and holds @p steps accepted steps.
///
/// The proposal is kept while its rounded endpoint precedes the frontier by more than four units
/// in the last place, and while the segment endpoint origin + (steps + 1) * proposal, rounded,
/// is not the frontier; a proposal whose endpoint is the frontier lands with it. These are the
/// rules of master's FixedDt (python/pops/runtime/_step_strategy.py@bb9), which absorb the
/// rounding that repeated additions accumulate. A landing attempt takes the binary64 distance
/// whose rounded sum with @p time is the frontier: the nearest rounding first, then the other
/// faithful one. When neither publishes the frontier, which happens only for an odd frontier
/// above a time with a half-unit bit in a lower binade, the attempt ends at the last
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
      origin + static_cast<double>(steps + 1) * proposal == frontier ||
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
