#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace pops::amr {

/// Three-valued result of one resolved AMR tagging expression.
///
/// Equality at a comparison leaf is deliberately Unknown: it is not coerced to either Boolean
/// branch before the complete refine/coarsen roots have been evaluated.
enum class TagTruth : std::uint8_t { False = 0, True = 1, Unknown = 2 };

[[nodiscard]] inline TagTruth tag_comparison(
    double sample, double threshold, bool greater) {
  if (!std::isfinite(sample) || !std::isfinite(threshold))
    throw std::domain_error("AMR Tagger rejected a non-finite indicator sample");
  if (sample == threshold)
    return TagTruth::Unknown;
  return (greater ? sample > threshold : sample < threshold)
             ? TagTruth::True
             : TagTruth::False;
}

[[nodiscard]] constexpr TagTruth tag_not(TagTruth value) noexcept {
  if (value == TagTruth::Unknown)
    return TagTruth::Unknown;
  return value == TagTruth::True ? TagTruth::False : TagTruth::True;
}

template <class Iterator>
[[nodiscard]] constexpr TagTruth tag_any(Iterator first, Iterator last) noexcept {
  bool unknown = false;
  for (; first != last; ++first) {
    if (*first == TagTruth::True)
      return TagTruth::True;
    unknown = unknown || *first == TagTruth::Unknown;
  }
  return unknown ? TagTruth::Unknown : TagTruth::False;
}

template <class Iterator>
[[nodiscard]] constexpr TagTruth tag_all(Iterator first, Iterator last) noexcept {
  bool unknown = false;
  for (; first != last; ++first) {
    if (*first == TagTruth::False)
      return TagTruth::False;
    unknown = unknown || *first == TagTruth::Unknown;
  }
  return unknown ? TagTruth::Unknown : TagTruth::True;
}

enum class TagEqualityPolicy : std::int32_t { Hold = 0, Refine = 1, Coarsen = 2 };
enum class TagConflictPolicy : std::int32_t {
  Error = 0,
  Hold = 1,
  RefineWins = 2,
  CoarsenWins = 3,
};

struct TagDecision {
  bool refine = false;
  bool coarsen = false;
  bool conflict_error = false;
};

/// Convert the two root truth values to actions.  Equality is mapped before conflict resolution,
/// including when one root is Unknown while the other root is already True.
[[nodiscard]] constexpr TagDecision resolve_tag_decision(
    TagTruth refine_root, TagTruth coarsen_root, TagEqualityPolicy equality_policy,
    TagConflictPolicy conflict_policy) noexcept {
  TagDecision decision{refine_root == TagTruth::True, coarsen_root == TagTruth::True, false};
  const bool has_unknown =
      refine_root == TagTruth::Unknown || coarsen_root == TagTruth::Unknown;
  if (has_unknown && equality_policy == TagEqualityPolicy::Refine)
    decision.refine = true;
  else if (has_unknown && equality_policy == TagEqualityPolicy::Coarsen)
    decision.coarsen = true;

  if (!decision.refine || !decision.coarsen)
    return decision;
  if (conflict_policy == TagConflictPolicy::Error) {
    decision.conflict_error = true;
  } else if (conflict_policy == TagConflictPolicy::Hold) {
    decision.refine = false;
    decision.coarsen = false;
  } else if (conflict_policy == TagConflictPolicy::RefineWins) {
    decision.coarsen = false;
  } else {
    decision.refine = false;
  }
  return decision;
}

}  // namespace pops::amr
