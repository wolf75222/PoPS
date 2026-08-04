/// @file
/// @brief Transaction-local, axis-qualified AMR face-flux ledger for dimensions 1..3.

#pragma once

#include <pops/mesh/index/index.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::amr::reflux {

/// Spatial centering is part of the persisted identity.  This ledger accepts only face-centered
/// numerical fluxes; Cell exists so attempts to route source terms fail explicitly at the boundary.
enum class FaceLedgerCentering : std::uint8_t { Face = 0, Cell = 1 };

enum class FaceLedgerRole : std::uint8_t { Coarse = 0, Fine = 1 };

/// Sources alter a cell volume and are never conservative face exchanges.  Keeping Source as an
/// explicit rejected value prevents a generic producer from silently recording it as a flux.
enum class FaceLedgerContribution : std::uint8_t { NumericalFlux = 0, Source = 1 };

struct LevelTransition {
  int coarse = 0;
  int fine = 1;

  constexpr bool operator==(const LevelTransition&) const = default;
};

namespace detail {

inline auto clock_coordinate(const ClockStamp& stamp) {
  return std::tuple{stamp.level, stamp.macro_step, stamp.phase.numerator, stamp.phase.denominator};
}

template <int Dim>
bool index_less(const Index<Dim>& left, const Index<Dim>& right) {
  for (int axis = 0; axis < Dim; ++axis) {
    if (left[axis] != right[axis])
      return left[axis] < right[axis];
  }
  return false;
}

template <int Dim>
bool index_equal(const Index<Dim>& left, const Index<Dim>& right) {
  return !index_less(left, right) && !index_less(right, left);
}

inline int checked_axis(int axis, int dimension) {
  if (axis < 0 || axis >= dimension)
    throw std::invalid_argument("ND face-flux ledger axis is outside its compile-time dimension");
  return axis;
}

}  // namespace detail

/// Complete identity of one stage-local face-flux fragment.  `face` is expressed in the index
/// space selected by role, while `coarse_face` is the common coarse-grid aggregation identity.
/// Exact clock coordinates, stage and attempt prevent contributions from retries or graph stages
/// from aliasing even when their floating-point times happen to compare equal.
template <int Dim>
struct FaceFluxFragmentKey {
  static_assert(Dim >= 1 && Dim <= 3, "ND face-flux keys support dimensions 1..3");

  std::string owner;
  std::string state;
  LevelTransition levels{};
  FaceLedgerCentering centering = FaceLedgerCentering::Face;
  int axis = 0;
  Index<Dim> face{};
  Index<Dim> coarse_face{};
  ClockStamp clock{};
  std::string stage;
  std::uint64_t attempt = 0;
  FaceLedgerRole role = FaceLedgerRole::Coarse;
  FaceLedgerContribution contribution = FaceLedgerContribution::NumericalFlux;

  friend bool operator<(const FaceFluxFragmentKey& left, const FaceFluxFragmentKey& right) {
    const auto left_prefix = std::tie(left.owner, left.state, left.levels.coarse, left.levels.fine,
                                      left.centering, left.axis);
    const auto right_prefix = std::tie(right.owner, right.state, right.levels.coarse,
                                       right.levels.fine, right.centering, right.axis);
    if (left_prefix != right_prefix)
      return left_prefix < right_prefix;
    if (!detail::index_equal(left.coarse_face, right.coarse_face))
      return detail::index_less(left.coarse_face, right.coarse_face);
    if (!detail::index_equal(left.face, right.face))
      return detail::index_less(left.face, right.face);
    const auto left_clock = detail::clock_coordinate(left.clock);
    const auto right_clock = detail::clock_coordinate(right.clock);
    if (left_clock != right_clock)
      return left_clock < right_clock;
    return std::tie(left.stage, left.attempt, left.role, left.contribution) <
           std::tie(right.stage, right.attempt, right.role, right.contribution);
  }
};

/// Metric and temporal measure of one physical flux density sample. The exact substep interval
/// authenticates temporal coverage; `substep_duration` is its physical duration. Geometry is
/// multiplied here exactly once because metric reflux compares integrated transport across coarse
/// and fine faces.
struct FaceFluxFragmentMeasure {
  Rational stage_weight{1, 1};
  Rational substep_begin{0, 1};
  Rational substep_end{0, 1};
  double substep_duration = 0.0;
  double face_measure = 0.0;
};

inline double weighted_face_flux_scale(const FaceFluxFragmentMeasure& measure) {
  return measure.stage_weight.value() * measure.substep_duration * measure.face_measure;
}

template <int Dim>
void validate_face_flux_fragment(const FaceFluxFragmentKey<Dim>& key,
                                 const FaceFluxFragmentMeasure& measure) {
  if (key.owner.empty() || key.state.empty() || key.stage.empty())
    throw std::invalid_argument("ND face-flux identity requires owner, state, and stage");
  if (key.levels.coarse < 0 || key.levels.coarse == std::numeric_limits<int>::max() ||
      key.levels.fine != key.levels.coarse + 1)
    throw std::invalid_argument("ND face-flux identity requires one adjacent level transition");
  detail::checked_axis(key.axis, Dim);
  if (key.centering != FaceLedgerCentering::Face)
    throw std::invalid_argument("ND face-flux ledger accepts only face-centered contributions");
  if (key.contribution != FaceLedgerContribution::NumericalFlux)
    throw std::invalid_argument("ND face-flux ledger explicitly excludes source contributions");

  int clock_level = -1;
  switch (key.role) {
    case FaceLedgerRole::Coarse:
      clock_level = key.levels.coarse;
      break;
    case FaceLedgerRole::Fine:
      clock_level = key.levels.fine;
      break;
    default:
      throw std::invalid_argument("ND face-flux identity has an invalid coarse/fine role");
  }
  if (key.clock.level != clock_level || key.clock.macro_step < 0 ||
      !std::isfinite(key.clock.physical_time))
    throw std::invalid_argument("ND face-flux clock is not qualified by its role and level");
  if (key.clock.phase.denominator <= 0)
    throw std::invalid_argument("ND face-flux clock phase is not a canonical exact rational");
  if (Rational{key.clock.phase.numerator, key.clock.phase.denominator} != key.clock.phase)
    throw std::invalid_argument("ND face-flux clock phase must retain canonical exact form");

  const double stage_weight = measure.stage_weight.value();
  if (measure.stage_weight.denominator <= 0 || measure.substep_begin.denominator <= 0 ||
      measure.substep_end.denominator <= 0 || !std::isfinite(stage_weight) ||
      !(measure.substep_begin < measure.substep_end) || key.clock.phase < measure.substep_begin ||
      measure.substep_end < key.clock.phase || !(measure.substep_duration > 0.0) ||
      !std::isfinite(measure.substep_duration) || !(measure.face_measure > 0.0) ||
      !std::isfinite(measure.face_measure))
    throw std::invalid_argument(
        "ND face-flux measure requires finite stage, time, and positive metric weights");
  if (Rational{measure.stage_weight.numerator, measure.stage_weight.denominator} !=
      measure.stage_weight)
    throw std::invalid_argument("ND face-flux stage weight must retain canonical exact form");
  if (Rational{measure.substep_begin.numerator, measure.substep_begin.denominator} !=
          measure.substep_begin ||
      Rational{measure.substep_end.numerator, measure.substep_end.denominator} !=
          measure.substep_end)
    throw std::invalid_argument("ND face-flux substep interval must retain canonical exact form");
  if (!std::isfinite(weighted_face_flux_scale(measure)))
    throw std::invalid_argument("ND face-flux weighted metric-time scale is not finite");
}

template <int Dim, class Payload>
struct FaceFluxFragment {
  FaceFluxFragmentKey<Dim> key;
  FaceFluxFragmentMeasure measure;
  Payload payload;
};

struct FaceFluxLedgerBudget {
  std::size_t max_pending_entries = 0;
  std::size_t max_published_entries = 0;
  std::size_t max_transaction_depth = 0;
};

/// One host-side ledger per normal axis.  Pending fragments remain transaction-local and are not
/// visible through published_entries().  The outer commit first builds a complete candidate copy,
/// then swaps it into place, preserving the accepted ledger if allocation or payload copy fails.
/// All retained work is explicitly bounded, and accepted attempts can be discarded after reflux.
template <int Dim, class Payload>
class TransactionalFaceFluxLedger {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "ND face-flux ledgers support dimensions 1..3");
  static_assert(std::is_copy_constructible_v<Payload>,
                "transactional ND face-flux payloads must support atomic commit copies");

  using Entry = FaceFluxFragment<Dim, Payload>;

  explicit TransactionalFaceFluxLedger(FaceFluxLedgerBudget budget) : budget_(budget) {
    if (budget_.max_pending_entries == 0 || budget_.max_published_entries == 0 ||
        budget_.max_transaction_depth == 0)
      throw std::invalid_argument("ND face-flux ledger budgets must be strictly positive");
  }

  void begin(std::uint64_t attempt) {
    const bool outer = !active_attempt_.has_value();
    if (!outer) {
      if (*active_attempt_ != attempt)
        throw std::invalid_argument(
            "nested ND face-flux transaction must retain the outer attempt identity");
    } else {
      if (last_closed_attempt_.has_value() && attempt <= *last_closed_attempt_)
        throw std::invalid_argument("ND face-flux attempt identities must increase monotonically");
    }

    if (savepoints_.size() >= budget_.max_transaction_depth)
      throw std::length_error("ND face-flux transaction depth exceeds its prepared budget");
    Savepoint savepoint{};
    for (int axis = 0; axis < Dim; ++axis)
      savepoint.pending_sizes[static_cast<std::size_t>(axis)] =
          pending_[static_cast<std::size_t>(axis)].size();
    savepoints_.push_back(savepoint);
    if (outer)
      active_attempt_ = attempt;
  }

  void commit() {
    require_transaction_("commit");
    if (savepoints_.size() > 1) {
      savepoints_.pop_back();
      return;
    }

    const std::size_t published_count = published_size();
    const std::size_t pending_count = pending_size();
    if (published_count > budget_.max_published_entries ||
        pending_count > budget_.max_published_entries - published_count)
      throw std::length_error("ND face-flux publication exceeds its prepared budget");

    auto candidate = published_;
    for (int axis = 0; axis < Dim; ++axis) {
      auto& destination = candidate[static_cast<std::size_t>(axis)];
      const auto& source = pending_[static_cast<std::size_t>(axis)];
      destination.reserve(destination.size() + source.size());
      for (const Entry& entry : source)
        destination.push_back(entry);
    }
    published_.swap(candidate);
    for (auto& entries : pending_)
      entries.clear();
    close_outer_transaction_();
  }

  void rollback() {
    require_transaction_("rollback");
    const Savepoint savepoint = savepoints_.back();
    for (int axis = 0; axis < Dim; ++axis)
      pending_[static_cast<std::size_t>(axis)].resize(
          savepoint.pending_sizes[static_cast<std::size_t>(axis)]);
    savepoints_.pop_back();
    if (savepoints_.empty()) {
      last_closed_attempt_ = active_attempt_;
      active_attempt_.reset();
    }
  }

  void clear() {
    if (in_transaction())
      throw std::runtime_error("cannot clear an active ND face-flux ledger transaction");
    for (auto& entries : pending_)
      entries.clear();
    for (auto& entries : published_)
      entries.clear();
    last_closed_attempt_.reset();
  }

  void accumulate(FaceFluxFragmentKey<Dim> key, FaceFluxFragmentMeasure measure, Payload payload) {
    require_transaction_("accumulation");
    if (key.attempt != *active_attempt_)
      throw std::invalid_argument("ND face-flux fragment uses a stale attempt identity");
    validate_face_flux_fragment(key, measure);
    if (pending_size() >= budget_.max_pending_entries)
      throw std::length_error("ND face-flux pending entries exceed their prepared budget");
    const std::size_t axis = static_cast<std::size_t>(key.axis);
    if (contains_identity_(pending_[axis], key) || contains_identity_(published_[axis], key))
      throw std::runtime_error(
          "ND face-flux transaction contains a duplicate clock-stage face identity");
    pending_[axis].push_back({std::move(key), measure, std::move(payload)});
  }

  bool in_transaction() const noexcept { return !savepoints_.empty(); }
  std::size_t transaction_depth() const noexcept { return savepoints_.size(); }
  std::optional<std::uint64_t> active_attempt() const noexcept { return active_attempt_; }

  std::size_t pending_size() const noexcept { return total_size_(pending_); }
  std::size_t published_size() const noexcept { return total_size_(published_); }

  const std::vector<Entry>& pending_entries(int axis) const {
    return pending_[static_cast<std::size_t>(detail::checked_axis(axis, Dim))];
  }

  const std::vector<Entry>& published_entries(int axis) const {
    return published_[static_cast<std::size_t>(detail::checked_axis(axis, Dim))];
  }

  std::size_t discard_published_attempt(std::uint64_t attempt) {
    if (in_transaction())
      throw std::runtime_error(
          "cannot discard published ND face fluxes during an active transaction");
    std::array<std::vector<Entry>, Dim> candidate;
    std::size_t removed = 0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto& source = published_[static_cast<std::size_t>(axis)];
      auto& destination = candidate[static_cast<std::size_t>(axis)];
      destination.reserve(source.size());
      for (const Entry& entry : source) {
        if (entry.key.attempt == attempt)
          ++removed;
        else
          destination.push_back(entry);
      }
    }
    published_.swap(candidate);
    return removed;
  }

 private:
  struct Savepoint {
    std::array<std::size_t, Dim> pending_sizes{};
  };

  static bool same_identity_(const FaceFluxFragmentKey<Dim>& left,
                             const FaceFluxFragmentKey<Dim>& right) {
    return !(left < right) && !(right < left);
  }

  static bool contains_identity_(const std::vector<Entry>& entries,
                                 const FaceFluxFragmentKey<Dim>& key) {
    for (const Entry& entry : entries)
      if (same_identity_(entry.key, key))
        return true;
    return false;
  }

  static std::size_t total_size_(const std::array<std::vector<Entry>, Dim>& entries) noexcept {
    std::size_t result = 0;
    for (const auto& axis : entries)
      result += axis.size();
    return result;
  }

  void require_transaction_(const char* operation) const {
    if (!in_transaction())
      throw std::runtime_error(std::string("ND face-flux ledger ") + operation +
                               " requires an active transaction");
  }

  void close_outer_transaction_() {
    savepoints_.pop_back();
    last_closed_attempt_ = active_attempt_;
    active_attempt_.reset();
  }

  std::array<std::vector<Entry>, Dim> pending_{};
  std::array<std::vector<Entry>, Dim> published_{};
  std::vector<Savepoint> savepoints_;
  std::optional<std::uint64_t> active_attempt_;
  std::optional<std::uint64_t> last_closed_attempt_;
  FaceFluxLedgerBudget budget_;
};

}  // namespace pops::amr::reflux
