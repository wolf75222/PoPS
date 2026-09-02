/// @file
/// @brief Transaction-local, axis-qualified AMR face-flux ledger for dimensions 1..3.

#pragma once

#include <pops/mesh/index/index.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
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

/// ``std::string::capacity`` includes its inline SSO buffer on common implementations.  The
/// enclosing object is already counted by callers, so only report storage whose data pointer
/// lies outside the string object itself.
inline std::uint64_t external_string_storage_bytes(const std::string& value) noexcept {
  const auto object_begin = reinterpret_cast<std::uintptr_t>(&value);
  const auto object_end = object_begin + sizeof(value);
  const auto data = reinterpret_cast<std::uintptr_t>(value.data());
  if (data >= object_begin && data < object_end)
    return 0;
  return static_cast<std::uint64_t>(value.capacity()) + 1U;
}

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

  /// Cold description of one topology-bound Program contribution slot.  Its identity strings,
  /// face coordinates and payload extent are allocated before an attempt starts; the hot path
  /// supplies only the time-varying clock, attempt, metric scalars and component span.
  struct PreparedSlot {
    FaceFluxFragmentKey<Dim> key;
    std::size_t payload_components = 0;
  };

  explicit TransactionalFaceFluxLedger(FaceFluxLedgerBudget budget) : budget_(budget) {
    if (budget_.max_pending_entries == 0 || budget_.max_published_entries == 0 ||
        budget_.max_transaction_depth == 0)
      throw std::invalid_argument("ND face-flux ledger budgets must be strictly positive");
    savepoints_.reserve(budget_.max_transaction_depth);
    for (int axis = 0; axis < Dim; ++axis) {
      pending_[static_cast<std::size_t>(axis)].reserve(budget_.max_pending_entries);
      published_[static_cast<std::size_t>(axis)].reserve(budget_.max_published_entries);
      publication_staging_[static_cast<std::size_t>(axis)].reserve(budget_.max_published_entries);
    }
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
          resident_slots_bound_ ? resident_pending_sizes_[static_cast<std::size_t>(axis)]
                                : pending_[static_cast<std::size_t>(axis)].size();
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

    if (resident_slots_bound_) {
      if constexpr (!std::is_copy_assignable_v<Entry>) {
        throw std::logic_error(
            "ND face-flux resident slots require an assignable preallocated payload");
      } else {
        for (int axis = 0; axis < Dim; ++axis) {
          const std::size_t dimension = static_cast<std::size_t>(axis);
          const std::size_t accepted = resident_published_sizes_[dimension];
          const std::size_t source = resident_pending_sizes_[dimension];
          const std::size_t required = accepted + source;
          if (required > resident_publication_staging_[dimension].size())
            throw std::logic_error("ND face-flux resident publication exceeded its frozen slots");
          for (std::size_t index = 0; index < accepted; ++index)
            require_preallocated_entry_contract_(resident_publication_staging_[dimension][index],
                                                 resident_published_[dimension][index]);
          for (std::size_t index = 0; index < source; ++index)
            require_preallocated_entry_contract_(
                resident_publication_staging_[dimension][accepted + index],
                resident_pending_[dimension][index]);
          for (std::size_t index = 0; index < accepted; ++index)
            resident_publication_staging_[dimension][index] = resident_published_[dimension][index];
          for (std::size_t index = 0; index < source; ++index)
            resident_publication_staging_[dimension][accepted + index] =
                resident_pending_[dimension][index];
          resident_published_sizes_[dimension] = required;
          resident_pending_sizes_[dimension] = 0;
        }
        resident_published_.swap(resident_publication_staging_);
        close_outer_transaction_();
        return;
      }
    }

    if constexpr (!std::is_copy_assignable_v<Entry>) {
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
      return;
    } else {
      for (int axis = 0; axis < Dim; ++axis) {
        auto& destination = publication_staging_[static_cast<std::size_t>(axis)];
        const auto& accepted = published_[static_cast<std::size_t>(axis)];
        const auto& source = pending_[static_cast<std::size_t>(axis)];
        const std::size_t required = accepted.size() + source.size();
        if (required > destination.capacity())
          throw std::logic_error("ND face-flux publication staging lost its prepared capacity");
        std::size_t index = 0;
        const auto overwrite = [&](const Entry& entry) {
          if (index < destination.size())
            destination[index] = entry;
          else
            destination.push_back(entry);
          ++index;
        };
        for (const Entry& entry : accepted)
          overwrite(entry);
        for (const Entry& entry : source)
          overwrite(entry);
        destination.resize(required);
      }
      published_.swap(publication_staging_);
      for (auto& entries : pending_)
        entries.clear();
      close_outer_transaction_();
    }
  }

  void rollback() {
    require_transaction_("rollback");
    const Savepoint savepoint = savepoints_.back();
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t dimension = static_cast<std::size_t>(axis);
      if (resident_slots_bound_)
        resident_pending_sizes_[dimension] = savepoint.pending_sizes[dimension];
      else
        pending_[dimension].resize(savepoint.pending_sizes[dimension]);
    }
    savepoints_.pop_back();
    if (savepoints_.empty()) {
      last_closed_attempt_ = active_attempt_;
      active_attempt_.reset();
    }
  }

  void clear() {
    if (in_transaction())
      throw std::runtime_error("cannot clear an active ND face-flux ledger transaction");
    if (resident_slots_bound_) {
      resident_pending_sizes_.fill(0);
      resident_published_sizes_.fill(0);
    } else {
      for (auto& entries : pending_)
        entries.clear();
      for (auto& entries : published_)
        entries.clear();
    }
    last_closed_attempt_.reset();
  }

  /// Freeze all Program-visible contributions before ``begin``.  This intentionally has no
  /// fallback allocation path: a late shape, owner, stage, or payload-size change is rejected.
  void prepare_resident_slots(std::span<const PreparedSlot> slots) {
    if (in_transaction())
      throw std::logic_error("ND face-flux resident slots cannot bind during a transaction");
    if (slots.size() > budget_.max_pending_entries || slots.size() > budget_.max_published_entries)
      throw std::length_error("ND face-flux resident slots exceed the prepared ledger budget");

    std::array<std::size_t, Dim> counts{};
    for (const PreparedSlot& slot : slots) {
      validate_prepared_slot_(slot);
      ++counts[static_cast<std::size_t>(slot.key.axis)];
    }
    std::array<std::vector<Entry>, Dim> pending;
    std::array<std::vector<Entry>, Dim> published;
    std::array<std::vector<Entry>, Dim> staging;
    std::vector<PreparedSlotLocation> locations;
    locations.resize(slots.size());
    std::array<std::size_t, Dim> cursors{};
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t dimension = static_cast<std::size_t>(axis);
      pending[dimension].resize(counts[dimension]);
      published[dimension].resize(counts[dimension]);
      staging[dimension].resize(counts[dimension]);
    }
    for (std::size_t slot_index = 0; slot_index < slots.size(); ++slot_index) {
      const PreparedSlot& description = slots[slot_index];
      const std::size_t dimension = static_cast<std::size_t>(description.key.axis);
      const std::size_t local = cursors[dimension]++;
      locations[slot_index] = {static_cast<std::uint32_t>(dimension), local};
      Entry entry{description.key, {}, Payload{}};
      entry.payload.resize(description.payload_components);
      pending[dimension][local] = entry;
      published[dimension][local] = entry;
      staging[dimension][local] = std::move(entry);
    }

    using std::swap;
    swap(resident_pending_, pending);
    swap(resident_published_, published);
    swap(resident_publication_staging_, staging);
    swap(resident_slot_locations_, locations);
    resident_pending_sizes_.fill(0);
    resident_published_sizes_.fill(0);
    resident_slots_bound_ = true;
  }

  /// Exact logical storage retained by this ledger, including the pre-slot retry images.  Vector
  /// allocation is charged by capacity; dynamic string and payload arenas are charged only for
  /// constructed entries.  This retains the three retry images and the compact slot-location
  /// index without treating an inline SSO buffer as an external allocation.
  [[nodiscard]] std::uint64_t retained_storage_bytes() const {
    const auto checked_add = [](std::uint64_t left, std::uint64_t right) {
      if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error("ND face-flux resident footprint overflows uint64");
      return left + right;
    };
    const auto entry_dynamic_bytes = [&](const Entry& entry) -> std::uint64_t {
      std::uint64_t bytes = 0;
      const auto add_string = [&](const std::string& value) {
        bytes = checked_add(bytes, detail::external_string_storage_bytes(value));
      };
      add_string(entry.key.owner);
      add_string(entry.key.state);
      add_string(entry.key.stage);
      if constexpr (requires {
                      entry.payload.capacity();
                      typename Payload::value_type;
                    }) {
        const auto capacity = static_cast<std::uint64_t>(entry.payload.capacity());
        if (capacity >
            std::numeric_limits<std::uint64_t>::max() / sizeof(typename Payload::value_type))
          throw std::overflow_error("ND face-flux resident payload footprint overflows uint64");
        bytes = checked_add(bytes, capacity * sizeof(typename Payload::value_type));
      }
      return bytes;
    };
    const auto vector_bytes = [&](std::size_t capacity, std::size_t element_size) {
      if (element_size != 0 && capacity > std::numeric_limits<std::uint64_t>::max() / element_size)
        throw std::overflow_error("ND face-flux resident vector footprint overflows uint64");
      return static_cast<std::uint64_t>(capacity) * element_size;
    };
    std::uint64_t result =
        vector_bytes(resident_slot_locations_.capacity(), sizeof(PreparedSlotLocation));
    result = checked_add(result, vector_bytes(savepoints_.capacity(), sizeof(Savepoint)));
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t dimension = static_cast<std::size_t>(axis);
      const auto append_image = [&](const std::vector<Entry>& entries) {
        result = checked_add(result, vector_bytes(entries.capacity(), sizeof(Entry)));
        for (const Entry& entry : entries)
          result = checked_add(result, entry_dynamic_bytes(entry));
      };
      append_image(resident_pending_[dimension]);
      append_image(resident_published_[dimension]);
      append_image(resident_publication_staging_[dimension]);
      // These legacy images remain allocated even when the prepared-slot path is selected.
      // Account for their retained capacities too; otherwise an exact Program ceiling would
      // omit a live arena that can survive across resident attempts.
      append_image(pending_[dimension]);
      append_image(published_[dimension]);
      append_image(publication_staging_[dimension]);
    }
    return result;
  }

  /// Exact retained storage of a cold-bound resident ledger.  The bound check preserves the
  /// generated-slot authority for callers that require it; unbound accepted retry images use
  /// ``retained_storage_bytes()`` during Program capacity accounting.
  [[nodiscard]] std::uint64_t resident_storage_bytes() const {
    if (!resident_slots_bound_)
      throw std::logic_error("ND face-flux resident footprint has no prepared slots");
    return retained_storage_bytes();
  }

  /// Verify that a bind-sealed ledger image can receive @p source without growing an entry,
  /// payload, identity, or transaction carrier.  This is the preflight half of accepted-context
  /// rollback: callers perform it for every ledger before changing a single mutable image.
  void require_preallocated_copy_from(const TransactionalFaceFluxLedger& source) const {
    if (this == &source)
      return;
    if (in_transaction() || source.in_transaction() || !resident_slots_bound_ ||
        !source.resident_slots_bound_ ||
        budget_.max_pending_entries != source.budget_.max_pending_entries ||
        budget_.max_published_entries != source.budget_.max_published_entries ||
        budget_.max_transaction_depth != source.budget_.max_transaction_depth ||
        resident_slot_locations_.size() != source.resident_slot_locations_.size())
      throw std::logic_error("ND face-flux ledger copy differs from its bind-sealed authority");
    for (std::size_t slot = 0; slot < resident_slot_locations_.size(); ++slot)
      if (resident_slot_locations_[slot].axis != source.resident_slot_locations_[slot].axis ||
          resident_slot_locations_[slot].index != source.resident_slot_locations_[slot].index)
        throw std::logic_error("ND face-flux ledger copy changed its resident slot ordering");
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t dimension = static_cast<std::size_t>(axis);
      const auto require_image = [&](const std::vector<Entry>& destination,
                                     const std::vector<Entry>& input) {
        if (destination.size() != input.size())
          throw std::logic_error("ND face-flux ledger copy changed its resident slot shape");
        for (std::size_t index = 0; index < input.size(); ++index)
          require_preallocated_entry_contract_(destination[index], input[index]);
      };
      require_image(resident_pending_[dimension], source.resident_pending_[dimension]);
      require_image(resident_published_[dimension], source.resident_published_[dimension]);
      require_image(resident_publication_staging_[dimension],
                    source.resident_publication_staging_[dimension]);
    }
  }

  /// Copy only mutable values into an already validated resident ledger image.  The complete
  /// slot topology remains immutable, so assignment stays within strings/payloads reserved at
  /// cold bind and cannot allocate during rollback.
  void copy_from_preallocated(const TransactionalFaceFluxLedger& source) {
    require_preallocated_copy_from(source);
    if (this == &source)
      return;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::size_t dimension = static_cast<std::size_t>(axis);
      resident_pending_[dimension] = source.resident_pending_[dimension];
      resident_published_[dimension] = source.resident_published_[dimension];
      resident_publication_staging_[dimension] = source.resident_publication_staging_[dimension];
      resident_pending_sizes_[dimension] = source.resident_pending_sizes_[dimension];
      resident_published_sizes_[dimension] = source.resident_published_sizes_[dimension];
    }
    last_closed_attempt_ = source.last_closed_attempt_;
    active_attempt_.reset();
    savepoints_.clear();
  }

  /// Commit a dense, cold-bound slot without reconstructing its key or payload container.
  /// ``slot`` must be emitted in the frozen per-axis order; that ordering is part of the exact
  /// Program/topology authority and makes duplicate and omission drift fail before publication.
  template <class Element>
  void accumulate_prepared(std::uint32_t slot, ClockStamp clock, std::uint64_t attempt,
                           FaceFluxFragmentMeasure measure, std::span<const Element> payload) {
    require_transaction_("prepared accumulation");
    if (!resident_slots_bound_ || slot >= resident_slot_locations_.size())
      throw std::invalid_argument("ND face-flux prepared accumulation has no resident slot");
    if (attempt != *active_attempt_)
      throw std::invalid_argument("ND face-flux prepared fragment uses a stale attempt identity");
    const PreparedSlotLocation location = resident_slot_locations_[slot];
    const std::size_t axis = location.axis;
    if (location.index != resident_pending_sizes_[axis])
      throw std::invalid_argument("ND face-flux prepared slot order differs from its frozen plan");
    Entry& entry = resident_pending_[axis][location.index];
    if (entry.payload.size() != payload.size())
      throw std::invalid_argument(
          "ND face-flux prepared payload shape differs from its frozen slot");
    entry.key.clock = clock;
    entry.key.attempt = attempt;
    entry.measure = measure;
    validate_face_flux_fragment(entry.key, entry.measure);
    std::copy(payload.begin(), payload.end(), entry.payload.begin());
    ++resident_pending_sizes_[axis];
  }

  void accumulate(FaceFluxFragmentKey<Dim> key, FaceFluxFragmentMeasure measure, Payload payload) {
    if (resident_slots_bound_)
      throw std::logic_error("ND face-flux resident ledger requires compact prepared slots");
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
  bool resident_slots_bound() const noexcept { return resident_slots_bound_; }
  const FaceFluxLedgerBudget& budget() const noexcept { return budget_; }
  std::size_t transaction_depth() const noexcept { return savepoints_.size(); }
  std::optional<std::uint64_t> active_attempt() const noexcept { return active_attempt_; }

  std::size_t pending_size() const noexcept {
    if (!resident_slots_bound_)
      return total_size_(pending_);
    return total_size_(resident_pending_sizes_);
  }
  std::size_t published_size() const noexcept {
    if (!resident_slots_bound_)
      return total_size_(published_);
    return total_size_(resident_published_sizes_);
  }

  std::span<const Entry> pending_entries(int axis) const {
    const std::size_t dimension = static_cast<std::size_t>(detail::checked_axis(axis, Dim));
    if (resident_slots_bound_)
      return {resident_pending_[dimension].data(), resident_pending_sizes_[dimension]};
    return pending_[dimension];
  }

  std::span<const Entry> published_entries(int axis) const {
    const std::size_t dimension = static_cast<std::size_t>(detail::checked_axis(axis, Dim));
    if (resident_slots_bound_)
      return {resident_published_[dimension].data(), resident_published_sizes_[dimension]};
    return published_[dimension];
  }

  /// Cold-only topology-bound slot image.  Unlike ``published_entries()``, this retains the
  /// complete resident shape even before the first accepted attempt.  Snapshot builders use it
  /// to prime their detached string/payload carriers; it must never be consulted from a hot
  /// attempt.
  std::span<const Entry> resident_slot_templates(int axis) const {
    const std::size_t dimension = static_cast<std::size_t>(detail::checked_axis(axis, Dim));
    if (!resident_slots_bound_)
      throw std::logic_error("ND face-flux resident templates were not bound");
    return resident_published_[dimension];
  }

  std::size_t discard_published_attempt(std::uint64_t attempt) {
    if (in_transaction())
      throw std::runtime_error(
          "cannot discard published ND face fluxes during an active transaction");
    if (resident_slots_bound_) {
      if constexpr (!std::is_copy_assignable_v<Entry>) {
        throw std::logic_error(
            "ND face-flux resident slots require an assignable preallocated payload");
      } else {
        std::size_t removed = 0;
        for (int axis = 0; axis < Dim; ++axis) {
          const std::size_t dimension = static_cast<std::size_t>(axis);
          std::size_t write = 0;
          for (std::size_t read = 0; read < resident_published_sizes_[dimension]; ++read) {
            Entry& entry = resident_published_[dimension][read];
            if (entry.key.attempt == attempt) {
              ++removed;
              continue;
            }
            if (write != read)
              require_preallocated_entry_contract_(resident_published_[dimension][write], entry);
            if (write != read)
              resident_published_[dimension][write] = entry;
            ++write;
          }
          resident_published_sizes_[dimension] = write;
        }
        return removed;
      }
    }
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
  struct PreparedSlotLocation {
    std::uint32_t axis = 0;
    std::size_t index = 0;
  };
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

  static std::size_t total_size_(const std::array<std::size_t, Dim>& entries) noexcept {
    std::size_t result = 0;
    for (const std::size_t axis : entries)
      result += axis;
    return result;
  }

  static void validate_prepared_slot_(const PreparedSlot& slot) {
    if (slot.key.owner.empty() || slot.key.state.empty() || slot.key.stage.empty() ||
        slot.payload_components == 0)
      throw std::invalid_argument("ND face-flux resident slot has incomplete identity or payload");
    if (slot.key.levels.coarse < 0 || slot.key.levels.fine != slot.key.levels.coarse + 1 ||
        slot.key.centering != FaceLedgerCentering::Face ||
        slot.key.contribution != FaceLedgerContribution::NumericalFlux)
      throw std::invalid_argument("ND face-flux resident slot has an invalid static route");
    detail::checked_axis(slot.key.axis, Dim);
    if ((slot.key.role == FaceLedgerRole::Coarse &&
         slot.key.clock.level != slot.key.levels.coarse) ||
        (slot.key.role == FaceLedgerRole::Fine && slot.key.clock.level != slot.key.levels.fine))
      throw std::invalid_argument("ND face-flux resident slot clock level differs from its role");
  }

  static void require_preallocated_entry_contract_(const Entry& destination, const Entry& source) {
    if (destination.key.owner.capacity() < source.key.owner.size() ||
        destination.key.state.capacity() < source.key.state.size() ||
        destination.key.stage.capacity() < source.key.stage.size())
      throw std::logic_error(
          "ND face-flux resident entry copy would exceed its frozen identity or payload capacity");
    if constexpr (requires {
                    destination.payload.capacity();
                    source.payload.size();
                  })
      if (destination.payload.capacity() < source.payload.size())
        throw std::logic_error(
            "ND face-flux resident entry copy would exceed its frozen payload capacity");
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
  std::array<std::vector<Entry>, Dim> publication_staging_{};
  std::array<std::vector<Entry>, Dim> resident_pending_{};
  std::array<std::vector<Entry>, Dim> resident_published_{};
  std::array<std::vector<Entry>, Dim> resident_publication_staging_{};
  std::array<std::size_t, Dim> resident_pending_sizes_{};
  std::array<std::size_t, Dim> resident_published_sizes_{};
  std::vector<PreparedSlotLocation> resident_slot_locations_;
  bool resident_slots_bound_ = false;
  std::vector<Savepoint> savepoints_;
  std::optional<std::uint64_t> active_attempt_;
  std::optional<std::uint64_t> last_closed_attempt_;
  FaceFluxLedgerBudget budget_;
};

}  // namespace pops::amr::reflux
