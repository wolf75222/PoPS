/// @file
/// @brief Metric coarse/fine face matching and conservative reflux for dimensions 1..3.

#pragma once

#include <pops/amr/reflux/face_flux_ledger.hpp>
#include <pops/amr/refinement_ratio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::amr::reflux {

/// Affine relation between the coarse and fine face index spaces.  The same mapping applies to
/// normal face coordinates and to tangential cell coordinates; the normal fine face has no child
/// offset, while tangential coordinates span their full anisotropic ratio.
template <int Dim>
struct FaceRefinementMapping {
  Index<Dim> coarse_origin{};
  Index<Dim> fine_origin{};

  constexpr bool operator==(const FaceRefinementMapping&) const = default;
};

/// Identity and exact macro-step window of the coarse face whose accepted fragments are reconciled.
template <int Dim>
struct CoarseFaceRefluxKey {
  std::string owner;
  std::string state;
  LevelTransition levels{};
  FaceLedgerCentering centering = FaceLedgerCentering::Face;
  int axis = 0;
  Index<Dim> coarse_face{};
  std::uint64_t attempt = 0;
  std::int64_t macro_step = 0;
  Rational window_begin{0, 1};
  Rational window_end{1, 1};
};

struct MetricRefluxBudget {
  std::size_t max_fine_faces = 0;
  std::size_t max_published_entries = 0;
  std::size_t max_clock_stage_slices = 0;
};

template <class Payload>
struct MetricFaceReflux {
  Payload coarse_integrated{};
  Payload fine_integrated{};
  Payload mismatch{};  ///< fine_integrated - coarse_integrated in canonical positive-axis units
  double coarse_weighted_measure = 0.0;
  double fine_weighted_measure = 0.0;
  std::size_t fine_face_count = 0;
};

enum class CoarseCellFaceSide : std::uint8_t { Lower = 0, Upper = 1 };

namespace detail {

template <int Dim>
std::array<int, Dim> coordinate_array(const Index<Dim>& index) {
  std::array<int, Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[static_cast<std::size_t>(axis)] = index[axis];
  return result;
}

inline int checked_fine_face_coordinate(std::int64_t value) {
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::overflow_error("ND metric reflux face mapping exceeds the signed index range");
  return static_cast<int>(value);
}

inline std::int64_t checked_fine_face_add(std::int64_t left, std::int64_t right) {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right))
    throw std::overflow_error("ND metric reflux face mapping exceeds int64_t");
  return left + right;
}

template <int Dim>
void validate_reflux_key(const CoarseFaceRefluxKey<Dim>& key) {
  if (key.owner.empty() || key.state.empty())
    throw std::invalid_argument("ND metric reflux requires qualified owner and state identities");
  if (key.levels.coarse < 0 || key.levels.coarse == std::numeric_limits<int>::max() ||
      key.levels.fine != key.levels.coarse + 1)
    throw std::invalid_argument("ND metric reflux requires one adjacent level transition");
  checked_axis(key.axis, Dim);
  if (key.centering != FaceLedgerCentering::Face)
    throw std::invalid_argument("ND metric reflux accepts only face-centered flux identities");
  if (key.macro_step < 0 || key.window_begin.denominator <= 0 || key.window_end.denominator <= 0 ||
      !(key.window_begin < key.window_end) ||
      Rational{key.window_begin.numerator, key.window_begin.denominator} != key.window_begin ||
      Rational{key.window_end.numerator, key.window_end.denominator} != key.window_end)
    throw std::invalid_argument("ND metric reflux requires one canonical exact clock window");
}

inline void validate_reflux_budget(const MetricRefluxBudget& budget) {
  if (budget.max_fine_faces == 0 || budget.max_published_entries == 0 ||
      budget.max_clock_stage_slices == 0)
    throw std::invalid_argument("ND metric reflux budgets must be strictly positive");
}

template <int Dim>
bool matches_reflux_key(const FaceFluxFragmentKey<Dim>& fragment,
                        const CoarseFaceRefluxKey<Dim>& query) {
  return fragment.owner == query.owner && fragment.state == query.state &&
         fragment.levels == query.levels && fragment.centering == query.centering &&
         fragment.axis == query.axis && fragment.coarse_face == query.coarse_face &&
         fragment.attempt == query.attempt && fragment.clock.macro_step == query.macro_step &&
         !(fragment.clock.phase < query.window_begin) && !(query.window_end < fragment.clock.phase);
}

using StageSlice =
    std::tuple<std::tuple<int, std::int64_t, std::int64_t, std::int64_t>, std::string>;

struct TemporalSliceMeasure {
  Rational stage_weight{0, 1};
  Rational substep_begin{0, 1};
  Rational substep_end{0, 1};
  double substep_duration = 0.0;
};

inline StageSlice stage_slice(const ClockStamp& clock, const std::string& stage) {
  return {clock_coordinate(clock), stage};
}

inline void register_temporal_slice(std::map<StageSlice, TemporalSliceMeasure>& slices,
                                    const StageSlice& slice, const FaceFluxFragmentMeasure& measure,
                                    std::size_t total_slice_count,
                                    const MetricRefluxBudget& budget) {
  const TemporalSliceMeasure candidate{measure.stage_weight, measure.substep_begin,
                                       measure.substep_end, measure.substep_duration};
  const auto existing = slices.find(slice);
  if (existing != slices.end()) {
    if (existing->second.stage_weight != candidate.stage_weight ||
        existing->second.substep_begin != candidate.substep_begin ||
        existing->second.substep_end != candidate.substep_end ||
        existing->second.substep_duration != candidate.substep_duration)
      throw std::runtime_error(
          "ND metric reflux clock-stage faces disagree on their temporal measure");
    return;
  }
  if (total_slice_count >= budget.max_clock_stage_slices)
    throw std::length_error("ND metric reflux clock-stage slices exceed their prepared budget");
  slices.emplace(slice, candidate);
}

struct ExactSubstep {
  Rational begin{0, 1};
  Rational end{0, 1};

  friend bool operator<(const ExactSubstep& left, const ExactSubstep& right) {
    return left.begin == right.begin ? left.end < right.end : left.begin < right.begin;
  }
};

struct SubstepQuadrature {
  Rational stage_weight_sum{0, 1};
  double duration = 0.0;
};

inline bool roundoff_equal(double left, double right, std::size_t operations) {
  if (left == right)
    return true;
  if (!std::isfinite(left) || !std::isfinite(right))
    return false;
  const double scale = std::max(std::abs(left), std::abs(right));
  const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() * scale *
                           static_cast<double>(std::max<std::size_t>(operations, 1));
  return std::abs(left - right) <= tolerance;
}

struct AuthenticatedWindow {
  double duration = 0.0;
  double duration_per_phase = 0.0;
  std::size_t substep_count = 0;
};

inline AuthenticatedWindow authenticated_window(
    const std::map<StageSlice, TemporalSliceMeasure>& slices, Rational window_begin,
    Rational window_end) {
  std::map<ExactSubstep, SubstepQuadrature> substeps;
  for (const auto& [slice, measure] : slices) {
    (void)slice;
    const ExactSubstep interval{measure.substep_begin, measure.substep_end};
    auto [position, inserted] =
        substeps.emplace(interval, SubstepQuadrature{Rational{0, 1}, measure.substep_duration});
    if (!inserted && position->second.duration != measure.substep_duration)
      throw std::runtime_error(
          "ND metric reflux stages disagree on their physical substep duration");
    position->second.stage_weight_sum = position->second.stage_weight_sum + measure.stage_weight;
  }

  Rational cursor = window_begin;
  AuthenticatedWindow result;
  for (const auto& [interval, quadrature] : substeps) {
    if (interval.begin != cursor || !(interval.begin < interval.end))
      throw std::runtime_error(
          "ND metric reflux substeps do not form a contiguous exact clock partition");
    if (quadrature.stage_weight_sum != Rational{1, 1})
      throw std::runtime_error("ND metric reflux stage weights do not close one accepted substep");
    const double phase_span = (interval.end - interval.begin).value();
    if (!(phase_span > 0.0) || !std::isfinite(phase_span))
      throw std::overflow_error("ND metric reflux physical clock rate is not finite");
    const double duration_per_phase = quadrature.duration / phase_span;
    if (!std::isfinite(duration_per_phase))
      throw std::overflow_error("ND metric reflux physical clock rate is not finite");
    if (result.substep_count == 0)
      result.duration_per_phase = duration_per_phase;
    else if (!roundoff_equal(result.duration_per_phase, duration_per_phase,
                             result.substep_count + 1))
      throw std::runtime_error("ND metric reflux substeps disagree on their physical clock rate");
    cursor = interval.end;
    result.duration += quadrature.duration;
    ++result.substep_count;
    if (!std::isfinite(result.duration))
      throw std::overflow_error("ND metric reflux physical window duration is not finite");
  }
  if (cursor != window_end)
    throw std::runtime_error(
        "ND metric reflux substeps do not cover the complete exact clock window");
  return result;
}

template <int Dim>
void validate_temporal_coverage(const CoarseFaceRefluxKey<Dim>& key,
                                const std::map<StageSlice, TemporalSliceMeasure>& coarse_slices,
                                const std::map<StageSlice, TemporalSliceMeasure>& fine_slices) {
  const AuthenticatedWindow coarse =
      authenticated_window(coarse_slices, key.window_begin, key.window_end);
  const AuthenticatedWindow fine =
      authenticated_window(fine_slices, key.window_begin, key.window_end);
  const std::size_t operations = coarse_slices.size() + fine_slices.size();
  if (!roundoff_equal(coarse.duration, fine.duration, operations) ||
      !roundoff_equal(coarse.duration_per_phase, fine.duration_per_phase, operations))
    throw std::runtime_error(
        "ND metric reflux coarse and fine physical clocks do not cover the same window");
}

template <int Dim>
std::set<std::array<int, Dim>> expected_fine_face_set(const CoarseFaceRefluxKey<Dim>& key,
                                                      const RefinementRatio<Dim>& ratio,
                                                      const FaceRefinementMapping<Dim>& mapping,
                                                      const MetricRefluxBudget& budget) {
  validate_reflux_budget(budget);
  if (!ratio.refines_any_axis())
    throw std::invalid_argument(
        "ND metric reflux requires a non-identity inter-level refinement ratio");
  std::size_t fine_face_count = 1;
  for (int direction = 0; direction < Dim; ++direction) {
    if (direction == key.axis)
      continue;
    const std::size_t axis_faces = static_cast<std::size_t>(ratio[direction]);
    if (axis_faces > std::numeric_limits<std::size_t>::max() / fine_face_count)
      throw std::length_error("ND metric reflux tangential face product exceeds size_t");
    fine_face_count *= axis_faces;
  }
  if (fine_face_count > budget.max_fine_faces)
    throw std::length_error("ND metric reflux tangential face product exceeds its prepared budget");
  Index<Dim> base{};
  for (int direction = 0; direction < Dim; ++direction) {
    const std::int64_t relative =
        static_cast<std::int64_t>(key.coarse_face[direction]) - mapping.coarse_origin[direction];
    if (relative > std::numeric_limits<std::int64_t>::max() / ratio[direction] ||
        relative < std::numeric_limits<std::int64_t>::min() / ratio[direction])
      throw std::overflow_error("ND metric reflux face mapping exceeds int64_t");
    const std::int64_t scaled = relative * ratio[direction];
    const std::int64_t fine = checked_fine_face_add(mapping.fine_origin[direction], scaled);
    base[direction] = checked_fine_face_coordinate(fine);
  }

  std::set<std::array<int, Dim>> result;
  Index<Dim> child{};
  for (;;) {
    Index<Dim> fine = base;
    for (int direction = 0; direction < Dim; ++direction)
      if (direction != key.axis)
        fine[direction] = checked_fine_face_coordinate(static_cast<std::int64_t>(base[direction]) +
                                                       child[direction]);
    result.insert(coordinate_array(fine));

    int direction = 0;
    for (; direction < Dim; ++direction) {
      if (direction == key.axis)
        continue;
      ++child[direction];
      if (child[direction] < ratio[direction])
        break;
      child[direction] = 0;
    }
    if (direction == Dim)
      break;
  }
  return result;
}

template <std::size_t Dim>
void require_complete_slices(const std::map<StageSlice, std::set<std::array<int, Dim>>>& slices,
                             const std::set<std::array<int, Dim>>& expected, const char* role) {
  if (slices.empty())
    throw std::runtime_error(std::string("ND metric reflux has no published ") + role +
                             " face fragments");
  for (const auto& [slice, faces] : slices) {
    (void)slice;
    if (faces != expected)
      throw std::runtime_error(std::string("ND metric reflux has an incomplete ") + role +
                               " tangential face product in one clock-stage slice");
  }
}

/// Payload storage used by the prepared route.  The generic path is deliberately restricted to
/// scalar payloads: arbitrary owning payload types cannot promise an allocation-free reset.  The
/// vector specialization retains the full component capacity fixed at bind.
template <class Payload>
struct PreparedMetricPayload {
  static constexpr bool supported = std::is_arithmetic_v<Payload>;

  static void prime(Payload& destination, const Payload&) { destination = Payload{}; }
  static void reset(Payload& destination) noexcept { destination = Payload{}; }
  static bool same_shape(const Payload&, const Payload&) noexcept { return true; }
  static bool has_capacity_for(const Payload&, const Payload&) noexcept { return true; }
  static std::size_t resident_storage_bytes(const Payload&) { return sizeof(Payload); }
};

template <class Value, class Allocator>
struct PreparedMetricPayload<std::vector<Value, Allocator>> {
  using Payload = std::vector<Value, Allocator>;
  static constexpr bool supported = std::is_nothrow_swappable_v<Payload>;

  static void prime(Payload& destination, const Payload& prototype) {
    destination.assign(prototype.size(), Value{});
  }
  static void reset(Payload& destination) noexcept {
    std::fill(destination.begin(), destination.end(), Value{});
  }
  static bool same_shape(const Payload& left, const Payload& right) noexcept {
    return left.size() == right.size();
  }
  static bool has_capacity_for(const Payload& destination, const Payload& source) noexcept {
    return destination.capacity() >= source.size() && destination.size() == source.size();
  }
  static std::size_t resident_storage_bytes(const Payload& payload) {
    if (payload.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(Value))
      throw std::overflow_error("ND prepared metric reflux payload storage overflows size_t");
    return payload.capacity() * sizeof(Value);
  }
};

}  // namespace detail

/// Enumerate the exact product of tangential fine faces covering one coarse face.  In 1D the
/// tangential product is one; in 2D it is the ratio of the other axis; in 3D it is the product of
/// both other-axis ratios.  The normal-axis ratio changes only the normal coordinate mapping.
template <int Dim>
std::vector<Index<Dim>> fine_faces_for_coarse_face(const CoarseFaceRefluxKey<Dim>& key,
                                                   const RefinementRatio<Dim>& ratio,
                                                   const FaceRefinementMapping<Dim>& mapping,
                                                   const MetricRefluxBudget& budget) {
  detail::validate_reflux_key(key);
  const auto expected = detail::expected_fine_face_set(key, ratio, mapping, budget);
  std::vector<Index<Dim>> result;
  result.reserve(expected.size());
  for (const auto& coordinate : expected) {
    Index<Dim> face{};
    for (int axis = 0; axis < Dim; ++axis)
      face[axis] = coordinate[static_cast<std::size_t>(axis)];
    result.push_back(face);
  }
  return result;
}

/// Integrate every accepted coarse and fine stage fragment using its exact rational stage weight,
/// authored substep duration and physical face measure.  Fine faces must form the complete
/// tangential product for every clock-stage slice. Pending/rejected fragments are never observed.
template <int Dim, class Payload, class Axpy>
MetricFaceReflux<Payload> metric_reflux(const TransactionalFaceFluxLedger<Dim, Payload>& ledger,
                                        const CoarseFaceRefluxKey<Dim>& key,
                                        const RefinementRatio<Dim>& ratio,
                                        const FaceRefinementMapping<Dim>& mapping,
                                        const MetricRefluxBudget& budget, Axpy&& axpy) {
  detail::validate_reflux_key(key);
  detail::validate_reflux_budget(budget);
  if (ledger.published_size() > budget.max_published_entries)
    throw std::length_error("ND metric reflux published entries exceed their prepared budget");
  const auto expected_fine = detail::expected_fine_face_set(key, ratio, mapping, budget);
  const std::set<std::array<int, Dim>> expected_coarse{detail::coordinate_array(key.coarse_face)};
  std::map<detail::StageSlice, std::set<std::array<int, Dim>>> coarse_slices;
  std::map<detail::StageSlice, std::set<std::array<int, Dim>>> fine_slices;
  std::map<detail::StageSlice, detail::TemporalSliceMeasure> coarse_temporal;
  std::map<detail::StageSlice, detail::TemporalSliceMeasure> fine_temporal;
  MetricFaceReflux<Payload> result;

  for (const auto& entry : ledger.published_entries(key.axis)) {
    if (!detail::matches_reflux_key(entry.key, key))
      continue;
    const double scale = weighted_face_flux_scale(entry.measure);
    const auto slice = detail::stage_slice(entry.key.clock, entry.key.stage);
    switch (entry.key.role) {
      case FaceLedgerRole::Coarse:
        coarse_slices[slice].insert(detail::coordinate_array(entry.key.face));
        detail::register_temporal_slice(coarse_temporal, slice, entry.measure,
                                        coarse_temporal.size() + fine_temporal.size(), budget);
        axpy(result.coarse_integrated, scale, entry.payload);
        result.coarse_weighted_measure += scale;
        if (!std::isfinite(result.coarse_weighted_measure))
          throw std::overflow_error("ND metric reflux coarse weighted measure is not finite");
        break;
      case FaceLedgerRole::Fine:
        fine_slices[slice].insert(detail::coordinate_array(entry.key.face));
        detail::register_temporal_slice(fine_temporal, slice, entry.measure,
                                        coarse_temporal.size() + fine_temporal.size(), budget);
        axpy(result.fine_integrated, scale, entry.payload);
        result.fine_weighted_measure += scale;
        if (!std::isfinite(result.fine_weighted_measure))
          throw std::overflow_error("ND metric reflux fine weighted measure is not finite");
        break;
      default:
        throw std::runtime_error("ND metric reflux observed an invalid published face role");
    }
  }

  detail::require_complete_slices(coarse_slices, expected_coarse, "coarse");
  detail::require_complete_slices(fine_slices, expected_fine, "fine");
  detail::validate_temporal_coverage(key, coarse_temporal, fine_temporal);
  if (!detail::roundoff_equal(result.coarse_weighted_measure, result.fine_weighted_measure,
                              ledger.published_size()))
    throw std::runtime_error(
        "ND metric reflux coarse and fine metric-time measures do not cover the same face");
  axpy(result.mismatch, 1.0, result.fine_integrated);
  axpy(result.mismatch, -1.0, result.coarse_integrated);
  result.fine_face_count = expected_fine.size();
  return result;
}

/// Bind-primed metric reflux authority for one static coarse/interface-face route.
///
/// `prepare()` is deliberately cold: it seals only topology identity, refinement/mapping, budget,
/// expected tangential faces and payload shape.  A reusable workspace may therefore reconcile
/// subsequent attempts and macro-step windows.  `reconcile()` authenticates those dynamic clock
/// fields every time and publishes its result only after every coverage and temporal check has
/// succeeded.  The supplied `axpy` must itself be allocation-free for the sealed payload shape.
template <int Dim, class Payload>
class PreparedMetricRefluxWorkspace {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "prepared metric reflux supports dimensions 1..3");
  static_assert(detail::PreparedMetricPayload<Payload>::supported,
                "prepared metric reflux supports scalar or noexcept-swappable vector payloads");
  static_assert(std::is_nothrow_swappable_v<Payload>,
                "prepared metric reflux payload swaps must not throw");

  PreparedMetricRefluxWorkspace() = default;
  PreparedMetricRefluxWorkspace(const PreparedMetricRefluxWorkspace&) = default;
  PreparedMetricRefluxWorkspace& operator=(const PreparedMetricRefluxWorkspace&) = default;
  PreparedMetricRefluxWorkspace(PreparedMetricRefluxWorkspace&&) noexcept = default;
  PreparedMetricRefluxWorkspace& operator=(PreparedMetricRefluxWorkspace&&) noexcept = default;

  friend void swap(PreparedMetricRefluxWorkspace& left,
                   PreparedMetricRefluxWorkspace& right) noexcept {
    using std::swap;
    swap(left.owner_, right.owner_);
    swap(left.state_, right.state_);
    swap(left.levels_, right.levels_);
    swap(left.centering_, right.centering_);
    swap(left.axis_, right.axis_);
    swap(left.coarse_face_, right.coarse_face_);
    swap(left.ratio_, right.ratio_);
    swap(left.mapping_, right.mapping_);
    swap(left.budget_, right.budget_);
    swap(left.expected_fine_, right.expected_fine_);
    swap(left.slices_, right.slices_);
    swap(left.substeps_, right.substeps_);
    swap(left.face_seen_, right.face_seen_);
    swap(left.result_, right.result_);
    swap(left.candidate_, right.candidate_);
    swap(left.correction_, right.correction_);
    swap(left.correction_candidate_, right.correction_candidate_);
    swap(left.slice_count_, right.slice_count_);
    swap(left.prepared_, right.prepared_);
  }

  [[nodiscard]] bool prepared() const noexcept { return prepared_; }

  [[nodiscard]] std::size_t resident_storage_bytes() const {
    const auto checked_add = [](std::size_t& total, std::size_t value) {
      if (value > std::numeric_limits<std::size_t>::max() - total)
        throw std::overflow_error("ND prepared metric reflux storage overflows size_t");
      total += value;
    };
    const auto checked_vector_bytes = [](std::size_t capacity, std::size_t element_size) {
      if (element_size != 0 && capacity > std::numeric_limits<std::size_t>::max() / element_size)
        throw std::overflow_error("ND prepared metric reflux vector storage overflows size_t");
      return capacity * element_size;
    };
    const auto external_string_bytes = [](const std::string& value) {
      const auto object_begin = reinterpret_cast<std::uintptr_t>(&value);
      const auto object_end = object_begin + sizeof(value);
      const auto data = reinterpret_cast<std::uintptr_t>(value.data());
      if (data >= object_begin && data < object_end)
        return std::size_t{0};
      if (value.capacity() == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("ND prepared metric reflux string storage overflows size_t");
      return value.capacity() + 1U;
    };
    std::size_t total = 0;
    checked_add(total, external_string_bytes(owner_));
    checked_add(total, external_string_bytes(state_));
    checked_add(total,
                checked_vector_bytes(expected_fine_.capacity(), sizeof(std::array<int, Dim>)));
    checked_add(total, checked_vector_bytes(slices_.capacity(), sizeof(Slice)));
    checked_add(total, checked_vector_bytes(substeps_.capacity(), sizeof(Substep)));
    checked_add(total, checked_vector_bytes(face_seen_.capacity(), sizeof(std::uint8_t)));
    for (const auto* payload :
         {&result_.coarse_integrated, &result_.fine_integrated, &result_.mismatch,
          &candidate_.coarse_integrated, &candidate_.fine_integrated, &candidate_.mismatch,
          &correction_, &correction_candidate_})
      checked_add(total, detail::PreparedMetricPayload<Payload>::resident_storage_bytes(*payload));
    return total;
  }

  /// Cold bind.  The query's attempt, macro-step and window are used only to discover the sealed
  /// payload layout; they are intentionally not retained as static identity.
  void prepare(const TransactionalFaceFluxLedger<Dim, Payload>& ledger,
               const CoarseFaceRefluxKey<Dim>& key, const RefinementRatio<Dim>& ratio,
               const FaceRefinementMapping<Dim>& mapping, const MetricRefluxBudget& budget) {
    detail::validate_reflux_key(key);
    detail::validate_reflux_budget(budget);
    if (ledger.published_size() > budget.max_published_entries)
      throw std::length_error("ND prepared metric reflux published entries exceed its budget");

    const Payload* prototype = nullptr;
    for (const auto& entry : ledger.published_entries(key.axis)) {
      if (!detail::matches_reflux_key(entry.key, key))
        continue;
      if (prototype == nullptr)
        prototype = &entry.payload;
      else if (!detail::PreparedMetricPayload<Payload>::same_shape(*prototype, entry.payload))
        throw std::invalid_argument("ND prepared metric reflux payload shapes disagree at bind");
    }
    // Immediately after prepare_resident_slots() the accepted publication is intentionally
    // empty.  The complete slot image is the cold payload authority in that state; do not seed a
    // synthetic accepted fragment merely to prime a Program workspace.
    if (prototype == nullptr) {
      for (const auto& entry : ledger.resident_slot_templates(key.axis)) {
        if (!detail::matches_reflux_key(entry.key, key))
          continue;
        if (prototype == nullptr)
          prototype = &entry.payload;
        else if (!detail::PreparedMetricPayload<Payload>::same_shape(*prototype, entry.payload))
          throw std::invalid_argument(
              "ND prepared metric reflux resident payload shapes disagree at bind");
      }
    }
    if (prototype == nullptr)
      throw std::runtime_error("ND prepared metric reflux requires one published bind prototype");

    bind_from_payload_prototype_(key, ratio, mapping, budget, *prototype);
  }

 private:
  void bind_from_payload_prototype_(const CoarseFaceRefluxKey<Dim>& key,
                                    const RefinementRatio<Dim>& ratio,
                                    const FaceRefinementMapping<Dim>& mapping,
                                    const MetricRefluxBudget& budget, const Payload& prototype) {
    const auto expected = fine_faces_for_coarse_face(key, ratio, mapping, budget);

    owner_ = key.owner;
    state_ = key.state;
    levels_ = key.levels;
    centering_ = key.centering;
    axis_ = key.axis;
    coarse_face_ = key.coarse_face;
    ratio_ = ratio;
    mapping_ = mapping;
    budget_ = budget;
    expected_fine_.clear();
    expected_fine_.reserve(expected.size());
    for (const auto& face : expected)
      expected_fine_.push_back(detail::coordinate_array(face));

    slices_.assign(budget.max_clock_stage_slices, Slice{});
    substeps_.assign(budget.max_clock_stage_slices, Substep{});
    face_seen_.assign(budget.max_clock_stage_slices * expected_fine_.size(), 0);
    prime_result_(result_, prototype);
    prime_result_(candidate_, prototype);
    detail::PreparedMetricPayload<Payload>::prime(correction_, prototype);
    detail::PreparedMetricPayload<Payload>::prime(correction_candidate_, prototype);
    clear_transient_();
    prepared_ = true;
  }

 public:
  /// Cold helper for snapshot/image construction.  It intentionally permits allocation while
  /// creating the detached copy; `copy_from_preallocated()` is the hot fail-closed counterpart.
  void prime_copied_capacities_from_cold_source(const PreparedMetricRefluxWorkspace& source) {
    if (!source.prepared_)
      throw std::logic_error("ND prepared metric reflux source is not bound");
    PreparedMetricRefluxWorkspace copy{source};
    copy.require_preallocated_copy_from(source);
    swap(*this, copy);
  }

  void require_preallocated_copy_from(const PreparedMetricRefluxWorkspace& source) const {
    if (!prepared_ || !source.prepared_ || !same_static_identity_(source))
      throw std::logic_error("ND prepared metric reflux copy has incompatible static identity");
    if (owner_.capacity() < source.owner_.size() || state_.capacity() < source.state_.size() ||
        expected_fine_.capacity() < source.expected_fine_.size() ||
        slices_.capacity() < source.slices_.size() ||
        substeps_.capacity() < source.substeps_.size() ||
        face_seen_.capacity() < source.face_seen_.size() ||
        !result_has_capacity_for_(result_, source.result_) ||
        !result_has_capacity_for_(candidate_, source.candidate_) ||
        !detail::PreparedMetricPayload<Payload>::has_capacity_for(correction_,
                                                                  source.correction_) ||
        !detail::PreparedMetricPayload<Payload>::has_capacity_for(correction_candidate_,
                                                                  source.correction_candidate_))
      throw std::length_error("ND prepared metric reflux copy would allocate after bind");
  }

  void copy_from_preallocated(const PreparedMetricRefluxWorkspace& source) {
    require_preallocated_copy_from(source);
    owner_ = source.owner_;
    state_ = source.state_;
    levels_ = source.levels_;
    centering_ = source.centering_;
    axis_ = source.axis_;
    coarse_face_ = source.coarse_face_;
    ratio_ = source.ratio_;
    mapping_ = source.mapping_;
    budget_ = source.budget_;
    expected_fine_ = source.expected_fine_;
    slices_ = source.slices_;
    substeps_ = source.substeps_;
    face_seen_ = source.face_seen_;
    result_ = source.result_;
    candidate_ = source.candidate_;
    correction_ = source.correction_;
    correction_candidate_ = source.correction_candidate_;
    slice_count_ = source.slice_count_;
    prepared_ = source.prepared_;
  }

  template <class Axpy>
  const MetricFaceReflux<Payload>& reconcile(
      const TransactionalFaceFluxLedger<Dim, Payload>& ledger, const CoarseFaceRefluxKey<Dim>& key,
      const RefinementRatio<Dim>& ratio, const FaceRefinementMapping<Dim>& mapping,
      const MetricRefluxBudget& budget, Axpy&& axpy) {
    require_dynamic_route_(ledger, key, ratio, mapping, budget);
    validate_payload_shapes_(ledger, key);
    clear_transient_();
    reset_result_(candidate_);
    struct TransientReset final {
      PreparedMetricRefluxWorkspace* workspace = nullptr;
      ~TransientReset() { workspace->clear_transient_(); }
    } transient_reset{this};

    for (const auto& entry : ledger.published_entries(axis_)) {
      if (!detail::matches_reflux_key(entry.key, key))
        continue;
      const std::size_t slice = acquire_slice_(entry.key, entry.measure);
      const std::size_t face = face_index_(entry.key);
      const std::size_t seen = slice * expected_fine_.size() + face;
      if (face_seen_.at(seen) != 0)
        throw std::runtime_error(
            "ND prepared metric reflux observed a duplicate face in one slice");
      face_seen_[seen] = 1;
      const double scale = weighted_face_flux_scale(entry.measure);
      switch (entry.key.role) {
        case FaceLedgerRole::Coarse:
          axpy(candidate_.coarse_integrated, scale, entry.payload);
          candidate_.coarse_weighted_measure += scale;
          if (!std::isfinite(candidate_.coarse_weighted_measure))
            throw std::overflow_error("ND metric reflux coarse weighted measure is not finite");
          break;
        case FaceLedgerRole::Fine:
          axpy(candidate_.fine_integrated, scale, entry.payload);
          candidate_.fine_weighted_measure += scale;
          if (!std::isfinite(candidate_.fine_weighted_measure))
            throw std::overflow_error("ND metric reflux fine weighted measure is not finite");
          break;
        default:
          throw std::runtime_error("ND metric reflux observed an invalid published face role");
      }
    }

    validate_coverage_and_temporal_(key);
    if (!detail::roundoff_equal(candidate_.coarse_weighted_measure,
                                candidate_.fine_weighted_measure, ledger.published_size()))
      throw std::runtime_error(
          "ND metric reflux coarse and fine metric-time measures do not cover the same face");
    axpy(candidate_.mismatch, 1.0, candidate_.fine_integrated);
    axpy(candidate_.mismatch, -1.0, candidate_.coarse_integrated);
    candidate_.fine_face_count = expected_fine_.size();
    using std::swap;
    swap(result_, candidate_);
    return result_;
  }

  template <class Axpy>
  const Payload& reconcile_coarse_cell_correction(double coarse_cell_measure,
                                                  CoarseCellFaceSide side, Axpy&& axpy) {
    if (!prepared_)
      throw std::logic_error("ND prepared metric reflux is not bound");
    if (!(coarse_cell_measure > 0.0) || !std::isfinite(coarse_cell_measure))
      throw std::invalid_argument(
          "ND metric reflux requires a finite positive coarse-cell measure");
    const double sign =
        side == CoarseCellFaceSide::Lower ? 1.0
        : side == CoarseCellFaceSide::Upper
            ? -1.0
            : throw std::invalid_argument("ND metric reflux has an invalid coarse-cell face side");
    const double coefficient = sign / coarse_cell_measure;
    if (!std::isfinite(coefficient))
      throw std::overflow_error(
          "ND metric reflux coarse-cell correction coefficient is not finite");
    detail::PreparedMetricPayload<Payload>::reset(correction_candidate_);
    axpy(correction_candidate_, coefficient, result_.mismatch);
    using std::swap;
    swap(correction_, correction_candidate_);
    return correction_;
  }

  [[nodiscard]] const MetricFaceReflux<Payload>& result() const {
    if (!prepared_)
      throw std::logic_error("ND prepared metric reflux is not bound");
    return result_;
  }

 private:
  using ClockCoordinate = std::tuple<int, std::int64_t, std::int64_t, std::int64_t>;

  struct Slice {
    FaceLedgerRole role = FaceLedgerRole::Coarse;
    ClockCoordinate clock{};
    std::string_view stage{};
    detail::TemporalSliceMeasure measure{};
    bool active = false;
  };

  struct Substep {
    detail::ExactSubstep interval{};
    detail::SubstepQuadrature quadrature{};
    bool active = false;
  };

  static void prime_result_(MetricFaceReflux<Payload>& result, const Payload& prototype) {
    detail::PreparedMetricPayload<Payload>::prime(result.coarse_integrated, prototype);
    detail::PreparedMetricPayload<Payload>::prime(result.fine_integrated, prototype);
    detail::PreparedMetricPayload<Payload>::prime(result.mismatch, prototype);
    result.coarse_weighted_measure = 0.0;
    result.fine_weighted_measure = 0.0;
    result.fine_face_count = 0;
  }

  static void reset_result_(MetricFaceReflux<Payload>& result) noexcept {
    detail::PreparedMetricPayload<Payload>::reset(result.coarse_integrated);
    detail::PreparedMetricPayload<Payload>::reset(result.fine_integrated);
    detail::PreparedMetricPayload<Payload>::reset(result.mismatch);
    result.coarse_weighted_measure = 0.0;
    result.fine_weighted_measure = 0.0;
    result.fine_face_count = 0;
  }

  static bool same_budget_(const MetricRefluxBudget& left,
                           const MetricRefluxBudget& right) noexcept {
    return left.max_fine_faces == right.max_fine_faces &&
           left.max_published_entries == right.max_published_entries &&
           left.max_clock_stage_slices == right.max_clock_stage_slices;
  }

  bool static_identity_matches_(const CoarseFaceRefluxKey<Dim>& key) const noexcept {
    return owner_ == key.owner && state_ == key.state && levels_ == key.levels &&
           centering_ == key.centering && axis_ == key.axis && coarse_face_ == key.coarse_face;
  }

  bool same_static_identity_(const PreparedMetricRefluxWorkspace& other) const noexcept {
    return owner_ == other.owner_ && state_ == other.state_ && levels_ == other.levels_ &&
           centering_ == other.centering_ && axis_ == other.axis_ &&
           coarse_face_ == other.coarse_face_ && ratio_ == other.ratio_ &&
           mapping_ == other.mapping_ && same_budget_(budget_, other.budget_) &&
           expected_fine_ == other.expected_fine_;
  }

  static bool result_has_capacity_for_(const MetricFaceReflux<Payload>& destination,
                                       const MetricFaceReflux<Payload>& source) noexcept {
    return detail::PreparedMetricPayload<Payload>::has_capacity_for(destination.coarse_integrated,
                                                                    source.coarse_integrated) &&
           detail::PreparedMetricPayload<Payload>::has_capacity_for(destination.fine_integrated,
                                                                    source.fine_integrated) &&
           detail::PreparedMetricPayload<Payload>::has_capacity_for(destination.mismatch,
                                                                    source.mismatch);
  }

  void require_dynamic_route_(const TransactionalFaceFluxLedger<Dim, Payload>& ledger,
                              const CoarseFaceRefluxKey<Dim>& key,
                              const RefinementRatio<Dim>& ratio,
                              const FaceRefinementMapping<Dim>& mapping,
                              const MetricRefluxBudget& budget) const {
    if (!prepared_)
      throw std::logic_error("ND prepared metric reflux is not bound");
    detail::validate_reflux_key(key);
    detail::validate_reflux_budget(budget);
    if (!static_identity_matches_(key) || !(ratio == ratio_) || mapping != mapping_ ||
        !same_budget_(budget, budget_))
      throw std::invalid_argument("ND prepared metric reflux static route drifted after bind");
    if (ledger.published_size() > budget_.max_published_entries)
      throw std::length_error("ND prepared metric reflux published entries exceed its budget");
  }

  void validate_payload_shapes_(const TransactionalFaceFluxLedger<Dim, Payload>& ledger,
                                const CoarseFaceRefluxKey<Dim>& key) const {
    for (const auto& entry : ledger.published_entries(axis_))
      if (detail::matches_reflux_key(entry.key, key) &&
          !detail::PreparedMetricPayload<Payload>::same_shape(result_.mismatch, entry.payload))
        throw std::invalid_argument("ND prepared metric reflux payload shape drifted after bind");
  }

  void clear_transient_() noexcept {
    slice_count_ = 0;
    for (auto& slice : slices_)
      slice = Slice{};
    for (auto& substep : substeps_)
      substep = Substep{};
    std::fill(face_seen_.begin(), face_seen_.end(), std::uint8_t{0});
  }

  std::size_t acquire_slice_(const FaceFluxFragmentKey<Dim>& key,
                             const FaceFluxFragmentMeasure& measure) {
    const ClockCoordinate clock = detail::clock_coordinate(key.clock);
    const detail::TemporalSliceMeasure temporal{measure.stage_weight, measure.substep_begin,
                                                measure.substep_end, measure.substep_duration};
    for (std::size_t index = 0; index < slice_count_; ++index) {
      Slice& slice = slices_[index];
      if (slice.role != key.role || slice.clock != clock || slice.stage != key.stage)
        continue;
      if (slice.measure.stage_weight != temporal.stage_weight ||
          slice.measure.substep_begin != temporal.substep_begin ||
          slice.measure.substep_end != temporal.substep_end ||
          slice.measure.substep_duration != temporal.substep_duration)
        throw std::runtime_error(
            "ND metric reflux clock-stage faces disagree on their temporal measure");
      return index;
    }
    if (slice_count_ >= budget_.max_clock_stage_slices)
      throw std::length_error("ND metric reflux clock-stage slices exceed their prepared budget");
    Slice& slice = slices_[slice_count_];
    slice.role = key.role;
    slice.clock = clock;
    slice.stage = key.stage;
    slice.measure = temporal;
    slice.active = true;
    return slice_count_++;
  }

  std::size_t face_index_(const FaceFluxFragmentKey<Dim>& key) const {
    const auto coordinate = detail::coordinate_array(key.face);
    if (key.role == FaceLedgerRole::Coarse) {
      if (coordinate != detail::coordinate_array(coarse_face_))
        throw std::runtime_error("ND metric reflux observed an unexpected coarse face");
      return 0;
    }
    if (key.role != FaceLedgerRole::Fine)
      throw std::runtime_error("ND metric reflux observed an invalid published face role");
    for (std::size_t index = 0; index < expected_fine_.size(); ++index)
      if (expected_fine_[index] == coordinate)
        return index;
    throw std::runtime_error("ND metric reflux observed a fine face outside the sealed product");
  }

  detail::AuthenticatedWindow authenticated_window_(FaceLedgerRole role,
                                                    const CoarseFaceRefluxKey<Dim>& key) {
    std::size_t count = 0;
    for (std::size_t index = 0; index < slice_count_; ++index) {
      const Slice& slice = slices_[index];
      if (slice.role != role)
        continue;
      const detail::ExactSubstep interval{slice.measure.substep_begin, slice.measure.substep_end};
      std::size_t position = 0;
      for (; position < count; ++position)
        if (substeps_[position].interval.begin == interval.begin &&
            substeps_[position].interval.end == interval.end)
          break;
      if (position == count) {
        Substep& substep = substeps_[count++];
        substep.interval = interval;
        substep.quadrature =
            detail::SubstepQuadrature{Rational{0, 1}, slice.measure.substep_duration};
        substep.active = true;
      } else if (substeps_[position].quadrature.duration != slice.measure.substep_duration) {
        throw std::runtime_error(
            "ND metric reflux stages disagree on their physical substep duration");
      }
      substeps_[position].quadrature.stage_weight_sum =
          substeps_[position].quadrature.stage_weight_sum + slice.measure.stage_weight;
    }
    if (count == 0)
      throw std::runtime_error("ND metric reflux has no published temporal slices");
    std::sort(
        substeps_.begin(), substeps_.begin() + static_cast<std::ptrdiff_t>(count),
        [](const Substep& left, const Substep& right) { return left.interval < right.interval; });
    Rational cursor = key.window_begin;
    detail::AuthenticatedWindow result;
    for (std::size_t index = 0; index < count; ++index) {
      const Substep& substep = substeps_[index];
      if (substep.interval.begin != cursor || !(substep.interval.begin < substep.interval.end))
        throw std::runtime_error(
            "ND metric reflux substeps do not form a contiguous exact clock partition");
      if (substep.quadrature.stage_weight_sum != Rational{1, 1})
        throw std::runtime_error(
            "ND metric reflux stage weights do not close one accepted substep");
      const double phase_span = (substep.interval.end - substep.interval.begin).value();
      if (!(phase_span > 0.0) || !std::isfinite(phase_span))
        throw std::overflow_error("ND metric reflux physical clock rate is not finite");
      const double duration_per_phase = substep.quadrature.duration / phase_span;
      if (!std::isfinite(duration_per_phase))
        throw std::overflow_error("ND metric reflux physical clock rate is not finite");
      if (result.substep_count == 0)
        result.duration_per_phase = duration_per_phase;
      else if (!detail::roundoff_equal(result.duration_per_phase, duration_per_phase,
                                       result.substep_count + 1))
        throw std::runtime_error("ND metric reflux substeps disagree on their physical clock rate");
      cursor = substep.interval.end;
      result.duration += substep.quadrature.duration;
      ++result.substep_count;
      if (!std::isfinite(result.duration))
        throw std::overflow_error("ND metric reflux physical window duration is not finite");
    }
    if (cursor != key.window_end)
      throw std::runtime_error(
          "ND metric reflux substeps do not cover the complete exact clock window");
    return result;
  }

  void validate_coverage_and_temporal_(const CoarseFaceRefluxKey<Dim>& key) {
    bool coarse_present = false;
    bool fine_present = false;
    for (std::size_t slice = 0; slice < slice_count_; ++slice) {
      const Slice& descriptor = slices_[slice];
      const std::size_t offset = slice * expected_fine_.size();
      if (descriptor.role == FaceLedgerRole::Coarse) {
        coarse_present = true;
        if (face_seen_[offset] == 0)
          throw std::runtime_error("ND metric reflux has an incomplete coarse face slice");
      } else if (descriptor.role == FaceLedgerRole::Fine) {
        fine_present = true;
        for (std::size_t face = 0; face < expected_fine_.size(); ++face)
          if (face_seen_[offset + face] == 0)
            throw std::runtime_error(
                "ND metric reflux has an incomplete fine tangential face product in one "
                "clock-stage slice");
      } else {
        throw std::runtime_error("ND metric reflux observed an invalid published face role");
      }
    }
    if (!coarse_present || !fine_present)
      throw std::runtime_error("ND metric reflux has no complete published coarse/fine slices");
    const detail::AuthenticatedWindow coarse = authenticated_window_(FaceLedgerRole::Coarse, key);
    const detail::AuthenticatedWindow fine = authenticated_window_(FaceLedgerRole::Fine, key);
    if (!detail::roundoff_equal(coarse.duration, fine.duration, slice_count_) ||
        !detail::roundoff_equal(coarse.duration_per_phase, fine.duration_per_phase, slice_count_))
      throw std::runtime_error(
          "ND metric reflux coarse and fine physical clocks do not cover the same window");
  }

  std::string owner_;
  std::string state_;
  LevelTransition levels_{};
  FaceLedgerCentering centering_ = FaceLedgerCentering::Face;
  int axis_ = 0;
  Index<Dim> coarse_face_{};
  RefinementRatio<Dim> ratio_{};
  FaceRefinementMapping<Dim> mapping_{};
  MetricRefluxBudget budget_{};
  std::vector<std::array<int, Dim>> expected_fine_;
  std::vector<Slice> slices_;
  std::vector<Substep> substeps_;
  std::vector<std::uint8_t> face_seen_;
  MetricFaceReflux<Payload> result_{};
  MetricFaceReflux<Payload> candidate_{};
  Payload correction_{};
  Payload correction_candidate_{};
  std::size_t slice_count_ = 0;
  bool prepared_ = false;
};

template <int Dim, class Payload, class Axpy>
const MetricFaceReflux<Payload>& metric_reflux_prepared(
    PreparedMetricRefluxWorkspace<Dim, Payload>& workspace,
    const TransactionalFaceFluxLedger<Dim, Payload>& ledger, const CoarseFaceRefluxKey<Dim>& key,
    const RefinementRatio<Dim>& ratio, const FaceRefinementMapping<Dim>& mapping,
    const MetricRefluxBudget& budget, Axpy&& axpy) {
  return workspace.reconcile(ledger, key, ratio, mapping, budget, std::forward<Axpy>(axpy));
}

/// Convert the integrated face mismatch into the correction of the adjacent coarse cell.  Fluxes
/// use canonical positive-axis orientation: replacing a lower face adds mismatch/volume, while
/// replacing an upper face subtracts it.  The opposite fine-side transport then closes composite
/// conservation to the arithmetic precision of the supplied payload axpy.
template <class Payload, class Axpy>
Payload coarse_cell_reflux_correction(const MetricFaceReflux<Payload>& reflux,
                                      double coarse_cell_measure, CoarseCellFaceSide side,
                                      Axpy&& axpy) {
  if (!(coarse_cell_measure > 0.0) || !std::isfinite(coarse_cell_measure))
    throw std::invalid_argument("ND metric reflux requires a finite positive coarse-cell measure");
  double sign = 0.0;
  switch (side) {
    case CoarseCellFaceSide::Lower:
      sign = 1.0;
      break;
    case CoarseCellFaceSide::Upper:
      sign = -1.0;
      break;
    default:
      throw std::invalid_argument("ND metric reflux has an invalid coarse-cell face side");
  }
  const double coefficient = sign / coarse_cell_measure;
  if (!std::isfinite(coefficient))
    throw std::overflow_error("ND metric reflux coarse-cell correction coefficient is not finite");
  Payload correction{};
  axpy(correction, coefficient, reflux.mismatch);
  return correction;
}

}  // namespace pops::amr::reflux
