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
#include <tuple>
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

/// Expected exact integral of an authored intermediate expression. The caller must
/// derive this contract from the immutable expression before consulting its ledger.
/// An absent contract retains the ordinary unit quadrature requirement.
struct RefluxTemporalInterval {
  Rational begin{0, 1};
  Rational end{0, 1};
  friend bool operator<(const RefluxTemporalInterval& left, const RefluxTemporalInterval& right) {
    return left.begin == right.begin ? left.end < right.end : left.begin < right.begin;
  }
};
using RefluxTemporalWeights =
    std::map<std::string, std::map<RefluxTemporalInterval, ::pops::ExactWeight>>;
struct MetricRefluxTemporalContract {
  RefluxTemporalWeights coarse;
  RefluxTemporalWeights fine;
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
  ::pops::ExactWeight stage_weight{0, 1};
  Rational substep_begin{0, 1};
  Rational substep_end{0, 1};
  double substep_duration = 0.0;
};

using TemporalSlices = std::map<StageSlice, TemporalSliceMeasure>;
using TemporalStageFamilies = std::map<std::string, TemporalSlices>;

inline StageSlice stage_slice(const ClockStamp& clock, const std::string& stage) {
  return {clock_coordinate(clock), stage};
}

inline void register_temporal_slice(TemporalSlices& slices, const StageSlice& slice,
                                    const FaceFluxFragmentMeasure& measure,
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

inline std::size_t temporal_slice_count(const TemporalStageFamilies& families) {
  std::size_t result = 0;
  for (const auto& [family, slices] : families) {
    (void)family;
    result += slices.size();
  }
  return result;
}

inline void register_temporal_family_slice(TemporalStageFamilies& families,
                                           const std::string& family, const StageSlice& slice,
                                           const FaceFluxFragmentMeasure& measure,
                                           std::size_t other_slice_count,
                                           const MetricRefluxBudget& budget) {
  auto& slices = families[family];
  register_temporal_slice(slices, slice, measure,
                          temporal_slice_count(families) + other_slice_count, budget);
}

using ExactSubstep = RefluxTemporalInterval;

struct SubstepQuadrature {
  ::pops::ExactWeight stage_weight_sum{0, 1};
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
    const TemporalSlices& slices, Rational window_begin, Rational window_end,
    const std::map<RefluxTemporalInterval, ::pops::ExactWeight>* expected = nullptr) {
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

  if (expected && expected->size() != substeps.size())
    throw std::runtime_error(
        "ND metric reflux substeps differ from the authored integral contract");
  Rational cursor = window_begin;
  AuthenticatedWindow result;
  for (const auto& [interval, quadrature] : substeps) {
    if (interval.begin != cursor || !(interval.begin < interval.end))
      throw std::runtime_error(
          "ND metric reflux substeps do not form a contiguous exact clock partition");
    if (expected) {
      const auto weight = expected->find(interval);
      if (weight == expected->end() || quadrature.stage_weight_sum != weight->second)
        throw std::runtime_error(
            "ND metric reflux stage weights differ from the authored integral contract");
    } else if (quadrature.stage_weight_sum != Rational{1, 1})
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
                                const TemporalStageFamilies& coarse_families,
                                const TemporalStageFamilies& fine_families,
                                const MetricRefluxTemporalContract* expected = nullptr) {
  if (expected && (expected->coarse.size() != coarse_families.size() ||
                   expected->fine.size() != fine_families.size()))
    throw std::runtime_error(
        "ND metric reflux families differ from the authored integral contract");
  if (coarse_families.size() != fine_families.size())
    throw std::runtime_error("ND metric reflux coarse and fine spatial provider identities differ");
  for (const auto& [family, coarse_slices] : coarse_families) {
    const auto fine = fine_families.find(family);
    if (fine == fine_families.end())
      throw std::runtime_error(
          "ND metric reflux coarse and fine spatial provider identities differ");
    if (expected && (!expected->coarse.contains(family) || !expected->fine.contains(family)))
      throw std::runtime_error(
          "ND metric reflux family differs from the authored integral contract");
    const AuthenticatedWindow coarse =
        authenticated_window(coarse_slices, key.window_begin, key.window_end,
                             expected ? &expected->coarse.at(family) : nullptr);
    const AuthenticatedWindow refined =
        authenticated_window(fine->second, key.window_begin, key.window_end,
                             expected ? &expected->fine.at(family) : nullptr);
    const std::size_t operations = coarse_slices.size() + fine->second.size();
    if (!roundoff_equal(coarse.duration, refined.duration, operations) ||
        !roundoff_equal(coarse.duration_per_phase, refined.duration_per_phase, operations))
      throw std::runtime_error(
          "ND metric reflux coarse and fine physical clocks do not cover the same window");
  }
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

namespace detail {

/// Integrate the published fragments that `for_each_candidate` visits; it visits, in ledger order,
/// published entries of `key.axis` that include every entry with the key's coarse face.
template <int Dim, class Payload, class Axpy, class ForEachCandidate>
MetricFaceReflux<Payload> metric_reflux_over(
    const TransactionalFaceFluxLedger<Dim, Payload>& ledger, const CoarseFaceRefluxKey<Dim>& key,
    const RefinementRatio<Dim>& ratio, const FaceRefinementMapping<Dim>& mapping,
    const MetricRefluxBudget& budget, Axpy& axpy, const MetricRefluxTemporalContract* expected,
    ForEachCandidate&& for_each_candidate) {
  validate_reflux_key(key);
  validate_reflux_budget(budget);
  if (ledger.published_size() > budget.max_published_entries)
    throw std::length_error("ND metric reflux published entries exceed their prepared budget");
  const auto expected_fine = expected_fine_face_set(key, ratio, mapping, budget);
  const std::set<std::array<int, Dim>> expected_coarse{coordinate_array(key.coarse_face)};
  std::map<StageSlice, std::set<std::array<int, Dim>>> coarse_slices;
  std::map<StageSlice, std::set<std::array<int, Dim>>> fine_slices;
  TemporalStageFamilies coarse_temporal;
  TemporalStageFamilies fine_temporal;
  MetricFaceReflux<Payload> result;

  for_each_candidate([&](const FaceFluxFragment<Dim, Payload>& entry) {
    if (!matches_reflux_key(entry.key, key))
      return;
    const double scale = weighted_face_flux_scale(entry.measure);
    const auto slice = stage_slice(entry.key.clock, entry.key.stage);
    switch (entry.key.role) {
      case FaceLedgerRole::Coarse:
        coarse_slices[slice].insert(coordinate_array(entry.key.face));
        register_temporal_family_slice(coarse_temporal, entry.key.temporal_family, slice,
                                       entry.measure, temporal_slice_count(fine_temporal), budget);
        axpy(result.coarse_integrated, scale, entry.payload);
        result.coarse_weighted_measure += scale;
        if (!std::isfinite(result.coarse_weighted_measure))
          throw std::overflow_error("ND metric reflux coarse weighted measure is not finite");
        break;
      case FaceLedgerRole::Fine:
        fine_slices[slice].insert(coordinate_array(entry.key.face));
        register_temporal_family_slice(fine_temporal, entry.key.temporal_family, slice,
                                       entry.measure, temporal_slice_count(coarse_temporal),
                                       budget);
        axpy(result.fine_integrated, scale, entry.payload);
        result.fine_weighted_measure += scale;
        if (!std::isfinite(result.fine_weighted_measure))
          throw std::overflow_error("ND metric reflux fine weighted measure is not finite");
        break;
      default:
        throw std::runtime_error("ND metric reflux observed an invalid published face role");
    }
  });

  require_complete_slices(coarse_slices, expected_coarse, "coarse");
  require_complete_slices(fine_slices, expected_fine, "fine");
  validate_temporal_coverage(key, coarse_temporal, fine_temporal, expected);
  if (!roundoff_equal(result.coarse_weighted_measure, result.fine_weighted_measure,
                      ledger.published_size()))
    throw std::runtime_error(
        "ND metric reflux coarse and fine metric-time measures do not cover the same face");
  axpy(result.mismatch, 1.0, result.fine_integrated);
  axpy(result.mismatch, -1.0, result.coarse_integrated);
  result.fine_face_count = expected_fine.size();
  return result;
}

}  // namespace detail

/// The published entries of one ledger grouped by axis and coarse face, each group in ledger
/// order. It reads the ledger once, so reconciling every coarse face of a transition stays linear
/// in its published fragments. The ledger must not change while the index is used.
template <int Dim, class Payload>
class PublishedCoarseFaceIndex {
 public:
  using Entry = FaceFluxFragment<Dim, Payload>;

  explicit PublishedCoarseFaceIndex(const TransactionalFaceFluxLedger<Dim, Payload>& ledger)
      : ledger_(&ledger) {
    for (int axis = 0; axis < Dim; ++axis)
      for (const Entry& entry : ledger.published_entries(axis))
        groups_[static_cast<std::size_t>(axis)][detail::coordinate_array(entry.key.coarse_face)]
            .push_back(&entry);
  }

  const TransactionalFaceFluxLedger<Dim, Payload>& ledger() const noexcept { return *ledger_; }

  /// The published entries of `axis` whose coarse face is `coarse_face`, in ledger order.
  const std::vector<const Entry*>& entries(int axis, const Index<Dim>& coarse_face) const {
    const auto& groups = groups_[static_cast<std::size_t>(detail::checked_axis(axis, Dim))];
    const auto group = groups.find(detail::coordinate_array(coarse_face));
    return group == groups.end() ? empty_ : group->second;
  }

 private:
  const TransactionalFaceFluxLedger<Dim, Payload>* ledger_;
  std::array<std::map<std::array<int, Dim>, std::vector<const Entry*>>, Dim> groups_{};
  std::vector<const Entry*> empty_;
};

/// Integrate every accepted coarse and fine stage fragment using its exact rational stage weight,
/// authored substep duration and physical face measure.  Fine faces must form the complete
/// tangential product for every clock-stage slice. Pending/rejected fragments are never observed.
template <int Dim, class Payload, class Axpy>
MetricFaceReflux<Payload> metric_reflux(const TransactionalFaceFluxLedger<Dim, Payload>& ledger,
                                        const CoarseFaceRefluxKey<Dim>& key,
                                        const RefinementRatio<Dim>& ratio,
                                        const FaceRefinementMapping<Dim>& mapping,
                                        const MetricRefluxBudget& budget, Axpy&& axpy,
                                        const MetricRefluxTemporalContract* expected = nullptr) {
  return detail::metric_reflux_over(ledger, key, ratio, mapping, budget, axpy, expected,
                                    [&](auto&& visit) {
                                      for (const auto& entry : ledger.published_entries(key.axis))
                                        visit(entry);
                                    });
}

/// The same reconciliation, reading only the indexed entries of the key's coarse face.
template <int Dim, class Payload, class Axpy>
MetricFaceReflux<Payload> metric_reflux(const PublishedCoarseFaceIndex<Dim, Payload>& index,
                                        const CoarseFaceRefluxKey<Dim>& key,
                                        const RefinementRatio<Dim>& ratio,
                                        const FaceRefinementMapping<Dim>& mapping,
                                        const MetricRefluxBudget& budget, Axpy&& axpy,
                                        const MetricRefluxTemporalContract* expected = nullptr) {
  return detail::metric_reflux_over(
      index.ledger(), key, ratio, mapping, budget, axpy, expected, [&](auto&& visit) {
        for (const auto* entry : index.entries(key.axis, key.coarse_face))
          visit(*entry);
      });
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
