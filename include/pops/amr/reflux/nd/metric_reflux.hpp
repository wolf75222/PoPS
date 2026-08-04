/// @file
/// @brief Metric coarse/fine face matching and conservative reflux for dimensions 1..3.

#pragma once

#include <pops/amr/reflux/nd/face_flux_ledger.hpp>
#include <pops/amr/transfer/nd/refinement_ratio.hpp>

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

namespace pops::amr::reflux::nd {

/// Affine relation between the coarse and fine face index spaces.  The same mapping applies to
/// normal face coordinates and to tangential cell coordinates; the normal fine face has no child
/// offset, while tangential coordinates span their full anisotropic ratio.
template <int Dim>
struct FaceRefinementMapping {
  Index<Dim> coarse_origin{};
  Index<Dim> fine_origin{};

  constexpr bool operator==(const FaceRefinementMapping&) const = default;
};

/// Identity of the coarse face whose accepted stage/substep fragments are to be reconciled.
template <int Dim>
struct CoarseFaceRefluxKey {
  std::string owner;
  std::string state;
  LevelTransition levels{};
  FaceLedgerCentering centering = FaceLedgerCentering::Face;
  int axis = 0;
  Index<Dim> coarse_face{};
  std::uint64_t attempt = 0;
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
}

template <int Dim>
bool matches_reflux_key(const FaceFluxFragmentKey<Dim>& fragment,
                        const CoarseFaceRefluxKey<Dim>& query) {
  return fragment.owner == query.owner && fragment.state == query.state &&
         fragment.levels == query.levels && fragment.centering == query.centering &&
         fragment.axis == query.axis && fragment.coarse_face == query.coarse_face &&
         fragment.attempt == query.attempt;
}

using StageSlice =
    std::tuple<std::tuple<int, std::int64_t, std::int64_t, std::int64_t>, std::string>;

inline StageSlice stage_slice(const ClockStamp& clock, const std::string& stage) {
  return {clock_coordinate(clock), stage};
}

template <int Dim>
std::set<std::array<int, Dim>> expected_fine_face_set(
    const CoarseFaceRefluxKey<Dim>& key, const transfer::nd::RefinementRatio<Dim>& ratio,
    const FaceRefinementMapping<Dim>& mapping) {
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

template <int Dim>
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
                                                   const transfer::nd::RefinementRatio<Dim>& ratio,
                                                   const FaceRefinementMapping<Dim>& mapping = {}) {
  detail::validate_reflux_key(key);
  const auto expected = detail::expected_fine_face_set(key, ratio, mapping);
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
                                        const transfer::nd::RefinementRatio<Dim>& ratio,
                                        const FaceRefinementMapping<Dim>& mapping, Axpy&& axpy) {
  detail::validate_reflux_key(key);
  const auto expected_fine = detail::expected_fine_face_set(key, ratio, mapping);
  const std::set<std::array<int, Dim>> expected_coarse{detail::coordinate_array(key.coarse_face)};
  std::map<detail::StageSlice, std::set<std::array<int, Dim>>> coarse_slices;
  std::map<detail::StageSlice, std::set<std::array<int, Dim>>> fine_slices;
  MetricFaceReflux<Payload> result;

  for (const auto& entry : ledger.published_entries(key.axis)) {
    if (!detail::matches_reflux_key(entry.key, key))
      continue;
    const double scale = weighted_face_flux_scale(entry.measure);
    const auto slice = detail::stage_slice(entry.key.clock, entry.key.stage);
    switch (entry.key.role) {
      case FaceLedgerRole::Coarse:
        coarse_slices[slice].insert(detail::coordinate_array(entry.key.face));
        axpy(result.coarse_integrated, scale, entry.payload);
        result.coarse_weighted_measure += scale;
        if (!std::isfinite(result.coarse_weighted_measure))
          throw std::overflow_error("ND metric reflux coarse weighted measure is not finite");
        break;
      case FaceLedgerRole::Fine:
        fine_slices[slice].insert(detail::coordinate_array(entry.key.face));
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
  axpy(result.mismatch, 1.0, result.fine_integrated);
  axpy(result.mismatch, -1.0, result.coarse_integrated);
  result.fine_face_count = expected_fine.size();
  return result;
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
  Payload correction{};
  axpy(correction, sign / coarse_cell_measure, reflux.mismatch);
  return correction;
}

}  // namespace pops::amr::reflux::nd
