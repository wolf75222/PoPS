/// @file
/// @brief Exact 2D periodic topology values shared by lowering and native halo execution.

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace pops {

/// Per-direction translation periodicity used by the historical axis-aligned halo scheduler.
struct Periodicity {
  bool x = false;
  bool y = false;
};

/// Exact topology equality shared by uniform, AMR and prepared-boundary validation.
constexpr bool same_periodicity(Periodicity left, Periodicity right) noexcept {
  return left.x == right.x && left.y == right.y;
}

/// One signed/permuted identification between two oriented Cartesian faces.
///
/// Face ordinals are xlo=0, xhi=1, ylo=2, yhi=3. For each source axis ``a``,
/// ``permutation[a]`` is its target axis and ``signs[a]`` is the signed orientation of that
/// coordinate. Translation is derived from the two endpoint faces and the execution domain; it is
/// deliberately absent from this topology-only value.
struct PeriodicIdentification2D {
  int source_face = -1;
  int target_face = -1;
  std::array<int, 2> permutation{{0, 1}};
  std::array<int, 2> signs{{1, 1}};

  bool operator==(const PeriodicIdentification2D&) const = default;

  bool is_translation_identity() const noexcept {
    return permutation == std::array<int, 2>{{0, 1}} && signs == std::array<int, 2>{{1, 1}};
  }

  void validate() const {
    if (source_face < 0 || source_face >= 4 || target_face < 0 || target_face >= 4 ||
        source_face == target_face)
      throw std::invalid_argument(
          "PeriodicIdentification2D requires two distinct xlo/xhi/ylo/yhi endpoints");
    if (!((permutation[0] == 0 && permutation[1] == 1) ||
          (permutation[0] == 1 && permutation[1] == 0)))
      throw std::invalid_argument(
          "PeriodicIdentification2D permutation must be the exact 2D axis permutation");
    if ((signs[0] != -1 && signs[0] != 1) || (signs[1] != -1 && signs[1] != 1))
      throw std::invalid_argument(
          "PeriodicIdentification2D signs must contain one -1/+1 per source axis");
    const int source_axis = source_face / 2;
    const int target_axis = target_face / 2;
    if (permutation[static_cast<std::size_t>(source_axis)] != target_axis)
      throw std::invalid_argument(
          "PeriodicIdentification2D source normal does not map to the target normal");
    const int source_outward = source_face % 2 == 0 ? -1 : 1;
    const int target_outward = target_face % 2 == 0 ? -1 : 1;
    const int required_normal_sign = -source_outward * target_outward;
    if (signs[static_cast<std::size_t>(source_axis)] != required_normal_sign)
      throw std::invalid_argument(
          "PeriodicIdentification2D normal sign does not map source interior to target exterior");
  }
};

/// Binding-friendly exact row decoder: source, target, permutation[0:2], signs[0:2].
inline std::vector<PeriodicIdentification2D> decode_periodic_identification_rows(
    const std::vector<std::array<int, 6>>& rows) {
  std::vector<PeriodicIdentification2D> result;
  result.reserve(rows.size());
  for (const auto& row : rows) {
    PeriodicIdentification2D identification{row[0], row[1], std::array<int, 2>{{row[2], row[3]}},
                                            std::array<int, 2>{{row[4], row[5]}}};
    identification.validate();
    result.push_back(identification);
  }
  return result;
}

}  // namespace pops
