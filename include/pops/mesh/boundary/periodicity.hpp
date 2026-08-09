/// @file
/// @brief Exact compile-time-ranked periodic topology values.

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
/// Face ordinals are axis-major: axis 0 lower/upper, axis 1 lower/upper, and so on. For each source
/// axis ``a``, ``permutation[a]`` is its target axis and ``signs[a]`` is the signed orientation of
/// that coordinate. Translation is derived from the endpoints and the execution domain; it is
/// deliberately absent from this topology-only value.
template <int Dim>
struct PeriodicIdentification {
  static_assert(Dim >= 1 && Dim <= 3,
                "PeriodicIdentification only supports dimensions 1, 2, and 3");

  int source_face = -1;
  int target_face = -1;
  std::array<int, Dim> permutation = [] {
    std::array<int, Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[static_cast<std::size_t>(axis)] = axis;
    return result;
  }();
  std::array<int, Dim> signs = [] {
    std::array<int, Dim> result{};
    result.fill(1);
    return result;
  }();

  bool operator==(const PeriodicIdentification&) const = default;

  bool is_translation_identity() const noexcept {
    for (int axis = 0; axis < Dim; ++axis)
      if (permutation[static_cast<std::size_t>(axis)] != axis ||
          signs[static_cast<std::size_t>(axis)] != 1)
        return false;
    return true;
  }

  void validate() const {
    if (source_face < 0 || source_face >= 2 * Dim || target_face < 0 || target_face >= 2 * Dim ||
        source_face == target_face)
      throw std::invalid_argument(
          "PeriodicIdentification requires two distinct ranked face endpoints");
    std::array<bool, Dim> assigned{};
    for (int source_axis = 0; source_axis < Dim; ++source_axis) {
      const int target_axis = permutation[static_cast<std::size_t>(source_axis)];
      if (target_axis < 0 || target_axis >= Dim || assigned[static_cast<std::size_t>(target_axis)])
        throw std::invalid_argument(
            "PeriodicIdentification permutation must be an exact ranked axis permutation");
      assigned[static_cast<std::size_t>(target_axis)] = true;
      if (signs[static_cast<std::size_t>(source_axis)] != -1 &&
          signs[static_cast<std::size_t>(source_axis)] != 1)
        throw std::invalid_argument(
            "PeriodicIdentification signs must contain one -1/+1 per source axis");
    }
    const int source_axis = source_face / 2;
    const int target_axis = target_face / 2;
    if (permutation[static_cast<std::size_t>(source_axis)] != target_axis)
      throw std::invalid_argument(
          "PeriodicIdentification source normal does not map to the target normal");
    const int source_outward = source_face % 2 == 0 ? -1 : 1;
    const int target_outward = target_face % 2 == 0 ? -1 : 1;
    const int required_normal_sign = -source_outward * target_outward;
    if (signs[static_cast<std::size_t>(source_axis)] != required_normal_sign)
      throw std::invalid_argument(
          "PeriodicIdentification normal sign does not map source interior to target exterior");
  }
};

/// Exact row decoder: source, target, permutation[0:Dim], signs[0:Dim].
template <int Dim>
inline std::vector<PeriodicIdentification<Dim>> decode_periodic_identification_rows(
    const std::vector<std::array<int, 2 + 2 * Dim>>& rows) {
  std::vector<PeriodicIdentification<Dim>> result;
  result.reserve(rows.size());
  for (const auto& row : rows) {
    PeriodicIdentification<Dim> identification;
    identification.source_face = row[0];
    identification.target_face = row[1];
    for (int axis = 0; axis < Dim; ++axis) {
      identification.permutation[static_cast<std::size_t>(axis)] =
          row[static_cast<std::size_t>(2 + axis)];
      identification.signs[static_cast<std::size_t>(axis)] =
          row[static_cast<std::size_t>(2 + Dim + axis)];
    }
    identification.validate();
    result.push_back(identification);
  }
  return result;
}

}  // namespace pops
