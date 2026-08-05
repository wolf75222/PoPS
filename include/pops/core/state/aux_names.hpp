/// @file
/// @brief Host-only exact-ranked canonical aux name/component tables.

#pragma once

#include <pops/core/state/state.hpp>

#include <array>
#include <cstddef>
#include <string_view>
#include <utility>

namespace pops {

namespace detail {

inline constexpr std::array<std::string_view, 3> kAuxGradientNames = {"grad_x", "grad_y", "grad_z"};

template <int Dim>
constexpr auto make_aux_canonical_names() {
  using layout = AuxComponentLayout<Dim>;
  std::array<std::pair<std::string_view, int>, layout::named_begin> result{};
  result[static_cast<std::size_t>(layout::phi)] = {"phi", layout::phi};
  for (int axis = 0; axis < Dim; ++axis)
    result[static_cast<std::size_t>(layout::gradient_begin + axis)] = {
        kAuxGradientNames[static_cast<std::size_t>(axis)], layout::gradient_begin + axis};
  result[static_cast<std::size_t>(layout::b_z)] = {"B_z", layout::b_z};
  result[static_cast<std::size_t>(layout::t_e)] = {"T_e", layout::t_e};
  return result;
}

}  // namespace detail

template <int Dim>
inline constexpr auto kAuxCanonicalNamesFor = detail::make_aux_canonical_names<Dim>();

/// Canonical table of this compiled native artifact.
inline constexpr auto kAuxCanonicalNames = kAuxCanonicalNamesFor<kNativeDimension>;

/// Component of canonical field `name` for rank `Dim`, or -1 for a model-named/unknown field.
template <int Dim = kNativeDimension>
constexpr int aux_canonical_index(std::string_view name) {
  for (const auto& [canonical_name, component] : kAuxCanonicalNamesFor<Dim>)
    if (canonical_name == name)
      return component;
  return -1;
}

/// Canonical field name at `component` for rank `Dim`, or an empty view outside that prefix.
template <int Dim = kNativeDimension>
constexpr std::string_view aux_canonical_name(int component) {
  for (const auto& [canonical_name, canonical_component] : kAuxCanonicalNamesFor<Dim>)
    if (canonical_component == component)
      return canonical_name;
  return {};
}

}  // namespace pops
