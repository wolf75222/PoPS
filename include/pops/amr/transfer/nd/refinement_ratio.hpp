/// @file
/// @brief Compatibility name for the common validated ND refinement ratio.

#pragma once

#include <pops/amr/nd/refinement_ratio.hpp>

namespace pops::amr::transfer::nd {

template <int Dim>
using RefinementRatio = ::pops::amr::nd::RefinementRatio<Dim>;

}  // namespace pops::amr::transfer::nd
