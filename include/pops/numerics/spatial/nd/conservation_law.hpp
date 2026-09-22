/// @file
/// @brief Model-independent contract consumed by dimension-generic finite volumes.

#pragma once

#include <pops/numerics/spatial/nd/state_conversion.hpp>

#include <concepts>
#include <type_traits>

namespace pops::nd {

template <int Dim, class Model>
concept ConservationLaw = Dim >= 1 && Dim <= 3 && Model::dimension == Dim && Model::n_vars >= 1 &&
                          std::is_trivially_copyable_v<Model> &&
                          requires(const Model& model, const typename Model::State& state) {
                            typename Model::Schema;
                            typename Model::Primitive;
                            {
                              model.recover(state)
                            } -> std::same_as<StateConversion<typename Model::Primitive>>;
                            { model.admissibility(state) } -> std::same_as<StateConversionStatus>;
                          };

}  // namespace pops::nd
