#pragma once

#include <pops/runtime/config/runtime_params.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace pops::compiled_model {

/// Whether one generated brick carries the fixed-size device RuntimeParams payload.
template <class Brick, class = void>
struct HasRuntimeParams : std::false_type {};

template <class Brick>
struct HasRuntimeParams<Brick, std::void_t<decltype(std::declval<Brick&>().params)>>
    : std::is_same<std::decay_t<decltype(std::declval<Brick&>().params)>, RuntimeParams> {};

template <class Brick>
inline void apply_runtime_params(Brick& brick, const RuntimeParams& params) {
  if constexpr (HasRuntimeParams<Brick>::value)
    brick.params = params;
}

template <class Brick>
inline int brick_runtime_param_count(const Brick& brick) {
  if constexpr (HasRuntimeParams<Brick>::value)
    return brick.params.count;
  (void)brick;
  return 0;
}

template <class Model>
inline RuntimeParams declaration_runtime_params(const Model& model) {
  if constexpr (HasRuntimeParams<std::decay_t<decltype(model.hyp)>>::value)
    return model.hyp.params;
  else if constexpr (HasRuntimeParams<std::decay_t<decltype(model.src)>>::value)
    return model.src.params;
  else if constexpr (HasRuntimeParams<std::decay_t<decltype(model.ell)>>::value)
    return model.ell.params;
  return RuntimeParams{};
}

template <class Model>
inline int runtime_param_count(const Model& model = Model{}) {
  const int hyp = brick_runtime_param_count(model.hyp);
  const int src = brick_runtime_param_count(model.src);
  const int ell = brick_runtime_param_count(model.ell);
  const int expected = hyp != 0 ? hyp : (src != 0 ? src : ell);
  if ((hyp != 0 && hyp != expected) || (src != 0 && src != expected) ||
      (ell != 0 && ell != expected))
    throw std::runtime_error("compiled model bricks disagree on the runtime parameter layout");
  if (expected < 0 || expected > kMaxRuntimeParams)
    throw std::runtime_error("compiled model runtime parameter count exceeds exact capacity");
  return expected;
}

template <class Model>
inline void declaration_runtime_param_defaults(double* output) {
  const RuntimeParams params = declaration_runtime_params(Model{});
  for (int index = 0; index < params.count; ++index)
    output[index] = static_cast<double>(params.values[index]);
}

/// Inject one complete BindSchema vector before native closures are constructed.
template <class Model>
inline Model bind_runtime_params(Model model, const double* values, int count) {
  const int expected = runtime_param_count(model);
  if (count != expected)
    throw std::runtime_error("native block parameter vector has " + std::to_string(count) +
                             " values but the compiled model requires " + std::to_string(expected));
  if (count > 0 && values == nullptr)
    throw std::runtime_error("native block parameter vector is null");
  RuntimeParams params{};
  params.count = count;
  for (int index = 0; index < count; ++index)
    params.values[index] = static_cast<Real>(values[index]);
  apply_runtime_params(model.hyp, params);
  apply_runtime_params(model.src, params);
  apply_runtime_params(model.ell, params);
  return model;
}

}  // namespace pops::compiled_model
