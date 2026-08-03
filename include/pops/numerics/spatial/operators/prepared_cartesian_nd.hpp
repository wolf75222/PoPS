#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/numerics/fv/flux_interfaces.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/provider_matrix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

/// @file
/// @brief Prepared, periodic Cartesian finite-volume residual for compile-time dimensions 1..3.
///
/// This is the dimension-generic local spatial-provider kernel.  It deliberately does not claim
/// that the current Box2D/MultiFab/AMR runtime can carry a 1D or 3D hierarchy: callers provide one
/// contiguous cell-major patch.  Metric preparation, reconstruction, typed Riemann evaluation and
/// conservative divergence are shared for every dimension.  Periodic indexing makes conservation
/// an executable kernel property without introducing a second physical-boundary authority.

namespace pops {

template <int Dimension>
struct PreparedCartesianMetric {
  static_assert(Dimension >= 1 && Dimension <= 3,
                "PreparedCartesianMetric supports compile-time dimensions 1..3");

  std::array<int, Dimension> extents{};
  std::array<Real, Dimension> spacing{};
  std::array<Real, Dimension> face_measure{};
  Real cell_measure = Real(0);
  std::size_t cells = 0;
};

namespace detail {

template <class Reconstruction>
consteval int periodic_reconstruction_minimum_extent() {
  if constexpr (CellValueReconstruction<Reconstruction>) {
    return 1;
  } else if constexpr (SlopeReconstruction<Reconstruction>) {
    return 3;
  } else {
    return Reconstruction::stencil_max_offset - Reconstruction::stencil_min_offset + 1;
  }
}

template <int Dimension>
PreparedCartesianMetric<Dimension> prepare_cartesian_metric(
    const std::array<int, Dimension>& extents, const std::array<Real, Dimension>& lower,
    const std::array<Real, Dimension>& upper, int minimum_extent) {
  PreparedCartesianMetric<Dimension> metric;
  metric.extents = extents;
  metric.cells = 1;
  metric.cell_measure = Real(1);
  for (int axis = 0; axis < Dimension; ++axis) {
    if (extents[axis] < minimum_extent)
      throw std::invalid_argument(
          "prepared Cartesian residual extent is smaller than its reconstruction stencil");
    if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || !(upper[axis] > lower[axis]))
      throw std::invalid_argument(
          "prepared Cartesian residual requires finite strictly ordered metric bounds");
    metric.spacing[axis] = (upper[axis] - lower[axis]) / static_cast<Real>(extents[axis]);
    metric.cell_measure *= metric.spacing[axis];
    metric.cells *= static_cast<std::size_t>(extents[axis]);
  }
  for (int axis = 0; axis < Dimension; ++axis)
    metric.face_measure[axis] = metric.cell_measure / metric.spacing[axis];
  return metric;
}

template <std::size_t Dimension>
std::array<int, Dimension> cartesian_index(std::size_t linear,
                                           const std::array<int, Dimension>& extents) {
  std::array<int, Dimension> index{};
  for (std::size_t axis = 0; axis < Dimension; ++axis) {
    index[axis] = static_cast<int>(linear % static_cast<std::size_t>(extents[axis]));
    linear /= static_cast<std::size_t>(extents[axis]);
  }
  return index;
}

template <std::size_t Dimension>
std::size_t cartesian_linear(const std::array<int, Dimension>& index,
                             const std::array<int, Dimension>& extents) {
  std::size_t linear = 0;
  std::size_t stride = 1;
  for (std::size_t axis = 0; axis < Dimension; ++axis) {
    linear += static_cast<std::size_t>(index[axis]) * stride;
    stride *= static_cast<std::size_t>(extents[axis]);
  }
  return linear;
}

inline int periodic_coordinate(int coordinate, int extent) {
  const int wrapped = coordinate % extent;
  return wrapped < 0 ? wrapped + extent : wrapped;
}

template <int Dimension, class Model>
typename Model::State load_periodic_state(std::span<const Real> state,
                                          const std::array<int, Dimension>& extents,
                                          std::array<int, Dimension> index, int axis = 0,
                                          int offset = 0) {
  index[axis] = periodic_coordinate(index[axis] + offset, extents[axis]);
  const std::size_t cell = cartesian_linear(index, extents);
  typename Model::State value{};
  for (int component = 0; component < Model::n_vars; ++component)
    value[component] =
        state[cell * static_cast<std::size_t>(Model::n_vars) + static_cast<std::size_t>(component)];
  return value;
}

template <int Dimension, class Model, class Reconstruction>
typename Model::State reconstruct_periodic_state(std::span<const Real> state,
                                                 const std::array<int, Dimension>& extents,
                                                 const std::array<int, Dimension>& index, int axis,
                                                 int orientation,
                                                 const Reconstruction& reconstruction) {
  typename Model::State result = load_periodic_state<Dimension, Model>(state, extents, index);
  for (int component = 0; component < Model::n_vars; ++component) {
    const auto sample = [&](int offset) {
      return load_periodic_state<Dimension, Model>(state, extents, index, axis, offset)[component];
    };
    const Real center = sample(0);
    if constexpr (CellValueReconstruction<Reconstruction>) {
      result[component] = reconstruction.cell_face_value(center);
    } else if constexpr (SlopeReconstruction<Reconstruction>) {
      result[component] =
          center + static_cast<Real>(orientation) * Real(0.5) *
                       reconstruction.limited_slope(center - sample(-1), sample(1) - center);
    } else if constexpr (StencilReconstruction<Reconstruction>) {
      const auto oriented_sample = [&](int offset) { return sample(orientation * offset); };
      result[component] = reconstruction.stencil_face_value(oriented_sample);
    }
  }
  return result;
}

}  // namespace detail

template <int Dimension, class Model, class Reconstruction = NoSlope,
          class NumericalFluxPolicy = RusanovFlux>
class PreparedPeriodicCartesianResidual {
 public:
  static_assert(Dimension >= 1 && Dimension <= 3,
                "PreparedPeriodicCartesianResidual supports dimensions 1..3");
  static_assert(ReconstructionPolicy<Reconstruction>,
                "PreparedPeriodicCartesianResidual requires one typed reconstruction policy");

  using State = typename Model::State;

  PreparedPeriodicCartesianResidual(const std::array<int, Dimension>& extents,
                                    const std::array<Real, Dimension>& lower,
                                    const std::array<Real, Dimension>& upper, Model model,
                                    Reconstruction reconstruction = {},
                                    NumericalFluxPolicy numerical_flux = {},
                                    FluxProviderValues<Model> constant_providers = {})
      : metric_(detail::prepare_cartesian_metric<Dimension>(
            extents, lower, upper,
            detail::periodic_reconstruction_minimum_extent<Reconstruction>())),
        model_(std::move(model)),
        reconstruction_(std::move(reconstruction)),
        numerical_flux_(std::move(numerical_flux)),
        constant_providers_(constant_providers) {}

  [[nodiscard]] static constexpr SpatialProviderCapabilities capabilities() {
    return make_cartesian_spatial_provider(Dimension);
  }

  [[nodiscard]] const PreparedCartesianMetric<Dimension>& metric() const noexcept {
    return metric_;
  }

  [[nodiscard]] std::size_t scalar_count() const noexcept {
    return metric_.cells * static_cast<std::size_t>(Model::n_vars);
  }

  /// Evaluate a conservative periodic residual.  State and residual may not alias.  A Riemann
  /// refusal clears the candidate residual before throwing; the accepted state is never mutated.
  void execute(std::span<const Real> state, std::span<Real> residual) const {
    if (state.size() != scalar_count() || residual.size() != scalar_count())
      throw std::invalid_argument(
          "prepared Cartesian residual buffers do not match extents x model components");
    if (state.data() == residual.data())
      throw std::invalid_argument(
          "prepared Cartesian residual requires distinct immutable state and output buffers");

    std::fill(residual.begin(), residual.end(), Real(0));
    const auto providers = bind_flux_providers<Model>(constant_providers_);
    for (std::size_t linear = 0; linear < metric_.cells; ++linear) {
      const auto index = detail::cartesian_index(linear, metric_.extents);
      for (int axis = 0; axis < Dimension; ++axis) {
        auto previous = index;
        auto next = index;
        previous[axis] = detail::periodic_coordinate(previous[axis] - 1, metric_.extents[axis]);
        next[axis] = detail::periodic_coordinate(next[axis] + 1, metric_.extents[axis]);

        const State minus_left = detail::reconstruct_periodic_state<Dimension, Model>(
            state, metric_.extents, previous, axis, +1, reconstruction_);
        const State minus_right = detail::reconstruct_periodic_state<Dimension, Model>(
            state, metric_.extents, index, axis, -1, reconstruction_);
        const State plus_left = detail::reconstruct_periodic_state<Dimension, Model>(
            state, metric_.extents, index, axis, +1, reconstruction_);
        const State plus_right = detail::reconstruct_periodic_state<Dimension, Model>(
            state, metric_.extents, next, axis, -1, reconstruction_);
        const FaceContext face = FaceContext::axis_aligned(
            axis, metric_.face_measure[axis], FaceOrientation::kPositive, metric_.cell_measure);
        const auto minus = evaluate_numerical_flux(numerical_flux_, model_, minus_left, providers,
                                                   minus_right, providers, face);
        const auto plus = evaluate_numerical_flux(numerical_flux_, model_, plus_left, providers,
                                                  plus_right, providers, face);
        if (!minus.succeeded() || !plus.succeeded()) {
          std::fill(residual.begin(), residual.end(), Real(0));
          throw std::runtime_error("prepared Cartesian residual numerical flux refused a face");
        }
        const State minus_integrated = apply_face_measure(minus.checked_density(), face).value;
        const State plus_integrated = apply_face_measure(plus.checked_density(), face).value;
        for (int component = 0; component < Model::n_vars; ++component)
          residual[linear * static_cast<std::size_t>(Model::n_vars) +
                   static_cast<std::size_t>(component)] -=
              (plus_integrated[component] - minus_integrated[component]) / metric_.cell_measure;
      }
    }
  }

 private:
  PreparedCartesianMetric<Dimension> metric_;
  Model model_;
  Reconstruction reconstruction_;
  NumericalFluxPolicy numerical_flux_;
  FluxProviderValues<Model> constant_providers_{};
};

}  // namespace pops
