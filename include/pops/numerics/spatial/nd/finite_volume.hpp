/// @file
/// @brief Axis-static numerical flux, metric divergence and CFL contracts for ND finite volume.

#pragma once

#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace pops::nd {

enum class FiniteVolumeStatus : std::uint8_t {
  Success = 0,
  NonFiniteState = 1,
  NonPositiveDensity = 2,
  NonPositivePressure = 3,
  InvalidEquationOfState = 4,
  InvalidMetric = 5,
  InvalidWaveSpeed = 6,
  NonFiniteFaceFlux = 7,
  InvalidCourantNumber = 8,
  InvalidFaceField = 9,
};

namespace finite_volume_detail {

POPS_HD constexpr FiniteVolumeStatus finite_volume_status(StateConversionStatus status) {
  switch (status) {
    case StateConversionStatus::Success:
      return FiniteVolumeStatus::Success;
    case StateConversionStatus::NonFiniteState:
      return FiniteVolumeStatus::NonFiniteState;
    case StateConversionStatus::NonPositiveDensity:
      return FiniteVolumeStatus::NonPositiveDensity;
    case StateConversionStatus::NonPositivePressure:
      return FiniteVolumeStatus::NonPositivePressure;
    case StateConversionStatus::InvalidEquationOfState:
      return FiniteVolumeStatus::InvalidEquationOfState;
  }
  return FiniteVolumeStatus::NonFiniteState;
}

POPS_HD constexpr std::uint32_t failure_reason(FiniteVolumeStatus status) {
  return UINT32_C(0x4e440000) | static_cast<std::uint32_t>(status);
}

template <class State>
POPS_HD bool finite_state(const State& state) {
  for (int component = 0; component < State::size(); ++component)
    if (!Kokkos::isfinite(state[component]))
      return false;
  return true;
}

template <class Numerical>
consteval RiemannSolverId solver_id() {
  if constexpr (requires { Numerical::solver_id; })
    return static_cast<RiemannSolverId>(Numerical::solver_id);
  return RiemannSolverId::kExternal;
}

struct NoFluxProviders {};

template <int Axis, class Model>
struct AxisPhysicalFlux {
  static_assert(Axis >= 0 && Axis < Model::dimension,
                "ND physical-flux axis is outside the conservation-law dimension");

  using State = typename Model::State;
  using ProviderPack = NoFluxProviders;
  using Trace = FaceTrace<State, ProviderPack>;
  static constexpr int n_vars = Model::n_vars;

  Model model;

  POPS_HD FluxDensity<State> evaluate(const Trace& trace, const FaceContext& face) const {
    State result = model.template flux<Axis>(trace.state);
    if (face.orientation == FaceOrientation::kNegative)
      for (int component = 0; component < n_vars; ++component)
        result[component] = -result[component];
    return {result};
  }

  POPS_HD StabilityBound stability(const Trace& trace, const FaceContext&) const {
    return {model.template max_wave_speed<Axis>(trace.state), StabilityUnit::kLengthPerTime,
            StabilityConvention::kNormalSpectralRadius};
  }

  POPS_HD void signed_wave_speeds(const Trace& trace, const FaceContext& face, Real& lower,
                                  Real& upper) const {
    model.template wave_speeds<Axis>(trace.state, lower, upper);
    if (face.orientation == FaceOrientation::kNegative) {
      const Real old_lower = lower;
      lower = -upper;
      upper = -old_lower;
    }
  }

  POPS_HD Real pressure(const State& state) const
    requires requires { model.pressure(state); }
  {
    return model.pressure(state);
  }

  POPS_HD Real contact_speed(const State& left, const State& right, Real pressure_left,
                             Real pressure_right, Real speed_left, Real speed_right,
                             const FaceContext&) const
    requires requires {
      model.template contact_speed<Axis>(left, right, pressure_left, pressure_right, speed_left,
                                         speed_right);
    }
  {
    return model.template contact_speed<Axis>(left, right, pressure_left, pressure_right,
                                              speed_left, speed_right);
  }

  POPS_HD State star_state(const State& state, Real pressure_value, Real speed, Real contact,
                           const FaceContext&) const
    requires requires { model.template star_state<Axis>(state, pressure_value, speed, contact); }
  {
    return model.template star_state<Axis>(state, pressure_value, speed, contact);
  }

  POPS_HD State roe_dissipation(const Trace& left, const Trace& right, const FaceContext&) const
    requires requires { model.template roe_dissipation<Axis>(left.state, right.state); }
  {
    return model.template roe_dissipation<Axis>(left.state, right.state);
  }
};

template <int Axis, class Model>
concept AxisConservationLaw =
    ConservationLaw<Model::dimension, Model> && Axis >= 0 && Axis < Model::dimension &&
    requires(const Model& model, const typename Model::State& state, Real& lower, Real& upper) {
      { model.template flux<Axis>(state) } -> std::same_as<typename Model::State>;
      { model.template max_wave_speed<Axis>(state) } -> std::convertible_to<Real>;
      model.template wave_speeds<Axis>(state, lower, upper);
    };

template <class Numerical, class State>
POPS_HD FluxEvaluation<State> reject_face_evaluation(FiniteVolumeStatus status) {
  return FluxEvaluation<State>::reject(failure_reason(status))
      .with_single_solver(solver_id<Numerical>());
}

template <class Numerical, class State>
POPS_HD FluxEvaluation<State> reject_face_evaluation(StateConversionStatus status) {
  return reject_face_evaluation<Numerical, State>(finite_volume_status(status));
}

template <int Axis, int Dim, class Metric>
POPS_HD Real face_measure(const Metric& metric, const Index<Dim>& cell, MetricFaceSide side) {
  static_assert(Axis >= 0 && Axis < Dim, "ND metric face axis is outside the dimension");
  typename Metric::PhysicalPoint area{};
  if (side == MetricFaceSide::Upper)
    area = metric.template oriented_face_area_vector<Axis, MetricFaceSide::Upper>(cell);
  else
    area = metric.template oriented_face_area_vector<Axis, MetricFaceSide::Lower>(cell);
  Real squared = Real(0);
  for (int physical_axis = 0; physical_axis < Metric::embedding_dimension; ++physical_axis)
    squared += area[physical_axis] * area[physical_axis];
  return Kokkos::sqrt(squared);
}

template <int Axis, int Dim, class Model>
POPS_HD void accumulate_cfl(const Model& model, const typename Model::State& state,
                            const auto& metric, const Index<Dim>& cell, Real inverse_volume,
                            Real& inverse_dt, FiniteVolumeStatus& status) {
  if (status != FiniteVolumeStatus::Success)
    return;
  const Real lower_area = face_measure<Axis>(metric, cell, MetricFaceSide::Lower);
  const Real upper_area = face_measure<Axis>(metric, cell, MetricFaceSide::Upper);
  const Real speed = model.template max_wave_speed<Axis>(state);
  if (!Kokkos::isfinite(lower_area) || !Kokkos::isfinite(upper_area) || lower_area < Real(0) ||
      upper_area < Real(0)) {
    status = FiniteVolumeStatus::InvalidMetric;
    return;
  }
  if (!Kokkos::isfinite(speed) || speed < Real(0)) {
    status = FiniteVolumeStatus::InvalidWaveSpeed;
    return;
  }
  inverse_dt += speed * Real(0.5) * (lower_area + upper_area) * inverse_volume;
  if (!Kokkos::isfinite(inverse_dt))
    status = FiniteVolumeStatus::InvalidWaveSpeed;
  if constexpr (Axis + 1 < Dim)
    accumulate_cfl<Axis + 1>(model, state, metric, cell, inverse_volume, inverse_dt, status);
}

template <int Axis, int Dim, int N, class T>
  requires std::same_as<std::remove_const_t<T>, Real>
POPS_HD void accumulate_divergence(const FaceFieldView<T, Dim>& faces, const Index<Dim>& cell,
                                   Real inverse_volume, StateVec<N>& divergence,
                                   FiniteVolumeStatus& status) {
  if (status != FiniteVolumeStatus::Success)
    return;
  if (cell[Axis] == std::numeric_limits<int>::max()) {
    status = FiniteVolumeStatus::InvalidFaceField;
    return;
  }
  Index<Dim> upper = cell;
  ++upper[Axis];
  for (int component = 0; component < N; ++component) {
    const Real lower_flux = faces.template operator()<Axis>(cell, component);
    const Real upper_flux = faces.template operator()<Axis>(upper, component);
    if (!Kokkos::isfinite(lower_flux) || !Kokkos::isfinite(upper_flux)) {
      status = FiniteVolumeStatus::NonFiniteFaceFlux;
      return;
    }
    divergence[component] += (upper_flux - lower_flux) * inverse_volume;
  }
  if constexpr (Axis + 1 < Dim)
    accumulate_divergence<Axis + 1>(faces, cell, inverse_volume, divergence, status);
}

template <int N, int Dim, class T>
  requires std::same_as<std::remove_const_t<T>, Real>
POPS_HD bool valid_face_field_layout(const FaceFieldView<T, Dim>& faces) {
  if (faces.ncomp != N || faces.cells.empty())
    return false;
  for (int axis = 0; axis < Dim; ++axis) {
    const auto& view = faces.axes[axis];
    if (view.data == nullptr || view.ncomp != N || view.origin != faces.cells.lo ||
        view.component_stride <= 0)
      return false;
    for (int direction = 0; direction < Dim; ++direction) {
      const std::int64_t expected =
          faces.cells.length(direction) + (direction == axis ? std::int64_t{1} : std::int64_t{0});
      if (view.extents[direction] != expected || view.strides[direction] <= 0)
        return false;
    }
  }
  return true;
}

}  // namespace finite_volume_detail

template <class State>
struct FiniteVolumeResult {
  State value{};
  FiniteVolumeStatus status = FiniteVolumeStatus::NonFiniteState;

  POPS_HD bool succeeded() const { return status == FiniteVolumeStatus::Success; }
};

struct CellCflResult {
  Real inverse_dt = Real(0);
  FiniteVolumeStatus status = FiniteVolumeStatus::InvalidWaveSpeed;

  POPS_HD bool succeeded() const { return status == FiniteVolumeStatus::Success; }
};

struct TimeStepResult {
  Real value = std::numeric_limits<Real>::quiet_NaN();
  FiniteVolumeStatus status = FiniteVolumeStatus::InvalidCourantNumber;

  POPS_HD bool succeeded() const { return status == FiniteVolumeStatus::Success; }
};

/// Context for the lower or upper geometric face, expressed in the canonical positive logical
/// orientation used by FaceField.  ``Side`` selects the metric location; it does not turn a stored
/// positive-axis flux into an outward flux for one particular cell.
template <int Axis, MetricFaceSide Side, int Dim, class Metric>
  requires PreparedMetricProvider<Dim, Metric>
POPS_HD FaceContext metric_face_context(const Metric& metric, const Index<Dim>& cell) {
  static_assert(Axis >= 0 && Axis < Dim, "ND metric face axis is outside the dimension");
  return FaceContext::axis_aligned(Axis,
                                   finite_volume_detail::face_measure<Axis>(metric, cell, Side),
                                   FaceOrientation::kPositive, metric.cell_measure(cell));
}

/// Evaluate one face with a compile-time normal axis.  The model is checked for admissibility
/// before the selected Riemann policy sees either trace; a failed conversion therefore cannot
/// publish a finite-looking flux or stability bound.
template <int Axis, class Numerical, class Model>
  requires finite_volume_detail::AxisConservationLaw<Axis, Model>
POPS_HD FluxEvaluation<typename Model::State> evaluate_axis_flux(
    const Numerical& numerical, const Model& model, const typename Model::State& left,
    const typename Model::State& right, Real face_measure = Real(1), Real cell_measure = Real(1)) {
  using Physical = finite_volume_detail::AxisPhysicalFlux<Axis, Model>;
  static_assert(NumericalFlux<Numerical, Physical>,
                "ND face evaluation requires a compatible typed numerical flux");
  constexpr RiemannSolverId solver = finite_volume_detail::solver_id<Numerical>();

  const StateConversionStatus left_status = model.admissibility(left);
  if (left_status != StateConversionStatus::Success)
    return finite_volume_detail::reject_face_evaluation<Numerical, typename Model::State>(
        left_status);
  const StateConversionStatus right_status = model.admissibility(right);
  if (right_status != StateConversionStatus::Success)
    return finite_volume_detail::reject_face_evaluation<Numerical, typename Model::State>(
        right_status);
  if (!Kokkos::isfinite(face_measure) || !Kokkos::isfinite(cell_measure) ||
      !(face_measure > Real(0)) || !(cell_measure > Real(0)))
    return finite_volume_detail::reject_face_evaluation<Numerical, typename Model::State>(
        FiniteVolumeStatus::InvalidMetric);

  const Physical physical{model};
  const typename Physical::Trace left_trace{left, {}};
  const typename Physical::Trace right_trace{right, {}};
  const FaceContext face =
      FaceContext::axis_aligned(Axis, face_measure, FaceOrientation::kPositive, cell_measure);
  auto result = numerical(physical, left_trace, right_trace, face);
  if (result.requested_solver == RiemannSolverId::kUnspecified)
    result = result.with_single_solver(solver);
  if (result.succeeded() && !finite_volume_detail::finite_state(result.checked_density().value))
    return finite_volume_detail::reject_face_evaluation<Numerical, typename Model::State>(
        FiniteVolumeStatus::NonFiniteFaceFlux);
  return result;
}

template <int Axis, MetricFaceSide Side, class Numerical, int Dim, class Model, class Metric>
  requires(Dim == Model::dimension && PreparedMetricProvider<Dim, Metric> &&
           finite_volume_detail::AxisConservationLaw<Axis, Model>)
POPS_HD FluxEvaluation<typename Model::State> evaluate_metric_face_flux(
    const Numerical& numerical, const Model& model, const typename Model::State& left,
    const typename Model::State& right, const Metric& metric, const Index<Dim>& cell) {
  if (!metric.identity().domain.contains(cell))
    return finite_volume_detail::reject_face_evaluation<Numerical, typename Model::State>(
        FiniteVolumeStatus::InvalidMetric);
  const FaceContext face = metric_face_context<Axis, Side>(metric, cell);
  return evaluate_axis_flux<Axis>(numerical, model, left, right, face.face_measure,
                                  face.cell_measure);
}

/// Conservative divergence of already integrated, positive-axis face fluxes.  Geometry enters
/// exactly once through the prepared cell measure; face integration is owned by
/// evaluate_metric_face_flux + apply_face_measure.
template <int N, int Dim, class Metric, class T>
  requires(std::same_as<std::remove_const_t<T>, Real> && PreparedMetricProvider<Dim, Metric>)
POPS_HD FiniteVolumeResult<StateVec<N>> conservative_residual(
    const Metric& metric, const FaceFieldView<T, Dim>& integrated_fluxes, const Index<Dim>& cell) {
  FiniteVolumeResult<StateVec<N>> result{};
  if (!integrated_fluxes.cells.contains(cell) ||
      !finite_volume_detail::valid_face_field_layout<N>(integrated_fluxes)) {
    result.status = FiniteVolumeStatus::InvalidFaceField;
    return result;
  }
  if (!metric.identity().domain.contains(integrated_fluxes.cells)) {
    result.status = FiniteVolumeStatus::InvalidMetric;
    return result;
  }
  const Real volume = metric.cell_measure(cell);
  if (!Kokkos::isfinite(volume) || !(volume > Real(0))) {
    result.status = FiniteVolumeStatus::InvalidMetric;
    return result;
  }
  result.status = FiniteVolumeStatus::Success;
  finite_volume_detail::accumulate_divergence<0>(integrated_fluxes, cell, Real(1) / volume,
                                                 result.value, result.status);
  if (!result.succeeded()) {
    result.value = {};
    return result;
  }
  for (int component = 0; component < N; ++component)
    result.value[component] = -result.value[component];
  return result;
}

template <int Dim, class Model, class Metric>
  requires(Dim == Model::dimension && PreparedMetricProvider<Dim, Metric> &&
           ConservationLaw<Dim, Model>)
POPS_HD CellCflResult cell_cfl_bound(const Model& model, const typename Model::State& state,
                                     const Metric& metric, const Index<Dim>& cell) {
  CellCflResult result{};
  if (!metric.identity().domain.contains(cell)) {
    result.status = FiniteVolumeStatus::InvalidMetric;
    return result;
  }
  const auto state_status = model.admissibility(state);
  if (state_status != StateConversionStatus::Success) {
    result.status = finite_volume_detail::finite_volume_status(state_status);
    return result;
  }
  const Real volume = metric.cell_measure(cell);
  if (!Kokkos::isfinite(volume) || !(volume > Real(0))) {
    result.status = FiniteVolumeStatus::InvalidMetric;
    return result;
  }
  result.status = FiniteVolumeStatus::Success;
  finite_volume_detail::accumulate_cfl<0>(model, state, metric, cell, Real(1) / volume,
                                          result.inverse_dt, result.status);
  if (!result.succeeded())
    result.inverse_dt = Real(0);
  return result;
}

template <int Dim, class Model, class Metric>
  requires(Dim == Model::dimension && PreparedMetricProvider<Dim, Metric> &&
           ConservationLaw<Dim, Model>)
POPS_HD TimeStepResult cell_time_step(const Model& model, const typename Model::State& state,
                                      const Metric& metric, const Index<Dim>& cell, Real courant) {
  TimeStepResult result{};
  if (!Kokkos::isfinite(courant) || !(courant > Real(0)))
    return result;
  const CellCflResult bound = cell_cfl_bound<Dim>(model, state, metric, cell);
  result.status = bound.status;
  if (!bound.succeeded())
    return result;
  result.value = bound.inverse_dt == Real(0) ? std::numeric_limits<Real>::infinity()
                                             : courant / bound.inverse_dt;
  return result;
}

}  // namespace pops::nd
