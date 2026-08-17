/// @file
/// @brief Prepared ND Cartesian transport restricted by a cell-centred active mask.

#pragma once

#include <pops/numerics/spatial/operators/cartesian_operator.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace pops::nd {

/// Boundary faces whose flux is owned by another prepared topology provider.
///
/// Axis and dimension are type properties.  The descriptor never decodes an x/y convention and
/// cannot become an alternate face-field authority.
template <int Dim>
struct BoundaryFaceOmission {
  Box<Dim> domain{};
  bool lower[Dim]{};
  bool upper[Dim]{};

  template <int Axis>
  POPS_HD bool omits(const FaceIndex<Dim, Axis>& face) const {
    static_assert(Axis >= 0 && Axis < Dim);
    return (lower[Axis] && face.coordinate[Axis] == domain.lo[Axis]) ||
           (upper[Axis] && face.coordinate[Axis] == domain.hi[Axis] + 1);
  }
};

namespace masked_operator_detail {

template <int Axis, ReconstructionVariables Variables, int Dim, class Model, class Metric,
          class Reconstruction, class NumericalFlux, class ProviderStorage>
struct MaterializeMaskedFaceFlux {
  Model model;
  Metric metric;
  Reconstruction reconstruction;
  NumericalFlux numerical_flux;
  Real positivity_floor = Real(0);
  int positivity_component = 0;
  FieldView<const Real, Dim> state{};
  ProviderStorage providers;
  FieldView<const Real, Dim> active_cells{};
  FaceFieldView<Real, Dim> integrated_fluxes{};
  FaceFieldView<Real, Dim> statuses{};
  BoundaryFaceOmission<Dim> omission{};

  POPS_HD void clear(const FaceIndex<Dim, Axis>& face, FiniteVolumeStatus status) const {
    for (int component = 0; component < Model::n_vars; ++component)
      integrated_fluxes.template operator()<Axis>(face.coordinate, component) = Real(0);
    statuses.template operator()<Axis>(face.coordinate) = static_cast<Real>(status);
  }

  POPS_HD void operator()(const FaceIndex<Dim, Axis>& face) const {
    Index<Dim> left_cell = face.coordinate;
    --left_cell[Axis];
    const Index<Dim> right_cell = face.coordinate;
    if (omission.template omits<Axis>(face) || active_cells(left_cell) < Real(0.5) ||
        active_cells(right_cell) < Real(0.5)) {
      clear(face, FiniteVolumeStatus::Success);
      return;
    }

    FaceContext context{};
    if (face.coordinate[Axis] == integrated_fluxes.cells.lo[Axis])
      context = metric_face_context<Axis, MetricFaceSide::Lower>(metric, right_cell);
    else
      context = metric_face_context<Axis, MetricFaceSide::Upper>(metric, left_cell);
    if (!Kokkos::isfinite(context.face_measure) || context.face_measure < Real(0) ||
        !Kokkos::isfinite(context.cell_measure) || !(context.cell_measure > Real(0))) {
      clear(face, FiniteVolumeStatus::InvalidMetric);
      return;
    }
    if (!(context.face_measure > Real(0))) {
      clear(face, FiniteVolumeStatus::Success);
      return;
    }

    auto traces = reconstruct_face_pair<Axis, Variables>(model, state, face, reconstruction);
    if (traces.left_status != StateConversionStatus::Success) {
      clear(face, finite_volume_detail::finite_volume_status(traces.left_status));
      return;
    }
    if (traces.right_status != StateConversionStatus::Success) {
      clear(face, finite_volume_detail::finite_volume_status(traces.right_status));
      return;
    }
    if (positivity_floor > Real(0)) {
      if (traces.left[positivity_component] < positivity_floor) {
        traces.left = load_state<Model>(state, left_cell);
        traces.left_status = model.admissibility(traces.left);
      }
      if (traces.right[positivity_component] < positivity_floor) {
        traces.right = load_state<Model>(state, right_cell);
        traces.right_status = model.admissibility(traces.right);
      }
      if (traces.left_status != StateConversionStatus::Success) {
        clear(face, finite_volume_detail::finite_volume_status(traces.left_status));
        return;
      }
      if (traces.right_status != StateConversionStatus::Success) {
        clear(face, finite_volume_detail::finite_volume_status(traces.right_status));
        return;
      }
    }

    const auto evaluation =
        evaluate_numerical_flux_at(numerical_flux, model, traces.left, providers, left_cell,
                                   traces.right, providers, right_cell, context);
    if (!evaluation.succeeded()) {
      clear(face, FiniteVolumeStatus::InvalidWaveSpeed);
      return;
    }
    const auto integrated = apply_face_measure(evaluation.checked_density(), context);
    for (int component = 0; component < Model::n_vars; ++component) {
      if (!Kokkos::isfinite(integrated.value[component])) {
        clear(face, FiniteVolumeStatus::NonFiniteFaceFlux);
        return;
      }
    }
    for (int component = 0; component < Model::n_vars; ++component)
      integrated_fluxes.template operator()<Axis>(face.coordinate, component) =
          integrated.value[component];
    statuses.template operator()<Axis>(face.coordinate) =
        static_cast<Real>(FiniteVolumeStatus::Success);
  }
};

template <int Axis, ReconstructionVariables Variables, int Dim, class Model, class Metric,
          class Reconstruction, class NumericalFlux, class ProviderStorage, class MemorySpace>
void materialize_axes(const Model& model, const Metric& metric,
                      const Reconstruction& reconstruction, const NumericalFlux& numerical_flux,
                      Real positivity_floor, int positivity_component,
                      const Fab<Dim, MemorySpace>& state, const Fab<Dim, MemorySpace>& active_cells,
                      const ProviderStorage& providers,
                      FaceField<Dim, MemorySpace>& integrated_fluxes,
                      FaceField<Dim, MemorySpace>& statuses,
                      const BoundaryFaceOmission<Dim>& omission) {
  for_each_face<Axis>(state.box(),
                      MaterializeMaskedFaceFlux<Axis, Variables, Dim, Model, Metric, Reconstruction,
                                                NumericalFlux, ProviderStorage>{
                          model, metric, reconstruction, numerical_flux, positivity_floor,
                          positivity_component, state.view(), providers, active_cells.view(),
                          integrated_fluxes.view(), statuses.view(), omission});
  if constexpr (Axis + 1 < Dim)
    materialize_axes<Axis + 1, Variables>(
        model, metric, reconstruction, numerical_flux, positivity_floor, positivity_component,
        state, active_cells, providers, integrated_fluxes, statuses, omission);
}

template <int Dim, class Metric, int N>
struct MaterializeMaskedResidual {
  Metric metric;
  FaceFieldView<const Real, Dim> integrated_fluxes{};
  FieldView<const Real, Dim> active_cells{};
  FieldView<Real, Dim> candidate{};
  FieldView<Real, Dim> statuses{};

  POPS_HD void operator()(const Index<Dim>& cell) const {
    if (active_cells(cell) < Real(0.5)) {
      for (int component = 0; component < N; ++component)
        candidate(cell, component) = Real(0);
      statuses(cell) = static_cast<Real>(FiniteVolumeStatus::Success);
      return;
    }
    const auto result = conservative_residual<N>(metric, integrated_fluxes, cell);
    if (!result.succeeded()) {
      for (int component = 0; component < N; ++component)
        candidate(cell, component) = Real(0);
      statuses(cell) = static_cast<Real>(result.status);
      return;
    }
    for (int component = 0; component < N; ++component)
      candidate(cell, component) = result.value[component];
    statuses(cell) = static_cast<Real>(FiniteVolumeStatus::Success);
  }
};

template <int Dim, class MemorySpace>
void require_active_mask(const Fab<Dim, MemorySpace>& state,
                         const Fab<Dim, MemorySpace>& active_cells) {
  if (active_cells.ncomp() != 1 || !(active_cells.box() == state.box()) ||
      !active_cells.grown_box().contains(state.box().grow(1)))
    throw std::invalid_argument(
        "prepared ND masked transport requires a scalar mask with one ghost layer");
}

template <int Dim, class MemorySpace>
void require_same_multifab_layout(const MultiFab<Dim, MemorySpace>& left,
                                  const MultiFab<Dim, MemorySpace>& right, const char* operation) {
  if (!(left.layout() == right.layout()) || !(left.distribution() == right.distribution()) ||
      !(left.local_rank() == right.local_rank()) || left.local_size() != right.local_size())
    throw std::invalid_argument(operation);
}

}  // namespace masked_operator_detail

/// Prepared Cartesian capability whose face set is gated by a ranked active-cell field.
///
/// Closed faces are rejected before reconstruction, so inactive sentinel storage never reaches a
/// primitive conversion or Riemann solver.  Open faces and the conservative divergence remain the
/// exact canonical ND implementation.
template <int Dim, class Model, class Metric, class Reconstruction = NoSlope,
          class NumericalFlux = RusanovFlux,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative>
  requires(ConservationLaw<Dim, Model> && PreparedMetricProvider<Dim, Metric> &&
           ReconstructionPolicy<Reconstruction>)
class PreparedMaskedCartesianOperator {
 public:
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Model::n_vars;

  PreparedMaskedCartesianOperator(Model model, Metric metric, Reconstruction reconstruction = {},
                                  NumericalFlux numerical_flux = {},
                                  Real positivity_floor = Real(0))
      : base_(model, metric, reconstruction, numerical_flux, positivity_floor),
        reconstruction_(std::move(reconstruction)),
        numerical_flux_(std::move(numerical_flux)),
        positivity_floor_(positivity_floor),
        positivity_component_(
            cartesian_operator_detail::resolve_positivity_component<Model>(positivity_floor)) {}

  const Model& model() const noexcept { return base_.model(); }
  const Metric& metric() const noexcept { return base_.metric(); }

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state,
                         const Fab<Dim, MemorySpace>& active_cells, Fab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const
    requires(flux_provider_count<Model> == 0)
  {
    assemble_residual_(state, cartesian_operator_detail::ProviderFreeStorage<Dim>{}, active_cells,
                       residual, omission);
  }

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state, const Fab<Dim, MemorySpace>& providers,
                         const Fab<Dim, MemorySpace>& active_cells, Fab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const {
    if (!(providers.box() == state.box()) || providers.ncomp() < flux_provider_count<Model> ||
        !providers.grown_box().contains(state.box().grow(1)))
      throw std::invalid_argument(
          "prepared ND masked provider patch does not cover model-qualified face traces");
    assemble_residual_(state, providers.view(), active_cells, residual, omission);
  }

  /// Plan-mapped provider route.  `ProviderStorageView` is a compact immutable consumer map,
  /// not an auxiliary field layout or a fallback storage allocation.
  template <class MemorySpace, int Count>
  void assemble_residual(const Fab<Dim, MemorySpace>& state,
                         const ProviderStorageView<Dim, Count>& providers,
                         const Fab<Dim, MemorySpace>& active_cells, Fab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const
    requires(Count == flux_provider_count<Model>)
  {
    assemble_residual_(state, providers, active_cells, residual, omission);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         MultiFab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const
    requires(flux_provider_count<Model> == 0)
  {
    require_multifab_layouts_(state, active_cells, residual);
    MultiFab<Dim, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                         residual.local_rank(), n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), active_cells.fab(local), candidate.fab(local), omission);
    publish_(state, candidate, residual);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& providers,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         MultiFab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const {
    require_multifab_layouts_(state, active_cells, residual);
    masked_operator_detail::require_same_multifab_layout(
        state, providers, "prepared ND masked state and provider layouts differ");
    if (providers.ncomp() < flux_provider_count<Model>)
      throw std::invalid_argument("prepared ND masked provider field is too narrow");
    MultiFab<Dim, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                         residual.local_rank(), n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), providers.fab(local), active_cells.fab(local),
                        candidate.fab(local), omission);
    publish_(state, candidate, residual);
  }

 private:
  template <class MemorySpace, class ProviderStorage>
  void assemble_residual_(const Fab<Dim, MemorySpace>& state, const ProviderStorage& providers,
                          const Fab<Dim, MemorySpace>& active_cells,
                          Fab<Dim, MemorySpace>& residual,
                          BoundaryFaceOmission<Dim> omission) const {
    if constexpr (DiffusiveModel<Model>)
      throw std::invalid_argument(
          "prepared ND masked transport does not support Fickian diffusion without EB face "
          "geometry");
    if (!base_.domain().contains(state.box()))
      throw std::invalid_argument("prepared ND masked state lies outside the metric domain");
    require_reconstruction_storage<Reconstruction>(state, state.box(), n_vars);
    masked_operator_detail::require_active_mask(state, active_cells);
    cartesian_operator_detail::require_residual_output(state, residual, n_vars);

    FaceField<Dim, MemorySpace> integrated_fluxes(state.box(), n_vars);
    FaceField<Dim, MemorySpace> statuses(state.box(), 1);
    masked_operator_detail::materialize_axes<0, Variables>(
        model(), metric(), reconstruction_, numerical_flux_, positivity_floor_,
        positivity_component_, state, active_cells, providers, integrated_fluxes, statuses,
        omission);
    const Real failure = cartesian_operator_detail::maximum_face_status<0>(statuses);
    if (failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error("prepared ND masked face evaluation refused publication");

    Fab<Dim, MemorySpace> candidate(state.box(), n_vars);
    Fab<Dim, MemorySpace> cell_statuses(state.box(), 1);
    for_each_cell(
        state.box(),
        masked_operator_detail::MaterializeMaskedResidual<Dim, Metric, n_vars>{
            metric(), static_cast<const FaceField<Dim, MemorySpace>&>(integrated_fluxes).view(),
            active_cells.view(), candidate.view(), cell_statuses.view()});
    const Real cell_failure = for_each_cell_reduce_max(
        state.box(), cartesian_operator_detail::FieldStatusMaximum<Dim>{
                         static_cast<const Fab<Dim, MemorySpace>&>(cell_statuses).view()});
    if (cell_failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error("prepared ND masked residual refused publication");
    for_each_cell(state.box(), cartesian_operator_detail::CopyCellField<Dim>{
                                   static_cast<const Fab<Dim, MemorySpace>&>(candidate).view(),
                                   residual.view(), n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void require_multifab_layouts_(const MultiFab<Dim, MemorySpace>& state,
                                 const MultiFab<Dim, MemorySpace>& active_cells,
                                 const MultiFab<Dim, MemorySpace>& residual) const {
    masked_operator_detail::require_same_multifab_layout(
        state, active_cells, "prepared ND masked state and mask layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, residual, "prepared ND masked state and residual layouts differ");
    if (state.ncomp() != n_vars || active_cells.ncomp() != 1 || residual.ncomp() != n_vars ||
        state.shares_storage_with(residual))
      throw std::invalid_argument("prepared ND masked MultiFab components differ or alias");
  }

  template <class MemorySpace>
  void publish_(const MultiFab<Dim, MemorySpace>& state,
                const MultiFab<Dim, MemorySpace>& candidate,
                MultiFab<Dim, MemorySpace>& residual) const {
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    cartesian_operator_detail::CopyCellField<Dim>{
                        static_cast<const Fab<Dim, MemorySpace>&>(candidate.fab(local)).view(),
                        residual.fab(local).view(), n_vars});
    device_fence();
  }

  PreparedCartesianOperator<Dim, Model, Metric, Reconstruction, NumericalFlux, Variables> base_;
  Reconstruction reconstruction_;
  NumericalFlux numerical_flux_;
  Real positivity_floor_ = Real(0);
  int positivity_component_ = 0;
};

template <int Dim, class Model, class Metric, class Reconstruction = NoSlope,
          class NumericalFlux = RusanovFlux,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative>
auto prepare_masked_cartesian_operator(Model model, Metric metric,
                                       Reconstruction reconstruction = {},
                                       NumericalFlux numerical_flux = {},
                                       Real positivity_floor = Real(0)) {
  return PreparedMaskedCartesianOperator<Dim, Model, Metric, Reconstruction, NumericalFlux,
                                         Variables>(std::move(model), std::move(metric),
                                                    std::move(reconstruction),
                                                    std::move(numerical_flux), positivity_floor);
}

}  // namespace pops::nd
