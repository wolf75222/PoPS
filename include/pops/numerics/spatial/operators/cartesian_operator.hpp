/// @file
/// @brief Prepared compile-time-ranked hyperbolic face flux and conservative residual operator.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/spatial/nd/finite_volume.hpp>
#include <pops/numerics/spatial/nd/reconstruction.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::nd {

/// Per-field storage for allocation-free prepared Cartesian face and divergence evaluation.
/// One generated level/block owns one instance and serializes its use with the surrounding
/// prepared evaluation session; no process-global or operator-static scratch is shared.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedCartesianOperatorScratch {
 public:
  explicit PreparedCartesianOperatorScratch(const MultiFab<Dim, MemorySpace>& prototype)
      : residual_candidate_(prototype.layout(), prototype.distribution(), prototype.local_rank(),
                            prototype.ncomp(), prototype.ghosts()),
        residual_status_(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                         prototype.ghosts()) {
    face_candidates_.reserve(prototype.local_size());
    face_statuses_.reserve(prototype.local_size());
    for (std::size_t local = 0; local < prototype.local_size(); ++local) {
      face_candidates_.emplace_back(prototype.box(local), prototype.ncomp());
      face_statuses_.emplace_back(prototype.box(local), 1);
    }
  }

  void require_layout(const MultiFab<Dim, MemorySpace>& field) const {
    if (field.layout() != residual_candidate_.layout() ||
        field.distribution() != residual_candidate_.distribution() ||
        field.local_rank() != residual_candidate_.local_rank() ||
        field.local_size() != residual_candidate_.local_size() ||
        field.ncomp() != residual_candidate_.ncomp() ||
        field.ghosts() != residual_candidate_.ghosts() ||
        face_candidates_.size() != field.local_size() ||
        face_statuses_.size() != field.local_size())
      throw std::invalid_argument(
          "prepared Cartesian operator scratch differs from its authenticated field layout");
  }

  FaceField<Dim, MemorySpace>& face_candidate(std::size_t local) {
    return face_candidates_.at(local);
  }
  FaceField<Dim, MemorySpace>& face_status(std::size_t local) { return face_statuses_.at(local); }
  MultiFab<Dim, MemorySpace>& residual_candidate() noexcept { return residual_candidate_; }
  MultiFab<Dim, MemorySpace>& residual_status() noexcept { return residual_status_; }

 private:
  std::vector<FaceField<Dim, MemorySpace>> face_candidates_;
  std::vector<FaceField<Dim, MemorySpace>> face_statuses_;
  MultiFab<Dim, MemorySpace> residual_candidate_;
  MultiFab<Dim, MemorySpace> residual_status_;
};

namespace cartesian_operator_detail {

template <int Dim>
struct FieldStatusMaximum {
  FieldView<const Real, Dim> status{};

  POPS_HD Real operator()(const Index<Dim>& index) const { return status(index); }
};

/// Device-clean provider storage for laws that explicitly declare no qualified provider rows.
/// It is selected by capability at compile time and never stands in for a missing authored field.
template <int Dim>
struct ProviderFreeStorage {
  POPS_HD Real operator()(const Index<Dim>&, int) const { return Real(0); }
};

template <class Model>
int resolve_positivity_component(Real floor) {
  if (!(floor > Real(0)))
    return 0;
  if (!std::isfinite(floor))
    throw std::invalid_argument("prepared ND positivity floor must be finite");
  if constexpr (requires { Model::conservative_vars(); }) {
    const int component = Model::conservative_vars().index_of(VariableRole::Density);
    if (component >= 0)
      return component;
    throw std::invalid_argument("prepared ND positivity requires a conservative Density variable");
  }
  throw std::invalid_argument(
      "prepared ND positivity requires conservative-variable introspection");
}

template <int Axis, int Dim>
struct CopyFaceAxis {
  FaceFieldView<const Real, Dim> source{};
  FaceFieldView<Real, Dim> destination{};
  int ncomp = 0;

  POPS_HD void operator()(const FaceIndex<Dim, Axis>& face) const {
    for (int component = 0; component < ncomp; ++component)
      destination.template operator()<Axis>(face.coordinate, component) =
          source.template operator()<Axis>(face.coordinate, component);
  }
};

template <int Dim>
struct CopyCellField {
  FieldView<const Real, Dim> source{};
  FieldView<Real, Dim> destination{};
  int ncomp = 0;

  POPS_HD void operator()(const Index<Dim>& cell) const {
    for (int component = 0; component < ncomp; ++component)
      destination(cell, component) = source(cell, component);
  }
};

template <int Axis, ReconstructionVariables Variables, int Dim, class Model, class Metric,
          class Reconstruction, class NumericalFlux, class ProviderStorage>
struct MaterializeFaceFlux {
  Model model;
  Metric metric;
  Reconstruction reconstruction;
  NumericalFlux numerical_flux;
  Real positivity_floor = Real(0);
  int positivity_component = 0;
  FieldView<const Real, Dim> state{};
  ProviderStorage providers;
  FaceFieldView<Real, Dim> integrated_fluxes{};
  FaceFieldView<Real, Dim> statuses{};

  POPS_HD void fail(const FaceIndex<Dim, Axis>& face, FiniteVolumeStatus status) const {
    for (int component = 0; component < Model::n_vars; ++component)
      integrated_fluxes.template operator()<Axis>(face.coordinate, component) = Real(0);
    statuses.template operator()<Axis>(face.coordinate) = static_cast<Real>(status);
  }

  POPS_HD void operator()(const FaceIndex<Dim, Axis>& face) const {
    auto traces = reconstruct_face_pair<Axis, Variables>(model, state, face, reconstruction);
    if (traces.left_status != StateConversionStatus::Success) {
      fail(face, finite_volume_detail::finite_volume_status(traces.left_status));
      return;
    }
    if (traces.right_status != StateConversionStatus::Success) {
      fail(face, finite_volume_detail::finite_volume_status(traces.right_status));
      return;
    }

    Index<Dim> left_cell = face.coordinate;
    --left_cell[Axis];
    const Index<Dim> right_cell = face.coordinate;
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
        fail(face, finite_volume_detail::finite_volume_status(traces.left_status));
        return;
      }
      if (traces.right_status != StateConversionStatus::Success) {
        fail(face, finite_volume_detail::finite_volume_status(traces.right_status));
        return;
      }
    }
    FaceContext context{};
    if (face[Axis] == integrated_fluxes.cells.lo[Axis])
      context = metric_face_context<Axis, MetricFaceSide::Lower>(metric, right_cell);
    else
      context = metric_face_context<Axis, MetricFaceSide::Upper>(metric, left_cell);

    const auto evaluation =
        evaluate_numerical_flux_at(numerical_flux, model, traces.left, providers, left_cell,
                                   traces.right, providers, right_cell, context);
    if (!evaluation.succeeded()) {
      fail(face, FiniteVolumeStatus::InvalidWaveSpeed);
      return;
    }
    const auto integrated = apply_face_measure(evaluation.checked_density(), context);
    for (int component = 0; component < Model::n_vars; ++component) {
      if (!Kokkos::isfinite(integrated.value[component])) {
        fail(face, FiniteVolumeStatus::NonFiniteFaceFlux);
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

template <int Dim, class Metric, int N>
struct MaterializeResidual {
  Metric metric;
  FaceFieldView<const Real, Dim> integrated_fluxes{};
  FieldView<Real, Dim> candidate{};
  FieldView<Real, Dim> statuses{};

  POPS_HD void operator()(const Index<Dim>& cell) const {
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

template <int Axis, ReconstructionVariables Variables, int Dim, class Model, class Metric,
          class Reconstruction, class NumericalFlux, class ProviderStorage, class MemorySpace>
void materialize_axes(const Model& model, const Metric& metric,
                      const Reconstruction& reconstruction, const NumericalFlux& numerical_flux,
                      Real positivity_floor, int positivity_component,
                      const Fab<Dim, MemorySpace>& state, const ProviderStorage& providers,
                      FaceField<Dim, MemorySpace>& integrated_fluxes,
                      FaceField<Dim, MemorySpace>& statuses) {
  for_each_face<Axis>(
      state.box(),
      MaterializeFaceFlux<Axis, Variables, Dim, Model, Metric, Reconstruction, NumericalFlux,
                          ProviderStorage>{model, metric, reconstruction, numerical_flux,
                                           positivity_floor, positivity_component, state.view(),
                                           providers, integrated_fluxes.view(), statuses.view()});
  if constexpr (Axis + 1 < Dim)
    materialize_axes<Axis + 1, Variables>(model, metric, reconstruction, numerical_flux,
                                          positivity_floor, positivity_component, state, providers,
                                          integrated_fluxes, statuses);
}

template <int Axis, int Dim, class MemorySpace>
Real maximum_face_status(const FaceField<Dim, MemorySpace>& statuses) {
  const auto view = statuses.template field<Axis>().view();
  const Real local = for_each_cell_reduce_max(statuses.template field<Axis>().box(),
                                              FieldStatusMaximum<Dim>{view});
  if constexpr (Axis + 1 < Dim) {
    const Real remaining = maximum_face_status<Axis + 1>(statuses);
    return local > remaining ? local : remaining;
  }
  return local;
}

template <int Axis, int Dim, class MemorySpace>
void copy_face_axes(const FaceField<Dim, MemorySpace>& source,
                    FaceField<Dim, MemorySpace>& destination, int ncomp) {
  for_each_face<Axis>(source.cell_box(),
                      CopyFaceAxis<Axis, Dim>{source.view(), destination.view(), ncomp});
  if constexpr (Axis + 1 < Dim)
    copy_face_axes<Axis + 1>(source, destination, ncomp);
}

template <int Dim, class MemorySpace>
void require_face_output(const FaceField<Dim, MemorySpace>& output, const Box<Dim>& cells,
                         int nvars) {
  if (!(output.cell_box() == cells) || output.ncomp() != nvars)
    throw std::invalid_argument(
        "prepared ND hyperbolic face output does not match the patch and conservation law");
}

template <int Dim, class MemorySpace>
void require_residual_shape(const Fab<Dim, MemorySpace>& residual, const Box<Dim>& cells,
                            int nvars) {
  if (!(residual.box() == cells) || residual.ncomp() != nvars)
    throw std::invalid_argument(
        "prepared ND hyperbolic residual does not match the state patch and conservation law");
}

template <int Dim, class MemorySpace>
void require_residual_output(const Fab<Dim, MemorySpace>& state,
                             const Fab<Dim, MemorySpace>& residual, int nvars) {
  require_residual_shape(residual, state.box(), nvars);
  if (state.view().data == residual.view().data)
    throw std::invalid_argument("prepared ND hyperbolic state and residual may not alias storage");
}

}  // namespace cartesian_operator_detail

/// One immutable hyperbolic numerical specialization over a global prepared metric.
///
/// `Dim`, the conservation law, reconstruction protocol, variables and Riemann solver are type
/// properties.  A call may operate on any local patch contained in the metric domain; the patch
/// carries the exact reconstruction ghosts and owns one axis-indexed FaceField.
template <int Dim, class Model, class Metric, class Reconstruction = NoSlope,
          class NumericalFlux = RusanovFlux,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative>
  requires(ConservationLaw<Dim, Model> && PreparedMetricProvider<Dim, Metric> &&
           ReconstructionPolicy<Reconstruction>)
class PreparedCartesianOperator {
 public:
  static_assert(stencil_envelope_fits_storage<Reconstruction>);
  static_assert(std::is_trivially_copyable_v<Reconstruction>);
  static_assert(std::is_trivially_copyable_v<NumericalFlux>);

  using State = typename Model::State;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Model::n_vars;
  static constexpr int ghost_depth = Reconstruction::n_ghost;
  static constexpr ReconstructionVariables reconstruction_variables = Variables;

  PreparedCartesianOperator(Model model, Metric metric, Reconstruction reconstruction = {},
                            NumericalFlux numerical_flux = {}, Real positivity_floor = Real(0))
      : model_(std::move(model)),
        metric_(std::move(metric)),
        reconstruction_(std::move(reconstruction)),
        numerical_flux_(std::move(numerical_flux)),
        positivity_floor_(positivity_floor),
        positivity_component_(
            cartesian_operator_detail::resolve_positivity_component<Model>(positivity_floor)) {
    if (metric_.identity().domain.empty())
      throw std::invalid_argument("prepared ND hyperbolic metric domain must be non-empty");
  }

  const Model& model() const noexcept { return model_; }
  const Metric& metric() const noexcept { return metric_; }
  Box<Dim> domain() const noexcept { return metric_.identity().domain; }

  template <class MemorySpace>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               FaceField<Dim, MemorySpace>& output) const
    requires(flux_provider_count<Model> == 0)
  {
    require_state_patch_(state);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);

    FaceField<Dim, MemorySpace> candidate(state.box(), n_vars);
    FaceField<Dim, MemorySpace> statuses(state.box(), 1);
    materialize_face_fluxes(state, output, candidate, statuses);
  }

  template <class MemorySpace>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               FaceField<Dim, MemorySpace>& output,
                               FaceField<Dim, MemorySpace>& candidate,
                               FaceField<Dim, MemorySpace>& statuses) const
    requires(flux_provider_count<Model> == 0)
  {
    if (&output == &candidate || &output == &statuses || &candidate == &statuses)
      throw std::invalid_argument("prepared ND hyperbolic face output and scratch must not alias");
    require_state_patch_(state);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(candidate, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(statuses, state.box(), 1);
    cartesian_operator_detail::materialize_axes<0, Variables>(
        model_, metric_, reconstruction_, numerical_flux_, positivity_floor_, positivity_component_,
        state, cartesian_operator_detail::ProviderFreeStorage<Dim>{}, candidate, statuses);
    const Real failure = cartesian_operator_detail::maximum_face_status<0>(statuses);
    if (failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error("prepared ND hyperbolic face evaluation refused publication");

    cartesian_operator_detail::copy_face_axes<0>(candidate, output, n_vars);
    device_fence();
  }

  template <class MemorySpace>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               const Fab<Dim, MemorySpace>& providers,
                               FaceField<Dim, MemorySpace>& output) const {
    require_state_patch_(state);
    require_provider_patch_(state, providers);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);

    FaceField<Dim, MemorySpace> candidate(state.box(), n_vars);
    FaceField<Dim, MemorySpace> statuses(state.box(), 1);
    materialize_face_fluxes(state, providers, output, candidate, statuses);
  }

  template <class MemorySpace>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               const Fab<Dim, MemorySpace>& providers,
                               FaceField<Dim, MemorySpace>& output,
                               FaceField<Dim, MemorySpace>& candidate,
                               FaceField<Dim, MemorySpace>& statuses) const {
    if (&output == &candidate || &output == &statuses || &candidate == &statuses)
      throw std::invalid_argument("prepared ND hyperbolic face output and scratch must not alias");
    require_state_patch_(state);
    require_provider_patch_(state, providers);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(candidate, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(statuses, state.box(), 1);
    cartesian_operator_detail::materialize_axes<0, Variables>(
        model_, metric_, reconstruction_, numerical_flux_, positivity_floor_, positivity_component_,
        state, providers.view(), candidate, statuses);
    const Real failure = cartesian_operator_detail::maximum_face_status<0>(statuses);
    if (failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error("prepared ND hyperbolic face evaluation refused publication");

    cartesian_operator_detail::copy_face_axes<0>(candidate, output, n_vars);
    device_fence();
  }

  /// Plan-mapped provider route.  The host has already validated the immutable consumer plan and
  /// gathered its storage-component map; this operator sees only dense local slots.
  template <class MemorySpace, int Count>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               const ProviderStorageView<Dim, Count>& providers,
                               FaceField<Dim, MemorySpace>& output) const
    requires(Count == flux_provider_count<Model>)
  {
    require_state_patch_(state);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);

    FaceField<Dim, MemorySpace> candidate(state.box(), n_vars);
    FaceField<Dim, MemorySpace> statuses(state.box(), 1);
    materialize_face_fluxes(state, providers, output, candidate, statuses);
  }

  template <class MemorySpace, int Count>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               const ProviderStorageView<Dim, Count>& providers,
                               FaceField<Dim, MemorySpace>& output,
                               FaceField<Dim, MemorySpace>& candidate,
                               FaceField<Dim, MemorySpace>& statuses) const
    requires(Count == flux_provider_count<Model>)
  {
    if (&output == &candidate || &output == &statuses || &candidate == &statuses)
      throw std::invalid_argument("prepared ND hyperbolic face output and scratch must not alias");
    require_state_patch_(state);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(candidate, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(statuses, state.box(), 1);
    cartesian_operator_detail::materialize_axes<0, Variables>(
        model_, metric_, reconstruction_, numerical_flux_, positivity_floor_, positivity_component_,
        state, providers, candidate, statuses);
    const Real failure = cartesian_operator_detail::maximum_face_status<0>(statuses);
    if (failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error("prepared ND hyperbolic face evaluation refused publication");

    cartesian_operator_detail::copy_face_axes<0>(candidate, output, n_vars);
    device_fence();
  }

  /// Assemble a conservative residual from one already integrated axis-indexed face field.  This
  /// explicit seam lets boundary topology apply post-Riemann conditions (notably NoFlux) without
  /// introducing a boundary type or a two-dimensional adapter into the numerical operator.
  template <class MemorySpace>
  void assemble_residual_from_face_fluxes(const FaceField<Dim, MemorySpace>& integrated_fluxes,
                                          Fab<Dim, MemorySpace>& residual) const {
    const Box<Dim>& cells = integrated_fluxes.cell_box();
    if (!domain().contains(cells))
      throw std::invalid_argument(
          "prepared ND hyperbolic face patch lies outside the metric domain");
    cartesian_operator_detail::require_face_output(integrated_fluxes, cells, n_vars);
    cartesian_operator_detail::require_residual_shape(residual, cells, n_vars);

    Fab<Dim, MemorySpace> candidate(cells, n_vars);
    Fab<Dim, MemorySpace> cell_statuses(cells, 1);
    assemble_residual_from_face_fluxes(integrated_fluxes, residual, candidate, cell_statuses);
  }

  template <class MemorySpace>
  void assemble_residual_from_face_fluxes(const FaceField<Dim, MemorySpace>& integrated_fluxes,
                                          Fab<Dim, MemorySpace>& residual,
                                          Fab<Dim, MemorySpace>& candidate,
                                          Fab<Dim, MemorySpace>& cell_statuses) const {
    if (&residual == &candidate || &residual == &cell_statuses || &candidate == &cell_statuses)
      throw std::invalid_argument(
          "prepared ND hyperbolic residual output and scratch must not alias");
    const Box<Dim>& cells = integrated_fluxes.cell_box();
    if (!domain().contains(cells))
      throw std::invalid_argument(
          "prepared ND hyperbolic face patch lies outside the metric domain");
    cartesian_operator_detail::require_face_output(integrated_fluxes, cells, n_vars);
    cartesian_operator_detail::require_residual_shape(residual, cells, n_vars);
    cartesian_operator_detail::require_residual_shape(candidate, cells, n_vars);
    cartesian_operator_detail::require_residual_shape(cell_statuses, cells, 1);
    for_each_cell(cells,
                  cartesian_operator_detail::MaterializeResidual<Dim, Metric, n_vars>{
                      metric_, integrated_fluxes.view(), candidate.view(), cell_statuses.view()});
    const Real cell_failure = for_each_cell_reduce_max(
        cells, cartesian_operator_detail::FieldStatusMaximum<Dim>{
                   static_cast<const Fab<Dim, MemorySpace>&>(cell_statuses).view()});
    if (cell_failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error("prepared ND hyperbolic residual refused publication");

    for_each_cell(cells, cartesian_operator_detail::CopyCellField<Dim>{
                             static_cast<const Fab<Dim, MemorySpace>&>(candidate).view(),
                             residual.view(), n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state, Fab<Dim, MemorySpace>& residual) const
    requires(flux_provider_count<Model> == 0)
  {
    require_state_patch_(state);
    cartesian_operator_detail::require_residual_output(state, residual, n_vars);

    FaceField<Dim, MemorySpace> integrated_fluxes(state.box(), n_vars);
    FaceField<Dim, MemorySpace> face_statuses(state.box(), 1);
    cartesian_operator_detail::materialize_axes<0, Variables>(
        model_, metric_, reconstruction_, numerical_flux_, positivity_floor_, positivity_component_,
        state, cartesian_operator_detail::ProviderFreeStorage<Dim>{}, integrated_fluxes,
        face_statuses);
    const Real face_failure = cartesian_operator_detail::maximum_face_status<0>(face_statuses);
    if (face_failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error("prepared ND hyperbolic face evaluation refused publication");
    assemble_residual_from_face_fluxes(integrated_fluxes, residual);
  }

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state, const Fab<Dim, MemorySpace>& providers,
                         Fab<Dim, MemorySpace>& residual) const {
    require_state_patch_(state);
    require_provider_patch_(state, providers);
    cartesian_operator_detail::require_residual_output(state, residual, n_vars);

    FaceField<Dim, MemorySpace> integrated_fluxes(state.box(), n_vars);
    materialize_face_fluxes(state, providers, integrated_fluxes);
    assemble_residual_from_face_fluxes(integrated_fluxes, residual);
  }

  template <class MemorySpace, int Count>
  void assemble_residual(const Fab<Dim, MemorySpace>& state,
                         const ProviderStorageView<Dim, Count>& providers,
                         Fab<Dim, MemorySpace>& residual) const
    requires(Count == flux_provider_count<Model>)
  {
    require_state_patch_(state);
    cartesian_operator_detail::require_residual_output(state, residual, n_vars);
    FaceField<Dim, MemorySpace> integrated_fluxes(state.box(), n_vars);
    materialize_face_fluxes(state, providers, integrated_fluxes);
    assemble_residual_from_face_fluxes(integrated_fluxes, residual);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         MultiFab<Dim, MemorySpace>& residual) const
    requires(flux_provider_count<Model> == 0)
  {
    if (state.ncomp() != n_vars || residual.ncomp() != n_vars ||
        !(state.layout() == residual.layout()) ||
        !(state.distribution() == residual.distribution()) ||
        !(state.local_rank() == residual.local_rank()) ||
        state.local_size() != residual.local_size() || state.shares_storage_with(residual))
      throw std::invalid_argument(
          "prepared ND hyperbolic MultiFab state and residual layouts differ or alias storage");
    MultiFab<Dim, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                         residual.local_rank(), n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), candidate.fab(local));
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    cartesian_operator_detail::CopyCellField<Dim>{
                        static_cast<const Fab<Dim, MemorySpace>&>(candidate.fab(local)).view(),
                        residual.fab(local).view(), n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& providers,
                         MultiFab<Dim, MemorySpace>& residual) const {
    require_multifab_layout_(state, residual);
    if (providers.layout() != state.layout() || providers.distribution() != state.distribution() ||
        providers.local_rank() != state.local_rank() ||
        providers.local_size() != state.local_size() ||
        providers.ncomp() < flux_provider_count<Model>)
      throw std::invalid_argument(
          "prepared ND provider field differs from the state layout or model contract");
    MultiFab<Dim, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                         residual.local_rank(), n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), providers.fab(local), candidate.fab(local));
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    cartesian_operator_detail::CopyCellField<Dim>{
                        static_cast<const Fab<Dim, MemorySpace>&>(candidate.fab(local)).view(),
                        residual.fab(local).view(), n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void assemble_residual_from_face_fluxes(
      const std::vector<FaceField<Dim, MemorySpace>>& integrated_fluxes,
      MultiFab<Dim, MemorySpace>& residual) const {
    if (residual.ncomp() != n_vars || integrated_fluxes.size() != residual.local_size())
      throw std::invalid_argument(
          "prepared ND hyperbolic face workspace does not match the residual MultiFab");
    for (std::size_t local = 0; local < residual.local_size(); ++local) {
      if (!(integrated_fluxes[local].cell_box() == residual.box(local)) ||
          !domain().contains(residual.box(local)))
        throw std::invalid_argument(
            "prepared ND hyperbolic face workspace patch does not match the residual layout");
      cartesian_operator_detail::require_face_output(integrated_fluxes[local], residual.box(local),
                                                     n_vars);
    }

    MultiFab<Dim, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                         residual.local_rank(), n_vars, residual.ghosts());
    MultiFab<Dim, MemorySpace> statuses(residual.layout(), residual.distribution(),
                                        residual.local_rank(), 1, residual.ghosts());
    assemble_residual_from_face_fluxes(integrated_fluxes, residual, candidate, statuses);
  }

  template <class MemorySpace>
  void assemble_residual_from_face_fluxes(
      const std::vector<FaceField<Dim, MemorySpace>>& integrated_fluxes,
      MultiFab<Dim, MemorySpace>& residual, MultiFab<Dim, MemorySpace>& candidate,
      MultiFab<Dim, MemorySpace>& statuses) const {
    if (&residual == &candidate || &residual == &statuses || &candidate == &statuses)
      throw std::invalid_argument(
          "prepared ND hyperbolic divergence output and scratch must not alias");
    if (residual.ncomp() != n_vars || integrated_fluxes.size() != residual.local_size() ||
        candidate.layout() != residual.layout() ||
        candidate.distribution() != residual.distribution() ||
        candidate.local_rank() != residual.local_rank() || candidate.ncomp() != n_vars ||
        candidate.ghosts() != residual.ghosts() || statuses.layout() != residual.layout() ||
        statuses.distribution() != residual.distribution() ||
        statuses.local_rank() != residual.local_rank() || statuses.ncomp() != 1 ||
        statuses.ghosts() != residual.ghosts())
      throw std::invalid_argument(
          "prepared ND hyperbolic divergence scratch differs from the residual layout");
    for (std::size_t local = 0; local < residual.local_size(); ++local) {
      if (!(integrated_fluxes[local].cell_box() == residual.box(local)) ||
          !domain().contains(residual.box(local)))
        throw std::invalid_argument(
            "prepared ND hyperbolic face workspace patch does not match the residual layout");
      cartesian_operator_detail::require_face_output(integrated_fluxes[local], residual.box(local),
                                                     n_vars);
    }
    for (std::size_t local = 0; local < residual.local_size(); ++local)
      assemble_residual_from_face_fluxes(integrated_fluxes[local], residual.fab(local),
                                         candidate.fab(local), statuses.fab(local));
  }

 private:
  template <class MemorySpace>
  void require_multifab_layout_(const MultiFab<Dim, MemorySpace>& state,
                                const MultiFab<Dim, MemorySpace>& residual) const {
    if (state.ncomp() != n_vars || residual.ncomp() != n_vars ||
        !(state.layout() == residual.layout()) ||
        !(state.distribution() == residual.distribution()) ||
        !(state.local_rank() == residual.local_rank()) ||
        state.local_size() != residual.local_size() || state.shares_storage_with(residual))
      throw std::invalid_argument(
          "prepared ND hyperbolic MultiFab state and residual layouts differ or alias storage");
  }

  template <class MemorySpace>
  void require_provider_patch_(const Fab<Dim, MemorySpace>& state,
                               const Fab<Dim, MemorySpace>& providers) const {
    if (!(providers.box() == state.box()) || providers.ncomp() < flux_provider_count<Model> ||
        !providers.grown_box().contains(state.box().grow(1)))
      throw std::invalid_argument(
          "prepared ND provider patch does not cover the model-qualified face traces");
  }

  template <class MemorySpace>
  void require_state_patch_(const Fab<Dim, MemorySpace>& state) const {
    if (!domain().contains(state.box()))
      throw std::invalid_argument(
          "prepared ND hyperbolic state patch lies outside the metric domain");
    require_reconstruction_storage<Reconstruction>(state, state.box(), n_vars);
  }

  Model model_;
  Metric metric_;
  Reconstruction reconstruction_;
  NumericalFlux numerical_flux_;
  Real positivity_floor_ = Real(0);
  int positivity_component_ = 0;
};

template <int Dim, class Model, class Metric, class Reconstruction = NoSlope,
          class NumericalFlux = RusanovFlux,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative>
  requires(ConservationLaw<Dim, Model> && PreparedMetricProvider<Dim, Metric> &&
           ReconstructionPolicy<Reconstruction>)
auto prepare_cartesian_operator(Model model, Metric metric, Reconstruction reconstruction = {},
                                NumericalFlux numerical_flux = {},
                                Real positivity_floor = Real(0)) {
  return PreparedCartesianOperator<Dim, Model, Metric, Reconstruction, NumericalFlux, Variables>(
      std::move(model), std::move(metric), std::move(reconstruction), std::move(numerical_flux),
      positivity_floor);
}

/// Convenience factory for the canonical Cartesian Geometry authority.
template <int Dim, class Model, class Reconstruction = NoSlope, class NumericalFlux = RusanovFlux,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative>
  requires ConservationLaw<Dim, Model>
auto prepare_cartesian_operator(const Geometry<Dim>& geometry, Model model,
                                Reconstruction reconstruction = {},
                                NumericalFlux numerical_flux = {},
                                Real positivity_floor = Real(0)) {
  RealVector<Dim> lengths{};
  for (int axis = 0; axis < Dim; ++axis)
    lengths[axis] = geometry.upper()[axis] - geometry.lower()[axis];
  const auto map = CartesianCoordinateMap<Dim>::make(geometry.lower(), lengths);
  auto metric = prepare_metric_provider(geometry.domain(), map);
  return prepare_cartesian_operator<Dim, Model, decltype(metric), Reconstruction, NumericalFlux,
                                    Variables>(std::move(model), std::move(metric),
                                               std::move(reconstruction), std::move(numerical_flux),
                                               positivity_floor);
}

template <int Dim, class MemorySpace>
std::vector<FaceField<Dim, MemorySpace>> make_face_flux_workspace(
    const MultiFab<Dim, MemorySpace>& state) {
  std::vector<FaceField<Dim, MemorySpace>> result;
  result.reserve(state.local_size());
  for (std::size_t local = 0; local < state.local_size(); ++local)
    result.emplace_back(state.box(local), state.ncomp());
  return result;
}

}  // namespace pops::nd
