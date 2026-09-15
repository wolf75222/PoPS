/// @file
/// @brief Prepared compile-time-ranked hyperbolic face flux and conservative residual operator.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/fan_li15_path_flux.hpp>
#include <pops/numerics/spatial/nd/finite_volume.hpp>
#include <pops/numerics/spatial/nd/reconstruction.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::nd {

inline std::string hyperbolic_publication_refusal(const char* what, Real failure) {
  return std::string(what) + " status=" + std::to_string(static_cast<int>(failure));
}

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

/// Patch-owned staging for a complete path face tuple. All allocations happen
/// during preparation; evaluation publishes none of F/L/R/speed on failure.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PreparedCartesianPathFaceScratch {
 public:
  PreparedCartesianPathFaceScratch(const Box<Dim>& cells, int nvars)
      : flux_(cells, nvars),
        left_ncp_(cells, nvars),
        right_ncp_(cells, nvars),
        speed_(cells, 1),
        status_(cells, 1) {}
  void require_layout(const Box<Dim>& cells, int nvars) const {
    if (!(flux_.cell_box() == cells) || flux_.ncomp() != nvars ||
        !(left_ncp_.cell_box() == cells) || left_ncp_.ncomp() != nvars ||
        !(right_ncp_.cell_box() == cells) || right_ncp_.ncomp() != nvars ||
        !(speed_.cell_box() == cells) || speed_.ncomp() != 1 || !(status_.cell_box() == cells) ||
        status_.ncomp() != 1)
      throw std::invalid_argument("prepared Cartesian path scratch differs from its patch layout");
  }
  FaceField<Dim, MemorySpace>& flux() { return flux_; }
  FaceField<Dim, MemorySpace>& left_ncp() { return left_ncp_; }
  FaceField<Dim, MemorySpace>& right_ncp() { return right_ncp_; }
  FaceField<Dim, MemorySpace>& speed() { return speed_; }
  FaceField<Dim, MemorySpace>& status() { return status_; }

 private:
  FaceField<Dim, MemorySpace> flux_, left_ncp_, right_ncp_, speed_, status_;
};

namespace cartesian_operator_detail {

/// Validate the same model-qualified slots sampled by bind_flux_providers_at in face kernels.
template <class Model, int Dim, int Count>
void require_provider_face_storage(const Box<Dim>& cells,
                                   const ProviderStorageView<Dim, Count>& providers) {
  // Provider values are sampled at the two adjacent cells, independently of the wider
  // reconstruction stencil for the state. A compact plan does not prove storage extent.
  const auto required = cells.grow(1);
  static_assert(Count == flux_provider_count<Model>);
  static_assert(qualified_flux_provider_requirements_valid<Model>());
  const auto require_slot = [&](int slot) {
    const auto& field = providers.storage[slot];
    const int component = providers.storage_components[slot];
    if (field.data == nullptr || component < 0 || component >= field.ncomp)
      throw std::invalid_argument("prepared ND provider map has an invalid storage component");
    for (int axis = 0; axis < Dim; ++axis) {
      if (field.extents[axis] <= 0 || field.origin[axis] > required.lo[axis] ||
          field.extents[axis] <= static_cast<std::int64_t>(required.hi[axis]) - field.origin[axis])
        throw std::invalid_argument(
            "prepared ND provider map does not cover the model-qualified face traces");
    }
  };
  if constexpr (has_qualified_flux_provider_requirements<Model>) {
    for (const auto& requirement : Model::flux_provider_requirements)
      require_slot(requirement.storage_slot);
  } else {
    // Hand-written models without a qualified consumer plan retain their complete slot ABI.
    for (int slot = 0; slot < Count; ++slot)
      require_slot(slot);
  }
}

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
  Box<Dim> domain{};
  std::array<bool, 2 * Dim> omitted_faces{};

  POPS_HD void fail(const FaceIndex<Dim, Axis>& face, FiniteVolumeStatus status) const {
    for (int component = 0; component < Model::n_vars; ++component)
      integrated_fluxes.template operator()<Axis>(face.coordinate, component) = Real(0);
    statuses.template operator()<Axis>(face.coordinate) = static_cast<Real>(status);
  }

  POPS_HD void operator()(const FaceIndex<Dim, Axis>& face) const {
    if ((omitted_faces[2 * Axis] && face[Axis] == domain.lo[Axis]) ||
        (omitted_faces[2 * Axis + 1] &&
         static_cast<std::int64_t>(face[Axis]) == static_cast<std::int64_t>(domain.hi[Axis]) + 1)) {
      fail(face, FiniteVolumeStatus::Success);
      return;
    }
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
    auto integrated = apply_face_measure(evaluation.checked_density(), context);
    if constexpr (DiffusiveModel<Model>) {
      const Real spacing = context.cell_measure / context.face_measure;
      const Real integrated_scale = context.face_measure * (-model.diffusivity()) / spacing;
      for (int component = 0; component < Model::n_vars; ++component)
        integrated.value[component] +=
            integrated_scale * (state(right_cell, component) - state(left_cell, component));
    }
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

/// Status prefixes preserve whether a refusal came from metric/finite-volume
/// checks (256), extra model conversion (512) or the exact Fan--Li policy (1024).
/// These small integers remain exactly representable in every supported Real.
template <int Axis, int Dim, class Model, class Metric, class ProviderStorage>
struct MaterializePathFace {
  Model model;
  Metric metric;
  FieldView<const Real, Dim> state;
  ProviderStorage providers;
  FaceFieldView<Real, Dim> flux, left_ncp, right_ncp, speed, statuses;
  Box<Dim> domain;

  POPS_HD void clear(const FaceIndex<Dim, Axis>& face, Real status) const {
    for (int k = 0; k < Model::n_vars; ++k) {
      flux.template operator()<Axis>(face.coordinate, k) = Real(0);
      left_ncp.template operator()<Axis>(face.coordinate, k) = Real(0);
      right_ncp.template operator()<Axis>(face.coordinate, k) = Real(0);
    }
    speed.template operator()<Axis>(face.coordinate) = Real(0);
    statuses.template operator()<Axis>(face.coordinate) = status;
  }
  POPS_HD void operator()(const FaceIndex<Dim, Axis>& face) const {
    constexpr auto zero_faces = Model::path_zero_measure_faces();
    if ((zero_faces[2 * Axis] && face[Axis] == domain.lo[Axis]) ||
        (zero_faces[2 * Axis + 1] &&
         static_cast<std::int64_t>(face[Axis]) == static_cast<std::int64_t>(domain.hi[Axis]) + 1)) {
      // Authored zero mapped measure, before reading state or provider ghosts.
      clear(face, Real(0));
      return;
    }
    Index<Dim> left_cell = face.coordinate;
    --left_cell[Axis];
    const Index<Dim> right_cell = face.coordinate;
    const FaceContext context =
        face[Axis] == flux.cells.lo[Axis]
            ? metric_face_context<Axis, MetricFaceSide::Lower>(metric, right_cell)
            : metric_face_context<Axis, MetricFaceSide::Upper>(metric, left_cell);
    if (!Kokkos::isfinite(context.face_measure) || !(context.face_measure > Real(0)) ||
        !Kokkos::isfinite(context.cell_measure) || !(context.cell_measure > Real(0))) {
      clear(face, Real(256) + static_cast<Real>(FiniteVolumeStatus::InvalidMetric));
      return;
    }
    // FirstOrder: exactly the two adjacent conservative stored states. The
    // path policy performs its own typed raw-SPD certificate on both traces.
    const auto left = load_state<Model>(state, left_cell),
               right = load_state<Model>(state, right_cell);
    const auto evaluation = evaluate_fan_li15_path_at<Axis>(
        FanLi15PathRusanovFlux{}, model, left, providers, left_cell, right, providers, right_cell);
    if (!evaluation.succeeded()) {
      clear(face, Real(1024) + static_cast<Real>(evaluation.status));
      return;
    }
    const auto left_status = model.admissibility(left), right_status = model.admissibility(right);
    if (left_status != StateConversionStatus::Success ||
        right_status != StateConversionStatus::Success) {
      clear(face, Real(512) + static_cast<Real>(left_status != StateConversionStatus::Success
                                                    ? left_status
                                                    : right_status));
      return;
    }
    Real f[15], l[15], r[15];
    for (int k = 0; k < 15; ++k) {
      f[k] = context.face_measure * evaluation.conservative_flux.values[k];
      l[k] = context.face_measure * evaluation.left_ncp.values[k];
      r[k] = context.face_measure * evaluation.right_ncp.values[k];
      if (!Kokkos::isfinite(f[k]) || !Kokkos::isfinite(l[k]) || !Kokkos::isfinite(r[k])) {
        clear(face, Real(256) + static_cast<Real>(FiniteVolumeStatus::NonFiniteFaceFlux));
        return;
      }
    }
    if (!Kokkos::isfinite(evaluation.speed_bound) || evaluation.speed_bound < Real(0)) {
      clear(face, Real(256) + static_cast<Real>(FiniteVolumeStatus::InvalidWaveSpeed));
      return;
    }
    for (int k = 0; k < 15; ++k) {
      flux.template operator()<Axis>(face.coordinate, k) = f[k];
      left_ncp.template operator()<Axis>(face.coordinate, k) = l[k];
      right_ncp.template operator()<Axis>(face.coordinate, k) = r[k];
    }
    speed.template operator()<Axis>(face.coordinate) = evaluation.speed_bound;
    statuses.template operator()<Axis>(face.coordinate) = Real(0);
  }
};

template <int Axis, int Dim, class Model, class Metric, class Storage, class MemorySpace>
void materialize_path_axes(const Model& model, const Metric& metric,
                           const Fab<Dim, MemorySpace>& state, const Storage& providers,
                           PreparedCartesianPathFaceScratch<Dim, MemorySpace>& candidate) {
  for_each_face<Axis>(state.box(), MaterializePathFace<Axis, Dim, Model, Metric, Storage>{
                                       model, metric, state.view(), providers,
                                       candidate.flux().view(), candidate.left_ncp().view(),
                                       candidate.right_ncp().view(), candidate.speed().view(),
                                       candidate.status().view(), metric.identity().domain});
  if constexpr (Axis + 1 < Dim)
    materialize_path_axes<Axis + 1>(model, metric, state, providers, candidate);
}

template <int Dim, class Metric, int N>
struct MaterializePathResidual {
  Metric metric;
  FaceFieldView<const Real, Dim> flux, left_ncp, right_ncp;
  FieldView<Real, Dim> candidate, statuses;
  POPS_HD void operator()(const Index<Dim>& cell) const {
    auto value = conservative_residual<N>(metric, flux, cell);
    if (value.succeeded()) {
      const Real inverse_volume = Real(1) / metric.cell_measure(cell);
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> upper = cell;
        ++upper[axis];
        for (int k = 0; k < N; ++k)
          value.value[k] +=
              inverse_volume * (right_ncp.axes[axis](cell, k) + left_ncp.axes[axis](upper, k));
      }
      for (int k = 0; k < N; ++k)
        if (!Kokkos::isfinite(value.value[k]))
          value.status = FiniteVolumeStatus::NonFiniteFaceFlux;
    }
    for (int k = 0; k < N; ++k)
      candidate(cell, k) = value.succeeded() ? value.value[k] : Real(0);
    statuses(cell) = static_cast<Real>(value.status);
  }
};

template <int Axis, ReconstructionVariables Variables, int Dim, class Model, class Metric,
          class Reconstruction, class NumericalFlux, class ProviderStorage, class MemorySpace>
void materialize_axes(const Model& model, const Metric& metric,
                      const Reconstruction& reconstruction, const NumericalFlux& numerical_flux,
                      Real positivity_floor, int positivity_component,
                      const Fab<Dim, MemorySpace>& state, const ProviderStorage& providers,
                      FaceField<Dim, MemorySpace>& integrated_fluxes,
                      FaceField<Dim, MemorySpace>& statuses,
                      const std::array<bool, 2 * Dim>& omitted_faces = {}) {
  for_each_face<Axis>(state.box(),
                      MaterializeFaceFlux<Axis, Variables, Dim, Model, Metric, Reconstruction,
                                          NumericalFlux, ProviderStorage>{
                          model, metric, reconstruction, numerical_flux, positivity_floor,
                          positivity_component, state.view(), providers, integrated_fluxes.view(),
                          statuses.view(), metric.identity().domain, omitted_faces});
  if constexpr (Axis + 1 < Dim)
    materialize_axes<Axis + 1, Variables>(model, metric, reconstruction, numerical_flux,
                                          positivity_floor, positivity_component, state, providers,
                                          integrated_fluxes, statuses, omitted_faces);
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

/// Path publication checks the actual axis allocations too: FaceField is an
/// owning preparation object; output references must not alias the transaction.
template <int Dim, class MemorySpace>
void require_path_face_output(const FaceField<Dim, MemorySpace>& output, const Box<Dim>& cells,
                              int nvars) {
  require_face_output(output, cells, nvars);
  const auto view = output.view();
  for (int axis = 0; axis < Dim; ++axis) {
    const auto expected = face_box(cells, axis);
    const auto& field = view.axes[axis];
    if (field.data == nullptr || field.ncomp != nvars)
      throw std::invalid_argument("prepared path face axis allocation is invalid");
    for (int d = 0; d < Dim; ++d)
      if (field.origin[d] != expected.lo[d] ||
          field.extents[d] != static_cast<std::int64_t>(expected.hi[d]) - expected.lo[d] + 1)
        throw std::invalid_argument("prepared path face axis allocation has the wrong extent");
  }
}

template <int Dim, class Storage>
bool path_provider_aliases(const Storage& storage, const Real* output) {
  if constexpr (requires { storage.data; }) {
    return storage.data == output;
  } else if constexpr (requires { storage.storage; }) {
    for (const auto& field : storage.storage)
      if (field.data == output)
        return true;
  }
  return false;
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
    if constexpr (DiffusiveModel<Model>) {
      if (Metric::capabilities().coordinate_map.kind != CoordinateMapKind::Cartesian)
        throw std::invalid_argument("prepared ND isotropic diffusion requires a Cartesian metric");
      const Real diffusivity = model_.diffusivity();
      if (!std::isfinite(diffusivity) || diffusivity < Real(0))
        throw std::invalid_argument(
            "prepared ND isotropic diffusion requires a finite non-negative diffusivity");
    }
  }

  const Model& model() const noexcept { return model_; }
  const Metric& metric() const noexcept { return metric_; }
  Box<Dim> domain() const noexcept { return metric_.identity().domain; }

  /// Publish one indivisible path tuple. F is a conservative face integral;
  /// L/R are separate integrated side residuals (-P/2), and speed is not integrated.
  template <class MemorySpace>
  void materialize_path_face_contributions(
      const Fab<Dim, MemorySpace>& state, FaceField<Dim, MemorySpace>& flux,
      FaceField<Dim, MemorySpace>& left_ncp, FaceField<Dim, MemorySpace>& right_ncp,
      FaceField<Dim, MemorySpace>& speed,
      PreparedCartesianPathFaceScratch<Dim, MemorySpace>& scratch,
      const std::array<bool, 2 * Dim>& omitted_faces = {}) const
    requires(path_conservative_model<Model> && flux_provider_count<Model> == 0)
  {
    materialize_path_faces_(state, cartesian_operator_detail::ProviderFreeStorage<Dim>{}, flux,
                            left_ncp, right_ncp, speed, scratch, omitted_faces);
  }

  template <class MemorySpace>
  void materialize_path_face_contributions(
      const Fab<Dim, MemorySpace>& state, const Fab<Dim, MemorySpace>& providers,
      FaceField<Dim, MemorySpace>& flux, FaceField<Dim, MemorySpace>& left_ncp,
      FaceField<Dim, MemorySpace>& right_ncp, FaceField<Dim, MemorySpace>& speed,
      PreparedCartesianPathFaceScratch<Dim, MemorySpace>& scratch,
      const std::array<bool, 2 * Dim>& omitted_faces = {}) const
    requires(path_conservative_model<Model>)
  {
    require_provider_patch_(state, providers);
    materialize_path_faces_(state, providers.view(), flux, left_ncp, right_ncp, speed, scratch,
                            omitted_faces);
  }

  template <class MemorySpace, int Count>
  void materialize_path_face_contributions(
      const Fab<Dim, MemorySpace>& state, const ProviderStorageView<Dim, Count>& providers,
      FaceField<Dim, MemorySpace>& flux, FaceField<Dim, MemorySpace>& left_ncp,
      FaceField<Dim, MemorySpace>& right_ncp, FaceField<Dim, MemorySpace>& speed,
      PreparedCartesianPathFaceScratch<Dim, MemorySpace>& scratch,
      const std::array<bool, 2 * Dim>& omitted_faces = {}) const
    requires(path_conservative_model<Model> && Count == flux_provider_count<Model>)
  {
    cartesian_operator_detail::require_provider_face_storage<Model>(state.box(), providers);
    materialize_path_faces_(state, providers, flux, left_ncp, right_ncp, speed, scratch,
                            omitted_faces);
  }

  /// The lower face contributes its right-cell residual; the upper face its
  /// left-cell residual. Only F belongs to the conservative flux register.
  template <class MemorySpace>
  void assemble_residual_from_path_faces(const FaceField<Dim, MemorySpace>& flux,
                                         const FaceField<Dim, MemorySpace>& left_ncp,
                                         const FaceField<Dim, MemorySpace>& right_ncp,
                                         Fab<Dim, MemorySpace>& residual,
                                         Fab<Dim, MemorySpace>& candidate,
                                         Fab<Dim, MemorySpace>& statuses) const
    requires(path_conservative_model<Model>)
  {
    require_path_route_({});
    const auto& cells = flux.cell_box();
    if (!domain().contains(cells))
      throw std::invalid_argument("prepared path face patch lies outside the metric domain");
    const std::array<const FaceField<Dim, MemorySpace>*, 3> inputs{&flux, &left_ncp, &right_ncp};
    for (const auto* input : inputs)
      cartesian_operator_detail::require_path_face_output(*input, cells, n_vars);
    cartesian_operator_detail::require_residual_shape(residual, cells, n_vars);
    cartesian_operator_detail::require_residual_shape(candidate, cells, n_vars);
    cartesian_operator_detail::require_residual_shape(statuses, cells, 1);
    const std::array<const Real*, 3> writes{residual.view().data, candidate.view().data,
                                            statuses.view().data};
    for (std::size_t i = 0; i < writes.size(); ++i) {
      for (std::size_t j = 0; j < i; ++j)
        if (writes[i] == writes[j])
          throw std::invalid_argument("prepared path residual output and scratch alias storage");
      for (const auto* input : inputs)
        for (const auto& axis : input->view().axes)
          if (writes[i] == axis.data)
            throw std::invalid_argument("prepared path residual writes alias a face input");
    }
    for_each_cell(cells, cartesian_operator_detail::MaterializePathResidual<Dim, Metric, n_vars>{
                             metric_, flux.view(), left_ncp.view(), right_ncp.view(),
                             candidate.view(), statuses.view()});
    const Real failure = for_each_cell_reduce_max(
        cells, cartesian_operator_detail::FieldStatusMaximum<Dim>{
                   static_cast<const Fab<Dim, MemorySpace>&>(statuses).view()});
    if (failure != Real(0))
      throw std::runtime_error(hyperbolic_publication_refusal(
          "prepared Cartesian path residual refused publication", failure));
    for_each_cell(cells, cartesian_operator_detail::CopyCellField<Dim>{
                             static_cast<const Fab<Dim, MemorySpace>&>(candidate).view(),
                             residual.view(), n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               FaceField<Dim, MemorySpace>& output,
                               const std::array<bool, 2 * Dim>& omitted_faces = {}) const
    requires(flux_provider_count<Model> == 0)
  {
    require_ordinary_route_();
    require_state_patch_(state);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);

    FaceField<Dim, MemorySpace> candidate(state.box(), n_vars);
    FaceField<Dim, MemorySpace> statuses(state.box(), 1);
    materialize_face_fluxes(state, output, candidate, statuses, omitted_faces);
  }

  template <class MemorySpace>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               FaceField<Dim, MemorySpace>& output,
                               FaceField<Dim, MemorySpace>& candidate,
                               FaceField<Dim, MemorySpace>& statuses,
                               const std::array<bool, 2 * Dim>& omitted_faces = {}) const
    requires(flux_provider_count<Model> == 0)
  {
    require_ordinary_route_();
    if (&output == &candidate || &output == &statuses || &candidate == &statuses)
      throw std::invalid_argument("prepared ND hyperbolic face output and scratch must not alias");
    require_state_patch_(state);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(candidate, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(statuses, state.box(), 1);
    cartesian_operator_detail::materialize_axes<0, Variables>(
        model_, metric_, reconstruction_, numerical_flux_, positivity_floor_, positivity_component_,
        state, cartesian_operator_detail::ProviderFreeStorage<Dim>{}, candidate, statuses,
        omitted_faces);
    const Real failure = cartesian_operator_detail::maximum_face_status<0>(statuses);
    if (failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error(hyperbolic_publication_refusal(
          "prepared ND hyperbolic face evaluation refused publication", failure));

    cartesian_operator_detail::copy_face_axes<0>(candidate, output, n_vars);
    device_fence();
  }

  template <class MemorySpace>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               const Fab<Dim, MemorySpace>& providers,
                               FaceField<Dim, MemorySpace>& output,
                               const std::array<bool, 2 * Dim>& omitted_faces = {}) const {
    require_ordinary_route_();
    require_state_patch_(state);
    require_provider_patch_(state, providers);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);

    FaceField<Dim, MemorySpace> candidate(state.box(), n_vars);
    FaceField<Dim, MemorySpace> statuses(state.box(), 1);
    materialize_face_fluxes(state, providers, output, candidate, statuses, omitted_faces);
  }

  template <class MemorySpace>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               const Fab<Dim, MemorySpace>& providers,
                               FaceField<Dim, MemorySpace>& output,
                               FaceField<Dim, MemorySpace>& candidate,
                               FaceField<Dim, MemorySpace>& statuses,
                               const std::array<bool, 2 * Dim>& omitted_faces = {}) const {
    require_ordinary_route_();
    if (&output == &candidate || &output == &statuses || &candidate == &statuses)
      throw std::invalid_argument("prepared ND hyperbolic face output and scratch must not alias");
    require_state_patch_(state);
    require_provider_patch_(state, providers);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(candidate, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(statuses, state.box(), 1);
    cartesian_operator_detail::materialize_axes<0, Variables>(
        model_, metric_, reconstruction_, numerical_flux_, positivity_floor_, positivity_component_,
        state, providers.view(), candidate, statuses, omitted_faces);
    const Real failure = cartesian_operator_detail::maximum_face_status<0>(statuses);
    if (failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error(hyperbolic_publication_refusal(
          "prepared ND hyperbolic face evaluation refused publication", failure));

    cartesian_operator_detail::copy_face_axes<0>(candidate, output, n_vars);
    device_fence();
  }

  /// Plan-mapped provider route.  The host has already validated the immutable consumer plan and
  /// gathered its storage-component map; this operator sees only dense local slots.
  template <class MemorySpace, int Count>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               const ProviderStorageView<Dim, Count>& providers,
                               FaceField<Dim, MemorySpace>& output,
                               const std::array<bool, 2 * Dim>& omitted_faces = {}) const
    requires(Count == flux_provider_count<Model>)
  {
    require_ordinary_route_();
    require_state_patch_(state);
    cartesian_operator_detail::require_provider_face_storage<Model>(state.box(), providers);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);

    FaceField<Dim, MemorySpace> candidate(state.box(), n_vars);
    FaceField<Dim, MemorySpace> statuses(state.box(), 1);
    materialize_face_fluxes(state, providers, output, candidate, statuses, omitted_faces);
  }

  template <class MemorySpace, int Count>
  void materialize_face_fluxes(const Fab<Dim, MemorySpace>& state,
                               const ProviderStorageView<Dim, Count>& providers,
                               FaceField<Dim, MemorySpace>& output,
                               FaceField<Dim, MemorySpace>& candidate,
                               FaceField<Dim, MemorySpace>& statuses,
                               const std::array<bool, 2 * Dim>& omitted_faces = {}) const
    requires(Count == flux_provider_count<Model>)
  {
    require_ordinary_route_();
    if (&output == &candidate || &output == &statuses || &candidate == &statuses)
      throw std::invalid_argument("prepared ND hyperbolic face output and scratch must not alias");
    require_state_patch_(state);
    cartesian_operator_detail::require_provider_face_storage<Model>(state.box(), providers);
    cartesian_operator_detail::require_face_output(output, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(candidate, state.box(), n_vars);
    cartesian_operator_detail::require_face_output(statuses, state.box(), 1);
    cartesian_operator_detail::materialize_axes<0, Variables>(
        model_, metric_, reconstruction_, numerical_flux_, positivity_floor_, positivity_component_,
        state, providers, candidate, statuses, omitted_faces);
    const Real failure = cartesian_operator_detail::maximum_face_status<0>(statuses);
    if (failure != static_cast<Real>(FiniteVolumeStatus::Success))
      throw std::runtime_error(hyperbolic_publication_refusal(
          "prepared ND hyperbolic face evaluation refused publication", failure));

    cartesian_operator_detail::copy_face_axes<0>(candidate, output, n_vars);
    device_fence();
  }

  /// Assemble a conservative residual from one already integrated axis-indexed face field.  This
  /// explicit seam lets boundary topology apply post-Riemann conditions (notably NoFlux) without
  /// introducing a boundary type or a two-dimensional adapter into the numerical operator.
  template <class MemorySpace>
  void assemble_residual_from_face_fluxes(const FaceField<Dim, MemorySpace>& integrated_fluxes,
                                          Fab<Dim, MemorySpace>& residual) const {
    require_ordinary_route_();
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
    require_ordinary_route_();
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
      throw std::runtime_error(hyperbolic_publication_refusal(
          "prepared ND hyperbolic residual refused publication", cell_failure));

    for_each_cell(cells, cartesian_operator_detail::CopyCellField<Dim>{
                             static_cast<const Fab<Dim, MemorySpace>&>(candidate).view(),
                             residual.view(), n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state, Fab<Dim, MemorySpace>& residual) const
    requires(flux_provider_count<Model> == 0)
  {
    require_ordinary_route_();
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
      throw std::runtime_error(hyperbolic_publication_refusal(
          "prepared ND hyperbolic face evaluation refused publication", face_failure));
    assemble_residual_from_face_fluxes(integrated_fluxes, residual);
  }

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state, const Fab<Dim, MemorySpace>& providers,
                         Fab<Dim, MemorySpace>& residual) const {
    require_ordinary_route_();
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
    require_ordinary_route_();
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
    require_ordinary_route_();
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
    require_ordinary_route_();
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
    require_ordinary_route_();
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
    require_ordinary_route_();
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
  void require_ordinary_route_() const {
    if constexpr (path_conservative_model<Model>)
      throw std::invalid_argument(
          "a path-conservative model requires the dedicated path face and residual APIs");
  }

  void require_path_route_(const std::array<bool, 2 * Dim>& omitted_faces) const
    requires(path_conservative_model<Model>)
  {
    static_assert(Dim == 2 && n_vars == 15,
                  "this native path evaluator implements full-temperature D2/M4 Fan-Li15");
    if constexpr (!std::is_same_v<Reconstruction, NoSlope> ||
                  Variables != ReconstructionVariables::Conservative ||
                  !std::is_same_v<NumericalFlux, RusanovFlux> || DiffusiveModel<Model>)
      throw std::invalid_argument(
          "Fan-Li15 path transport requires FirstOrder conservative Rusanov without diffusion");
    if (!std::isfinite(positivity_floor_) || positivity_floor_ != Real(0))
      throw std::invalid_argument("Fan-Li15 path transport does not permit a positivity floor");
    if (Model::path_operator_identity().empty())
      throw std::invalid_argument("Fan-Li15 path transport requires its exact operator identity");
    constexpr auto zero_faces = Model::path_zero_measure_faces();
    for (int face = 0; face < 2 * Dim; ++face)
      if (omitted_faces[face] && !zero_faces[face])
        throw std::invalid_argument(
            "only an authored zero-mapped-measure path face may omit the full face tuple");
  }

  template <class MemorySpace, class Storage>
  void materialize_path_faces_(const Fab<Dim, MemorySpace>& state, const Storage& providers,
                               FaceField<Dim, MemorySpace>& flux,
                               FaceField<Dim, MemorySpace>& left_ncp,
                               FaceField<Dim, MemorySpace>& right_ncp,
                               FaceField<Dim, MemorySpace>& speed,
                               PreparedCartesianPathFaceScratch<Dim, MemorySpace>& scratch,
                               const std::array<bool, 2 * Dim>& omitted_faces) const
    requires(path_conservative_model<Model>)
  {
    require_path_route_(omitted_faces);
    require_state_patch_(state);
    scratch.require_layout(state.box(), n_vars);
    const std::array<FaceField<Dim, MemorySpace>*, 9> fields{&flux,
                                                             &left_ncp,
                                                             &right_ncp,
                                                             &speed,
                                                             &scratch.flux(),
                                                             &scratch.left_ncp(),
                                                             &scratch.right_ncp(),
                                                             &scratch.speed(),
                                                             &scratch.status()};
    std::array<const Real*, 9 * Dim> writes{};
    std::size_t count = 0;
    for (std::size_t f = 0; f < fields.size(); ++f) {
      cartesian_operator_detail::require_path_face_output(*fields[f], state.box(),
                                                          (f == 3 || f >= 7) ? 1 : n_vars);
      for (const auto& axis : fields[f]->view().axes) {
        if (axis.data == state.view().data ||
            cartesian_operator_detail::path_provider_aliases<Dim>(providers, axis.data))
          throw std::invalid_argument("prepared path face writes alias an input allocation");
        for (std::size_t i = 0; i < count; ++i)
          if (axis.data == writes[i])
            throw std::invalid_argument("prepared path tuple output and scratch alias storage");
        writes[count++] = axis.data;
      }
    }
    cartesian_operator_detail::materialize_path_axes<0>(model_, metric_, state, providers, scratch);
    const Real failure = cartesian_operator_detail::maximum_face_status<0>(scratch.status());
    if (failure != Real(0))
      throw std::runtime_error(hyperbolic_publication_refusal(
          "prepared Cartesian path face tuple refused publication", failure));
    cartesian_operator_detail::copy_face_axes<0>(scratch.flux(), flux, n_vars);
    cartesian_operator_detail::copy_face_axes<0>(scratch.left_ncp(), left_ncp, n_vars);
    cartesian_operator_detail::copy_face_axes<0>(scratch.right_ncp(), right_ncp, n_vars);
    cartesian_operator_detail::copy_face_axes<0>(scratch.speed(), speed, 1);
    device_fence();
  }

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
