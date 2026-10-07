/// @file
/// @brief Final exact-ranked Uniform block factory used by generated native packages.

#pragma once

#include <pops/core/model/physical_model.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/prepared_boundary_component.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/path_flux.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/embedded_boundary/operator.hpp>
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>
#include <pops/numerics/spatial/operators/masked_operator.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/recovery/uniform_recovery_consumer.hpp>
#include <pops/runtime/program/prepared_scalar_boundary_session.hpp>
#include <pops/runtime/system/provider_storage_binding.hpp>
#include <pops/runtime/system/prepared_field_rhs_inputs.hpp>
#include <pops/runtime/system/auxiliary_ghost_fill.hpp>
#include <pops/runtime/system/system_block_closures.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {
namespace generated_system_detail {

/// Convert one primitive cell without exposing a rejected or partially materialized candidate.
///
/// Generated Uniform and AMR blocks share this publication seam.  The model-owned inverse is
/// evaluated before the caller's buffer is touched so a provider that reports success while
/// producing a non-finite or non-recoverable conservative state still fails closed.
template <class Model>
void publish_conservative_state(const Model& model, const double* primitive, double* conservative) {
  if (primitive == nullptr || conservative == nullptr)
    throw std::invalid_argument(
        "generated primitive-to-conservative conversion requires valid buffers");

  typename Model::Primitive input{};
  for (int component = 0; component < Model::n_vars; ++component) {
    const Real value = static_cast<Real>(primitive[component]);
    if (!std::isfinite(value))
      throw std::runtime_error(
          "generated primitive-to-conservative conversion rejected its candidate");
    input[component] = value;
  }

  const auto converted = model.make_conservative(input);
  if (!converted.succeeded())
    throw std::runtime_error(
        "generated primitive-to-conservative conversion rejected its candidate");

  const auto recovered = model.recover(converted.value);
  if (!recovered.succeeded())
    throw std::runtime_error(
        "generated primitive-to-conservative conversion failed its recovery roundtrip");

  double candidate[Model::n_vars]{};
  for (int component = 0; component < Model::n_vars; ++component) {
    const Real converted_value = converted.value[component];
    const Real recovered_value = recovered.value[component];
    if (!std::isfinite(converted_value) || !std::isfinite(recovered_value))
      throw std::runtime_error(
          "generated primitive-to-conservative conversion failed its recovery roundtrip");
    candidate[component] = static_cast<double>(converted_value);
    if (!std::isfinite(candidate[component]))
      throw std::runtime_error(
          "generated primitive-to-conservative conversion exceeds publication precision");
  }

  for (int component = 0; component < Model::n_vars; ++component)
    conservative[component] = candidate[component];
}

template <int Dim, class Model>
concept GeneratedSourceModel = PhysicalSourceFor<Model, Dim>;

template <int Dim, class Model>
concept GeneratedEllipticRhsModel = PhysicalEllipticRhsFor<Model, Dim>;

template <int Dim>
struct CopyValidField {
  FieldView<const Real, Dim> source{};
  FieldView<Real, Dim> destination{};
  int ncomp = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    for (int component = 0; component < ncomp; ++component)
      destination(index, component) = source(index, component);
  }
};

template <int Dim, class Model>
  requires PhysicalSourceFor<Model, Dim>
struct MaterializeSource {
  static constexpr int provider_count = provider_count_for<Model, Dim>();
  Model model;
  FieldView<const Real, Dim> state{};
  ProviderStorageView<Dim, provider_count> providers{};
  FieldView<Real, Dim> source{};
  FieldView<Real, Dim> status{};
  FieldView<const Real, Dim> active{};
  bool has_active = false;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (has_active && active(index, 0) < Real(0.5)) {
      for (int component = 0; component < Model::n_vars; ++component)
        source(index, component) = Real(0);
      status(index) = Real(0);
      return;
    }
    const auto value = model.source(load_state<Model>(state, index),
                                    load_provider_values<provider_count>(providers, index));
    Real failure = Real(0);
    for (int component = 0; component < Model::n_vars; ++component) {
      source(index, component) = value[component];
      if (!Kokkos::isfinite(value[component]))
        failure = Real(1);
    }
    status(index) = failure;
  }
};

template <int Dim, class Model>
struct MaterializePointwiseProjection {
  static constexpr int provider_count = provider_count_for<Model, Dim>();
  Model model;
  FieldView<const Real, Dim> source{};
  FieldView<Real, Dim> destination{};
  ProviderStorageView<Dim, provider_count> providers{};
  FieldView<const Real, Dim> active{};
  FieldView<Real, Dim> status{};
  bool has_active_mask = false;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (has_active_mask && active(index) < Real(0.5)) {
      for (int component = 0; component < Model::n_vars; ++component)
        destination(index, component) = source(index, component);
      status(index) = Real(0);
      return;
    }
    const auto projected = model.project(load_state<Model>(source, index),
                                         load_provider_values<provider_count>(providers, index));
    Real failure = Real(0);
    for (int component = 0; component < Model::n_vars; ++component) {
      destination(index, component) = projected[component];
      if (!Kokkos::isfinite(projected[component]))
        failure = Real(1);
    }
    status(index) = failure;
  }
};

template <int Dim, class Model>
  requires PhysicalEllipticRhsFor<Model, Dim>
struct MaterializePoissonRhs {
  Model model;
  FieldView<const Real, Dim> state{};
  FieldView<Real, Dim> rhs{};
  FieldView<Real, Dim> status{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    const Real value = model.elliptic_rhs(load_state<Model>(state, index));
    status(index) = Kokkos::isfinite(value) ? Real(0) : Real(1);
    rhs(index) = value;
  }
};

template <int Dim, class Model, int Count>
struct MaterializePoissonRhsV2 {
  Model model;
  FieldView<const Real, Dim> state{};
  ProviderStorageView<Dim, Count> providers{};
  FieldView<Real, Dim> rhs{}, status{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    const auto input = load_state<Model>(state, index);
    Real value;
    if constexpr (Count == 0)
      value = model.elliptic_rhs(input);
    else
      value = model.elliptic_rhs(input, load_provider_values<Count>(providers, index));
    rhs(index) = value;
    status(index) = Kokkos::isfinite(value) ? Real(0) : Real(1);
  }
};

template <int Axis, int Dim, class Model>
POPS_HD Real maximum_axis_speed(const Model& model, const typename Model::State& state,
                                const BoundFluxProviders<Model>& providers) {
  // This reduction selects the timestep bound, not the physical Riemann wave speed.
  // Retain the exact model-qualified provider pack used by the selected physical component.
  Real local;
  if constexpr (requires {
                  {
                    model.template stability_speed<Axis>(state, providers)
                  } -> std::convertible_to<Real>;
                })
    local = model.template stability_speed<Axis>(state, providers);
  else if constexpr (requires {
                       {
                         model.stability_speed(state, providers, Axis)
                       } -> std::convertible_to<Real>;
                     })
    local = model.stability_speed(state, providers, Axis);
  else
    local = detail::model_max_wave_speed_at<Axis>(model, state, providers);
  // An invalid axis must survive both the axis maximum and the later cell/rank reduction.
  if (!Kokkos::isfinite(local) || local < Real(0))
    return std::numeric_limits<Real>::infinity();
  if constexpr (Axis + 1 < Dim) {
    const Real remainder = maximum_axis_speed<Axis + 1, Dim>(model, state, providers);
    return local > remainder ? local : remainder;
  }
  return local;
}

template <int Dim, class Model>
struct MaterializeMaximumSpeed {
  static constexpr int provider_count = flux_provider_count<Model>;
  Model model;
  FieldView<const Real, Dim> state{};
  ProviderStorageView<Dim, provider_count> providers{};
  FieldView<Real, Dim> speed{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    const auto bound = bind_flux_providers_at<Model>(providers, index);
    speed(index) = maximum_axis_speed<0, Dim>(model, load_state<Model>(state, index), bound);
  }
};

template <int Dim, class Model>
struct MaterializeSourceFrequency {
  static constexpr int provider_count = provider_count_for<Model, Dim>();
  Model model;
  FieldView<const Real, Dim> state{};
  ProviderStorageView<Dim, provider_count> providers{};
  FieldView<Real, Dim> value{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    value(index) = model.source_frequency(load_state<Model>(state, index),
                                          load_provider_values<provider_count>(providers, index));
  }
};

template <int Dim, class Model>
struct MaterializeStabilityDt {
  static constexpr int provider_count = provider_count_for<Model, Dim>();
  Model model;
  FieldView<const Real, Dim> state{};
  ProviderStorageView<Dim, provider_count> providers{};
  FieldView<Real, Dim> value{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    value(index) = model.stability_dt(load_state<Model>(state, index),
                                      load_provider_values<provider_count>(providers, index));
  }
};

/// Return the immutable q = 2 nu sum_a h_a^-2 for a prepared Cartesian metric.  q is independent
/// of state and providers, so generated closures capture it once rather than allocating a field or
/// reducing it during step_cfl.
template <int Dim, class Model, class Metric>
Real parabolic_frequency(const Model& model, const Metric& metric) {
  const auto bound = nd::cell_parabolic_frequency<Dim>(model, metric, metric.identity().domain.lo);
  const Real frequency = bound.succeeded() ? bound.value : std::numeric_limits<Real>::quiet_NaN();
  if (!std::isfinite(frequency) || frequency < Real(0))
    throw std::runtime_error("generated parabolic frequency is invalid");
  return frequency;
}

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(label);
  return left * right;
}

template <int Dim>
HaloScheduleBudget halo_budget(const MultiFab<Dim>& field, const Box<Dim>& domain,
                               const BoundaryTopology<Dim>& topology, const Extent<Dim>& ghosts) {
  const std::size_t boxes = field.layout().size();
  const std::size_t pairs = checked_product(boxes, boxes, "generated halo pair budget overflow");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    std::size_t count = 1;
    if (topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) && ghosts[axis] > 0) {
      const std::int64_t length = domain.length(axis);
      if (length <= 0)
        throw std::invalid_argument("generated halo domain has an empty periodic axis");
      const std::int64_t wraps = 1 + (ghosts[axis] - 1) / length;
      count = 1 + checked_product(2, static_cast<std::size_t>(wraps),
                                  "generated halo image budget overflow");
    }
    images = checked_product(images, count, "generated halo image budget overflow");
  }
  const std::size_t work = checked_product(pairs, images, "generated halo work budget overflow");
  const std::size_t jobs = checked_product(work, static_cast<std::size_t>(2 * Dim),
                                           "generated halo job budget overflow");
  const std::int64_t signed_cells = domain.numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("generated halo domain must be non-empty");
  const std::size_t elements = checked_product(
      checked_product(jobs, static_cast<std::size_t>(signed_cells),
                      "generated halo element budget overflow"),
      static_cast<std::size_t>(field.ncomp()), "generated halo component budget overflow");
  return {{boxes, pairs},
          work,
          jobs,
          images,
          checked_product(boxes, std::size_t{2}, "generated halo peer budget overflow"),
          elements,
          elements,
          elements};
}

template <int Dim>
void require_same_layout(const MultiFab<Dim>& state, const MultiFab<Dim>& other,
                         int other_components, const char* operation) {
  if (state.layout() != other.layout() || state.distribution() != other.distribution() ||
      state.local_rank() != other.local_rank() || state.local_size() != other.local_size() ||
      other.ncomp() != other_components)
    throw std::invalid_argument(std::string(operation) +
                                ": field differs from the exact ranked block layout");
}

template <int Dim>
void copy_valid(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  require_same_layout(source, destination, source.ncomp(), "generated block publication");
  for (std::size_t local = 0; local < source.local_size(); ++local)
    for_each_cell(source.box(local),
                  CopyValidField<Dim>{source.fab(local).view(), destination.fab(local).view(),
                                      source.ncomp()});
  device_fence();
}

template <int Dim, class Model>
MultiFab<Dim> materialize_source(
    const Model& model, const MultiFab<Dim>& state,
    const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
    const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* plan,
    const MultiFab<Dim>* active_cells = nullptr) {
  constexpr int provider_count = provider_count_for<Model, Dim>();
  if constexpr (provider_count > 0) {
    if (provider_storage == nullptr)
      throw std::invalid_argument("generated source requires resolved provider storage");
    runtime::system::require_pointwise_provider_groups<Dim, provider_count>(
        state, provider_storage, plan, "generated source providers");
  }
  if (active_cells != nullptr)
    require_same_layout(state, *active_cells, 1, "generated source active mask");
  MultiFab<Dim> candidate(state.layout(), state.distribution(), state.local_rank(), Model::n_vars,
                          state.ghosts());
  if constexpr (GeneratedSourceModel<Dim, Model>) {
    MultiFab<Dim> status(state.layout(), state.distribution(), state.local_rank(), 1,
                         state.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    MaterializeSource<Dim, Model>{
                        model, state.fab(local).view(),
                        runtime::system::bind_provider_storage_view<Dim, provider_count>(
                            plan, provider_storage, local),
                        candidate.fab(local).view(), status.fab(local).view(),
                        active_cells == nullptr ? FieldView<const Real, Dim>{}
                                                : active_cells->fab(local).view(),
                        active_cells != nullptr});
    if (reduce_max(status) != Real(0))
      throw std::runtime_error("generated source produced a non-finite component");
  } else {
    candidate.set_val(Real(0));
  }
  return candidate;
}

/// Allocation-free source materialization into storage owned by the prepared block session.
/// The caller performs exact-lane status consensus before any subsequent collective phase.
template <int Dim, class Model>
void materialize_source_into(const Model& model, const MultiFab<Dim>& state,
                             const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
                             const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* plan,
                             MultiFab<Dim>& candidate, MultiFab<Dim>& status) {
  constexpr int provider_count = provider_count_for<Model, Dim>();
  require_same_layout(state, candidate, Model::n_vars, "generated prepared source candidate");
  require_same_layout(state, status, 1, "generated prepared source status");
  if (state.shares_storage_with(candidate) || state.shares_storage_with(status) ||
      candidate.shares_storage_with(status))
    throw std::invalid_argument("generated prepared source output and scratch must not alias");
  if constexpr (provider_count > 0) {
    if (provider_storage == nullptr)
      throw std::invalid_argument("generated source requires resolved provider storage");
    runtime::system::require_pointwise_provider_groups<Dim, provider_count>(
        state, provider_storage, plan, "generated source providers");
  }
  if constexpr (GeneratedSourceModel<Dim, Model>) {
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    MaterializeSource<Dim, Model>{
                        model, state.fab(local).view(),
                        runtime::system::bind_provider_storage_view<Dim, provider_count>(
                            plan, provider_storage, local),
                        candidate.fab(local).view(), status.fab(local).view()});
  } else {
    candidate.set_val(Real(0));
    status.set_val(Real(0));
  }
}

template <class Operation>
void prepared_boundary_collective_phase(const ExecutionLane& lane, Operation&& operation,
                                        const char* failure_message) {
  runtime::program::collective_boundary_provider_phase(lane, failure_message,
                                                       std::forward<Operation>(operation));
}

template <int Dim, class Model>
MultiFab<Dim> materialize_masked_source(
    const Model& model, const MultiFab<Dim>& state,
    const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
    const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* plan,
    const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>& embedded) {
  const MultiFab<Dim>& active = embedded.active_mask();
  require_same_layout(state, active, 1, "generated embedded-boundary active mask");
  return materialize_source<Dim>(model, state, provider_storage, plan, &active);
}

template <int Dim>
using EmbeddedResidualFamily = typename SystemBlockClosures<Dim>::EmbeddedResidualFamily;

template <int Dim, class Flux>
EmbeddedResidualFamily<Dim> make_embedded_residual_family(
    Flux flux, typename SystemBlockClosures<Dim>::EmbeddedResidual source) {
  EmbeddedResidualFamily<Dim> family;
  family.flux_only = flux;
  family.source_only = source;
  family.full = [flux, source](MultiFab<Dim>& state, MultiFab<Dim>& residual,
                               const auto& embedded) {
    MultiFab<Dim> candidate(residual.layout(), residual.distribution(), residual.local_rank(),
                            residual.ncomp(), residual.ghosts());
    flux(state, candidate, embedded);
    MultiFab<Dim> source_candidate(residual.layout(), residual.distribution(),
                                   residual.local_rank(), residual.ncomp(), residual.ghosts());
    source(state, source_candidate, embedded);
    saxpy(candidate, Real(1), source_candidate);
    copy_valid(candidate, residual);
  };
  return family;
}

template <int Dim, class Model, class MaskedOperator>
void assemble_masked_residual_with_plan(
    const MaskedOperator& masked, const MultiFab<Dim>& state,
    const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
    const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* plan,
    const MultiFab<Dim>& active_cells, MultiFab<Dim>& residual,
    nd::BoundaryFaceOmission<Dim> omission) {
  constexpr int provider_count = flux_provider_count<Model>;
  if constexpr (provider_count == 0) {
    masked.assemble_residual(state, active_cells, residual, omission);
  } else {
    MultiFab<Dim> candidate(residual.layout(), residual.distribution(), residual.local_rank(),
                            residual.ncomp(), residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      masked.assemble_residual(state.fab(local),
                               runtime::system::bind_provider_storage_view<Dim, provider_count>(
                                   plan, provider_storage, local),
                               active_cells.fab(local), candidate.fab(local), omission);
    copy_valid(candidate, residual);
  }
}

template <int Dim, class Model, class EmbeddedOperator>
void assemble_embedded_residual_with_plan(
    const EmbeddedOperator& embedded_operator, const MultiFab<Dim>& state,
    const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
    const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* plan,
    const MultiFab<Dim>& active_cells, const MultiFab<Dim>& inverse_volume_fraction,
    const MultiFab<Dim>& face_aperture_lower, const MultiFab<Dim>& face_aperture_upper,
    MultiFab<Dim>& residual, nd::BoundaryFaceOmission<Dim> omission) {
  constexpr int provider_count = flux_provider_count<Model>;
  if constexpr (provider_count == 0) {
    embedded_operator.assemble_residual(state, active_cells, inverse_volume_fraction,
                                        face_aperture_lower, face_aperture_upper, residual,
                                        omission);
  } else {
    MultiFab<Dim> candidate(residual.layout(), residual.distribution(), residual.local_rank(),
                            residual.ncomp(), residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      embedded_operator.assemble_residual(
          state.fab(local),
          runtime::system::bind_provider_storage_view<Dim, provider_count>(plan, provider_storage,
                                                                           local),
          active_cells.fab(local), inverse_volume_fraction.fab(local),
          face_aperture_lower.fab(local), face_aperture_upper.fab(local), candidate.fab(local),
          omission);
    copy_valid(candidate, residual);
  }
}

template <int Dim, nd::ReconstructionVariables Variables, class Model, class Metric,
          class Reconstruction, class Numerical, class PrepareState>
EmbeddedResidualFamily<Dim> make_cut_cell_residual_family(
    const Model& model, const Metric& metric, const Reconstruction& reconstruction,
    const Numerical& numerical, Real positivity_floor, PrepareState prepare_state,
    std::shared_ptr<const runtime::system::AuxiliaryStorageGroups<Dim>> provider_storage_owner,
    std::shared_ptr<const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>> provider_plan_owner,
    const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
    const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* provider_plan,
    nd::BoundaryFaceOmission<Dim> omission,
    typename SystemBlockClosures<Dim>::EmbeddedResidual source) {
  auto embedded_operator =
      nd::prepare_embedded_boundary_operator<Model, Metric, Reconstruction, Numerical, Variables>(
          model, metric, reconstruction, numerical, positivity_floor);
  auto flux = [embedded_operator, prepare_state, provider_storage_owner, provider_plan_owner,
               provider_storage, provider_plan,
               omission](MultiFab<Dim>& state, MultiFab<Dim>& residual, const auto& embedded) {
    prepare_state(state, nullptr);
    assemble_embedded_residual_with_plan<Dim, Model>(
        embedded_operator, state, provider_storage, provider_plan, embedded.active_mask(),
        embedded.inverse_volume_fraction(), embedded.face_aperture_lower(),
        embedded.face_aperture_upper(), residual, omission);
  };
  return make_embedded_residual_family<Dim>(std::move(flux), std::move(source));
}

// The AMR generated block image still owns its established hierarchy-wide reduction contract and calls
// this shared pointwise evaluator directly. Uniform System packages never bind this overload: their
// installed closure below requires the authenticated System ExecutionLane explicitly.
template <int Dim, class Model>
Real maximum_speed(const Model& model, const MultiFab<Dim>& state,
                   const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
                   const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* plan) {
  constexpr int provider_count = flux_provider_count<Model>;
  if constexpr (provider_count > 0) {
    if (provider_storage == nullptr)
      throw std::invalid_argument("generated speed requires resolved provider storage");
    runtime::system::require_pointwise_provider_groups<Dim, provider_count>(
        state, provider_storage, plan, "generated speed providers");
  }
  MultiFab<Dim> values(state.layout(), state.distribution(), state.local_rank(), 1, state.ghosts());
  for (std::size_t local = 0; local < state.local_size(); ++local)
    for_each_cell(state.box(local),
                  MaterializeMaximumSpeed<Dim, Model>{
                      model, state.fab(local).view(),
                      runtime::system::bind_provider_storage_view<Dim, provider_count>(
                          plan, provider_storage, local),
                      values.fab(local).view()});
  const Real result = reduce_max(values);
  if (!std::isfinite(result) || result < Real(0))
    throw std::runtime_error("generated model produced an invalid maximum speed");
  return result;
}

template <int Dim, class Model>
Real maximum_speed(const Model& model, const MultiFab<Dim>& state,
                   const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
                   const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* plan,
                   const ExecutionLane& lane) {
  constexpr int provider_count = flux_provider_count<Model>;
  std::optional<MultiFab<Dim>> values;
  Real local_result = -std::numeric_limits<Real>::infinity();
  prepared_boundary_collective_phase(
      lane,
      [&] {
        if constexpr (provider_count > 0) {
          if (provider_storage == nullptr)
            throw std::invalid_argument("generated speed requires resolved provider storage");
          runtime::system::require_pointwise_provider_groups<Dim, provider_count>(
              state, provider_storage, plan, "generated speed providers");
        }
        values.emplace(state.layout(), state.distribution(), state.local_rank(), 1, state.ghosts());
        for (std::size_t local = 0; local < state.local_size(); ++local)
          for_each_cell(state.box(local),
                        MaterializeMaximumSpeed<Dim, Model>{
                            model, state.fab(local).view(),
                            runtime::system::bind_provider_storage_view<Dim, provider_count>(
                                plan, provider_storage, local),
                            values->fab(local).view()});
        device_fence();
        local_result = reduce_max_local(*values);
        if (state.local_size() != 0 && (!std::isfinite(local_result) || local_result < Real(0)))
          throw std::runtime_error("generated model produced an invalid local maximum speed");
      },
      "generated model maximum-speed preparation failed collectively");
  const Real result = static_cast<Real>(all_reduce_max(static_cast<double>(local_result), lane));
  if (!std::isfinite(result) || result < Real(0))
    throw std::runtime_error("generated model produced an invalid maximum speed");
  return result;
}

template <int Dim, class Model>
void apply_pointwise_projection(
    const Model& model, MultiFab<Dim>& state,
    const runtime::system::AuxiliaryStorageGroups<Dim>* provider_storage,
    const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>* plan,
    const MultiFab<Dim>* active_cells, const ExecutionLane& lane) {
  constexpr int provider_count = provider_count_for<Model, Dim>();
  std::optional<MultiFab<Dim>> candidate;
  std::optional<MultiFab<Dim>> status;
  Real local_status = Real(0);
  prepared_boundary_collective_phase(
      lane,
      [&] {
        if constexpr (provider_count > 0) {
          if (provider_storage == nullptr)
            throw std::invalid_argument("generated projection requires resolved provider storage");
          runtime::system::require_pointwise_provider_groups<Dim, provider_count>(
              state, provider_storage, plan, "generated projection providers");
        }
        if (active_cells != nullptr)
          require_same_layout(state, *active_cells, 1, "generated projection active mask");
        candidate.emplace(state.layout(), state.distribution(), state.local_rank(), Model::n_vars,
                          state.ghosts());
        status.emplace(state.layout(), state.distribution(), state.local_rank(), 1, state.ghosts());
        for (std::size_t local = 0; local < state.local_size(); ++local)
          for_each_cell(
              state.box(local),
              MaterializePointwiseProjection<Dim, Model>{
                  model, std::as_const(state).fab(local).view(), candidate->fab(local).view(),
                  runtime::system::bind_provider_storage_view<Dim, provider_count>(
                      plan, provider_storage, local),
                  active_cells == nullptr ? FieldView<const Real, Dim>{}
                                          : active_cells->fab(local).view(),
                  status->fab(local).view(), active_cells != nullptr});
        device_fence();
        if (state.local_size() != 0)
          local_status = reduce_max_local(*status);
      },
      "generated pointwise projection failed collectively");
  if (all_reduce_max(local_status, lane) != Real(0))
    throw std::runtime_error("generated pointwise projection produced a non-finite value");
  prepared_boundary_collective_phase(
      lane,
      [&] {
        if (!candidate.has_value())
          throw std::logic_error("generated pointwise projection lost its unpublished candidate");
        copy_valid(*candidate, state);
      },
      "generated pointwise projection publication failed collectively");
}

template <int Dim, class Model>
void add_poisson_rhs(const Model& model, const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
  // This provider is deliberately rank-local. ExactNamedField owns the surrounding exact-lane
  // failure consensus after every rank returns (or throws) from this callback.
  if (rhs.ncomp() != 1)
    throw std::invalid_argument("generated Poisson RHS destination must have one component");
  require_same_layout(state, rhs, 1, "generated Poisson RHS");
  if constexpr (requires(const Model& provider, const typename Model::State& value) {
                  provider.elliptic_rhs(value);
                }) {
    static_assert(GeneratedEllipticRhsModel<Dim, Model>,
                  "declared elliptic RHS requires its exact physical rank and scalar result");
    MultiFab<Dim> candidate(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
    MultiFab<Dim> status(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local), MaterializePoissonRhs<Dim, Model>{
                                          model, state.fab(local).view(),
                                          candidate.fab(local).view(), status.fab(local).view()});
    device_fence();
    const Real local_status = reduce_max_local(status);
    if (state.local_size() != 0 && local_status != Real(0))
      throw std::runtime_error("generated Poisson RHS produced a non-finite value");
    saxpy(rhs, Real(1), candidate);
  }
}

template <int Dim, class Model, class Reconstruction, class Numerical,
          nd::ReconstructionVariables Variables, class Request>
PreparedSystemBlock<Dim> materialize_block(Request request, Reconstruction reconstruction,
                                           Numerical numerical) {
  static_assert(Model::dimension == Dim);
  constexpr int provider_count = provider_count_for<Model, Dim>();
  if constexpr (provider_count > 0) {
    if (!request.provider_plan ||
        request.provider_plan->value_count() != static_cast<std::size_t>(provider_count))
      throw std::invalid_argument("generated System block provider plan differs from its model");
    if (!request.provider_storage)
      throw std::invalid_argument("generated System block requires accepted provider storage");
  } else if (request.provider_plan || request.provider_storage) {
    throw std::invalid_argument(
        "provider-free generated System block cannot retain provider state");
  }

  constexpr int storage_ghost_depth = [] {
    if constexpr (requires { Model::program_state_ghost_depth; }) {
      static_assert(Model::program_state_ghost_depth >= 0);
      return std::max(Reconstruction::n_ghost, Model::program_state_ghost_depth);
    }
    return Reconstruction::n_ghost;
  }();
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = storage_ghost_depth;
  const auto spatial =
      nd::prepare_cartesian_operator<Dim, Model, Reconstruction, Numerical, Variables>(
          request.geometry, request.model, reconstruction, numerical,
          request.routes.positivity_floor);
  const Model model = request.model;
  const auto provider_storage_owner = request.provider_storage;
  const auto provider_plan_owner = request.provider_plan;
  const runtime::system::AuxiliaryStorageGroups<Dim>* const provider_storage =
      provider_storage_owner.get();
  const auto* const provider_plan = provider_plan_owner.get();
  const Geometry<Dim> geometry = request.geometry;
  const BoundaryTopology<Dim> topology = request.topology;

  auto prepare_state = [model, provider_storage_owner, provider_plan_owner, provider_storage,
                        provider_plan, geometry, topology, ghosts](
                           MultiFab<Dim>& state, const PreparedHyperbolicBoundary<Dim>* boundary) {
    if (boundary != nullptr) {
      boundary->template require_model_qualified_characteristic_provider<Model>();
      if (boundary->has_characteristic_no_inflow())
        throw std::logic_error("characteristic no-inflow requires the prepared boundary transport");
    }
    const auto state_schedule = HaloSchedule<Dim>(
        state.layout(), state.distribution(), state.local_rank(), geometry.domain(), ghosts,
        topology, state.ncomp(), halo_budget(state, geometry.domain(), topology, ghosts));
    fill_boundary(state, state_schedule);
    if (provider_storage != nullptr) {
      runtime::system::require_pointwise_provider_groups<Dim, provider_count>(
          state, provider_storage, provider_plan, "generated flux providers");
    }
    if (boundary != nullptr)
      boundary->fill_physical_model_qualified(state, geometry, model);
  };

  auto prepare_state_with_transport =
      [model, provider_storage_owner, provider_plan_owner, provider_storage, provider_plan,
       geometry](const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state,
                 const PreparedHyperbolicBoundary<Dim>* boundary,
                 const ExecutionLane& lane,
                 const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        prepared_boundary_collective_phase(
            lane,
            [&] {
              if (boundary != nullptr)
                boundary->template require_model_qualified_characteristic_provider<Model>();
            },
            "generated prepared boundary model authentication failed collectively");
        prepared_boundary_collective_phase(
            lane,
            [&] {
              if (boundary != nullptr)
                transport.fill_halo(state);
              else
                transport.fill(state);
              if (provider_storage != nullptr) {
                runtime::system::require_pointwise_provider_groups<Dim, provider_count>(
                    state, provider_storage, provider_plan, "generated flux providers");
              }
            },
            "generated prepared boundary state preparation failed collectively");
        if (boundary != nullptr) {
          prepared_boundary_collective_phase(
              lane,
              [&] {
                transport.with_characteristic_candidate(state, [&](MultiFab<Dim>& candidate) {
                  boundary->fill_physical_model_qualified(
                      state, geometry, model, lane, candidate,
                      static_cast<Real>(point.physical_time));
                });
              },
              "generated prepared physical boundary fill failed collectively");
        }
      };

  auto flux = [spatial, prepare_state, provider_storage_owner, provider_plan_owner,
               provider_storage, provider_plan,
               geometry](MultiFab<Dim>& state, MultiFab<Dim>& residual,
                         const PreparedHyperbolicBoundary<Dim>* boundary) {
    require_same_layout(state, residual, Model::n_vars, "generated flux residual");
    prepare_state(state, boundary);

    const auto omitted_faces =
        boundary == nullptr ? std::array<bool, 2 * Dim>{} : boundary->omitted_interface_faces();
    auto faces = nd::make_face_flux_workspace(state);
    for (std::size_t local = 0; local < state.local_size(); ++local) {
      if constexpr (flux_provider_count<Model> == 0)
        spatial.materialize_face_fluxes(state.fab(local), faces[local], omitted_faces);
      else
        spatial.materialize_face_fluxes(
            state.fab(local),
            runtime::system::bind_provider_storage_view<Dim, flux_provider_count<Model>>(
                provider_plan, provider_storage, local),
            faces[local], omitted_faces);
      if (boundary != nullptr)
        boundary->apply_physical_flux_conditions(faces[local], geometry.domain());
    }
    spatial.assemble_residual_from_face_fluxes(faces, residual);
  };

  auto full = [flux, model, provider_storage_owner, provider_plan_owner, provider_storage,
               provider_plan](MultiFab<Dim>& state, MultiFab<Dim>& residual,
                              const PreparedHyperbolicBoundary<Dim>* boundary) {
    MultiFab<Dim> candidate(residual.layout(), residual.distribution(), residual.local_rank(),
                            residual.ncomp(), residual.ghosts());
    flux(state, candidate, boundary);
    MultiFab<Dim> source = materialize_source<Dim>(model, state, provider_storage, provider_plan);
    saxpy(candidate, Real(1), source);
    copy_valid(candidate, residual);
  };
  auto external_boundary_flux =
      std::make_shared<typename SystemBlockClosures<Dim>::BoundaryFluxTransform>();
  auto external_ghost_boundary =
      std::make_shared<typename SystemBlockClosures<Dim>::ExternalGhostBoundary>();
  auto external_field_boundary_residual =
      std::make_shared<typename SystemBlockClosures<Dim>::PreparedPointBoundaryResidual>();
  auto external_field_boundary_jvp =
      std::make_shared<typename SystemBlockClosures<Dim>::PreparedPointJvp>();
  auto prepare_state_with_external =
      [prepare_state_with_transport, external_ghost_boundary, geometry](
          const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state,
          const PreparedHyperbolicBoundary<Dim>* boundary, const ExecutionLane& lane,
          const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        prepare_state_with_transport(point, state, boundary, lane, transport);
        if (boundary != nullptr && *external_ghost_boundary)
          (*external_ghost_boundary)(point, state, geometry, lane);
      };
  auto flux_with_transport = [spatial, prepare_state_with_external, provider_storage_owner,
                              provider_plan_owner, provider_storage, provider_plan, geometry,
                              external_boundary_flux](
                                 const runtime::multiblock::BoundaryEvaluationPoint& point,
                                 MultiFab<Dim>& state, MultiFab<Dim>& residual,
                                 const PreparedHyperbolicBoundary<Dim>* boundary,
                                 const ExecutionLane& lane,
                                 const runtime::program::PreparedScalarBoundarySession<Dim>&
                                     transport) {
    prepared_boundary_collective_phase(
        lane,
        [&] { require_same_layout(state, residual, Model::n_vars, "generated flux residual"); },
        "generated prepared flux preflight failed collectively");
    prepare_state_with_external(point, state, boundary, lane, transport);
    transport.with_boundary_scratch(state, [&](auto& scratch) {
      auto& faces = scratch.generated_faces;
      const auto omitted_faces =
          boundary == nullptr ? std::array<bool, 2 * Dim>{} : boundary->omitted_interface_faces();
      prepared_boundary_collective_phase(
          lane,
          [&] {
            for (std::size_t local = 0; local < state.local_size(); ++local) {
              if constexpr (flux_provider_count<Model> == 0)
                spatial.materialize_face_fluxes(state.fab(local), faces[local],
                                                scratch.cartesian_operator.face_candidate(local),
                                                scratch.cartesian_operator.face_status(local),
                                                omitted_faces,
                                                &scratch.cartesian_operator.face_speed(local));
              else
                spatial.materialize_face_fluxes(
                    state.fab(local),
                    runtime::system::bind_provider_storage_view<Dim, flux_provider_count<Model>>(
                        provider_plan, provider_storage, local),
                    faces[local], scratch.cartesian_operator.face_candidate(local),
                    scratch.cartesian_operator.face_status(local), omitted_faces,
                    &scratch.cartesian_operator.face_speed(local));
              if (boundary != nullptr)
                boundary->apply_physical_flux_conditions(faces[local], geometry.domain());
            }
          },
          "generated prepared face materialization failed collectively");
      if (boundary != nullptr && *external_boundary_flux) {
        prepared_boundary_collective_phase(
            lane, [&] { (*external_boundary_flux)(point, state, faces, geometry, lane); },
            "generated prepared boundary-flux component failed collectively");
      }
      prepared_boundary_collective_phase(
          lane,
          [&] {
            spatial.assemble_residual_from_face_fluxes(
                faces, residual, scratch.cartesian_operator.residual_candidate(),
                scratch.cartesian_operator.residual_status());
          },
          "generated prepared divergence materialization failed collectively");
    });
  };
  auto full_with_transport =
      [flux_with_transport, model, provider_storage_owner, provider_plan_owner, provider_storage,
       provider_plan](const runtime::multiblock::BoundaryEvaluationPoint& point,
                      MultiFab<Dim>& state, MultiFab<Dim>& residual,
                      const PreparedHyperbolicBoundary<Dim>* boundary, const ExecutionLane& lane,
                      const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        transport.with_boundary_scratch(state, [&](auto& scratch) {
          flux_with_transport(point, state, scratch.generated_candidate, boundary, lane, transport);
          Real local_status = Real(0);
          prepared_boundary_collective_phase(
              lane,
              [&] {
                materialize_source_into<Dim>(model, state, provider_storage, provider_plan,
                                             scratch.generated_source,
                                             scratch.generated_source_status);
                local_status = reduce_max_local(scratch.generated_source_status);
              },
              "generated prepared source materialization failed collectively");
          if (all_reduce_max(local_status, lane) != Real(0))
            throw std::runtime_error("generated prepared source produced a non-finite component");
          prepared_boundary_collective_phase(
              lane,
              [&] {
                saxpy(scratch.generated_candidate, Real(1), scratch.generated_source);
                copy_valid(scratch.generated_candidate, residual);
              },
              "generated prepared source accumulation failed collectively");
        });
      };
  auto source = [model, provider_storage_owner, provider_plan_owner, provider_storage,
                 provider_plan](MultiFab<Dim>& state, MultiFab<Dim>& residual) {
    MultiFab<Dim> candidate =
        materialize_source<Dim>(model, state, provider_storage, provider_plan);
    copy_valid(candidate, residual);
  };
  typename SystemBlockClosures<Dim>::EmbeddedResidual embedded_source =
      [model, provider_storage_owner, provider_plan_owner, provider_storage, provider_plan](
          MultiFab<Dim>& state, MultiFab<Dim>& residual, const auto& embedded) {
        MultiFab<Dim> candidate =
            materialize_masked_source<Dim>(model, state, provider_storage, provider_plan, embedded);
        copy_valid(candidate, residual);
      };

  nd::BoundaryFaceOmission<Dim> physical_omission;
  physical_omission.domain = request.geometry.domain();
  for (int axis = 0; axis < Dim; ++axis) {
    const bool periodic = request.topology.is_periodic(Face<Dim>{axis, BoundarySide::lower});
    physical_omission.lower[axis] = !periodic;
    physical_omission.upper[axis] = !periodic;
  }
  auto masked_operator =
      nd::prepare_masked_cartesian_operator<Dim, Model,
                                            std::remove_cvref_t<decltype(spatial.metric())>,
                                            Reconstruction, Numerical, Variables>(
          model, spatial.metric(), reconstruction, numerical, request.routes.positivity_floor);
  auto staircase_flux = [masked_operator, prepare_state, provider_storage_owner,
                         provider_plan_owner, provider_storage, provider_plan, physical_omission](
                            MultiFab<Dim>& state, MultiFab<Dim>& residual, const auto& embedded) {
    prepare_state(state, nullptr);
    assemble_masked_residual_with_plan<Dim, Model>(masked_operator, state, provider_storage,
                                                   provider_plan, embedded.active_mask(), residual,
                                                   physical_omission);
  };

  PreparedSystemBlock<Dim> result;
  result.provider_identity = "pops.generated.cartesian.nd/" + std::to_string(Dim) + "/" +
                             request.routes.limiter + "/" + request.routes.riemann + "/" +
                             request.routes.reconstruction;
  result.provider_components = provider_count;
  result.ghosts = ghosts;

  result.closures.rhs_into = [full](MultiFab<Dim>& state, MultiFab<Dim>& residual) {
    full(state, residual, nullptr);
  };
  result.closures.rhs_flux_only = [flux](MultiFab<Dim>& state, MultiFab<Dim>& residual) {
    flux(state, residual, nullptr);
  };
  result.closures.source_only = source;
  result.closures.source_only_masked = source;
  result.closures.solve_implicit_source =
      [model, provider_storage_owner, provider_plan_owner, provider_storage, provider_plan](
          MultiFab<Dim>& state, Real dt, const NewtonOptions& options, const ExecutionLane& lane) {
        (void)provider_storage_owner;
        (void)provider_plan_owner;
        const auto provider_at = [provider_storage, provider_plan](std::size_t local) {
          if constexpr (provider_count == 0)
            return ProviderStorageView<Dim, 0>{};
          else
            return runtime::system::bind_provider_storage_view<Dim, provider_count>(
                provider_plan, provider_storage, local);
        };
        if constexpr (generated_system_detail::GeneratedSourceModel<Dim, Model>) {
          return backward_euler_source(model, provider_at, state, dt, options, lane);
        } else {
          return SolveOutcome::collective_lane(SolveReport::capability_failure(), lane);
        }
      };
  result.closures.staircase =
      make_embedded_residual_family<Dim>(std::move(staircase_flux), embedded_source);
  result.closures.cut_cell = make_cut_cell_residual_family<Dim, Variables>(
      model, spatial.metric(), reconstruction, numerical, request.routes.positivity_floor,
      prepare_state, provider_storage_owner, provider_plan_owner, provider_storage, provider_plan,
      physical_omission, embedded_source);
  result.closures.rhs_at_point = [full](const auto&, MultiFab<Dim>& state,
                                        MultiFab<Dim>& residual) {
    full(state, residual, nullptr);
  };
  result.closures.rhs_flux_only_at_point = [flux](const auto&, MultiFab<Dim>& state,
                                                  MultiFab<Dim>& residual) {
    flux(state, residual, nullptr);
  };
  result.closures.rhs_without_prepared_interfaces = result.closures.rhs_at_point;
  result.closures.rhs_flux_only_without_prepared_interfaces =
      result.closures.rhs_flux_only_at_point;
  result.closures.rhs_core_at_point = result.closures.rhs_at_point;
  result.closures.rhs_flux_only_core_at_point = result.closures.rhs_flux_only_at_point;
  result.closures.rhs_core_at_point_prepared =
      [full](const auto&, MultiFab<Dim>& state, MultiFab<Dim>& residual,
             const PreparedHyperbolicBoundary<Dim>& boundary) { full(state, residual, &boundary); };
  result.closures.rhs_flux_only_core_at_point_prepared =
      [flux](const auto&, MultiFab<Dim>& state, MultiFab<Dim>& residual,
             const PreparedHyperbolicBoundary<Dim>& boundary) { flux(state, residual, &boundary); };
  result.closures.boundary_full_at_point_prepared =
      [full_with_transport](const auto& point, MultiFab<Dim>& state, MultiFab<Dim>& residual,
                            const PreparedHyperbolicBoundary<Dim>& boundary,
                            const ExecutionLane& lane,
                            const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        full_with_transport(point, state, residual, &boundary, lane, transport);
      };
  result.closures.boundary_core_at_point_prepared =
      [full_with_transport](const auto& point, MultiFab<Dim>& state, MultiFab<Dim>& residual,
                            const PreparedHyperbolicBoundary<Dim>&, const ExecutionLane& lane,
                            const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        full_with_transport(point, state, residual, nullptr, lane, transport);
      };
  result.closures.boundary_flux_full_at_point_prepared =
      [flux_with_transport](const auto& point, MultiFab<Dim>& state, MultiFab<Dim>& residual,
                            const PreparedHyperbolicBoundary<Dim>& boundary,
                            const ExecutionLane& lane,
                            const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        flux_with_transport(point, state, residual, &boundary, lane, transport);
      };
  result.closures.boundary_flux_core_at_point_prepared =
      [flux_with_transport](const auto& point, MultiFab<Dim>& state, MultiFab<Dim>& residual,
                            const PreparedHyperbolicBoundary<Dim>&, const ExecutionLane& lane,
                            const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        flux_with_transport(point, state, residual, nullptr, lane, transport);
      };
  result.closures.periodic_full_at_point_prepared =
      [full_with_transport](const auto& point, MultiFab<Dim>& state, MultiFab<Dim>& residual,
                            const ExecutionLane& lane,
                            const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        full_with_transport(point, state, residual, nullptr, lane, transport);
      };
  result.closures.periodic_flux_at_point_prepared =
      [flux_with_transport](const auto& point, MultiFab<Dim>& state, MultiFab<Dim>& residual,
                            const ExecutionLane& lane,
                            const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        flux_with_transport(point, state, residual, nullptr, lane, transport);
      };
  auto compiled_boundary_residual =
      make_prepared_boundary_residual<Dim>(result.closures.boundary_full_at_point_prepared,
                                           result.closures.boundary_core_at_point_prepared);
  auto compiled_boundary_jvp = make_prepared_boundary_jvp<Dim>(compiled_boundary_residual);
  result.closures.boundary_residual_at_point_prepared = bind_external_field_boundary_residual<Dim>(
      std::move(compiled_boundary_residual), external_field_boundary_residual);
  result.closures.boundary_jvp_at_point_prepared = bind_external_field_boundary_jvp<Dim>(
      std::move(compiled_boundary_jvp), external_field_boundary_jvp);
  result.closures.external_boundary_flux = std::move(external_boundary_flux);
  result.closures.external_field_boundary_residual = std::move(external_field_boundary_residual);
  result.closures.external_field_boundary_jvp = std::move(external_field_boundary_jvp);
  result.closures.prepare_generated_state_at_point =
      [prepare_state](const auto&, MultiFab<Dim>& state) { prepare_state(state, nullptr); };
  result.closures.prepare_generated_state_at_point_prepared =
      [prepare_state](const auto&, MultiFab<Dim>& state,
                      const PreparedHyperbolicBoundary<Dim>& boundary) {
        prepare_state(state, &boundary);
      };
  result.closures.prepare_generated_state_with_transport_prepared =
      [prepare_state_with_external](
          const auto& point, MultiFab<Dim>& state, const PreparedHyperbolicBoundary<Dim>& boundary,
          const ExecutionLane& lane,
          const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
        prepare_state_with_external(point, state, &boundary, lane, transport);
      };
  result.closures.external_ghost_boundary = std::move(external_ghost_boundary);

  if constexpr (HasPointwiseProjection<Model>) {
    auto cartesian_projection = [model, provider_storage_owner, provider_plan_owner,
                                 provider_storage,
                                 provider_plan](MultiFab<Dim>& state, const ExecutionLane& lane) {
      apply_pointwise_projection<Dim, Model>(model, state, provider_storage, provider_plan, nullptr,
                                             lane);
    };
    result.closures.project = cartesian_projection;
    result.closures.project_masked = cartesian_projection;
    auto embedded_projection =
        [model, provider_storage_owner, provider_plan_owner, provider_storage, provider_plan](
            MultiFab<Dim>& state, const auto& embedded, const ExecutionLane& lane) {
          apply_pointwise_projection<Dim, Model>(model, state, provider_storage, provider_plan,
                                                 &embedded.active_mask(), lane);
        };
    result.closures.staircase.project = embedded_projection;
    result.closures.cut_cell.project = embedded_projection;
  }

  result.maximum_speed = [model, provider_storage_owner, provider_plan_owner, provider_storage,
                          provider_plan](const MultiFab<Dim>& state, const ExecutionLane& lane) {
    return maximum_speed<Dim>(model, state, provider_storage, provider_plan, lane);
  };
  result.poisson_rhs = [model](const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
    add_poisson_rhs<Dim>(model, state, rhs);
  };
  result.primitive_to_conservative = [model](const double* primitive, double* conservative) {
    generated_system_detail::publish_conservative_state(model, primitive, conservative);
  };
  auto recovery = std::make_shared<PreparedModelVariableInversionRecovery<Model>>(model);
  result.conservative_to_primitive = [recovery](const double* conservative, double* primitive) {
    Real input[Model::n_vars]{};
    for (int component = 0; component < Model::n_vars; ++component)
      input[component] = static_cast<Real>(conservative[component]);
    const auto prepared = recovery->recover(input);
    const RecoveryOutcome<Model::n_vars>& outcome = prepared.outcome;
    if (outcome.publication_permitted())
      for (int component = 0; component < Model::n_vars; ++component)
        primitive[component] = static_cast<double>(outcome.value[component]);
    return recovery_report(outcome);
  };
  result.batch_conservative_to_primitive = make_uniform_variable_inversion_consumer(recovery);

  if constexpr (requires(const Model& value, const typename Model::State& state,
                         const ProviderValues<provider_count>& providers) {
                  value.source_frequency(state, providers);
                }) {
    result.source_frequency = [model, provider_storage_owner, provider_plan_owner, provider_storage,
                               provider_plan](const MultiFab<Dim>& state) {
      MultiFab<Dim> values(state.layout(), state.distribution(), state.local_rank(), 1,
                           state.ghosts());
      for (std::size_t local = 0; local < state.local_size(); ++local)
        for_each_cell(state.box(local),
                      MaterializeSourceFrequency<Dim, Model>{
                          model, state.fab(local).view(),
                          runtime::system::bind_provider_storage_view<Dim, provider_count>(
                              provider_plan, provider_storage, local),
                          values.fab(local).view()});
      const Real frequency = reduce_max(values);
      if (!std::isfinite(frequency) || frequency < Real(0))
        throw std::runtime_error("generated source frequency is invalid");
      return frequency;
    };
  }
  if constexpr (DiffusiveModel<Model>) {
    const Real q = parabolic_frequency<Dim>(model, spatial.metric());
    result.parabolic_frequency = q;
  }
  if constexpr (requires(const Model& value, const typename Model::State& state,
                         const ProviderValues<provider_count>& providers) {
                  value.stability_dt(state, providers);
                }) {
    auto model_stability_dt = [model, provider_storage_owner, provider_plan_owner, provider_storage,
                               provider_plan](const MultiFab<Dim>& state) {
      MultiFab<Dim> values(state.layout(), state.distribution(), state.local_rank(), 1,
                           state.ghosts());
      for (std::size_t local = 0; local < state.local_size(); ++local)
        for_each_cell(state.box(local),
                      MaterializeStabilityDt<Dim, Model>{
                          model, state.fab(local).view(),
                          runtime::system::bind_provider_storage_view<Dim, provider_count>(
                              provider_plan, provider_storage, local),
                          values.fab(local).view()});
      const Real dt = reduce_min(values);
      if (!std::isfinite(dt) || !(dt > Real(0)))
        throw std::runtime_error("generated stability dt is invalid");
      return dt;
    };
    result.stability_dt = std::move(model_stability_dt);
  }
  return result;
}

/// Storage and conversion support for a Program-owned spatial operator. No transport law
/// is installed by this adapter: selecting a legacy residual is a capability error.
template <int Dim, class Request>
PreparedSystemBlock<Dim> materialize_state_block(Request request) {
  using Model = std::remove_cvref_t<decltype(request.model)>;
  static_assert(PhysicalStateFor<Model, Dim>);
  if constexpr (path_conservative_model<Model>) {
    const std::string expected_face = coordinated_face_model<Model>
        ? std::string("coordinated_face:v1:") + std::string(Model::path_operator_identity())
        : std::string("rusanov");
    if (request.routes.limiter != "none" || request.routes.riemann != expected_face ||
        request.routes.reconstruction != "conservative" ||
        !std::isfinite(request.routes.positivity_floor) ||
        request.routes.positivity_floor != Real(0) || Model::path_operator_identity().empty())
      throw std::invalid_argument(
          "Uniform path transport requires exact first-order conservative face authority");
  } else {
    if (request.routes.limiter != "state_storage" || request.routes.riemann != "unavailable" ||
        request.routes.reconstruction != "conservative")
      throw std::invalid_argument("Program-only model requires the exact state-storage route");
  }
  constexpr int provider_count = provider_count_for<Model, Dim>();
  if constexpr (provider_count > 0) {
    if (!request.provider_plan || !request.provider_storage ||
        request.provider_plan->value_count() != static_cast<std::size_t>(provider_count))
      throw std::invalid_argument("state-storage model requires its exact provider plan");
  } else if (request.provider_plan || request.provider_storage) {
    throw std::invalid_argument("provider-free state storage cannot retain provider state");
  }
  const auto model = request.model;
  const auto geometry = request.geometry;
  const auto topology = request.topology;
  Extent<Dim> ghosts{};
  static_assert(Model::program_state_ghost_depth >= 1,
                "Program state storage requires its resolved positive ghost depth");
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = Model::program_state_ghost_depth;
  auto prepare = [geometry, topology, ghosts](MultiFab<Dim>& state) {
    const HaloSchedule<Dim> schedule(state.layout(), state.distribution(), state.local_rank(),
                                     geometry.domain(), ghosts, topology, state.ncomp(),
                                     halo_budget(state, geometry.domain(), topology, ghosts));
    fill_boundary(state, schedule);
  };
  PreparedSystemBlock<Dim> result;
  result.provider_identity =
      std::string(path_conservative_model<Model> ? "pops.generated.path.nd/"
                                                : "pops.generated.program-state.nd/") +
      std::to_string(Dim);
  result.provider_components = provider_count;
  result.ghosts = ghosts;
  auto unavailable = [](auto&&...) -> void {
    throw std::logic_error("state-storage block has no selected legacy spatial residual");
  };
  result.closures.rhs_into = unavailable;
  result.closures.rhs_flux_only = unavailable;
  result.closures.source_only = unavailable;
  result.closures.source_only_masked = unavailable;
  result.closures.rhs_at_point = unavailable;
  result.closures.rhs_flux_only_at_point = unavailable;
  result.closures.rhs_core_at_point = unavailable;
  result.closures.rhs_flux_only_core_at_point = unavailable;
  result.closures.rhs_without_prepared_interfaces = unavailable;
  result.closures.rhs_flux_only_without_prepared_interfaces = unavailable;
  result.closures.rhs_core_at_point_prepared = unavailable;
  result.closures.rhs_flux_only_core_at_point_prepared = unavailable;
  result.closures.prepare_generated_state_at_point = [prepare](const auto&, MultiFab<Dim>& state) {
    prepare(state);
  };
  result.closures.prepare_generated_state_at_point_prepared =
      [](const auto&, MultiFab<Dim>&, const PreparedHyperbolicBoundary<Dim>&) {
        throw std::logic_error("state storage cannot consume a hyperbolic boundary law");
      };
  result.closures.prepare_generated_state_with_transport_prepared = unavailable;
  result.closures.external_ghost_boundary =
      std::make_shared<typename SystemBlockClosures<Dim>::ExternalGhostBoundary>();
  result.maximum_speed = [](const MultiFab<Dim>&, const ExecutionLane&) -> Real {
    throw std::logic_error("state storage has no hyperbolic wave-speed provider");
  };
  // Storage is a state capability, not an elliptic source. An empty callback
  // lets the field planner distinguish absence from an authored RHS provider.
  result.poisson_rhs = {};
  result.primitive_to_conservative = [model](const double* primitive, double* conservative) {
    publish_conservative_state(model, primitive, conservative);
  };
  auto recovery = std::make_shared<PreparedModelVariableInversionRecovery<Model>>(model);
  result.conservative_to_primitive = [recovery](const double* conservative, double* primitive) {
    Real input[Model::n_vars]{};
    for (int component = 0; component < Model::n_vars; ++component)
      input[component] = static_cast<Real>(conservative[component]);
    const auto prepared = recovery->recover(input);
    const RecoveryOutcome<Model::n_vars>& outcome = prepared.outcome;
    if (outcome.publication_permitted())
      for (int component = 0; component < Model::n_vars; ++component)
        primitive[component] = static_cast<double>(outcome.value[component]);
    return recovery_report(outcome);
  };
  result.batch_conservative_to_primitive = make_uniform_variable_inversion_consumer(recovery);
  if constexpr (path_conservative_model<Model>) {
    result.physical_boundary_route = PreparedPhysicalBoundaryRoute::path_residual;
    const auto spatial = nd::prepare_cartesian_operator<
        Dim, Model, NoSlope, ModelPathFlux<Model>,
        nd::ReconstructionVariables::Conservative>(
        geometry, model, NoSlope{}, ModelPathFlux<Model>{});
    const auto provider_storage_owner = request.provider_storage;
    const auto provider_plan_owner = request.provider_plan;
    const auto* provider_storage = provider_storage_owner.get();
    const auto* provider_plan = provider_plan_owner.get();
    result.maximum_speed = [model, provider_storage_owner, provider_plan_owner,
                            provider_storage, provider_plan](const MultiFab<Dim>& state,
                                                             const ExecutionLane& lane) {
      return maximum_speed<Dim>(model, state, provider_storage, provider_plan, lane);
    };
    result.closures.path_rhs_at_point_prepared =
        [model, spatial, geometry, provider_storage_owner, provider_plan_owner,
         provider_storage, provider_plan](
            const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state,
            MultiFab<Dim>& residual,
            const PreparedHyperbolicBoundary<Dim>* boundary, const ExecutionLane& lane,
            const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
          constexpr int flux_count = flux_provider_count<Model>;
          std::array<bool, 2 * Dim> omitted{};
          prepared_boundary_collective_phase(
              lane,
              [&] {
                require_same_layout(state, residual, Model::n_vars,
                                    "generated Uniform path residual");
                if (boundary != nullptr) {
                  boundary->template require_model_qualified_characteristic_provider<Model>();
                  omitted = boundary->omitted_interface_faces();
                }
                if constexpr (flux_count > 0)
                  runtime::system::require_pointwise_provider_groups<Dim, flux_count>(
                      state, provider_storage, provider_plan, "generated Uniform path providers");
                for (int axis = 0; axis < Dim; ++axis)
                  if (!std::isfinite(geometry.spacing(axis)) ||
                      !(geometry.spacing(axis) > Real(0)))
                    throw std::invalid_argument("Uniform path geometry has invalid spacing");
              },
              "generated Uniform path preflight failed collectively");
          prepared_boundary_collective_phase(
              lane,
              [&] {
                if (boundary != nullptr)
                  transport.fill_halo(state);
                else
                  transport.fill(state);
              },
              "generated Uniform path halo preparation failed collectively");
          if (boundary != nullptr)
            prepared_boundary_collective_phase(
                lane,
                [&] {
                  transport.with_characteristic_candidate(state, [&](MultiFab<Dim>& candidate) {
                    boundary->fill_physical_model_qualified(
                        state, geometry, model, lane, candidate,
                        static_cast<Real>(point.physical_time));
                  });
                },
                "generated Uniform path physical boundary failed collectively");
          Real local_frequency = Real(0);
          prepared_boundary_collective_phase(
              lane,
              [&] {
                for (std::size_t local = 0; local < state.local_size(); ++local) {
                  const auto& patch = state.fab(local);
                  nd::FaceField<Dim> flux(patch.box(), Model::n_vars);
                  nd::FaceField<Dim> left(patch.box(), Model::n_vars);
                  nd::FaceField<Dim> right(patch.box(), Model::n_vars);
                  nd::FaceField<Dim> speed(patch.box(), 1);
                  nd::PreparedCartesianPathFaceScratch<Dim> face_scratch(patch.box(),
                                                                          Model::n_vars);
                  if constexpr (flux_count == 0)
                    spatial.materialize_path_face_contributions(
                        patch, flux, left, right, speed, face_scratch, omitted);
                  else
                    spatial.materialize_path_face_contributions(
                        patch,
                        runtime::system::bind_provider_storage_view<Dim, flux_count>(
                            provider_plan, provider_storage, local),
                        flux, left, right, speed, face_scratch, omitted);
                  Fab<Dim> candidate(patch.box(), Model::n_vars, residual.ghosts());
                  Fab<Dim> statuses(patch.box(), 1, residual.ghosts());
                  spatial.assemble_residual_from_path_faces(
                      flux, left, right, residual.fab(local), candidate, statuses);
                  const auto speeds = speed.view();
                  RealVector<Dim> inverse_spacing{};
                  for (int axis = 0; axis < Dim; ++axis)
                    inverse_spacing[axis] = Real(1) / geometry.spacing(axis);
                  const Real frequency = for_each_cell_reduce_max(
                      patch.box(), [=] POPS_HD(const Index<Dim>& cell) {
                        Real value = Real(0);
                        for (int axis = 0; axis < Dim; ++axis) {
                          Index<Dim> upper = cell;
                          ++upper[axis];
                          const Real lower_speed = speeds.axes[axis](cell, 0);
                          const Real upper_speed = speeds.axes[axis](upper, 0);
                          if (!Kokkos::isfinite(lower_speed) ||
                              !Kokkos::isfinite(upper_speed) ||
                              lower_speed < Real(0) || upper_speed < Real(0))
                            return std::numeric_limits<Real>::infinity();
                          value += (lower_speed + upper_speed) * inverse_spacing[axis];
                        }
                        return Kokkos::isfinite(value) ? value
                                                       : std::numeric_limits<Real>::infinity();
                      });
                  local_frequency = std::max(local_frequency, frequency);
                }
                device_fence();
              },
              "generated Uniform path face/residual materialization failed collectively");
          const Real frequency = all_reduce_max(local_frequency, lane);
          if (!std::isfinite(frequency) || frequency < Real(0))
            throw std::runtime_error(
                "Uniform path RHS has an invalid incident-face stability frequency");
          // A residual has no time-step weight. Its explicit consumer supplies
          // the Shu--Osher state/rate weights and checks this actual face bound.
          return frequency;
        };
  }
  return result;
}

template <int Dim, nd::ReconstructionVariables Variables, class Request, class Reconstruction>
PreparedSystemBlock<Dim> select_riemann(Request request, Reconstruction reconstruction) {
  using Model = std::remove_cvref_t<decltype(request.model)>;
  switch (parse_riemann_route(request.routes.riemann, "generated System block")) {
    case RiemannRouteId::kRusanov:
      return materialize_block<Dim, Model, Reconstruction, RusanovFlux, Variables>(
          std::move(request), reconstruction, RusanovFlux{});
    case RiemannRouteId::kHll:
      if constexpr (detail::wave_speeds_all_axes<Model>())
        return materialize_block<Dim, Model, Reconstruction, HLLFlux, Variables>(
            std::move(request), reconstruction, HLLFlux{});
      break;
    case RiemannRouteId::kHllc:
      if constexpr (HasHLLCStructure<Model>)
        return materialize_block<Dim, Model, Reconstruction, HLLCFlux, Variables>(
            std::move(request), reconstruction, HLLCFlux{});
      break;
    case RiemannRouteId::kRoe:
      if constexpr (HasRoeDissipation<Model>)
        return materialize_block<Dim, Model, Reconstruction, RoeFlux, Variables>(
            std::move(request), reconstruction, RoeFlux{});
      break;
    case RiemannRouteId::kRoeHllRusanovRecovery:
      if constexpr (HasRoeDissipation<Model> && detail::wave_speeds_all_axes<Model>())
        return materialize_block<Dim, Model, Reconstruction, RoeHllRusanovRecoveryPolicy,
                                 Variables>(std::move(request), reconstruction,
                                            RoeHllRusanovRecoveryPolicy{});
      break;
  }
  throw std::invalid_argument("generated model does not satisfy the requested Riemann capability");
}

template <int Dim, nd::ReconstructionVariables Variables, class Request>
PreparedSystemBlock<Dim> select_reconstruction(Request request) {
  switch (parse_limiter_route(request.routes.limiter, "generated System block")) {
    case LimiterRouteId::kNone:
      return select_riemann<Dim, Variables>(std::move(request), NoSlope{});
    case LimiterRouteId::kMinmod:
      return select_riemann<Dim, Variables>(std::move(request), Minmod{});
    case LimiterRouteId::kVanLeer:
      return select_riemann<Dim, Variables>(std::move(request), VanLeer{});
    case LimiterRouteId::kWeno5:
      return select_riemann<Dim, Variables>(std::move(request), configured_reconstruction<Weno5>());
    case LimiterRouteId::kMc:
      return select_riemann<Dim, Variables>(std::move(request), MC{});
    case LimiterRouteId::kSuperbee:
      return select_riemann<Dim, Variables>(std::move(request), Superbee{});
  }
  throw std::logic_error("generated limiter route escaped its exhaustive selector");
}

template <int Dim, nd::ReconstructionVariables Variables, class Request, class Numerical>
PreparedSystemBlock<Dim> select_reconstruction_with_face(Request request, Numerical numerical) {
  using Model = std::remove_cvref_t<decltype(request.model)>;
  switch (parse_limiter_route(request.routes.limiter, "generated System face block")) {
    case LimiterRouteId::kNone:
      return materialize_block<Dim, Model, NoSlope, Numerical, Variables>(
          std::move(request), NoSlope{}, numerical);
    case LimiterRouteId::kMinmod:
      return materialize_block<Dim, Model, Minmod, Numerical, Variables>(
          std::move(request), Minmod{}, numerical);
    case LimiterRouteId::kVanLeer:
      return materialize_block<Dim, Model, VanLeer, Numerical, Variables>(
          std::move(request), VanLeer{}, numerical);
    case LimiterRouteId::kWeno5:
      return materialize_block<Dim, Model, Weno5, Numerical, Variables>(
          std::move(request), configured_reconstruction<Weno5>(), numerical);
    case LimiterRouteId::kMc:
      return materialize_block<Dim, Model, MC, Numerical, Variables>(
          std::move(request), MC{}, numerical);
    case LimiterRouteId::kSuperbee:
      return materialize_block<Dim, Model, Superbee, Numerical, Variables>(
          std::move(request), Superbee{}, numerical);
  }
  throw std::logic_error("generated face limiter route escaped its exhaustive selector");
}

}  // namespace generated_system_detail

/// Materialize the exact-ranked elliptic RHS closure owned by one bound generated model. Native
/// System and AMR packages use the same typed closure; only the host field layout differs.
template <class Model>
constexpr unsigned poisson_rhs_read_contract_version() {
  if constexpr (requires { Model::prepared_field_rhs_read_contract_version; })
    return Model::prepared_field_rhs_read_contract_version;
  return 0;
}
template <class Model>
constexpr unsigned poisson_rhs_state_read_cells() {
  if constexpr (requires { Model::prepared_field_rhs_state_read_cells; })
    return Model::prepared_field_rhs_state_read_cells;
  return 0; // Meaningful only when the explicit contract version is nonzero.
}

template <class Model>
  requires PhysicalEllipticRhsFor<Model, kNativeDimension>
auto make_poisson_rhs(Model model) {
  return [model = std::move(model)](const MultiFab<kNativeDimension>& state,
                                    MultiFab<kNativeDimension>& rhs) {
    generated_system_detail::add_poisson_rhs<kNativeDimension>(model, state, rhs);
  };
}

/// Provider-aware Field density on separately authenticated State and readonly Aux images.
/// Every allocation/view is prepared and voted before the first rank launches a kernel.
template <class Model>
auto make_poisson_rhs_v2(Model model) {
  constexpr int Dim = kNativeDimension;
  constexpr int Count = [] {
    if constexpr (requires { Model::n_providers; }) return Model::n_providers;
    else return 0;
  }();
  static_assert(Count >= 0);
  return [model = std::move(model)](
      const runtime::system::PreparedFieldRhsInputs<Dim>& inputs, MultiFab<Dim>& rhs) {
    const ExecutionLane& lane = inputs.consensus_lane();
    struct Workspace {
      Kokkos::DefaultExecutionSpace execution;
      std::optional<MultiFab<Dim>> candidate, status;
      std::vector<generated_system_detail::MaterializePoissonRhsV2<Dim, Model, Count>> kernels;
      std::vector<std::pair<FieldView<Real, Dim>, FieldView<const Real, Dim>>> outputs;
      std::vector<typename Fab<Dim>::storage_type> output_allocations;
    };
    std::shared_ptr<Workspace> workspace;
    std::exception_ptr error;
    try {
      workspace = std::make_shared<Workspace>();
      workspace->execution = inputs.execution();
      inputs.template validate_provider_inputs<Count>();
      const auto& state = inputs.state();
      if (rhs.ncomp() != 1 || state.ncomp() != Model::n_vars)
        throw std::invalid_argument("Field RHS V2 requires its exact scalar/physical-State arity");
      generated_system_detail::require_same_layout(state, rhs, 1, "Field RHS V2");
      workspace->candidate.emplace(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
      workspace->status.emplace(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
      workspace->kernels.reserve(state.local_size());
      workspace->outputs.reserve(state.local_size());
      workspace->output_allocations.reserve(state.local_size());
      for (std::size_t local = 0; local < state.local_size(); ++local) {
        workspace->kernels.push_back({model, state.fab(local).view(),
            inputs.template provider_values_view<Count>(local),
            workspace->candidate->fab(local).view(), workspace->status->fab(local).view()});
        workspace->output_allocations.push_back(rhs.fab(local).storage());
        workspace->outputs.emplace_back(rhs.fab(local).view(),
            std::as_const(*workspace->candidate).fab(local).view());
      }
      inputs.retain_execution_owner(workspace);
    } catch (...) { error = std::current_exception(); }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        error, &lane, "Field RHS V2 input/allocation preparation failed collectively");
    try {
      for (std::size_t local = 0; local < workspace->kernels.size(); ++local)
        for_each_cell(workspace->execution, workspace->candidate->box(local), workspace->kernels[local]);
      workspace->execution.fence();
      if (!workspace->kernels.empty() && reduce_max_local(*workspace->status) != Real(0))
        throw std::runtime_error("Field RHS V2 produced a non-finite candidate");
    } catch (...) {
      error = std::current_exception();
      try { workspace->execution.fence(); }
      catch (...) { error = std::current_exception(); }
    }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        error, &lane, "Field RHS V2 candidate failed collectively before publication");
    try {
      for (std::size_t local = 0; local < workspace->outputs.size(); ++local) {
        const auto target = workspace->outputs[local].first;
        const auto candidate = workspace->outputs[local].second;
        for_each_cell(workspace->execution, workspace->candidate->box(local),
            [=] POPS_HD(const Index<Dim>& index) { target(index) += candidate(index); });
      }
      workspace->execution.fence();
    } catch (...) {
      error = std::current_exception();
      try { workspace->execution.fence(); }
      catch (...) { error = std::current_exception(); }
    }
    runtime::system::auxiliary_ghost_detail::rethrow_collective_failure(
        error, &lane, "Field RHS V2 publication failed collectively");
  };
}

/// Build the single exact numerical specialization requested by a generated package.
template <class Request>
auto prepare_generated_system_block(Request request) -> PreparedSystemBlock<Request::dimension> {
  constexpr int Dim = Request::dimension;
  using Model = std::remove_cvref_t<decltype(request.model)>;
  static_assert(Model::dimension == Dim,
                "generated System request and physical model have different ranks");
  constexpr bool storage_only = [] {
    if constexpr (requires { Model::program_only_storage; })
      return static_cast<bool>(Model::program_only_storage);
    return false;
  }();
  if constexpr (storage_only || path_conservative_model<Model>) {
    return generated_system_detail::materialize_state_block<Dim>(std::move(request));
  } else {
    switch (parse_recon_route(request.routes.reconstruction, "generated System block")) {
      case ReconRouteId::kConservative:
        return generated_system_detail::select_reconstruction<
            Dim, nd::ReconstructionVariables::Conservative>(std::move(request));
      case ReconRouteId::kPrimitive:
        return generated_system_detail::select_reconstruction<
            Dim, nd::ReconstructionVariables::Primitive>(std::move(request));
    }
    throw std::logic_error("generated reconstruction route escaped its exhaustive selector");
  }
}

/// A source-authored policy is a type in this exact compiled package. Its wire
/// spelling authenticates that type's body identity; no catalogue token may
/// select an unrelated reconstruction implementation.
template <class Request, class Reconstruction>
  requires ReconstructionPolicy<Reconstruction>
auto prepare_generated_system_block(Request request, Reconstruction reconstruction)
    -> PreparedSystemBlock<Request::dimension> {
  constexpr int Dim = Request::dimension;
  using Model = std::remove_cvref_t<decltype(request.model)>;
  static_assert(Model::dimension == Dim);
  static_assert(StencilReconstruction<Reconstruction>);
  static_assert(stencil_envelope_fits_storage<Reconstruction>);
  if (request.routes.limiter !=
      std::string("source_stencil:") + Reconstruction::source_identity)
    throw std::invalid_argument("generated System reconstruction source identity differs from package");
  if constexpr (path_conservative_model<Model>) {
    throw std::invalid_argument("source reconstruction needs a composed path face policy");
  } else {
    switch (parse_recon_route(request.routes.reconstruction, "generated System block")) {
      case ReconRouteId::kConservative:
        return generated_system_detail::select_riemann<
            Dim, nd::ReconstructionVariables::Conservative>(
            std::move(request), reconstruction);
      case ReconRouteId::kPrimitive:
        return generated_system_detail::select_riemann<
            Dim, nd::ReconstructionVariables::Primitive>(
            std::move(request), reconstruction);
    }
    throw std::logic_error("generated reconstruction route escaped its exhaustive selector");
  }
}

/// A source-authored face body is instantiated only by its owning package and
/// still flows through the ordinary shared conservative face operator.
template <class Request, class Numerical>
  requires requires { Numerical::source_identity; } && (!ReconstructionPolicy<Numerical>)
auto prepare_generated_system_block(Request request, Numerical numerical)
    -> PreparedSystemBlock<Request::dimension> {
  constexpr int Dim = Request::dimension;
  using Model = std::remove_cvref_t<decltype(request.model)>;
  if (request.routes.riemann != std::string("source_face:") + Numerical::source_identity)
    throw std::invalid_argument("generated System face source identity differs from package");
  if constexpr (path_conservative_model<Model>) {
    throw std::invalid_argument("source face needs a composed path face policy");
  } else {
    switch (parse_recon_route(request.routes.reconstruction, "generated System face block")) {
      case ReconRouteId::kConservative:
        return generated_system_detail::select_reconstruction_with_face<
            Dim, nd::ReconstructionVariables::Conservative>(std::move(request), numerical);
      case ReconRouteId::kPrimitive:
        return generated_system_detail::select_reconstruction_with_face<
            Dim, nd::ReconstructionVariables::Primitive>(std::move(request), numerical);
    }
    throw std::logic_error("generated face reconstruction route escaped its exhaustive selector");
  }
}

template <class Request, class Reconstruction, class Numerical>
  requires ReconstructionPolicy<Reconstruction> && requires { Numerical::source_identity; }
auto prepare_generated_system_block(Request request, Reconstruction reconstruction,
                                    Numerical numerical)
    -> PreparedSystemBlock<Request::dimension> {
  constexpr int Dim = Request::dimension;
  using Model = std::remove_cvref_t<decltype(request.model)>;
  static_assert(StencilReconstruction<Reconstruction>);
  static_assert(stencil_envelope_fits_storage<Reconstruction>);
  if (request.routes.limiter !=
      std::string("source_stencil:") + Reconstruction::source_identity ||
      request.routes.riemann != std::string("source_face:") + Numerical::source_identity)
    throw std::invalid_argument("generated System numerical source identity differs from package");
  if constexpr (path_conservative_model<Model>) {
    throw std::invalid_argument("source face/reconstruction need a composed path face policy");
  } else {
    switch (parse_recon_route(request.routes.reconstruction, "generated System source block")) {
      case ReconRouteId::kConservative:
        return generated_system_detail::materialize_block<
            Dim, Model, Reconstruction, Numerical, nd::ReconstructionVariables::Conservative>(
            std::move(request), reconstruction, numerical);
      case ReconRouteId::kPrimitive:
        return generated_system_detail::materialize_block<
            Dim, Model, Reconstruction, Numerical, nd::ReconstructionVariables::Primitive>(
            std::move(request), reconstruction, numerical);
    }
    throw std::logic_error("generated source reconstruction route escaped its exhaustive selector");
  }
}

}  // namespace pops
