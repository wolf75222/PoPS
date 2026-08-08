/// @file
/// @brief Final exact-ranked Uniform block factory used by generated native packages.

#pragma once

#include <pops/core/model/physical_model.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/fv/numerical_flux.hpp>
#include <pops/numerics/fv/reconstruction.hpp>
#include <pops/numerics/spatial/operators/cartesian_operator.hpp>
#include <pops/numerics/spatial/primitives/state_access.hpp>
#include <pops/runtime/config/route_ids.hpp>
#include <pops/runtime/recovery/uniform_recovery_consumer.hpp>
#include <pops/runtime/system/system_block_closures.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace pops {
namespace generated_system_detail {

/// Convert one primitive cell without exposing a rejected or partially materialized candidate.
///
/// Generated Uniform and AMR blocks share this publication seam.  The model-owned inverse is
/// evaluated before the caller's buffer is touched so a provider that reports success while
/// producing a non-finite or non-recoverable conservative state still fails closed.
template <class Model>
void publish_conservative_state(const Model& model, const double* primitive,
                                double* conservative) {
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
concept GeneratedSourceModel =
    requires(const Model& model, const typename Model::State& state,
             const AuxState<Dim>& auxiliary) {
      { model.source(state, auxiliary) } -> std::same_as<typename Model::State>;
    };

template <class Model>
concept GeneratedEllipticRhsModel =
    requires(const Model& model, const typename Model::State& state) {
      { model.elliptic_rhs(state) } -> std::convertible_to<Real>;
    };

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
struct MaterializeSource {
  Model model;
  FieldView<const Real, Dim> state{};
  FieldView<const Real, Dim> auxiliary{};
  FieldView<Real, Dim> source{};
  FieldView<Real, Dim> status{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    const auto value = model.source(load_state<Model>(state, index),
                                    load_aux<aux_comps_for<Model, Dim>()>(auxiliary, index));
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

template <int Axis, int Dim, class Model>
POPS_HD Real maximum_axis_speed(const Model& model, const typename Model::State& state,
                                const BoundFluxProviders<Model>& providers) {
  const Real local = detail::model_max_wave_speed_at<Axis>(model, state, providers);
  if constexpr (Axis + 1 < Dim) {
    const Real remainder = maximum_axis_speed<Axis + 1, Dim>(model, state, providers);
    return local > remainder ? local : remainder;
  }
  return local;
}

template <int Dim, class Model>
struct MaterializeMaximumSpeed {
  Model model;
  FieldView<const Real, Dim> state{};
  FieldView<const Real, Dim> auxiliary{};
  FieldView<Real, Dim> speed{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    const auto providers = bind_flux_providers_at<Model>(auxiliary, index);
    speed(index) = maximum_axis_speed<0, Dim>(model, load_state<Model>(state, index), providers);
  }
};

template <int Dim, class Model>
struct MaterializeSourceFrequency {
  Model model;
  FieldView<const Real, Dim> state{};
  FieldView<const Real, Dim> auxiliary{};
  FieldView<Real, Dim> value{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    value(index) = model.source_frequency(
        load_state<Model>(state, index),
        load_aux<aux_comps_for<Model, Dim>()>(auxiliary, index));
  }
};

template <int Dim, class Model>
struct MaterializeStabilityDt {
  Model model;
  FieldView<const Real, Dim> state{};
  FieldView<const Real, Dim> auxiliary{};
  FieldView<Real, Dim> value{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    value(index) = model.stability_dt(load_state<Model>(state, index),
                                      load_aux<aux_comps_for<Model, Dim>()>(auxiliary, index));
  }
};

inline std::size_t checked_product(std::size_t left, std::size_t right, const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(label);
  return left * right;
}

template <int Dim>
HaloScheduleBudget halo_budget(const MultiFab<Dim>& field, const Box<Dim>& domain,
                               const BoundaryTopology<Dim>& topology,
                               const Extent<Dim>& ghosts) {
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
  return {{boxes, boxes > 1 ? boxes * (boxes - 1) / 2 : 0}, work, jobs, images};
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
MultiFab<Dim> materialize_source(const Model& model, const MultiFab<Dim>& state,
                                 const MultiFab<Dim>& auxiliary) {
  require_same_layout(state, auxiliary, auxiliary.ncomp(), "generated source auxiliary");
  if (auxiliary.ncomp() < aux_comps_for<Model, Dim>())
    throw std::invalid_argument("generated source auxiliary field is too narrow");
  MultiFab<Dim> candidate(state.layout(), state.distribution(), state.local_rank(), Model::n_vars,
                          state.ghosts());
  if constexpr (GeneratedSourceModel<Dim, Model>) {
    MultiFab<Dim> status(state.layout(), state.distribution(), state.local_rank(), 1,
                         state.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    MaterializeSource<Dim, Model>{model, state.fab(local).view(),
                                                  auxiliary.fab(local).view(),
                                                  candidate.fab(local).view(),
                                                  status.fab(local).view()});
    if (reduce_max(status) != Real(0))
      throw std::runtime_error("generated source produced a non-finite component");
  } else {
    candidate.set_val(Real(0));
  }
  return candidate;
}

template <int Dim, class Model>
Real maximum_speed(const Model& model, const MultiFab<Dim>& state,
                   const MultiFab<Dim>& auxiliary) {
  require_same_layout(state, auxiliary, auxiliary.ncomp(), "generated speed auxiliary");
  if (auxiliary.ncomp() < flux_provider_count<Model>)
    throw std::invalid_argument("generated speed auxiliary field is too narrow");
  MultiFab<Dim> values(state.layout(), state.distribution(), state.local_rank(), 1, state.ghosts());
  for (std::size_t local = 0; local < state.local_size(); ++local)
    for_each_cell(state.box(local), MaterializeMaximumSpeed<Dim, Model>{
                                         model, state.fab(local).view(),
                                         auxiliary.fab(local).view(), values.fab(local).view()});
  const Real result = reduce_max(values);
  if (!std::isfinite(result) || result < Real(0))
    throw std::runtime_error("generated model produced an invalid maximum speed");
  return result;
}

template <int Dim, class Model>
void add_poisson_rhs(const Model& model, const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
  if (rhs.ncomp() != 1)
    throw std::invalid_argument("generated Poisson RHS destination must have one component");
  require_same_layout(state, rhs, 1, "generated Poisson RHS");
  if constexpr (GeneratedEllipticRhsModel<Model>) {
    MultiFab<Dim> candidate(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
    MultiFab<Dim> status(rhs.layout(), rhs.distribution(), rhs.local_rank(), 1, rhs.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local), MaterializePoissonRhs<Dim, Model>{
                                           model, state.fab(local).view(),
                                           candidate.fab(local).view(), status.fab(local).view()});
    if (reduce_max(status) != Real(0))
      throw std::runtime_error("generated Poisson RHS produced a non-finite value");
    saxpy(rhs, Real(1), candidate);
  }
}

template <int Dim, class Model, class Reconstruction, class Numerical,
          nd::ReconstructionVariables Variables, class Request>
PreparedSystemBlock<Dim> materialize_block(Request request, Reconstruction reconstruction,
                                           Numerical numerical) {
  static_assert(Model::dimension == Dim);
  if (request.auxiliary == nullptr)
    throw std::invalid_argument("generated System block requires the shared auxiliary owner");

  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = Reconstruction::n_ghost;
  const auto state_schedule = HaloSchedule<Dim>(
      request.auxiliary->layout(), request.auxiliary->distribution(),
      request.auxiliary->local_rank(), request.geometry.domain(), ghosts, request.topology,
      Model::n_vars,
      halo_budget(*request.auxiliary, request.geometry.domain(), request.topology, ghosts));
  const auto auxiliary_schedule = prepare_halo_schedule(
      *request.auxiliary, request.geometry.domain(), request.topology,
      halo_budget(*request.auxiliary, request.geometry.domain(), request.topology,
                  request.auxiliary->ghosts()));
  const auto spatial = nd::prepare_cartesian_operator<Dim, Model, Reconstruction, Numerical,
                                                       Variables>(
      request.geometry, request.model, reconstruction, numerical,
      request.routes.positivity_floor);
  const Model model = request.model;
  MultiFab<Dim>* const auxiliary = request.auxiliary;
  const Geometry<Dim> geometry = request.geometry;

  auto prepare_state = [state_schedule, auxiliary_schedule, auxiliary, geometry](
                           MultiFab<Dim>& state,
                           const PreparedHyperbolicBoundary<Dim>* boundary) {
    require_same_layout(state, *auxiliary, auxiliary->ncomp(), "generated flux auxiliary");
    fill_boundary(state, state_schedule);
    fill_boundary(*auxiliary, auxiliary_schedule);
    if (boundary != nullptr)
      boundary->fill_physical(state, geometry);
  };

  auto flux = [spatial, prepare_state, auxiliary, geometry](
                  MultiFab<Dim>& state, MultiFab<Dim>& residual,
                  const PreparedHyperbolicBoundary<Dim>* boundary) {
    require_same_layout(state, residual, Model::n_vars, "generated flux residual");
    prepare_state(state, boundary);

    auto faces = nd::make_face_flux_workspace(state);
    for (std::size_t local = 0; local < state.local_size(); ++local) {
      spatial.materialize_face_fluxes(state.fab(local), auxiliary->fab(local), faces[local]);
      if (boundary != nullptr)
        boundary->apply_physical_flux_conditions(faces[local], geometry.domain());
    }
    spatial.assemble_residual_from_face_fluxes(faces, residual);
  };

  auto full = [flux, model, auxiliary](MultiFab<Dim>& state, MultiFab<Dim>& residual,
                                      const PreparedHyperbolicBoundary<Dim>* boundary) {
    MultiFab<Dim> candidate(residual.layout(), residual.distribution(), residual.local_rank(),
                            residual.ncomp(), residual.ghosts());
    flux(state, candidate, boundary);
    MultiFab<Dim> source = materialize_source<Dim>(model, state, *auxiliary);
    saxpy(candidate, Real(1), source);
    copy_valid(candidate, residual);
  };
  auto source = [model, auxiliary](MultiFab<Dim>& state, MultiFab<Dim>& residual) {
    MultiFab<Dim> candidate = materialize_source<Dim>(model, state, *auxiliary);
    copy_valid(candidate, residual);
  };

  PreparedSystemBlock<Dim> result;
  result.provider_identity = "pops.generated.cartesian.nd/" + std::to_string(Dim) + "/" +
                             request.routes.limiter + "/" + request.routes.riemann + "/" +
                             request.routes.reconstruction;
  result.aux_components = aux_comps_for<Model, Dim>();
  result.ghosts = ghosts;

  result.closures.rhs_into = [full](MultiFab<Dim>& state, MultiFab<Dim>& residual) {
    full(state, residual, nullptr);
  };
  result.closures.rhs_flux_only = [flux](MultiFab<Dim>& state, MultiFab<Dim>& residual) {
    flux(state, residual, nullptr);
  };
  result.closures.source_only = source;
  result.closures.source_only_masked = source;
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
             const PreparedHyperbolicBoundary<Dim>& boundary) {
        full(state, residual, &boundary);
      };
  result.closures.rhs_flux_only_core_at_point_prepared =
      [flux](const auto&, MultiFab<Dim>& state, MultiFab<Dim>& residual,
             const PreparedHyperbolicBoundary<Dim>& boundary) {
        flux(state, residual, &boundary);
      };
  result.closures.prepare_generated_state_at_point =
      [prepare_state](const auto&, MultiFab<Dim>& state) { prepare_state(state, nullptr); };
  result.closures.prepare_generated_state_at_point_prepared =
      [prepare_state](const auto&, MultiFab<Dim>& state,
                      const PreparedHyperbolicBoundary<Dim>& boundary) {
        prepare_state(state, &boundary);
      };

  result.maximum_speed = [model, auxiliary](const MultiFab<Dim>& state) {
    return maximum_speed<Dim>(model, state, *auxiliary);
  };
  result.poisson_rhs = [model](const MultiFab<Dim>& state, MultiFab<Dim>& rhs) {
    add_poisson_rhs<Dim>(model, state, rhs);
  };
  result.primitive_to_conservative = [model](const double* primitive, double* conservative) {
    generated_system_detail::publish_conservative_state(model, primitive, conservative);
  };
  const auto recovery_plan = prepare_model_variable_recovery(model);
  result.conservative_to_primitive =
      [recovery_plan](const double* conservative, double* primitive) {
        Real input[Model::n_vars]{};
        Real initial[Model::n_vars]{};
        for (int component = 0; component < Model::n_vars; ++component)
          input[component] = static_cast<Real>(conservative[component]);
        const auto outcome = recover_prepared_variable(recovery_plan, input, initial);
        if (outcome.publication_permitted())
          for (int component = 0; component < Model::n_vars; ++component)
            primitive[component] = static_cast<double>(outcome.value[component]);
        return recovery_report(outcome);
      };
  result.batch_conservative_to_primitive = make_uniform_recovery_consumer(model);

  if constexpr (requires(const Model& value, const typename Model::State& state,
                         const typename Model::Aux& aux) {
                  value.source_frequency(state, aux);
                }) {
    result.source_frequency = [model, auxiliary](const MultiFab<Dim>& state) {
      MultiFab<Dim> values(state.layout(), state.distribution(), state.local_rank(), 1,
                           state.ghosts());
      for (std::size_t local = 0; local < state.local_size(); ++local)
        for_each_cell(state.box(local), MaterializeSourceFrequency<Dim, Model>{
                                             model, state.fab(local).view(),
                                             auxiliary->fab(local).view(),
                                             values.fab(local).view()});
      const Real frequency = reduce_max(values);
      if (!std::isfinite(frequency) || frequency < Real(0))
        throw std::runtime_error("generated source frequency is invalid");
      return frequency;
    };
  }
  if constexpr (requires(const Model& value, const typename Model::State& state,
                         const typename Model::Aux& aux) {
                  value.stability_dt(state, aux);
                }) {
    result.stability_dt = [model, auxiliary](const MultiFab<Dim>& state) {
      MultiFab<Dim> values(state.layout(), state.distribution(), state.local_rank(), 1,
                           state.ghosts());
      for (std::size_t local = 0; local < state.local_size(); ++local)
        for_each_cell(state.box(local), MaterializeStabilityDt<Dim, Model>{
                                             model, state.fab(local).view(),
                                             auxiliary->fab(local).view(),
                                             values.fab(local).view()});
      const Real dt = reduce_min(values);
      if (!std::isfinite(dt) || !(dt > Real(0)))
        throw std::runtime_error("generated stability dt is invalid");
      return dt;
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

}  // namespace generated_system_detail

/// Build the single exact numerical specialization requested by a generated package.
template <class Request>
auto prepare_generated_system_block(Request request)
    -> PreparedSystemBlock<Request::dimension> {
  constexpr int Dim = Request::dimension;
  using Model = std::remove_cvref_t<decltype(request.model)>;
  static_assert(Model::dimension == Dim,
                "generated System request and physical model have different ranks");
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

}  // namespace pops
