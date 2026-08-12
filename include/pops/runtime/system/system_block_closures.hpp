/// @file
/// @brief Type-erased spatial operations retaining one immutable compile-time rank.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/recovery/uniform_recovery_consumer.hpp>
#include <pops/runtime/system/prepared_embedded_boundary.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {

class ExecutionLane;
namespace runtime::program {
template <int Dim>
class PreparedScalarBoundarySession;
}

/// Prepared block capabilities accepted by the exact-ranked Uniform core.
///
/// Geometry-specific builders may own richer private state, but they must lower to these ranked
/// closures before installation. No legacy mesh view, scalar axis extent, or dynamic dimension tag
/// crosses this boundary.
template <int Dim>
struct SystemBlockClosures {
  static_assert(Dim >= 1 && Dim <= 3, "SystemBlockClosures only supports dimensions 1, 2, and 3");

  using field_type = MultiFab<Dim>;
  using boundary_type = PreparedHyperbolicBoundary<Dim>;
  using point_type = runtime::multiblock::BoundaryEvaluationPoint;
  using Residual = std::function<void(field_type&, field_type&)>;
  using ConstResidual = std::function<void(const field_type&, field_type&)>;
  using PointResidual = std::function<void(const point_type&, field_type&, field_type&)>;
  using PreparedPointResidual =
      std::function<void(const point_type&, field_type&, field_type&, const boundary_type&)>;
  using PreparedPointBoundaryResidual = std::function<void(
      const point_type&, field_type&, field_type&, const boundary_type&, const ExecutionLane&,
      const runtime::program::PreparedScalarBoundarySession<Dim>&)>;
  using PreparedPointJvp = std::function<void(
      const point_type&, field_type&, const field_type&, field_type&, const boundary_type&,
      const ExecutionLane&, const runtime::program::PreparedScalarBoundarySession<Dim>&)>;
  using BoundaryFluxTransform =
      std::function<void(const point_type&, const field_type&, std::vector<nd::FaceField<Dim>>&,
                         const Geometry<Dim>&, const ExecutionLane&)>;
  using PointStatePreparation = std::function<void(const point_type&, field_type&)>;
  using PreparedPointStatePreparation =
      std::function<void(const point_type&, field_type&, const boundary_type&)>;
  using PreparedPointStateTransport =
      std::function<void(const point_type&, field_type&, const boundary_type&, const ExecutionLane&,
                         const runtime::program::PreparedScalarBoundarySession<Dim>&)>;
  using ExternalGhostBoundary = std::function<void(const point_type&, field_type&,
                                                   const Geometry<Dim>&, const ExecutionLane&)>;
  using embedded_geometry_type = runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>;
  using EmbeddedResidual =
      std::function<void(field_type&, field_type&, const embedded_geometry_type&)>;
  using EmbeddedProjection = std::function<void(field_type&, const embedded_geometry_type&)>;

  struct EmbeddedResidualFamily {
    EmbeddedResidual full;
    EmbeddedResidual flux_only;
    EmbeddedResidual source_only;
    EmbeddedProjection project;
  };

  Residual rhs_into;
  Residual rhs_flux_only;
  Residual source_only;
  Residual source_only_masked;

  PointResidual rhs_at_point;
  PointResidual rhs_flux_only_at_point;
  PointResidual rhs_without_prepared_interfaces;
  PointResidual rhs_flux_only_without_prepared_interfaces;
  PointResidual rhs_core_at_point;
  PointResidual rhs_flux_only_core_at_point;
  PreparedPointResidual rhs_core_at_point_prepared;
  PreparedPointResidual rhs_flux_only_core_at_point_prepared;
  PreparedPointBoundaryResidual boundary_full_at_point_prepared;
  PreparedPointBoundaryResidual boundary_core_at_point_prepared;
  PreparedPointBoundaryResidual boundary_flux_full_at_point_prepared;
  PreparedPointBoundaryResidual boundary_flux_core_at_point_prepared;
  PreparedPointBoundaryResidual boundary_residual_at_point_prepared;
  PreparedPointJvp boundary_jvp_at_point_prepared;
  std::shared_ptr<BoundaryFluxTransform> external_boundary_flux;
  std::shared_ptr<PreparedPointBoundaryResidual> external_field_boundary_residual;
  std::shared_ptr<PreparedPointJvp> external_field_boundary_jvp;

  /// Fill the exact same-level and physical halos consumed by generated pointwise stencils.
  /// The closure is prepared by the dimension-qualified block package; Program code never
  /// reconstructs topology or boundary laws from scalar metadata.
  PointStatePreparation prepare_generated_state_at_point;
  PreparedPointStatePreparation prepare_generated_state_at_point_prepared;
  /// Transport-aware generated preparation used by detached routed images. It owns no image and
  /// accepts the exact source block boundary/session supplied by the caller.
  PreparedPointStateTransport prepare_generated_state_with_transport_prepared;
  std::shared_ptr<ExternalGhostBoundary> external_ghost_boundary;

  std::function<void(field_type&)> project;
  std::function<void(field_type&)> project_masked;
  EmbeddedResidualFamily staircase;
  EmbeddedResidualFamily cut_cell;
};

template <int Dim>
typename SystemBlockClosures<Dim>::PreparedPointBoundaryResidual
bind_external_field_boundary_residual(
    typename SystemBlockClosures<Dim>::PreparedPointBoundaryResidual compiled,
    std::shared_ptr<typename SystemBlockClosures<Dim>::PreparedPointBoundaryResidual> external) {
  if (!compiled || !external)
    throw std::invalid_argument(
        "prepared boundary residual extension requires complete callable storage");
  return [compiled = std::move(compiled), external = std::move(external)](
             const auto& point, MultiFab<Dim>& state, MultiFab<Dim>& result,
             const PreparedHyperbolicBoundary<Dim>& boundary, const ExecutionLane& lane,
             const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
    compiled(point, state, result, boundary, lane, transport);
    if (*external)
      (*external)(point, state, result, boundary, lane, transport);
  };
}

template <int Dim>
typename SystemBlockClosures<Dim>::PreparedPointJvp bind_external_field_boundary_jvp(
    typename SystemBlockClosures<Dim>::PreparedPointJvp compiled,
    std::shared_ptr<typename SystemBlockClosures<Dim>::PreparedPointJvp> external) {
  if (!compiled || !external)
    throw std::invalid_argument(
        "prepared boundary JVP extension requires complete callable storage");
  return [compiled = std::move(compiled), external = std::move(external)](
             const auto& point, MultiFab<Dim>& state, const MultiFab<Dim>& direction,
             MultiFab<Dim>& result, const PreparedHyperbolicBoundary<Dim>& boundary,
             const ExecutionLane& lane,
             const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
    compiled(point, state, direction, result, boundary, lane, transport);
    if (*external)
      (*external)(point, state, direction, result, boundary, lane, transport);
  };
}

/// Build the one prepared boundary residual as a true detached difference between a
/// boundary-filled evaluation and a core evaluation whose physical ghosts start from zero.  Neither
/// branch can observe ghosts published by the other branch or by an earlier RHS invocation.
template <int Dim>
typename SystemBlockClosures<Dim>::PreparedPointBoundaryResidual make_prepared_boundary_residual(
    typename SystemBlockClosures<Dim>::PreparedPointBoundaryResidual boundary_full,
    typename SystemBlockClosures<Dim>::PreparedPointBoundaryResidual boundary_core) {
  if (!boundary_full || !boundary_core)
    throw std::invalid_argument(
        "prepared boundary residual requires complete full and core authorities");
  return [boundary_full = std::move(boundary_full), boundary_core = std::move(boundary_core)](
             const auto& point, MultiFab<Dim>& state, MultiFab<Dim>& result,
             const PreparedHyperbolicBoundary<Dim>& boundary, const ExecutionLane& lane,
             const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
    if (state.layout() != result.layout() || state.distribution() != result.distribution() ||
        state.local_rank() != result.local_rank() || state.ncomp() != result.ncomp() ||
        state.ghosts() != result.ghosts())
      throw std::invalid_argument(
          "prepared boundary residual fields differ from the authenticated block contract");

    transport.with_boundary_scratch(state, [&](auto& scratch) {
      scratch.residual_boundary_state.set_val(Real(0));
      scratch.residual_core_state.set_val(Real(0));
      lincomb(scratch.residual_boundary_state, Real(1), state, Real(0), state);
      lincomb(scratch.residual_core_state, Real(1), state, Real(0), state);
      boundary_full(point, scratch.residual_boundary_state, scratch.residual_boundary_total,
                    boundary, lane, transport);
      boundary_core(point, scratch.residual_core_state, scratch.residual_core_total, boundary, lane,
                    transport);
      lincomb(result, Real(1), scratch.residual_boundary_total, Real(-1),
              scratch.residual_core_total);
    });
  };
}

/// Materialize the one generic finite-difference linearization of a prepared boundary residual.
/// Rebuilding this closure after an external GhostBoundary is installed makes both the base and
/// displaced evaluations traverse the same authenticated component and lane.
template <int Dim>
typename SystemBlockClosures<Dim>::PreparedPointJvp make_prepared_boundary_jvp(
    typename SystemBlockClosures<Dim>::PreparedPointBoundaryResidual boundary_residual) {
  if (!boundary_residual)
    throw std::invalid_argument("prepared boundary JVP requires one complete residual authority");
  return [boundary_residual = std::move(boundary_residual)](
             const auto& point, MultiFab<Dim>& state, const MultiFab<Dim>& direction,
             MultiFab<Dim>& result, const PreparedHyperbolicBoundary<Dim>& boundary,
             const ExecutionLane& lane,
             const runtime::program::PreparedScalarBoundarySession<Dim>& transport) {
    if (state.layout() != direction.layout() || state.distribution() != direction.distribution() ||
        state.local_rank() != direction.local_rank() || state.ncomp() != direction.ncomp() ||
        state.ghosts() != direction.ghosts() || state.layout() != result.layout() ||
        state.distribution() != result.distribution() ||
        state.local_rank() != result.local_rank() || state.ncomp() != result.ncomp() ||
        state.ghosts() != result.ghosts())
      throw std::invalid_argument(
          "prepared boundary JVP fields differ from the authenticated block contract");

    Real local_state_scale = Real(0);
    Real local_direction_scale = Real(0);
    for (int component = 0; component < state.ncomp(); ++component) {
      const Real state_component_scale = norm_inf(state, component);
      const Real direction_component_scale = norm_inf(direction, component);
      if (!std::isfinite(state_component_scale))
        local_state_scale = std::numeric_limits<Real>::infinity();
      else if (std::isfinite(local_state_scale))
        local_state_scale = std::max(local_state_scale, state_component_scale);
      if (!std::isfinite(direction_component_scale))
        local_direction_scale = std::numeric_limits<Real>::infinity();
      else if (std::isfinite(local_direction_scale))
        local_direction_scale = std::max(local_direction_scale, direction_component_scale);
    }
    const Real state_scale =
        static_cast<Real>(all_reduce_max(static_cast<double>(local_state_scale), lane));
    const Real direction_scale =
        static_cast<Real>(all_reduce_max(static_cast<double>(local_direction_scale), lane));
    if (!std::isfinite(state_scale))
      throw std::runtime_error("prepared boundary JVP rejected a non-finite state norm");
    if (!std::isfinite(direction_scale))
      throw std::runtime_error("prepared boundary JVP rejected a non-finite direction norm");
    if (direction_scale == Real(0)) {
      result.set_val(Real(0));
      return;
    }

    const Real increment =
        std::sqrt(std::numeric_limits<Real>::epsilon()) * (Real(1) + state_scale) / direction_scale;
    if (!std::isfinite(increment) || !(increment > Real(0)))
      throw std::runtime_error("prepared boundary JVP produced an invalid finite-difference step");
    transport.with_boundary_scratch(state, [&](auto& scratch) {
      scratch.jvp_perturbed.set_val(Real(0));
      lincomb(scratch.jvp_perturbed, Real(1), state, increment, direction);
      boundary_residual(point, state, scratch.jvp_base, boundary, lane, transport);
      boundary_residual(point, scratch.jvp_perturbed, scratch.jvp_displaced, boundary, lane,
                        transport);
      lincomb(scratch.jvp_displaced, Real(1) / increment, scratch.jvp_displaced,
              Real(-1) / increment, scratch.jvp_base);
      lincomb(result, Real(1), scratch.jvp_displaced, Real(0), scratch.jvp_displaced);
    });
  };
}

/// Complete immutable block image prepared by one dimension-qualified native package.
///
/// The image crosses the loader boundary once and is committed as one structural mutation.  This
/// prevents the former install-then-patch sequence (ghosts, conversion, recovery, dt bounds) from
/// leaving a partially installed block after a later preparation failure.
template <int Dim>
struct PreparedSystemBlock {
  static_assert(Dim >= 1 && Dim <= 3, "PreparedSystemBlock only supports dimensions 1, 2, and 3");

  using field_type = MultiFab<Dim>;

  std::string name;
  std::string provider_identity;
  int ncomp = 0;
  /// Number of values in this block's local compact provider pack.  It is zero for a model with no
  /// providers; it is never a width request for a shared physical ``aux`` field.
  int provider_components = 0;
  VariableSet conservative_variables{};
  VariableSet primitive_variables{};
  double gamma = 1.0;
  Extent<Dim> ghosts{};
  int substeps = 1;
  bool evolve = true;
  int stride = 1;

  SystemBlockClosures<Dim> closures;
  std::function<Real(const field_type&)> maximum_speed;
  std::function<void(const field_type&, field_type&)> poisson_rhs;
  std::function<void(const double*, double*)> primitive_to_conservative;
  std::function<RecoveryReport(const double*, double*)> conservative_to_primitive;
  UniformCellRecovery batch_conservative_to_primitive;
  std::function<Real(const field_type&)> source_frequency;
  std::function<Real(const field_type&)> stability_dt;
};

/// Exact-ranked shared-interface execution provider installed after every endpoint block exists.
///
/// Interface geometry and numerical fluxes remain owned by the authenticated component that
/// prepared them.  The Uniform core only supplies the simultaneous ranked state/output packs and
/// never reconstructs an axis-aligned two-dimensional route from scalar metadata.
template <int Dim>
struct SystemInterfaceProvider {
  static_assert(Dim >= 1 && Dim <= 3,
                "SystemInterfaceProvider only supports dimensions 1, 2, and 3");

  using field_type = MultiFab<Dim>;
  using point_type = runtime::multiblock::BoundaryEvaluationPoint;

  /// Stable key and complete binary contract of the detached provider registry.  The key is a
  /// digest-qualified identity; the contract authenticates route order, components, execution
  /// authority and the native spatial rank before this provider is published.
  std::string provider_identity;
  std::string collective_contract;
  std::function<void(const point_type&, const std::vector<field_type*>&,
                     const std::vector<field_type*>&, const std::vector<int>&)>
      evaluate_rhs;
  std::function<void(const point_type&, const std::vector<field_type*>&,
                     const std::vector<field_type*>&, const std::vector<int>&)>
      evaluate_core;
  std::function<std::size_t(const std::string&, int)> evaluation_count;
  std::function<bool(int)> has_interfaces;
  std::function<void()> discard;
};

}  // namespace pops
