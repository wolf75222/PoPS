/// @file
/// @brief Type-erased spatial operations retaining one immutable compile-time rank.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/state/variables.hpp>
#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/nonlinear/prepared_variable_recovery.hpp>
#include <pops/runtime/multiblock/evaluation_point.hpp>
#include <pops/runtime/recovery/uniform_recovery_consumer.hpp>
#include <pops/runtime/system/prepared_embedded_boundary.hpp>

#include <functional>
#include <string>
#include <vector>

namespace pops {

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
  using PointJvp =
      std::function<void(const point_type&, field_type&, const field_type&, field_type&)>;
  using PreparedPointJvp = std::function<void(const point_type&, field_type&, const field_type&,
                                              field_type&, const boundary_type&)>;
  using PointStatePreparation = std::function<void(const point_type&, field_type&)>;
  using PreparedPointStatePreparation =
      std::function<void(const point_type&, field_type&, const boundary_type&)>;
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
  PointResidual boundary_residual_at_point;
  PointJvp boundary_jvp_at_point;

  PreparedPointResidual rhs_core_at_point_prepared;
  PreparedPointResidual rhs_flux_only_core_at_point_prepared;
  PreparedPointResidual boundary_residual_at_point_prepared;
  PreparedPointJvp boundary_jvp_at_point_prepared;

  /// Fill the exact same-level and physical halos consumed by generated pointwise stencils.
  /// The closure is prepared by the dimension-qualified block package; Program code never
  /// reconstructs topology or boundary laws from scalar metadata.
  PointStatePreparation prepare_generated_state_at_point;
  PreparedPointStatePreparation prepare_generated_state_at_point_prepared;

  std::function<void(field_type&)> project;
  std::function<void(field_type&)> project_masked;
  EmbeddedResidualFamily staircase;
  EmbeddedResidualFamily cut_cell;
};

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
