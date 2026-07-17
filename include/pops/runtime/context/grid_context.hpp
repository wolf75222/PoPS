#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/boundary/prepared_boundary_plan.hpp>
#include <pops/numerics/spatial/embedded_boundary/domain.hpp>  // detail::DiscDomain (built-in level-set domain instance)

#include <functional>
#include <memory>

/// @file
/// @brief Block grid context plus closures, shared between System (which installs them) and
///        block_builder.hpp (which builds them from a compiled model). LIGHT header (mesh plus
///        std::function, no numerics) so it can be included in the System public API without
///        pulling in assemble_rhs / flux / steppers.

namespace pops {

/// TRANSPORT GEOMETRY MODE of the macro-step (T5-PR3 effort, disc wiring in System::step).
///  - None: full Cartesian domain (default). Transport uses assemble_rhs (historical
///                path). BIT-IDENTICAL to history as long as no disc is set.
///  - Staircase: disc approximated by a cell-centered 0/1 MASK (active/inactive face gate,
///                staircase boundary). Transport uses assemble_rhs_masked (T2 effort).
///  - CutCell: disc as cut-cell / embedded-boundary (continuous alpha_f apertures plus volume
///                fraction kappa). Transport uses assemble_rhs_eb (T5-PR1/PR2 efforts).
/// The mode is held by the System (set_disc_domain mode= / set_geometry_mode) and read by the stepper
/// to DISPATCH each block transport advance. None stays the untouched production path.
enum class GeometryMode { None, Staircase, CutCell };

/// Mesh + transport BC + aux shared by a block closures. @c aux is NOT owned:
/// it points to the System aux (lifetime longer than the block, stable address).
///
/// EMBEDDED BOUNDARY / LEVEL-SET DOMAIN (T5-PR3 effort): @c domain_mask and @c eb_domain point (NOT
/// owned) to the 0/1 mask and the level-set domain descriptor of the System (members with STABLE
/// address). They are used ONLY to build the optional embedded-boundary transport advances
/// (build_block); read BY POINTER at the step, the order add_block / set_disc_domain does not matter.
/// nullptr -> no embedded-boundary advance (stepper on advance, bit-identical). The mask is
/// materialized / the descriptor is set by set_disc_domain (the disc is one instance of the contract,
/// cf. numerics/embedded_boundary.hpp).
struct GridContext {
  Box2D dom;                              ///< domain (without ghost)
  BCRec bc;                               ///< transport BC
  Geometry geom;                          ///< geometry (dx, dy, bounds)
  MultiFab* aux = nullptr;                ///< System aux (phi, grad phi); NOT owned
  const MultiFab* domain_mask = nullptr;  ///< 0/1 domain mask (Impl::domain_mask_); NOT owned
  const detail::DiscDomain* eb_domain =
      nullptr;  ///< level-set domain descriptor (Impl::eb_domain_); NOT owned
  // ADC-615: cut-cell / EB thresholds (kappa_min, face_open_eps, cut_theta_min), by value so this
  // header stays light. Defaults are the historical constants (kEbKappaMin / kEbFaceOpenEps /
  // kEbCutFractionFloor), so an unconfigured context builds the bit-identical EB advance. Set from
  // Impl::eb_thresholds_ at grid_ctx() time; read when building the EB transport advance.
  Real eb_kappa_min = Real(1e-2);
  Real eb_face_open_eps = Real(1e-6);
  Real eb_cut_theta_min = Real(1e-3);
  /// Exact per-block ghost-production authority. Empty only for legacy low-level construction;
  /// resolved Case installation always supplies one before closures are built.
  std::shared_ptr<const PreparedBoundaryPlan> boundary_plan{};
  /// Open N-ary storage-binding protocol.  A runtime that owns several states/fields/outputs binds
  /// their exact qualified identities here; the boundary executor remains independent of System,
  /// AMR, field registries and storage classes.  Empty selects the ordinary one-state/one-aux
  /// convenience adapter and never fabricates aliases for an N-ary request.
  using BoundaryFieldRegistryFactory = std::function<detail::BoundaryFieldRegistry(
      const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, const MultiFab*, MultiFab*)>;
  BoundaryFieldRegistryFactory boundary_field_registry{};
};

/// The single transport ghost-fill entry used by compiled block closures.  The historical BCRec
/// path remains for low-level native construction with no resolved boundary authority; a resolved
/// plan never falls back because its pointer is captured when the block is built.
inline void fill_grid_ghosts(MultiFab& state, const GridContext& context) {
  if (context.boundary_plan) {
    context.boundary_plan->fill_same_level_and_physical(state, context.dom);
    return;
  }
  fill_ghosts(state, context.dom, context.bc);
}

inline void fill_grid_ghosts(MultiFab& state, const GridContext& context,
                             const runtime::multiblock::BoundaryEvaluationPoint& point) {
  if (context.boundary_plan) {
    if (context.boundary_field_registry) {
      auto fields = context.boundary_field_registry(point, state, nullptr, nullptr);
      context.boundary_plan->fill_same_level_and_physical(state, fields, context.geom, point);
    } else {
      context.boundary_plan->fill_same_level_and_physical(state, context.aux, context.geom, point);
    }
    return;
  }
  fill_ghosts(state, context.dom, context.bc);
}

inline void add_grid_boundary_residual(MultiFab& state, MultiFab& residual,
                                       const GridContext& context,
                                       const runtime::multiblock::BoundaryEvaluationPoint& point) {
  if (!context.boundary_plan)
    return;
  if (context.boundary_field_registry) {
    auto fields = context.boundary_field_registry(point, state, nullptr, &residual);
    context.boundary_plan->add_residual(point, fields, context.geom);
  } else {
    context.boundary_plan->add_residual(point, state, context.aux, context.geom, residual);
  }
}

inline void apply_grid_boundary_jvp(MultiFab& state, const MultiFab& direction, MultiFab& output,
                                    const GridContext& context,
                                    const runtime::multiblock::BoundaryEvaluationPoint& point) {
  if (!context.boundary_plan)
    return;
  if (context.boundary_field_registry) {
    auto fields = context.boundary_field_registry(point, state, &direction, &output);
    context.boundary_plan->apply_jvp(point, fields, context.geom);
  } else {
    context.boundary_plan->apply_jvp(point, state, direction, context.aux, context.geom, output);
  }
}

/// Compiled block closures, frozen at add time.
///
/// advance is the transport advance of the DEFAULT path (assemble_rhs, full Cartesian). The two
/// optional DISC advances (T5-PR3 effort) mimic advance EXACTLY (same RK / IMEX scheme,
/// same limiter / flux) but dispatch the transport residual to the disc operator:
///   - advance_masked: assemble_rhs_masked (0/1 mask, Staircase mode);
///   - advance_eb: assemble_rhs_eb (cut-cell EB, CutCell mode).
/// They read the System mask / level set BY POINTER at step time (not at
/// construction), so the order add_block / set_disc_domain does not matter. Empty (default) as long as
/// the block does not support disc routing: the stepper then falls back to advance (bit-identical).
struct BlockClosures {
  std::function<void(MultiFab&, Real, int)> advance;  ///< (U, dt, n): n substeps of dt/n
  std::function<void(MultiFab&, Real, int)>
      advance_masked;                                    ///< same, residual via assemble_rhs_masked
  std::function<void(MultiFab&, Real, int)> advance_eb;  ///< same, residual via assemble_rhs_eb
  std::function<void(MultiFab&, MultiFab&)> rhs_into;    ///< R <- -div F + S (Poisson frozen)
  /// FLUX-ONLY residual R <- -div F(U) (NO default/composite source), Poisson frozen (ADC-425). The
  /// SAME transport assembly as @ref rhs_into evaluated on SourceFreeModel<Model> (zero source), so the
  /// flux / ghost / geometry handling is bit-identical -- only the source is dropped. A compiled time
  /// Program's hyperbolic stage (P.rhs(flux=True, sources without "default")) reads it so a Lie/Strang
  /// split assembles "flux but no source" without the default source leaking in (spec criterion 17:
  /// sources are explicit, never summed implicitly). OPTIONAL (empty for block paths that do not build
  /// it, e.g. the host .so prototype loader): System::block_neg_div_flux_into fails loud then.
  std::function<void(MultiFab&, MultiFab&)> rhs_flux_only;
  /// SOURCE-ONLY residual R <- S(U, aux) (the model's default/composite source, NO flux divergence),
  /// Poisson frozen (ADC-430). The exact MIRROR of @ref rhs_flux_only: together they split @ref rhs_into
  /// (-div F + S) into its two halves. Evaluates m.source per cell (the SAME source term assemble_rhs
  /// adds) with no numerical-flux dispatch, so it is flux-template agnostic and bit-identical to the
  /// source half of rhs_into. A compiled time Program's source stage (P.rhs(flux=False, sources with
  /// "default")) reads it so a Lie/Strang split assembles "the default source but no flux" without the
  /// -div F base leaking in (spec: rhs flux=False is source-only). OPTIONAL (empty for block paths that
  /// do not build it, e.g. the host .so prototype loader): System::block_source_into fails loud then.
  std::function<void(MultiFab&, MultiFab&)> source_only;
  /// Point-qualified full/flux-only residuals used by every compiled Program rate.  These are not
  /// optional aliases of the legacy closures: a prepared native boundary component requires the
  /// exact clock/stage/dt carried by BoundaryEvaluationPoint and the legacy unqualified route fails.
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
      rhs_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
      rhs_flux_only_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
      rhs_without_prepared_interfaces;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
      rhs_flux_only_without_prepared_interfaces;
  /// Core residuals exclude additive FieldBoundary contributions but retain ghost producers and
  /// shared-interface face ownership.  Residual/JVP closures expose that boundary term separately.
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
      rhs_core_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
      rhs_flux_only_core_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
      boundary_residual_at_point;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, MultiFab&)>
      boundary_jvp_at_point;
  /// dt_hotspot diagnostic (ADC-182): (U, w, i, j) -> GLOBAL cell dominating the transport
  /// CFL and its speed. OPTIONAL (empty = block without diagnostic, e.g. historical
  /// unrewired paths); never called by step/step_cfl (off the hot path).
  std::function<void(const MultiFab&, Real&, int&, int&)> hotspot;
  /// PROJECTION PONCTUELLE post-pas (ADC-177) : U <- project(U, aux) sur les cellules VALIDES du
  /// bloc, appliquee par le stepper a la FIN de chaque macro-pas ENTIER (jamais par etage RK).
  /// OPTIONNELLE (vide = bloc sans projection : jamais interrogee, cout nul, bit-identique).
  std::function<void(MultiFab&)> project;
};

}  // namespace pops
