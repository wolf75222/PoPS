#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/boundary/prepared_boundary_plan.hpp>
#include <pops/numerics/spatial/embedded_boundary/domain.hpp>  // detail::DiscDomain (built-in level-set domain instance)
#include <pops/parallel/execution_lane.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
  using BoundaryFieldRegistryFactory =
      std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                         const MultiFab*, MultiFab*, detail::BoundaryFieldRegistry&)>;
  BoundaryFieldRegistryFactory boundary_field_registry{};
  /// Optional hierarchy producer that runs before same-level/physical filling. Uniform contexts
  /// leave it empty; an AMR block/level context binds its exact prepared coarse/fine authority.
  std::function<void(MultiFab&)> coarse_fine_fill{};
};

/// Ephemeral, allocation-free read view over one boundary-session binding epoch.
///
/// Prepared read tokens carry their exact plan owner and component-table revision. A view also
/// captures the registry epoch, so retaining it across another boundary operation fails explicitly
/// instead of observing a silently rebound storage pointer.
class PreparedBoundaryReadView final {
 public:
  [[nodiscard]] const MultiFab& state(const PreparedBoundaryStateRead& read) const {
    validate_(read.owner_, read.component_revision_);
    return registry_->state_at(read.slot_);
  }

  [[nodiscard]] const MultiFab& field(const PreparedBoundaryFieldRead& read) const {
    validate_(read.owner_, read.component_revision_);
    return registry_->field_at(read.slot_);
  }

 private:
  friend class PreparedGridBoundarySession;

  PreparedBoundaryReadView(const PreparedBoundaryPlan* plan,
                           const detail::BoundaryFieldRegistry* registry,
                           std::uint64_t binding_epoch)
      : plan_(plan), registry_(registry), binding_epoch_(binding_epoch) {}

  void validate_(const PreparedBoundaryPlan* owner, std::size_t component_revision) const {
    if (plan_ == nullptr || registry_ == nullptr)
      throw std::logic_error("prepared boundary read view is empty");
    if (owner != plan_)
      throw std::invalid_argument("prepared boundary read token belongs to another plan");
    if (component_revision != plan_->component_revision_)
      throw std::logic_error("prepared boundary read token predates the current plan revision");
    if (binding_epoch_ != registry_->binding_epoch())
      throw std::logic_error("prepared boundary read view is stale after a later binding");
  }

  const PreparedBoundaryPlan* plan_;
  const detail::BoundaryFieldRegistry* registry_;
  std::uint64_t binding_epoch_;
};

/// Move-only execution state for one exact GridContext on one ExecutionLane.
///
/// A resolved boundary plan may own native component state.  That state is materialized here once,
/// before a prepared operator enters its numerical loop, and is then reused by every ghost,
/// residual and JVP call.  A context without a resolved plan remains an explicit mesh-only
/// authority (BCRec + geometry); it does not probe or borrow a plan from another block.
class PreparedGridBoundarySession final {
 public:
  PreparedGridBoundarySession(GridContext context, const ExecutionLane& lane)
      : context_(std::move(context)), lane_(&lane) {
    if (context_.boundary_plan && context_.boundary_plan->has_component_boundaries())
      throw std::invalid_argument(
          "component boundary session requires its exact prototype and preparation point");
    if (context_.boundary_plan) {
      plan_session_.emplace(context_.boundary_plan->make_session(lane));
      configure_registry_();
      if (!required_states_.empty() || !required_fields_.empty())
        throw std::invalid_argument(
            "boundary read dependencies require an exact prototype and preparation point");
    }
  }

  PreparedGridBoundarySession(GridContext context, const ExecutionLane& lane, MultiFab& prototype,
                              const runtime::multiblock::BoundaryEvaluationPoint& preparation_point)
      : context_(std::move(context)), lane_(&lane) {
    if (!context_.boundary_plan)
      return;
    plan_session_.emplace(context_.boundary_plan->make_session(lane));
    configure_registry_();
    // Resolve every declared read route once while the session is materialized. Subsequent RHS
    // applications only advance the registry epoch and rebind pointers into these stable slots.
    if (!required_states_.empty() || !required_fields_.empty() ||
        context_.boundary_plan->has_component_boundaries())
      bind_registry_(preparation_point, prototype, nullptr, nullptr);
    // Built-in BCs have no component executor workspace to materialize. Keep their prepared plan
    // session and read registry, but do not invent component work for them.
    if (!context_.boundary_plan->has_component_boundaries())
      return;
    plan_session_->prepare_ghost_executor(prototype, registry_, context_.geom);
    if (!residual_outputs_.empty()) {
      bind_registry_(preparation_point, prototype, nullptr, &prototype);
      plan_session_->prepare_residual_executor(registry_, context_.geom);
    }
    if (!jvp_outputs_.empty()) {
      bind_registry_(preparation_point, prototype, &prototype, &prototype);
      plan_session_->prepare_jvp_executor(registry_, context_.geom);
    }
  }

  PreparedGridBoundarySession(const PreparedGridBoundarySession&) = delete;
  PreparedGridBoundarySession& operator=(const PreparedGridBoundarySession&) = delete;
  PreparedGridBoundarySession(PreparedGridBoundarySession&&) noexcept = default;
  PreparedGridBoundarySession& operator=(PreparedGridBoundarySession&&) noexcept = default;
  ~PreparedGridBoundarySession() = default;

  const GridContext& context() const noexcept { return context_; }
  bool has_resolved_plan() const noexcept { return plan_session_.has_value(); }
  const PreparedBoundaryPlan* resolved_plan() const noexcept {
    return context_.boundary_plan.get();
  }

  /// Rebind the exact read-only state/field dependencies for one evaluation point.
  ///
  /// The returned view aliases session-owned storage and remains valid only until the next boundary
  /// operation on this lane. Slot identities and capacity are prepared by the constructor; this
  /// method performs no allocation or dependency discovery in the numerical path.
  PreparedBoundaryReadView bind_reads(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                      MultiFab& state) const {
    if (!plan_session_)
      throw std::logic_error("boundary read dependencies require a resolved prepared plan");
    plan_session_->validate_current();
    bind_registry_(point, state, nullptr, nullptr);
    return PreparedBoundaryReadView(context_.boundary_plan.get(), &registry_,
                                    registry_.binding_epoch());
  }

  /// Mesh-only/operator fill.  A component-backed plan deliberately refuses this route because an
  /// exact BoundaryEvaluationPoint is part of its authored contract.
  void fill(MultiFab& state) const {
    if (context_.coarse_fine_fill)
      context_.coarse_fine_fill(state);
    if (plan_session_) {
      plan_session_->fill_same_level_and_physical(state, context_.dom);
      return;
    }
    fill_ghosts(state, context_.dom, context_.bc, lane());
  }

  void fill(MultiFab& state, const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    if (context_.coarse_fine_fill)
      context_.coarse_fine_fill(state);
    fill_same_level_and_physical(state, point);
  }

  /// Apply only same-level and physical boundary production.  AMR reflux first installs an exact
  /// temporal coarse/fine snapshot and must not have that snapshot overwritten by the ordinary
  /// hierarchy callback.  The component registry and executor workspaces remain the same prepared
  /// objects used by fill().
  void fill_same_level_and_physical(
      MultiFab& state, const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    if (!plan_session_) {
      fill_ghosts(state, context_.dom, context_.bc, lane());
      return;
    }
    if (!context_.boundary_plan->has_component_boundaries()) {
      plan_session_->fill_same_level_and_physical(state, context_.dom);
      return;
    }
    bind_registry_(point, state, nullptr, nullptr);
    plan_session_->fill_same_level_and_physical(state, registry_, context_.geom, point);
  }

  void add_residual(MultiFab& state, MultiFab& residual,
                    const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    if (!plan_session_ || !context_.boundary_plan->has_component_boundaries())
      return;
    bind_registry_(point, state, nullptr, &residual);
    plan_session_->add_residual(point, registry_, context_.geom);
  }

  void apply_jvp(MultiFab& state, const MultiFab& direction, MultiFab& output,
                 const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    if (!plan_session_ || !context_.boundary_plan->has_component_boundaries())
      return;
    bind_registry_(point, state, &direction, &output);
    plan_session_->apply_jvp(point, registry_, context_.geom);
  }

 private:
  void configure_registry_() {
    required_states_ = context_.boundary_plan->required_state_identities();
    required_directions_ = context_.boundary_plan->required_direction_identities();
    required_fields_ = context_.boundary_plan->required_field_identities();
    residual_outputs_ = context_.boundary_plan->residual_output_identities();
    jvp_outputs_ = context_.boundary_plan->jvp_output_identities();
    all_outputs_ = context_.boundary_plan->all_output_identities();
    registry_.configure_states(required_states_);
    registry_.configure_directions(required_directions_);
    registry_.configure_fields(required_fields_);
    registry_.configure_outputs(all_outputs_);
    if (residual_outputs_.size() == 1)
      residual_output_slot_ = registry_.output_index(residual_outputs_.front());
    if (jvp_outputs_.size() == 1)
      jvp_output_slot_ = registry_.output_index(jvp_outputs_.front());
  }

  const ExecutionLane& lane() const {
    if (lane_ == nullptr)
      throw std::logic_error("PreparedGridBoundarySession is empty after move");
    return *lane_;
  }

  void bind_registry_(const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab& state,
                      const MultiFab* direction, MultiFab* output) const {
    if (!context_.boundary_plan)
      return;
    registry_.begin_binding();
    if (required_states_.empty() && required_fields_.empty() && direction == nullptr &&
        output == nullptr)
      return;
    if (context_.boundary_field_registry) {
      context_.boundary_field_registry(point, state, direction, output, registry_);
      return;
    }
    if (!required_states_.empty()) {
      if (required_states_.size() != 1 ||
          required_states_.front() != context_.boundary_plan->state_identity())
        throw std::runtime_error(
            "multi-state boundary session requires an exact runtime storage binder");
      registry_.bind_state_slot(0, state);
    }
    if (!required_fields_.empty()) {
      if (required_fields_.size() != 1 || context_.aux == nullptr)
        throw std::runtime_error(
            "multi-field boundary session requires an exact runtime storage binder");
      registry_.bind_field_slot(0, *context_.aux);
    }
    if (direction != nullptr) {
      if (required_directions_.size() != 1)
        throw std::runtime_error(
            "multi-direction boundary session requires an exact runtime storage binder");
      registry_.bind_direction_slot(0, *direction);
    }
    if (output != nullptr) {
      const auto& outputs = direction == nullptr ? residual_outputs_ : jvp_outputs_;
      const auto& output_slot = direction == nullptr ? residual_output_slot_ : jvp_output_slot_;
      if (outputs.size() != 1 || !output_slot.has_value())
        throw std::runtime_error(
            "multi-output boundary session requires an exact runtime storage binder");
      registry_.bind_output_slot(*output_slot, *output);
    }
  }

  GridContext context_;
  const ExecutionLane* lane_ = nullptr;
  mutable detail::BoundaryFieldRegistry registry_;
  std::optional<PreparedBoundaryPlan::Session> plan_session_;
  std::vector<std::string> required_states_;
  std::vector<std::string> required_directions_;
  std::vector<std::string> required_fields_;
  std::vector<std::string> residual_outputs_;
  std::vector<std::string> jvp_outputs_;
  std::vector<std::string> all_outputs_;
  std::optional<std::size_t> residual_output_slot_;
  std::optional<std::size_t> jvp_output_slot_;
};

/// One-shot control/diagnostic adapter. Prepared numerical closures retain a
/// PreparedGridBoundarySession and call bind_reads() instead.
inline detail::BoundaryFieldRegistry bind_grid_boundary_fields(
    const GridContext& context, const runtime::multiblock::BoundaryEvaluationPoint& point,
    MultiFab& state, const MultiFab* direction, MultiFab* output) {
  detail::BoundaryFieldRegistry fields;
  if (!context.boundary_plan || !context.boundary_field_registry)
    return fields;
  fields.configure_states(context.boundary_plan->required_state_identities());
  fields.configure_directions(context.boundary_plan->required_direction_identities());
  fields.configure_fields(context.boundary_plan->required_field_identities());
  fields.configure_outputs(context.boundary_plan->all_output_identities());
  fields.begin_binding();
  context.boundary_field_registry(point, state, direction, output, fields);
  return fields;
}

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
                             const ExecutionLane& lane) {
  if (context.boundary_plan) {
    context.boundary_plan->fill_same_level_and_physical(state, context.dom, lane);
    return;
  }
  fill_ghosts(state, context.dom, context.bc, lane);
}

inline void fill_grid_ghosts(MultiFab& state, const GridContext& context,
                             const runtime::multiblock::BoundaryEvaluationPoint& point) {
  if (context.boundary_plan) {
    if (context.boundary_field_registry) {
      auto fields = bind_grid_boundary_fields(context, point, state, nullptr, nullptr);
      context.boundary_plan->fill_same_level_and_physical_control(state, fields, context.geom,
                                                                  point);
    } else {
      context.boundary_plan->fill_same_level_and_physical_control(state, context.aux, context.geom,
                                                                  point);
    }
    return;
  }
  fill_ghosts(state, context.dom, context.bc);
}

inline void fill_grid_ghosts(MultiFab& state, const GridContext& context,
                             const runtime::multiblock::BoundaryEvaluationPoint& point,
                             const ExecutionLane& lane) {
  if (context.boundary_plan) {
    if (context.boundary_field_registry) {
      auto fields = bind_grid_boundary_fields(context, point, state, nullptr, nullptr);
      context.boundary_plan->fill_same_level_and_physical_control(state, fields, context.geom,
                                                                  point, lane);
    } else {
      context.boundary_plan->fill_same_level_and_physical_control(state, context.aux, context.geom,
                                                                  point, lane);
    }
    return;
  }
  fill_ghosts(state, context.dom, context.bc, lane);
}

inline void fill_grid_ghosts(MultiFab& state, const PreparedGridBoundarySession& session) {
  session.fill(state);
}

inline void fill_grid_ghosts(MultiFab& state, const PreparedGridBoundarySession& session,
                             const runtime::multiblock::BoundaryEvaluationPoint& point) {
  session.fill(state, point);
}

inline void add_grid_boundary_residual(MultiFab& state, MultiFab& residual,
                                       const GridContext& context,
                                       const runtime::multiblock::BoundaryEvaluationPoint& point) {
  if (!context.boundary_plan)
    return;
  if (context.boundary_field_registry) {
    auto fields = bind_grid_boundary_fields(context, point, state, nullptr, &residual);
    context.boundary_plan->add_residual_control(point, fields, context.geom);
  } else {
    context.boundary_plan->add_residual_control(point, state, context.aux, context.geom, residual);
  }
}

inline void add_grid_boundary_residual(MultiFab& state, MultiFab& residual,
                                       const PreparedGridBoundarySession& session,
                                       const runtime::multiblock::BoundaryEvaluationPoint& point) {
  session.add_residual(state, residual, point);
}

inline void apply_grid_boundary_jvp(MultiFab& state, const MultiFab& direction, MultiFab& output,
                                    const GridContext& context,
                                    const runtime::multiblock::BoundaryEvaluationPoint& point) {
  if (!context.boundary_plan)
    return;
  if (context.boundary_field_registry) {
    auto fields = bind_grid_boundary_fields(context, point, state, &direction, &output);
    context.boundary_plan->apply_jvp_control(point, fields, context.geom);
  } else {
    context.boundary_plan->apply_jvp_control(point, state, direction, context.aux, context.geom,
                                             output);
  }
}

inline void apply_grid_boundary_jvp(MultiFab& state, const MultiFab& direction, MultiFab& output,
                                    const PreparedGridBoundarySession& session,
                                    const runtime::multiblock::BoundaryEvaluationPoint& point) {
  session.apply_jvp(state, direction, output, point);
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
  /// Prepared-operator twins. The lane-bound boundary session is materialized by the operator
  /// factory and retained by its workspace; these closures therefore never construct component
  /// state or select a boundary authority during a Krylov apply.
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&,
                     const PreparedGridBoundarySession&)>
      rhs_core_at_point_prepared;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&,
                     const PreparedGridBoundarySession&)>
      rhs_flux_only_core_at_point_prepared;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&,
                     const PreparedGridBoundarySession&)>
      boundary_residual_at_point_prepared;
  std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                     const MultiFab&, MultiFab&, const PreparedGridBoundarySession&)>
      boundary_jvp_at_point_prepared;
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
