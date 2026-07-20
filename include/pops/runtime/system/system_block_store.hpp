#pragma once

#include <pops/core/state/state.hpp>  // kAuxBaseComps (default aux channel of the Schur stage: B_z)
#include <pops/core/foundation/types.hpp>  // Real
#include <pops/core/state/variables.hpp>   // VariableSet (role descriptor carried by each block)
#include <pops/mesh/index/box2d.hpp>       // Box2D
#include <pops/mesh/execution/for_each.hpp>  // device_fence (marshaling synchronizes the device before reading the host)
#include <pops/mesh/storage/multifab.hpp>         // MultiFab, Array4, ConstArray4
#include <pops/runtime/context/grid_context.hpp>  // GeometryMode + point-qualified geometry residuals
#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

/// @file
/// @brief SystemBlockStore: the BLOCK MANAGEMENT responsibility extracted from the god-class System::Impl
///        (audit Lot B.3, last P0 extraction; follows SystemFieldSolver #176 and SystemStepper).
///        Extracted VERBATIM from python/system.cpp: no change to numerics, layout, iteration order,
///        indexing, or error message. STRICTLY bit-identical -- the code is moved as is.
///
/// CONTRACT / INVARIANTS
/// - OWNS: the BlockState struct (formerly Species: state U + descriptors + closures of the block) and the
///   ordered registry of blocks (vector<BlockState>), the UNIQUE source of truth populated by all
///   native install paths (`install_block` / `add_native_block` via the authenticated loader).
/// - EXPOSES: index(name) (0-based index or error), find(name) (const + non const, reference to the block or
///   error), and the state MARSHALING helpers copy_comp0 / copy_state / write_state (host <-> MultiFab copy,
///   device_fence included). The insertion ORDER fixes indexing and thus iteration: it is
///   PRESERVED (push_back at the tail, never sorted nor reshuffled).
/// - ERROR MESSAGES UNCHANGED: "System: bloc inconnu '...'" (index/find) and
///   "System::set_state: taille != ncomp*n*n" (write_state) are kept VERBATIM (the non-regression of the
///   facade tests depends on it).
/// - DOES NOT OWN: the domain (ba/dm/dom/geom/pgeom_), the aux and its width, the Poisson/elliptic, the
///   couplings, t/macro_step_. These concerns stay in System::Impl (or in SystemFieldSolver /
///   SystemStepper); the store knows nothing about them.
///
/// The `blocks` registry is PUBLIC: System::Impl exposes it as is via a reference member `sp` (alias),
/// so that the already-extracted header templates (SystemFieldSolver, SystemStepper, native_loader) that
/// iterate `owner_->sp` / `P->sp` and name `Impl::Species` stay UNCHANGED and bit-identical. The
/// struct is named BlockState (clearer meaning than the historical "Species"); Impl keeps the alias
/// `using Species = SystemBlockStore::BlockState;` for template compatibility.

namespace pops {

class PreparedGridBoundarySession;
class ExecutionLane;

/// ORDERED registry of the System blocks + state marshaling helpers. See contract above (OWNS
/// BlockState + the vector; EXPOSES index/find + copy/write_state; DOES NOT OWN the domain/aux/Poisson).
class SystemBlockStore {
 public:
  /// Type-erase of the POINTWISE (one cell) cons <-> prim conversion of a block: in/out are
  /// arrays of ncomp doubles. SAME type as System::CellConvert (identical std::function): assignment
  /// from set_block_conversion / native_loader stays a trivial move.
  using CellConvert = std::function<void(const double* in, double* out)>;

  /// Compiled closures frozen at block add time (composite model + spatial scheme + time).
  /// Type-erased ONLY at the block list level; the kernel stays compiled.
  /// MEMBER ORDER FROZEN: install_block (python/system.cpp) and native_loader (push_dynamic /
  /// add_compiled_model) initialize this struct by positional AGGREGATE
  /// {name, U, ncomp, substeps, evolve, stride, gamma, advance, rhs_into, max_speed, add_poisson_rhs};
  /// do not reorder these members nor insert any before add_poisson_rhs.
  ///
  /// DESIGN DECISION (ADC-610, named-option-groups audit): this struct is DELIBERATELY kept flat, unlike
  /// AmrBuildParams / AmrCompiledHooks which were regrouped into named sub-structs. Three constraints make
  /// a regroup net-negative here:
  ///   (1) it is a POSITIONAL AGGREGATE (the frozen brace-init above): grouping members into sub-structs
  ///       would break every {..} construction site (install_block, native_loader push_dynamic /
  ///       add_compiled_model) with no ordering safety gained;
  ///   (2) the SystemStepper templates read these members by name on the hot path, so a regroup churns
  ///       the stepper without changing ownership;
  ///   (3) unlike AmrBuildParams, BlockState does NOT cross the dlopen `.so` boundary by value, so there is
  ///       no ABI-versioning payoff.
  /// The members below are therefore left flat but ANNOTATED with named-group comments (IDENTITY / SCHEME /
  /// MASKED TRANSPORT ...) so the ownership is legible without paying the regroup cost. This
  /// is the honest per-target decision the audit calls for, not an omission.
  struct BlockState {
    // --- IDENTITY + SCHEME (positional-aggregate head; order frozen) ---
    std::string name;
    MultiFab U;
    int ncomp;
    int substeps;    // static substeps (add_block)
    bool evolve;     // false = frozen species (fixed background, not advanced)
    int stride = 1;  // cadence: advance once every stride macro-steps
    double gamma;    // for the rest energy (4 var)
    std::function<void(MultiFab&, Real, int)> advance;   // (U, dt, n): n substeps of dt/n
    std::function<void(MultiFab&, MultiFab&)> rhs_into;  // R <- -div F + S (Poisson frozen)
    std::function<Real(const MultiFab&)> max_speed;      // max |wave speed| of the block
    std::function<void(const MultiFab&, MultiFab&)> add_poisson_rhs;  // += elliptic_rhs(U)
    // Descriptor of the conservative / primitive variables (names + physical ROLES) of the block.
    // The roles (provided by M::conservative_vars()) let inter-species couplings target a component
    // by its MEANING (momentum, energy) instead of a hard-coded index u[1]/u[3].
    VariableSet cons_vars, prim_vars;
    // POINTWISE cons <-> prim conversions OF THE BLOCK MODEL (one cell, ncomp doubles in/out).
    // Set at add time (install_block / push_dynamic) from the real model; empty -> identity (the
    // model exposes no conversion, e.g. pure scalar or .so generated before this work).
    // Consumed by set_primitive_state / get_primitive_state (init/diagnostic in primitive).
    CellConvert prim_to_cons, cons_to_prim;
    // --- EMBEDDED-BOUNDARY TRANSPORT (opt-in; empty default -> Cartesian path) ---
    // GEOMETRY-AWARE TRANSPORT ADVANCES. Empty (default) -> no embedded-boundary routing:
    // the stepper advances via `advance` (full Cartesian path, BIT-IDENTICAL). Non empty, they MIMIC
    // `advance` (same RK / IMEX scheme, same limiter / flux) but dispatch the transport residual
    // to the selected metric operator, and are SELECTED only if the System is in Staircase mode
    // (resp. CutCell) AND a signed embedded boundary is installed. Built at block add time
    // AT THE SAME TIME as `advance`, they read the System mask / level set by pointer at step time
    // (stable address): block/geometry authoring order is irrelevant. Trailing + empty default:
    // the positional aggregate init of the other members stays unchanged.
    std::function<void(MultiFab&, Real, int)>
        advance_masked;  // residual via assemble_rhs_masked (Staircase)
    std::function<void(MultiFab&, Real, int)> advance_eb;  // residual via assemble_rhs_eb (CutCell)
    // dt_hotspot DIAGNOSTIC (ADC-182): (U, w, i, j) -> GLOBAL cell dominating the transport CFL bound
    // of the block + its speed w = max(wx, wy). ON DEMAND only (System::dt_hotspot):
    // never queried by step/step_cfl (hot path bit-identical). Trailing + empty default.
    std::function<void(const MultiFab&, Real&, int&, int&)> hotspot;
    // OPTIONAL STEP BOUNDS of the block (audit 2026-06, step_cfl work). EMPTY (default) -> the
    // stepper does not query them: STRICTLY historical step policy (transport only,
    // bit-identical). Set by set_block_dt_bounds when the model declares the traits
    // HasSourceFrequency / HasStabilityDt (cf. core/physical_model.hpp for the semantics).
    // Trailing + empty default: the positional aggregate init of the other members stays unchanged.
    std::function<Real(const MultiFab&)>
        source_frequency;  // max over cells of mu [1/s] (0 = no constraint)
    std::function<Real(const MultiFab&)>
        stability_dt;  // min over cells of the admissible step (0 = no constraint)
    // PROJECTION PONCTUELLE post-pas (ADC-177) : U <- project(U, aux) sur les cellules VALIDES,
    // appliquee par le stepper a la FIN de chaque macro-pas ENTIER (apres transport + etage source +
    // couplages ; jamais par etage RK). VIDE (defaut) -> jamais interrogee (cout nul, chemin
    // bit-identique). Trailing + defaut vide : l'init par agregat positionnel reste inchangee.
    std::function<void(MultiFab&)> project;
    // Geometry-aware projection: same pointwise model projection, restricted to the prepared
    // active-cell mask.  Empty is an unsupported provider, never permission to project inactive
    // storage when an embedded boundary is active.
    std::function<void(MultiFab&)> project_masked;
    // FLUX-ONLY residual R <- -div F(U) (NO default/composite source), Poisson frozen (ADC-425). The
    // SAME transport assembly as rhs_into evaluated on SourceFreeModel<Model> (zero source), so the
    // flux / ghost / geometry handling is bit-identical -- only the source is dropped (with
    // limiter='none'; the HLL wave-speed cache -- rejected on the aot/production backends compiled
    // Programs use -- is the only path where cached cell-center speeds differ from the per-face
    // reconstruction). Read by
    // System::block_neg_div_flux_into, which a compiled time Program's hyperbolic stage calls so a
    // Lie/Strang split assembles "flux but no source" (spec criterion 17). EMPTY (default) for paths
    // that do not build it (the host .so prototype loader); block_neg_div_flux_into fails loud then.
    // Trailing + empty default: the positional aggregate init of the other members stays unchanged.
    std::function<void(MultiFab&, MultiFab&)> rhs_flux_only;
    // NAMED elliptic-field RHS closures (ADC-428): field name -> (+= elliptic_field_rhs(U)). A model
    // declaring m.elliptic_field("phi2", rhs=...) carries here a SECOND Poisson right-hand side
    // (distinct from add_poisson_rhs, the default Poisson coupling), assembled the same way (host loop,
    // += per cell). EMPTY (default) -> no named elliptic field: bit-identical to the historical block.
    // The SystemFieldSolver gathers these per field (sum over blocks) into a SEPARATE elliptic solve
    // whose phi/grad are written to the field's OWN aux channel. Trailing + empty default: the
    // positional aggregate init of the other members stays unchanged.
    std::map<std::string, std::function<void(const MultiFab&, MultiFab&)>> named_poisson_rhs;
    // SOURCE-ONLY residual R <- S(U, aux) (the default/composite source, NO flux divergence), Poisson
    // frozen (ADC-430). The exact MIRROR of rhs_flux_only: together they split rhs_into (-div F + S).
    // SourceInto evaluates m.source per cell (the SAME source term assemble_rhs adds) with no
    // numerical-flux dispatch, so it is bit-identical to the source half of rhs_into. Read by
    // System::block_source_into, which a compiled time Program's source stage calls so a Lie/Strang
    // split assembles "the default source but no flux" (spec: rhs flux=False is source-only). EMPTY
    // (default) for paths that do not build it (the host .so prototype loader); block_source_into fails
    // loud then. Trailing + empty default: the positional aggregate init of the other members stays
    // unchanged.
    std::function<void(MultiFab&, MultiFab&)> source_only;
    // Geometry-aware source-only twin.  It writes S(U, aux) on active cells and exact zero on
    // inactive cells for both Staircase and CutCell Program splits.
    std::function<void(MultiFab&, MultiFab&)> source_only_masked;
    // Point-qualified residuals are the sole compiled-Program route once a prepared boundary
    // component exists.  They retain the exact StagePoint instead of reconstructing time in a
    // closure or calling the legacy unqualified RHS.
    std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
        rhs_at_point;
    std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
        rhs_flux_only_at_point;
    // Interface-aware residual used by the multi-block pair executor.  It assembles every volume,
    // source and non-interface face term, but deliberately omits every face owned by a prepared
    // shared interface.  The scheduler then inserts the unique pair flux.  Reusing rhs_into here
    // would double-count its physical-BC/ghost flux, so installation fails unless this closure exists.
    std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
        rhs_without_prepared_interfaces;
    std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
        rhs_flux_only_without_prepared_interfaces;
    std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
        rhs_core_at_point;
    std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
        rhs_flux_only_core_at_point;
    std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&, MultiFab&)>
        boundary_residual_at_point;
    std::function<void(const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab&,
                       const MultiFab&, MultiFab&)>
        boundary_jvp_at_point;
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
    PointQualifiedResidualClosures staircase_residuals;
    PointQualifiedResidualClosures cutcell_residuals;
    // Frozen numerical-provider capability. Kept at the aggregate tail so the positional head used
    // by install_block remains ABI/source compatible.
    std::uint8_t supported_geometry_modes = kCartesianGeometrySupport;
    /// Sequential runtime session materialized once at bind, after block layouts and qualified
    /// storage routes are frozen. Prepared Krylov workspaces own distinct lane-private sessions.
    std::shared_ptr<ExecutionLane> boundary_lane;
    std::shared_ptr<PreparedGridBoundarySession> boundary_session;
    /// Exact owner-qualified state Handle.  Installed from the compiled block plan rather than
    /// inferred from the optional physical-boundary authority.
    std::string state_identity;
  };

  /// ORDERED registry of the blocks (UNIQUE source of truth). PUBLIC: Impl aliases it as `sp` for the
  /// already-extracted templates (SystemFieldSolver / SystemStepper / native_loader) that iterate owner_->sp.
  std::vector<BlockState> blocks;

  // --- access by NAME (0-based indexing, insertion order) ------------------------------------------
  /// Reference to block @p name (for writing). @throws std::runtime_error "System: bloc inconnu '...'".
  BlockState& find(const std::string& name) {
    for (auto& s : blocks)
      if (s.name == name)
        return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  /// Reference to block @p name (for reading). @throws std::runtime_error "System: bloc inconnu '...'".
  const BlockState& find(const std::string& name) const {
    for (auto& s : blocks)
      if (s.name == name)
        return s;
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }
  /// 0-based index of block @p name (insertion order). @throws std::runtime_error if unknown.
  int index(const std::string& name) const {
    for (std::size_t k = 0; k < blocks.size(); ++k)
      if (blocks[k].name == name)
        return static_cast<int>(k);
    throw std::runtime_error("System : bloc inconnu '" + name + "'");
  }

  /// Number of registered blocks.
  int size() const { return static_cast<int>(blocks.size()); }

  /// Install one prepared shared-interface flux between two Uniform runtime blocks.  The blocks may
  /// own distinct layouts/geometries; the scheduler proves the supported equal face discretisation
  /// before retaining the route.
  void install_interface_flux(
      runtime::multiblock::AxisAlignedInterface route, const Geometry& left_geometry,
      const Geometry& right_geometry, const PopsExecutionContextV1& execution,
      runtime::multiblock::InterfaceFluxEvaluatorFactory evaluator_factory) {
    if (route.left_block >= blocks.size() || route.right_block >= blocks.size())
      throw std::out_of_range("SystemBlockStore interface block index is out of range");
    const std::size_t left = route.left_block;
    const std::size_t right = route.right_block;
    if (!blocks[left].rhs_without_prepared_interfaces ||
        !blocks[right].rhs_without_prepared_interfaces ||
        !blocks[left].rhs_flux_only_without_prepared_interfaces ||
        !blocks[right].rhs_flux_only_without_prepared_interfaces)
      throw std::invalid_argument(
          "SystemBlockStore interface blocks lack full/flux-only interface-omitting residuals");
    interface_scheduler_.install(std::move(route), blocks[left].U, left_geometry, blocks[right].U,
                                 right_geometry, execution, std::move(evaluator_factory));
  }

  void install_interface_flux(runtime::multiblock::AxisAlignedInterface route,
                              const Geometry& left_geometry, const Geometry& right_geometry,
                              const PopsExecutionContextV1& execution,
                              runtime::multiblock::InterfaceFluxEvaluator evaluator) {
    install_interface_flux(
        std::move(route), left_geometry, right_geometry, execution,
        runtime::multiblock::InterfaceFluxEvaluatorFactory(
            [evaluator = std::move(evaluator)]() mutable { return std::move(evaluator); }));
  }

  /// Real Uniform multi-block residual executor: assemble every block RHS first, then invoke every
  /// shared interface exactly once and scatter its conservative pair contribution before any caller
  /// consumes either residual.
  void evaluate_rhs_with_interfaces(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                    const std::vector<MultiFab*>& states,
                                    const std::vector<MultiFab*>& rhs,
                                    const std::vector<int>& flux_only = {},
                                    GeometryMode geometry_mode = GeometryMode::None) {
    if (geometry_mode != GeometryMode::None && interface_scheduler_.has_interfaces(point.level))
      throw std::runtime_error(
          "SystemBlockStore embedded-boundary Program RHS cannot use a shared interface: "
          "the pair scheduler has no signed-mask/volume-metric contract");
    if (states.size() != blocks.size() || rhs.size() != blocks.size() ||
        (!flux_only.empty() && flux_only.size() != blocks.size()))
      throw std::invalid_argument("SystemBlockStore multi-block RHS vector size mismatch");
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if ((states[block] == nullptr) != (rhs[block] == nullptr))
        throw std::invalid_argument(
            "SystemBlockStore sparse RHS group has only one state/output pointer");
      if (states[block] != nullptr)
        require_geometry_provider(blocks[block], geometry_mode);
    }
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if ((states[block] == nullptr) != (rhs[block] == nullptr))
        throw std::invalid_argument(
            "SystemBlockStore sparse RHS group has only one state/output pointer");
      if (states[block] == nullptr)
        continue;
      const bool flux = !flux_only.empty() && flux_only[block] != 0;
      if (blocks[block].boundary_session) {
        if (geometry_mode != GeometryMode::None) {
          auto& prepared = select_prepared_full(blocks[block], geometry_mode, flux);
          if (!prepared)
            throw_missing_geometry_residual(blocks[block], geometry_mode,
                                            "prepared full Program RHS");
          prepared(point, *states[block], *rhs[block], *blocks[block].boundary_session);
          continue;
        }
        auto& core = flux ? blocks[block].rhs_flux_only_core_at_point_prepared
                          : blocks[block].rhs_core_at_point_prepared;
        if (!core || !blocks[block].boundary_residual_at_point_prepared)
          throw std::runtime_error(
              "SystemBlockStore block lacks its persistent prepared boundary closures");
        core(point, *states[block], *rhs[block], *blocks[block].boundary_session);
        blocks[block].boundary_residual_at_point_prepared(point, *states[block], *rhs[block],
                                                          *blocks[block].boundary_session);
        continue;
      }
      if (interface_scheduler_.participates(block, point.level)) {
        auto& closure = select_full(blocks[block], geometry_mode, flux,
                                    /*omit_prepared_interfaces=*/true);
        if (!closure)
          throw_missing_geometry_residual(blocks[block], geometry_mode,
                                          "interface-omitting Program RHS");
        closure(point, *states[block], *rhs[block]);
      } else {
        auto& closure = select_full(blocks[block], geometry_mode, flux,
                                    /*omit_prepared_interfaces=*/false);
        if (!closure)
          throw_missing_geometry_residual(blocks[block], geometry_mode,
                                          "point-qualified Program RHS");
        closure(point, *states[block], *rhs[block]);
      }
    }
    interface_scheduler_.apply(point, states, rhs);
  }

  /// Core-only twin for implicit linearization: ghost/volume/source/shared-interface terms are
  /// assembled exactly as above, while additive FieldBoundary residuals stay in their own closure.
  void evaluate_rhs_core_with_interfaces(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                         const std::vector<MultiFab*>& states,
                                         const std::vector<MultiFab*>& rhs,
                                         const std::vector<int>& flux_only = {},
                                         GeometryMode geometry_mode = GeometryMode::None) {
    if (geometry_mode != GeometryMode::None && interface_scheduler_.has_interfaces(point.level))
      throw std::runtime_error(
          "SystemBlockStore embedded-boundary core Program RHS cannot use a shared interface: "
          "the pair scheduler has no signed-mask/volume-metric contract");
    if (states.size() != blocks.size() || rhs.size() != blocks.size() ||
        (!flux_only.empty() && flux_only.size() != blocks.size()))
      throw std::invalid_argument("SystemBlockStore core RHS vector size mismatch");
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if ((states[block] == nullptr) != (rhs[block] == nullptr))
        throw std::invalid_argument(
            "SystemBlockStore sparse core RHS has only one state/output pointer");
      if (states[block] != nullptr)
        require_geometry_provider(blocks[block], geometry_mode);
    }
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if ((states[block] == nullptr) != (rhs[block] == nullptr))
        throw std::invalid_argument(
            "SystemBlockStore sparse core RHS has only one state/output pointer");
      if (states[block] == nullptr)
        continue;
      const bool flux = !flux_only.empty() && flux_only[block] != 0;
      if (blocks[block].boundary_session) {
        auto& prepared = select_prepared_core(blocks[block], geometry_mode, flux);
        if (!prepared)
          throw_missing_geometry_residual(blocks[block], geometry_mode,
                                          "prepared core Program RHS");
        prepared(point, *states[block], *rhs[block], *blocks[block].boundary_session);
        continue;
      }
      auto& closure = select_core(blocks[block], geometry_mode, flux);
      if (!closure)
        throw_missing_geometry_residual(blocks[block], geometry_mode,
                                        "point-qualified core Program RHS");
      closure(point, *states[block], *rhs[block]);
    }
    interface_scheduler_.apply(point, states, rhs);
  }

  /// Allocation-free scalar core route used by a prepared matrix-free operator. A block taking part
  /// in a shared interface cannot be linearized independently: the coupled operator must supply every
  /// participating state/output together, so this seam refuses instead of constructing a sparse
  /// temporary vector and evaluating an incomplete interface batch.
  void evaluate_rhs_core(const runtime::multiblock::BoundaryEvaluationPoint& point,
                         std::size_t block, MultiFab& state, MultiFab& rhs, bool flux_only,
                         GeometryMode geometry_mode = GeometryMode::None) {
    if (block >= blocks.size())
      throw std::out_of_range("SystemBlockStore core RHS block index is out of range");
    require_geometry_provider(blocks[block], geometry_mode);
    if (interface_scheduler_.participates(block, point.level))
      throw std::runtime_error(
          "System implicit core RHS requires a coupled shared-interface solve");
    auto& closure = select_core(blocks[block], geometry_mode, flux_only);
    if (blocks[block].boundary_session) {
      auto& prepared = select_prepared_core(blocks[block], geometry_mode, flux_only);
      if (!prepared)
        throw_missing_geometry_residual(blocks[block], geometry_mode,
                                        "persistent prepared core Program RHS");
      prepared(point, state, rhs, *blocks[block].boundary_session);
      return;
    }
    if (!closure)
      throw_missing_geometry_residual(blocks[block], geometry_mode,
                                      "point-qualified core Program RHS");
    closure(point, state, rhs);
  }

  void evaluate_rhs_core_prepared(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                  std::size_t block, MultiFab& state, MultiFab& rhs, bool flux_only,
                                  const PreparedGridBoundarySession& boundary,
                                  GeometryMode geometry_mode = GeometryMode::None) {
    if (block >= blocks.size())
      throw std::out_of_range("SystemBlockStore prepared core RHS block index is out of range");
    require_geometry_provider(blocks[block], geometry_mode);
    if (interface_scheduler_.participates(block, point.level))
      throw std::runtime_error(
          "System implicit core RHS requires a coupled shared-interface solve");
    auto& closure = select_prepared_core(blocks[block], geometry_mode, flux_only);
    if (!closure)
      throw_missing_geometry_residual(blocks[block], geometry_mode, "prepared core Program RHS");
    closure(point, state, rhs, boundary);
  }

  std::size_t interface_evaluation_count(const std::string& identity, int level) const {
    return interface_scheduler_.evaluation_count(identity, level);
  }

  bool has_interfaces(int level) const { return interface_scheduler_.has_interfaces(level); }

  void discard_interface_fluxes() { interface_scheduler_.clear(); }
  /// Block names, in insertion order (UNIQUE source: all add paths appear here).
  std::vector<std::string> names() const {
    // Reads the UNIQUE block registry populated by the native install paths.
    std::vector<std::string> out;
    out.reserve(blocks.size());
    for (const auto& s : blocks)
      out.push_back(s.name);
    return out;
  }

  // --- state marshaling (host <-> MultiFab copy; device_fence included) ---------------------------
  /// Copies component 0 of fab(0) (density) in row-major (j slow, i fast). device_fence beforehand:
  /// the marshaling reads the host, so the device must have finished writing U.
  std::vector<double> copy_comp0(const MultiFab& mf) const {
    device_fence();
    // MPI single-box: the box lives on the owner rank (rank 0). A rank without a box (local_size()==0)
    // has NO fab(0) -> return EMPTY rather than an OUT-OF-BOUNDS access (UB). Single-rank: local_size()
    // is always 1, behavior UNCHANGED. For the global multi-rank field: System::density_global.
    if (mf.local_size() == 0)
      return {};
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(v.nx()) * v.ny());
    for (int j = v.lo[1]; j <= v.hi[1]; ++j)
      for (int i = v.lo[0]; i <= v.hi[0]; ++i)
        out.push_back(u(i, j, 0));
    return out;
  }
  /// Copies the ncomp components of fab(0) in component-major layout (c slow, then j, then i).
  std::vector<double> copy_state(const MultiFab& mf, int ncomp) const {
    device_fence();
    // Rank without a box (MPI single-box, non-owner): return EMPTY (no fab(0)). Cf. copy_comp0;
    // the global multi-rank field goes through System::state_global (collective gather).
    if (mf.local_size() == 0)
      return {};
    const ConstArray4 u = mf.fab(0).const_array();
    const Box2D v = mf.box(0);
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(ncomp) * v.nx() * v.ny());
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out.push_back(u(i, j, c));
    return out;
  }
  /// Writes the ncomp components into fab(0) from a component-major buffer (same layout as
  /// copy_state). @throws std::runtime_error "System::set_state: taille != ncomp*n*n" if the size
  /// does not match ncomp*nx*ny (message unchanged).
  void write_state(MultiFab& mf, int ncomp, const std::vector<double>& in) {
    // Rank without a box (MPI single-box, non-owner): NO-OP (no fab(0) to write). Lets
    // sim.set_state / sim.restart be called on ALL ranks with the GLOBAL field: only the
    // owner rank (rank 0, box = full domain) writes, the others do nothing. Single-rank:
    // local_size()==1, validation + write UNCHANGED (bit-identical).
    if (mf.local_size() == 0)
      return;
    const Box2D v = mf.box(0);
    const std::size_t need = static_cast<std::size_t>(ncomp) * v.nx() * v.ny();
    if (in.size() != need)
      throw std::runtime_error("System::set_state : taille != ncomp*n*n");
    Array4 u = mf.fab(0).array();
    std::size_t k = 0;
    for (int c = 0; c < ncomp; ++c)
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          u(i, j, c) = in[k++];
  }

 private:
  static void require_geometry_provider(const BlockState& block, GeometryMode mode) {
    if (!supports_geometry_mode(block.supported_geometry_modes, mode))
      throw std::runtime_error("SystemBlockStore block '" + block.name +
                               "' has no numerical provider for geometry policy '" +
                               geometry_token(mode) + "'");
    if (mode == GeometryMode::None || !block.boundary_session)
      return;
    const PreparedBoundaryPlan* plan = block.boundary_session->resolved_plan();
    if (plan != nullptr && plan->has_component_boundaries())
      throw std::runtime_error(
          "SystemBlockStore embedded-boundary block '" + block.name +
          "' cannot execute a native boundary component without an active-cell or cut-cell "
          "metric provider");
  }

  static PointQualifiedResidualClosures& embedded_residuals(BlockState& block, GeometryMode mode) {
    return mode == GeometryMode::Staircase ? block.staircase_residuals : block.cutcell_residuals;
  }

  static PointQualifiedResidualClosures::AtPoint& select_full(BlockState& block, GeometryMode mode,
                                                              bool flux_only,
                                                              bool omit_prepared_interfaces) {
    if (mode == GeometryMode::None) {
      if (omit_prepared_interfaces)
        return flux_only ? block.rhs_flux_only_without_prepared_interfaces
                         : block.rhs_without_prepared_interfaces;
      return flux_only ? block.rhs_flux_only_at_point : block.rhs_at_point;
    }
    auto& family = embedded_residuals(block, mode);
    if (omit_prepared_interfaces)
      return flux_only ? family.flux_only_without_prepared_interfaces
                       : family.without_prepared_interfaces;
    return flux_only ? family.flux_only : family.full;
  }

  static PointQualifiedResidualClosures::AtPoint& select_core(BlockState& block, GeometryMode mode,
                                                              bool flux_only) {
    if (mode == GeometryMode::None)
      return flux_only ? block.rhs_flux_only_core_at_point : block.rhs_core_at_point;
    auto& family = embedded_residuals(block, mode);
    return flux_only ? family.flux_only_core : family.core;
  }

  static PointQualifiedResidualClosures::PreparedAtPoint& select_prepared_full(BlockState& block,
                                                                               GeometryMode mode,
                                                                               bool flux_only) {
    if (mode == GeometryMode::None) {
      // The Cartesian prepared full route is deliberately composed from its core and additive
      // boundary closures by evaluate_rhs_with_interfaces, as it was before geometry routing.
      static PointQualifiedResidualClosures::PreparedAtPoint empty;
      return empty;
    }
    auto& family = embedded_residuals(block, mode);
    return flux_only ? family.flux_only_full_prepared : family.full_prepared;
  }

  static PointQualifiedResidualClosures::PreparedAtPoint& select_prepared_core(BlockState& block,
                                                                               GeometryMode mode,
                                                                               bool flux_only) {
    if (mode == GeometryMode::None)
      return flux_only ? block.rhs_flux_only_core_at_point_prepared
                       : block.rhs_core_at_point_prepared;
    auto& family = embedded_residuals(block, mode);
    return flux_only ? family.flux_only_core_prepared : family.core_prepared;
  }

  static const char* geometry_token(GeometryMode mode) {
    switch (mode) {
      case GeometryMode::None:
        return "cartesian";
      case GeometryMode::Staircase:
        return "staircase";
      case GeometryMode::CutCell:
        return "cutcell";
    }
    return "unknown";
  }

  [[noreturn]] static void throw_missing_geometry_residual(const BlockState& block,
                                                           GeometryMode mode, const char* route) {
    throw std::runtime_error("SystemBlockStore block '" + block.name + "' lacks " + route +
                             " for geometry policy '" + geometry_token(mode) + "'");
  }

  runtime::multiblock::InterfaceFluxScheduler interface_scheduler_;
};

}  // namespace pops
