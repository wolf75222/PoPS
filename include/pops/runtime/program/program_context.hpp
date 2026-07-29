#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pops/core/foundation/types.hpp>     // Real, POPS_HD
#include <pops/runtime/program/profiler.hpp>  // Profiler / ProfileScope (per-node timing, ADC-459)
#include <pops/runtime/program/step_transaction.hpp>
#include <pops/runtime/program/wire_ids.hpp>   // stable compiled-Program numeric protocol
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts (periodic / physical halo exchange)
#include <pops/mesh/execution/for_each.hpp>  // for_each_cell (per-cell coeff / reconstruct kernels + negated divergence copy)
#include <pops/mesh/geometry/geometry.hpp>  // Geometry (mesh metric of the Laplacian / gradient)
#include <pops/mesh/layout/field_distribution.hpp>  // FieldDistribution
#include <pops/mesh/storage/fab2d.hpp>              // Array4 / ConstArray4 (per-cell handles)
#include <pops/mesh/storage/mf_arith.hpp>           // saxpy (linear combine over a MultiFab)
#include <pops/mesh/storage/multifab.hpp>           // MultiFab
#include <pops/parallel/execution_lane.hpp>
#include <pops/parallel/solve_report_consensus.hpp>
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess (centered gradient)
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/numerics/elliptic/linear/vector_distribution.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (shared 5-point matvec)
#include <pops/numerics/elliptic/polar/polar_tensor_operator.hpp>  // metric-aware generated tensor solve
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams (compiled-Program runtime params, ADC-510)
#include <pops/runtime/context/grid_context.hpp>    // GridContext (System aux seam)
#include <pops/runtime/program/cache_manager.hpp>   // CacheManager (held-node value cache, ADC-458)
#include <pops/runtime/program/clock_schedule.hpp>  // nested logical-clock cursor validation
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/system.hpp>  // System (the runtime this facade forwards to)

/// @file
/// @brief ProgramContext -- the C++-side facade a generated problem.so calls to run a compiled time
///        Program during sim.step(dt) (epic ADC-399, ADC-401 Phase 2b).
///
/// It REIMPLEMENTS NOTHING. Each method forwards to an existing pops::System primitive:
///   install(fn)          -> System::install_program_step(fn)   (registers the macro-step body)
///   solve_fields()       -> System::solve_fields()             (elliptic solve + aux at current U)
///   solve_fields_from_state(b, U) -> System::solve_fields_from_state(b, U) (aux at a stage state)
///   n_blocks()           -> System::n_blocks()
///   state(b)             -> System::block_state(b)             (the block's live MultiFab, zero-copy)
///   rhs_into(b, U, R, rate_id) -> System::block_rhs_into_at(...) (point-qualified -div F + S)
///   neg_div_flux_default_into(b, U, R, rate_id) -> point-qualified -div F with no source
///   axpy(U, a, R)        -> pops::saxpy(U, a, R)                (U <- U + a R, device-dispatched)
///
/// The Program composes the chain (e.g. Forward Euler = solve_fields(); for each block:
/// rhs_into(b, U, R, rate_id); axpy(U, dt, R)) and installs it via install(...). The .so NEVER touches
/// System::Impl / Array4 / fill_boundary / the elliptic solver / Kokkos / MPI / CFL / substeps.
///
/// IDIOM: ProgramContext is a plain (non-template) class holding a System*. A generated .so receives
/// the System as a flat void* across the dlopen boundary (like the native loader's `void* self`) and
/// wraps it here; it reaches per-block storage through the System's public accessors because
/// System::Impl is private to the _pops translation unit.
namespace pops {
namespace runtime {
namespace program {

class ProgramContext : public ProgramExecutionServices<ProgramContext> {
 public:
  explicit ProgramContext(System* sys) : sys_(sys) {}
  /// Wraps a System passed as a flat void* (what pops_install_program(void* sys) receives).
  explicit ProgramContext(void* sys) : sys_(static_cast<System*>(sys)) {}

  /// Register the macro-step body. @p step advances ONE macro-step over dt (it owns solve_fields,
  /// the RHS, the linear combine and the commit). An empty std::function is rejected.
  void install(std::function<void(double)> step) const {
    sys_->install_program_step(std::move(step));
  }

  /// Start one generated Program body.  The native stepper supplies the accepted local dt; every
  /// boundary evaluation in the body derives its physical time from this exact value and the
  /// authored rational stage fraction.
  void begin_step(double dt) const {
    if (!std::isfinite(dt) || dt <= 0.0)
      throw std::invalid_argument("Program boundary clock requires a finite positive dt");
    current_dt_ = dt;
    stage_time_ = amr::Rational(0, 1);
    logical_phase_begin_ = amr::Rational(0, 1);
    logical_phase_span_ = amr::Rational(1, 1);
    logical_physical_time_offset_ = 0.0;
  }

  SolveOutcome solve_fields() const {
    // No count_kernel() here: System's private in-place default provider seam already counts it.
    // The from_state/from_blocks/named seams below do not, so those routes count explicitly.
    sys_->prepare_default_field_publication_storage_();
    return run_field_solve_transaction_([&]() { return sys_->solve_fields_in_place_(); });
  }
  /// Per-stage field solve (ADC-409): re-solve the elliptic fields and re-fill the shared aux from
  /// block @p b's STAGE state @p u_stage (not its live state), so a field-coupled multi-stage
  /// Program's stage k reads phi solved from stage k's own state. Forwards to
  /// System::solve_fields_from_state. With b = 0 and u_stage = U^n (the first stage) it matches
  /// solve_fields(); the codegen lowers every solve_fields op to this, passing the stage's state var.
  SolveOutcome solve_fields_from_state(int b, MultiFab& u_stage) const {
    count_kernel();
    sys_->prepare_default_field_publication_storage_();
    return run_field_solve_transaction_(
        [&]() { return sys_->solve_fields_from_state_in_place_(sys_block(b), u_stage); });
  }
  SolveOutcome solve_fields_from_state_at(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                          const std::string& provider_slot, int b,
                                          MultiFab& u_stage) const {
    count_kernel();
    if (provider_slot.empty())
      throw std::invalid_argument(
          "System::solve_fields_from_state_at requires an exact provider slot");
    sys_->prepare_named_field_publication_storage_(provider_slot);
    return run_field_solve_transaction_([&]() {
      return sys_->solve_fields_from_state_at_in_place_(point, provider_slot, sys_block(b),
                                                        u_stage);
    });
  }
  /// Named multi-elliptic field solve (ADC-428): re-solve the SECOND elliptic field @p field from block
  /// @p b's stage state @p u_stage and write its phi (+ centered grad) into the field's OWN aux
  /// components (distinct from the shared phi/grad the default solve_fields fills). Forwards to
  /// System::solve_fields_from_state(field, b, u_stage). The codegen lowers
  /// P.solve_fields(field=name, state=U) to this; a default (unnamed) solve_fields keeps the overload
  /// above, byte-identical.
  SolveOutcome solve_fields_from_state(const std::string& field, int b, MultiFab& u_stage) const {
    count_kernel();
    sys_->prepare_named_field_publication_storage_(field);
    return run_field_solve_transaction_(
        [&]() { return sys_->solve_fields_from_state_in_place_(field, sys_block(b), u_stage); });
  }
  /// Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): re-solve the elliptic fields and
  /// re-fill the shared aux from the SIMULTANEOUS stage states of MULTIPLE blocks at once -- the system
  /// Poisson RHS is Sum_s elliptic_rhs_s(U_s), every coupled block reading its OWN stage state (not a
  /// single-target override). @p u_stages is indexed BY BLOCK INDEX (size == n_blocks()); a nullptr
  /// entry uses that block's live state. Forwards to System::solve_fields_from_blocks. The codegen
  /// Manual callers may provide the historical pointer vector. Generated Programs use the exact-IR
  /// initializer-list overload below, which fills the same context-owned workspace without allocating
  /// a pointer vector in the step body. This is the multi-target counterpart of solve_fields_from_state.
  SolveOutcome solve_fields_from_blocks(const std::vector<const MultiFab*>& u_stages) const {
    count_kernel();
    // The codegen builds @p u_stages indexed BY PROGRAM block index (a stage state slotted at its own
    // Program index, the rest nullptr). The System solver expects it indexed by SYSTEM block index, so
    // re-slot each Program entry p at its name-matched System index sys_block(p) (Spec 3 criterion 23,
    // ADC-457). Even an order-matching Program carries an explicit identity map.
    const std::vector<int>& block_map = sys_->program_block_map();
    if (block_map.empty())
      throw block_map_error_(
          "ProgramContext::solve_fields_from_blocks: no explicit program-to-system block map is "
          "installed; positional block identity is not supported");
    if (u_stages.size() < block_map.size())
      throw block_map_error_("ProgramContext::solve_fields_from_blocks: received " +
                             std::to_string(u_stages.size()) +
                             " Program stage slots for an explicit block map with " +
                             std::to_string(block_map.size()) + " entries");
    FieldSolveWorkspace& workspace = manual_default_field_solve_workspace_();
    fill_manual_field_stages_(workspace, u_stages, /*require_exact_size=*/false);
    // Iterate the PROGRAM block indices [0, m.size()) -- NOT u_stages.size(), which is the larger
    // SYSTEM block count. The codegen sizes u_stages to ctx.n_blocks() but only fills Program slots
    // [0, n_program_blocks); when the System has MORE blocks than the Program declares (a subset
    // install), walking the System-sized range would re-map the nullptr padding through the identity
    // fallthrough and clobber real entries. m[p] is Program block p's System index (install-validated
    // in range); the unlisted System slots stay nullptr = their live state. sys_block validates every
    // mapped value before it is used as a vector index.
    sys_->prepare_default_field_publication_storage_();
    return run_field_solve_transaction_(
        [&]() { return solve_default_field_workspace_(workspace); });
  }

  SolveOutcome solve_fields_from_blocks(const std::string& field,
                                        const std::vector<const MultiFab*>& u_stages) const {
    count_kernel();
    FieldSolveWorkspace& workspace = manual_named_field_solve_workspace_(field);
    fill_manual_field_stages_(workspace, u_stages, /*require_exact_size=*/true);
    sys_->prepare_named_field_publication_storage_(field);
    return run_field_solve_transaction_(
        [&]() { return solve_named_field_workspace_(field, workspace); });
  }

  /// Allocation-free generated route.  The exact IR identity owns one context-local pointer/snapshot
  /// workspace; @p field and the ordered Program block pack are authenticated on every replay.  The
  /// old vector overloads above remain available for manual C++ callers.
  SolveOutcome solve_fields_from_blocks(std::int64_t value_id, std::string_view field,
                                        std::initializer_list<FieldStageOverride> overrides) const {
    count_kernel();
    FieldSolveWorkspace& workspace = generated_field_solve_workspace_(value_id, field, overrides);
    sys_->prepare_named_field_publication_storage_(workspace.generated_field_identity);
    return run_field_solve_transaction_([&]() {
      return solve_named_field_workspace_(workspace.generated_field_identity, workspace);
    });
  }
  MultiFab& state(int b) const { return sys_->block_state(sys_block(b)); }
  /// Evaluate one authored rate at its exact, stable node identity.  There is deliberately no
  /// sentinel/default identity: shared-interface assembly and boundary callbacks authenticate this
  /// value as part of BoundaryEvaluationPoint, so an anonymous rate would be temporally ambiguous.
  void rhs_into(int b, MultiFab& u, MultiFab& r, int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel();
    sys_->block_rhs_into_at(boundary_point_(rate_id), sys_block(b), u, r);
  }
  runtime::multiblock::BoundaryEvaluationPoint boundary_evaluation_point(int stage_id) const {
    return boundary_point_(stage_id);
  }
  bool has_boundary_linearization(int b) const {
    return sys_->block_has_boundary_linearization(sys_block(b));
  }
  void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                        MultiFab& u, MultiFab& r, bool flux_only) const {
    count_kernel();
    sys_->block_rhs_core_into_at(point, sys_block(b), u, r, flux_only);
  }
  void rhs_core_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                        MultiFab& u, MultiFab& r, bool flux_only,
                        const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    sys_->block_rhs_core_into_at(point, sys_block(b), u, r, flux_only, boundary);
  }
  void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                                 MultiFab& u, MultiFab& c) const {
    count_kernel();
    sys_->block_boundary_residual_into_at(point, sys_block(b), u, c);
  }
  void boundary_residual_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                                 MultiFab& u, MultiFab& c,
                                 const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    sys_->block_boundary_residual_into_at(point, sys_block(b), u, c, boundary);
  }
  void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                            MultiFab& u, const MultiFab& v, MultiFab& j) const {
    count_kernel();
    sys_->block_boundary_jvp_into_at(point, sys_block(b), u, v, j);
  }
  void boundary_jvp_into_at(const runtime::multiblock::BoundaryEvaluationPoint& point, int b,
                            MultiFab& u, const MultiFab& v, MultiFab& j,
                            const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    sys_->block_boundary_jvp_into_at(point, sys_block(b), u, v, j, boundary);
  }

  struct RhsGroupRequest {
    RhsGroupRequest(int block_value, MultiFab* state_value, MultiFab* rhs_value, int rate_id_value,
                    int flux_only_value)
        : block(block_value),
          state(state_value),
          rhs(rhs_value),
          rate_id(rate_id_value),
          flux_only(flux_only_value) {}

    int block;
    MultiFab* state;
    MultiFab* rhs;
    int rate_id;
    int flux_only;
  };

  /// Simultaneous multi-block rate evaluation.  @p group_id is the exact authored identity of this
  /// atomic evaluation and is deliberately distinct from every request's rate-node identity.  The
  /// generated Program emits one group only for RHS nodes authenticated at the same exact StagePoint;
  /// System then executes each installed interface once before any group result can be consumed.
  void rhs_group(int group_id, std::initializer_list<RhsGroupRequest> requests) const {
    require_group_identity_(group_id);
    if (requests.size() == 0)
      throw std::invalid_argument("Program RHS group cannot be empty");
    std::vector<int> rate_ids;
    rate_ids.reserve(requests.size());
    for (const auto& request : requests) {
      require_rate_identity_(request.rate_id);
      if (request.rate_id == group_id ||
          std::find(rate_ids.begin(), rate_ids.end(), request.rate_id) != rate_ids.end())
        throw std::invalid_argument(
            "Program RHS group and member rate identities must be distinct");
      if (request.state == nullptr || request.rhs == nullptr ||
          (request.flux_only != 0 && request.flux_only != 1))
        throw std::invalid_argument("Program RHS group contains an invalid request");
      rate_ids.push_back(request.rate_id);
    }
    std::vector<int> blocks;
    std::vector<MultiFab*> states;
    std::vector<MultiFab*> rhs;
    std::vector<int> flux_only;
    blocks.reserve(requests.size());
    states.reserve(requests.size());
    rhs.reserve(requests.size());
    flux_only.reserve(requests.size());
    for (const auto& request : requests) {
      count_kernel();
      blocks.push_back(sys_block(request.block));
      states.push_back(request.state);
      rhs.push_back(request.rhs);
      flux_only.push_back(request.flux_only);
    }
    sys_->block_rhs_group(boundary_point_(group_id), blocks, states, rhs, flux_only);
  }

  /// r <- -div F(u) for block @p b -- the SAME flux divergence as @ref rhs_into but WITHOUT the model's
  /// default/composite source (Poisson frozen). Forwards to System::block_neg_div_flux_into (the block's
  /// SourceFreeModel<Model> rhs path, bit-identical to rhs_into minus the source). The codegen lowers a
  /// hyperbolic stage that excludes the default source (P.rhs(flux=True, sources without "default"),
  /// incl. the empty list) to this, so a Lie/Strang split assembles "flux but no source" without the
  /// default source leaking in (epic ADC-399 / ADC-425, spec criterion 17). Header-inline forwarder,
  /// like @ref rhs_into.
  void neg_div_flux_default_into(int b, MultiFab& u, MultiFab& r, int rate_id) const {
    require_rate_identity_(rate_id);
    count_kernel();
    sys_->block_neg_div_flux_into_at(boundary_point_(rate_id), sys_block(b), u, r);
  }

  /// r <- S(u, aux) for block @p b -- the model's default/composite SOURCE only, WITHOUT the flux
  /// divergence (the exact MIRROR of @ref neg_div_flux_default_into). Forwards to
  /// System::block_source_into (the block's SourceInto path, bit-identical to the source half of
  /// rhs_into). The codegen lowers a SOURCE stage (P.rhs(flux=False, sources with "default")) to this, so
  /// a Lie/Strang split assembles "the default source but no flux" without the -div F base leaking in
  /// (epic ADC-399 / ADC-430, spec: rhs flux=False is source-only). Header-inline forwarder, like @ref
  /// neg_div_flux_default_into.
  void source_default_into(int b, MultiFab& u, MultiFab& r) const {
    count_kernel();
    sys_->block_source_into(sys_block(b), u, r);
  }

  /// Fail before a generated pointwise operator touches storage when an embedded boundary is active.
  /// Default-source and transport residuals have native geometry-aware providers; arbitrary generated
  /// expressions and local solves do not yet, and cannot be repaired by post-zeroing their outputs.
  void require_cartesian_generated_operator(int b, const std::string& operation) const {
    sys_->require_cartesian_generated_operator(sys_block(b), operation);
  }

  /// Return the prepared active-cell mask for a pointwise generated operator owned by Program block
  /// @p b.  A Cartesian block has no mask (nullptr means every valid cell is active).  Embedded
  /// geometries expose the same stable, block-qualified mask used by the native residual and field
  /// algebra.  The field layout is authenticated here, before generated code launches a device
  /// kernel; pointwise providers therefore never infer geometry from a case name or mode.
  const MultiFab* pointwise_active_mask(int b, const MultiFab& field) const {
    const GridContext context = sys_->grid_context(sys_block(b));
    if (context.embedded_boundary_set == nullptr || !*context.embedded_boundary_set ||
        context.geometry_mode == nullptr || *context.geometry_mode == GeometryMode::None)
      return nullptr;
    if (context.domain_mask == nullptr)
      throw std::runtime_error(
          "ProgramContext pointwise operator has no prepared active-cell mask");
    pops::detail::validate_relative_cell_measure(field,
                                                 RelativeCellMeasure{context.domain_mask, nullptr},
                                                 "ProgramContext pointwise active-cell mask");
    return context.domain_mask;
  }

  /// Collective fail-closed reduction for a pointwise generated operator.  @p active_cells must be
  /// the exact block-qualified mask returned by pointwise_active_mask for @p status; this prevents a
  /// generated kernel from evaluating one physical domain and validating another.
  Real pointwise_status_max(int b, const MultiFab& status, const MultiFab* active_cells) const {
    const MultiFab* expected = pointwise_active_mask(b, status);
    if (expected != active_cells)
      throw std::invalid_argument(
          "ProgramContext pointwise status reduction received a different active-cell mask");
    const Real reduced = pops::reduce_max(status, 0, RelativeCellMeasure{active_cells, nullptr});
    return reduced == -std::numeric_limits<Real>::infinity() ? Real(0) : reduced;
  }

  /// The MIN physical cell size of the grid (Cartesian min(dx, dy); polar min(dr, r_min*dtheta)) -- the
  /// SAME hmin the native CFL uses. Forwards to System::cfl_min_dx. A compiled time Program's dt bound
  /// (epic ADC-399 / ADC-417, spec s18) reads it to express e.g. cfl * hmin / max_wave_speed.
  Real hmin() const { return sys_->cfl_min_dx(); }

  /// The maximum |wave speed| of block @p b on the state @p u: the SAME per-block reduction step_cfl
  /// reads (BlockState::max_speed). Forwards to System::block_max_speed -- it REUSES the block's
  /// wave-speed closure, it does not recompute the speed. @p u is the state the bound is evaluated on
  /// (the block's current state for a CFL bound). The dt_bound expression uses it as the denominator of
  /// cfl * hmin / max_wave_speed (epic ADC-399 / ADC-417, spec s18).
  Real max_wave_speed(int b, const MultiFab& u) const {
    return sys_->block_max_speed(sys_block(b), u);
  }

  /// The System aux MultiFab (phi=0, grad_x=1, grad_y=2, B_z=3, T_e=4, named fields from
  /// kAuxNamedBase). NOT owned by the context: it is the live System aux (stable address), the same
  /// channel solve_fields() fills. A generated local-linear-solve kernel reads the operator
  /// coefficients (e.g. B_z) from it. Forwards to System::grid_context().aux.
  MultiFab& aux() const { return *sys_->grid_context().aux; }

  /// The System grid context (transport BC + mesh geometry + the live aux pointer). BY VALUE:
  /// grid_context() returns a temporary. A generic seam accessor forwarding to
  /// System::grid_context(), used by out-of-line runtime operators (the coupled elliptic operator
  /// modules) that assemble coefficient / flux halos from the transport BC without reaching into
  /// System::Impl -- the SAME channel geom() / aux() expose, bundled.
  GridContext grid_context() const { return sys_->grid_context(); }

  /// Materialize one lane-private mesh authority for a prepared operator that is not attached to a
  /// conservative block (for example, a scalar elliptic field).  This deliberately uses the
  /// unqualified mesh BC and cannot borrow a block's native boundary components.
  std::shared_ptr<PreparedGridBoundarySession> prepare_mesh_boundary_session(
      const MultiFab&, const ExecutionLane& lane) const {
    return std::make_shared<PreparedGridBoundarySession>(sys_->grid_context(), lane);
  }

  /// Materialize the exact boundary authority of one authenticated Program block.  The Program
  /// index is resolved through the installed name map before any component state is prepared.
  std::shared_ptr<PreparedGridBoundarySession> prepare_block_boundary_session(
      int block, MultiFab& prototype, const runtime::multiblock::BoundaryEvaluationPoint& point,
      const ExecutionLane& lane) const {
    return std::make_shared<PreparedGridBoundarySession>(sys_->grid_context(sys_block(block)), lane,
                                                         prototype, point);
  }

  /// The MultiFab a per-level coefficient / RHS assembly kernel should WRITE its field into (ADC-633).
  /// On the uniform System the answer is always the passed field itself -- an IDENTITY hook, so a
  /// templated assembly free function writes straight into the level-0-bound scratch the codegen
  /// allocated, byte-for-byte as before. The opaque prepared field-slot identity is ignored here; it
  /// exists so an AMR provider can redirect the write to its own per-level storage without extending
  /// this context for every new operator envelope.
  MultiFab& assembly_target(MultiFab& field, std::string_view field_slot_identity) const {
    validate_prepared_field_slot(field_slot_identity, "ProgramContext::assembly_target");
    return field;
  }

  /// The MultiFab a per-level reconstruction should READ its solved field from (ADC-633). Identity on
  /// the uniform System (the field passed is the level-0 solution the emitted solve wrote); the AMR
  /// ProgramContext redirects the READ to the current level's published composite field on a refined
  /// hierarchy. Trivial + inline so the uniform .so is byte-for-byte unchanged.
  MultiFab& assembly_source(MultiFab& field, std::string_view field_slot_identity) const {
    validate_prepared_field_slot(field_slot_identity, "ProgramContext::assembly_source");
    return field;
  }
  /// Uniform counterpart of AmrProgramContext::linear_solution: one grid has one solve field.
  MultiFab& linear_solution(MultiFab& field) const { return field; }

  /// Authenticate the exact operator evaluation point. Generated code supplies a canonical 256-bit
  /// Program/operator authority plus the prepared field/resource identities; the context supplies the
  /// monotonic evaluation revision and exact native clock values.
  OperatorEvaluationSnapshot operator_evaluation_snapshot(OperatorFingerprint authority,
                                                          const MultiFab& prototype,
                                                          OperatorFingerprint resources) const {
    if (!std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::logic_error("operator snapshot requested outside a prepared Program step");
    const GridContext gc = sys_->grid_context();
    OperatorFingerprint topology =
        ::pops::detail::layout_fingerprint(prototype, program_resource_vector_distribution());
    if (sys_->program_is_polar())
      ::pops::detail::fingerprint_geometry(topology, sys_->program_polar_geometry());
    else
      ::pops::detail::fingerprint_geometry(topology, gc.geom);
    ::pops::detail::fingerprint_boundary(topology, gc.bc);
    if (gc.boundary_plan) {
      ::pops::detail::fingerprint_mix(topology, gc.boundary_plan->identity());
      ::pops::detail::fingerprint_mix(topology, gc.boundary_plan->state_identity());
      ::pops::detail::fingerprint_mix(
          topology, static_cast<std::uint64_t>(gc.boundary_plan->required_depth()));
    } else {
      ::pops::detail::fingerprint_mix(topology, "legacy-bcrec-boundary");
    }
    if (operator_snapshot_revision_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("Program operator snapshot revision exhausted");
    const std::uint64_t revision = ++operator_snapshot_revision_;
    invalidate_active_operator_snapshot_();
    OperatorEvaluationSnapshot snapshot =
        operator_evaluation_snapshot_(authority, topology, resources, revision);
    active_operator_snapshot_revision_ = revision;
    return snapshot;
  }

  /// Recompute the current native identity without advancing the monotonic counter. A requested
  /// revision is reproduced only while it is the context's active mint; logical-scope entry/exit
  /// clears that authority, so an exactly restored outer clock still probes unequal until reminted.
  /// The uniform mesh fingerprint remains reusable, keeping the Krylov probe free of layout walks.
  OperatorEvaluationSnapshot probe_operator_evaluation(OperatorFingerprint authority,
                                                       OperatorFingerprint topology,
                                                       OperatorFingerprint resources,
                                                       std::uint64_t revision) const {
    const std::uint64_t probe_revision =
        revision == active_operator_snapshot_revision_ ? revision : UINT64_C(0);
    return operator_evaluation_snapshot_(authority, topology, resources, probe_revision);
  }

 private:
  void invalidate_active_operator_snapshot_() const noexcept {
    active_operator_snapshot_revision_ = 0;
  }

  OperatorEvaluationSnapshot operator_evaluation_snapshot_(OperatorFingerprint authority,
                                                           OperatorFingerprint topology,
                                                           OperatorFingerprint resources,
                                                           std::uint64_t revision) const {
    if (!std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::logic_error("operator snapshot requested outside a prepared Program step");
    const amr::Rational evaluation_stage = logical_phase_begin_ + stage_time_ * logical_phase_span_;
    const double evaluation_time = static_cast<double>(physical_time()) +
                                   logical_physical_time_offset_ +
                                   stage_time_.value() * current_dt_;
    return {authority,
            revision,
            static_cast<std::int64_t>(macro_step()),
            evaluation_stage.numerator,
            evaluation_stage.denominator,
            std::bit_cast<std::uint64_t>(current_dt_),
            std::bit_cast<std::uint64_t>(evaluation_time),
            UINT64_C(1),
            topology,
            resources};
  }

 public:
  /// Capability for an operator body emitted inside this compiled Program artifact. Direct C++
  /// extensions cannot construct the token and therefore remain on the verified apply path.
  ::pops::detail::AuthenticatedProgramApplyToken authenticated_program_apply_token(
      OperatorFingerprint authority) const {
    if (sys_ == nullptr || !sys_->program_owns_operator_authority(authority))
      throw std::invalid_argument(
          "compiled Program requested an operator authority not owned by its installed artifact");
    return ::pops::detail::AuthenticatedProgramApplyToken(authority);
  }

  /// Execute an already prepared affine problem with its bound persistent workspace. The raw callback,
  /// integer method wire, lazy preconditioner path and per-call scratch allocations no longer exist.
  SolveOutcome solve_prepared_linear(const PreparedAffineLinearProblem& problem,
                                     KrylovWorkspace& workspace, MultiFab& sol, const MultiFab& rhs,
                                     const KrylovControls& controls) const {
    return pops::solve_prepared_affine_outcome(problem, workspace, sol, rhs, controls);
  }

  /// Physical ownership of fields allocated for one generated Program resource bundle. A uniform
  /// System partitions its one mesh across communicator ranks; exposing the descriptor through the
  /// same context protocol as AMR keeps generic codegen independent from the runtime target.
  const PreparedVectorDistribution& program_resource_vector_distribution() const noexcept {
    return PreparedVectorDistribution::Distributed;
  }
  FieldDistribution program_resource_field_storage_distribution() const noexcept {
    return FieldDistribution::Distributed;
  }
  int program_resource_field_level() const noexcept { return 0; }
  /// Resolve the topology-dependent part of an authored field-nullspace plan at the same boundary
  /// that allocates its persistent Program resource. Generic codegen supplies only the mathematical
  /// basis/gauge; the context owns layout scope, physical measure and communicator ownership.
  void configure_program_resource_field_nullspace(FieldNullspacePlan& plan) const {
    const GridContext gc = grid_context();
    const Real measure = gc.geom.dx() * gc.geom.dy();
    for (FieldNullspaceBasis& basis : plan.bases)
      basis.cell_measure = {measure};
  }

  /// A fresh scalar field co-distributed with the System mesh (block 0's box array / distribution),
  /// @p n_comp components, @p n_ghost ghost layers, zero-initialized. Forwards to
  /// System::alloc_scalar_field. The scratch fields (residual, search direction, solution) a
  /// matrix-free Krylov solve allocates -- a 1-component field is distinct from the n_cons block state,
  /// but shares its (ba, dm) so laplacian / gradient pair it with the state and aux by local fab index.
  MultiFab alloc_scalar_field(int n_comp = 1, int n_ghost = 1) const {
    return sys_->alloc_scalar_field(n_comp, n_ghost);
  }

  /// The System mesh geometry (index domain + physical bounds, dx/dy). BY VALUE: grid_context()
  /// returns a temporary, so a reference to its @c geom member would dangle. The metric the matrix-free
  /// Laplacian / gradient read.
  Geometry geom() const { return sys_->grid_context().geom; }
  /// Metric facts captured by generated kernels before entering device lambdas.  Cartesian and polar
  /// Programs share one emitted body; only these geometry-level values select the coordinate metric.
  bool is_polar_geometry() const { return sys_->program_is_polar(); }
  Real radial_origin() const {
    return sys_->program_is_polar() ? sys_->program_polar_geometry().r_min : Real(0);
  }
  Real radial_spacing() const {
    return sys_->program_is_polar() ? sys_->program_polar_geometry().dr() : geom().dx();
  }

  /// out = Lap(in): fill @p in's ghosts (transport BC, periodic by default) then apply the SHARED
  /// discrete 5-point Laplacian (pops::apply_laplacian, all optional coefficients null -> the bare
  /// bit-identical Laplacian). @p in is non-const because the ghost fill WRITES its halos (the valid
  /// cells are unchanged); this is the same matvec idiom the matrix-free Krylov test
  /// (tests/test_generic_krylov.cpp) wraps in its ApplyFn. The compiled program forms an operator
  /// A(in) = in - alpha*Lap(in) by combining this with ctx.lincomb.
  void laplacian(MultiFab& out, MultiFab& in) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(in, gc.geom.domain, gc.bc);
    if (sys_->program_is_polar()) {
      if (!polar_unit_rr_) {
        polar_unit_rr_ = std::make_shared<MultiFab>(in.box_array(), in.dmap(), 1, 1);
        polar_unit_tt_ = std::make_shared<MultiFab>(in.box_array(), in.dmap(), 1, 1);
        polar_unit_rr_->set_val(Real(1));
        polar_unit_tt_->set_val(Real(1));
      }
      apply_polar_tensor(in, sys_->program_polar_geometry(), out, polar_unit_rr_.get(),
                         polar_unit_tt_.get(), nullptr, nullptr);
    } else {
      apply_laplacian(in, gc.geom, out);  // all optional pointers null -> bare 5-point Laplacian
    }
  }

  void laplacian(MultiFab& out, MultiFab& in, const ExecutionLane& lane) const {
    if (sys_->program_is_polar())
      throw std::logic_error(
          "lane-isolated ProgramContext::laplacian requires a prepared polar operator session");
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(in, gc.geom.domain, gc.bc, lane);
    apply_laplacian(in, gc.geom, out);
  }

  void laplacian(MultiFab& out, MultiFab& in, const PreparedGridBoundarySession& boundary) const {
    if (sys_->program_is_polar())
      throw std::logic_error(
          "prepared ProgramContext::laplacian requires a polar operator provider");
    count_kernel();
    boundary.fill(in);
    apply_laplacian(in, boundary.context().geom, out);
  }

  void laplacian(MultiFab& out, MultiFab& in, const PreparedGridBoundarySession& boundary,
                 const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    if (sys_->program_is_polar())
      throw std::logic_error(
          "prepared ProgramContext::laplacian requires a polar operator provider");
    count_kernel();
    boundary.fill(in, point);
    apply_laplacian(in, boundary.context().geom, out);
  }

  /// Metric-aware tensor div(A grad(in)). The authored ApplyFn remains the sole mathematical
  /// operator on Cartesian and polar meshes; solver dispatch never swaps it for a second loop with
  /// different tolerances, preconditioning or residual semantics.
  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_grid_ghosts(in, gc);
    if (sys_->program_is_polar()) {
      apply_polar_tensor(in, sys_->program_polar_geometry(), out, &a_xx, &a_yy, &a_xy, &a_yx);
    } else {
      apply_laplacian(in, gc.geom, out, nullptr, &a_xx, nullptr, &a_yy, &a_xy, &a_yx);
    }
  }

  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx,
                        const ExecutionLane& lane) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_grid_ghosts(in, gc, lane);
    if (sys_->program_is_polar()) {
      apply_polar_tensor(in, sys_->program_polar_geometry(), out, &a_xx, &a_yy, &a_xy, &a_yx);
    } else {
      apply_laplacian(in, gc.geom, out, nullptr, &a_xx, nullptr, &a_yy, &a_xy, &a_yx);
    }
  }

  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx,
                        const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    boundary.fill(in);
    const GridContext& gc = boundary.context();
    if (sys_->program_is_polar()) {
      apply_polar_tensor(in, sys_->program_polar_geometry(), out, &a_xx, &a_yy, &a_xy, &a_yx);
    } else {
      apply_laplacian(in, gc.geom, out, nullptr, &a_xx, nullptr, &a_yy, &a_xy, &a_yx);
    }
  }

  void tensor_laplacian(MultiFab& out, MultiFab& in, const MultiFab& a_xx, const MultiFab& a_yy,
                        const MultiFab& a_xy, const MultiFab& a_yx,
                        const PreparedGridBoundarySession& boundary,
                        const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    count_kernel();
    boundary.fill(in, point);
    const GridContext& gc = boundary.context();
    if (sys_->program_is_polar()) {
      apply_polar_tensor(in, sys_->program_polar_geometry(), out, &a_xx, &a_yy, &a_xy, &a_yx);
    } else {
      apply_laplacian(in, gc.geom, out, nullptr, &a_xx, nullptr, &a_yy, &a_xy, &a_yx);
    }
  }

  /// out = grad(@p phi) by centered differences: out(.,0) = d phi/dx, out(.,1) = d phi/dy (@p out
  /// needs >= 2 components). Fills @p phi's ghosts then forwards to pops::field_postprocess with
  /// store_phi=false (the gradient lands in components 0/1) and the centered factors cx = 1/(2 dx),
  /// cy = 1/(2 dy) -- the same derivation the elliptic aux post-process uses (+grad sign).
  void gradient(MultiFab& out, MultiFab& phi) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(phi, gc.geom.domain, gc.bc);
    const Real cx = Real(1) / (Real(2) * gc.geom.dx());
    const Real cy = Real(1) / (Real(2) * gc.geom.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }

  void gradient(MultiFab& out, MultiFab& phi, const ExecutionLane& lane) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(phi, gc.geom.domain, gc.bc, lane);
    const Real cx = Real(1) / (Real(2) * gc.geom.dx());
    const Real cy = Real(1) / (Real(2) * gc.geom.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }

  void gradient(MultiFab& out, MultiFab& phi, const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    boundary.fill(phi);
    const GridContext& gc = boundary.context();
    const Real cx = Real(1) / (Real(2) * gc.geom.dx());
    const Real cy = Real(1) / (Real(2) * gc.geom.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }

  void gradient(MultiFab& out, MultiFab& phi, const PreparedGridBoundarySession& boundary,
                const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    count_kernel();
    boundary.fill(phi, point);
    const GridContext& gc = boundary.context();
    const Real cx = Real(1) / (Real(2) * gc.geom.dx());
    const Real cy = Real(1) / (Real(2) * gc.geom.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }

  /// out = div(@p fx, @p fy) by centered differences: out = d fx/dx + d fy/dy (component 0). The x-flux
  /// is read from component 0 of @p fx and the y-flux from component 1 of @p fy, the SAME layout
  /// @ref gradient writes (d/dx in component 0, d/dy in component 1) -- so chaining ctx.gradient(g, phi)
  /// then ctx.divergence(out, g, g) recovers the 5-point Laplacian. Fills the ghosts of @p fx and @p fy
  /// (transport BC, periodic by default) then forwards to pops::apply_divergence -- the exact inverse
  /// stencil of @ref gradient and the same centered FV divergence the coupled elliptic operator
  /// modules assemble. @p fx and @p fy are non-const because the ghost fill WRITES their halos (the
  /// valid cells are unchanged). A compiled Program forms a tensor flux operator
  /// A(phi) = phi - alpha*div(grad phi) by chaining ctx.gradient then ctx.divergence inside a
  /// matrix-free apply.
  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(fx, gc.geom.domain, gc.bc);
    if (&fy != &fx)
      fill_ghosts(fy, gc.geom.domain, gc.bc);  // skip the redundant halo fill when fy aliases fx
    apply_divergence(fx, fy, gc.geom, out, /*cx=*/0, /*cy=*/1);
  }

  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy, const ExecutionLane& lane) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(fx, gc.geom.domain, gc.bc, lane);
    if (&fy != &fx)
      fill_ghosts(fy, gc.geom.domain, gc.bc, lane);
    apply_divergence(fx, fy, gc.geom, out, /*cx=*/0, /*cy=*/1);
  }

  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy,
                  const PreparedGridBoundarySession& boundary) const {
    count_kernel();
    boundary.fill(fx);
    if (&fy != &fx)
      boundary.fill(fy);
    apply_divergence(fx, fy, boundary.context().geom, out, /*cx=*/0, /*cy=*/1);
  }

  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy,
                  const PreparedGridBoundarySession& boundary,
                  const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    count_kernel();
    boundary.fill(fx, point);
    if (&fy != &fx)
      boundary.fill(fy, point);
    apply_divergence(fx, fy, boundary.context().geom, out, /*cx=*/0, /*cy=*/1);
  }

  /// r <- -div(fx, fy) per conservative component (ADC-419 named fluxes): r(.,c) = -(d fx(.,c)/dx +
  /// d fy(.,c)/dy), centered FV, for every component c of @p r. @p fx and @p fy hold the n_cons x- and
  /// y-flux fields a compiled Program's named-flux kernel wrote (component c = the flux of conservative
  /// component c). REUSES pops::apply_divergence component-by-component (the SAME centered stencil as
  /// @ref divergence, the inverse of @ref gradient -- no new differencing): the ghosts are filled once
  /// per field, then each component's divergence lands in a 1-component scratch and is copied with a
  /// sign flip into @p r. @p fx / @p fy are non-const because the ghost fill writes their halos (the
  /// valid cells are unchanged). This semi-discrete -div F is LINEAR in the flux, so the -div of a SUM
  /// of named fluxes equals the sum of their -div (the named-flux parity guarantee).
  void neg_div_flux_into(MultiFab& r, MultiFab& fx, MultiFab& fy, MultiFab& divc) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(fx, gc.geom.domain, gc.bc);
    fill_ghosts(fy, gc.geom.domain, gc.bc);
    if (!scratch_layout_matches_(divc, r, 1, 0))
      throw std::invalid_argument(
          "Program named-flux divergence scratch must match the RHS distributed layout");
    for (int c = 0; c < r.ncomp(); ++c) {
      apply_divergence(fx, fy, gc.geom, divc, /*cx=*/c, /*cy=*/c);  // divc(.,0) = div(fx_c, fy_c)
      for (int li = 0; li < r.local_size(); ++li) {
        const ConstArray4 d = divc.fab(li).const_array();
        Array4 rv = r.fab(li).array();
        const int comp = c;
        for_each_cell(r.box(li), [=] POPS_HD(int i, int j) { rv(i, j, comp) = -d(i, j, 0); });
      }
    }
  }

  /// Historical/manual C++ convenience overload. Generated Programs pass a context-owned persistent
  /// divergence scratch through the overload above, so their per-step path performs no allocation.
  void neg_div_flux_into(MultiFab& r, MultiFab& fx, MultiFab& fy) const {
    MultiFab divc(r.box_array(), r.dmap(), 1, 0);
    neg_div_flux_into(r, fx, fy, divc);
  }

  void neg_div_flux_into(MultiFab& r, MultiFab& fx, MultiFab& fy, MultiFab& divc,
                         const ExecutionLane& lane) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(fx, gc.geom.domain, gc.bc, lane);
    fill_ghosts(fy, gc.geom.domain, gc.bc, lane);
    if (!scratch_layout_matches_(divc, r, 1, 0))
      throw std::invalid_argument(
          "Program named-flux divergence scratch must match the RHS distributed layout");
    for (int c = 0; c < r.ncomp(); ++c) {
      apply_divergence(fx, fy, gc.geom, divc, /*cx=*/c, /*cy=*/c);
      for (int li = 0; li < r.local_size(); ++li) {
        const ConstArray4 d = divc.fab(li).const_array();
        Array4 rv = r.fab(li).array();
        const int comp = c;
        for_each_cell(r.box(li), [=] POPS_HD(int i, int j) { rv(i, j, comp) = -d(i, j, 0); });
      }
    }
  }

  void neg_div_flux_into(MultiFab& r, MultiFab& fx, MultiFab& fy, const ExecutionLane& lane) const {
    MultiFab divc(r.box_array(), r.dmap(), 1, 0);
    neg_div_flux_into(r, fx, fy, divc, lane);
  }

  /// u <- u + a r over the valid cells (linear combine; forwards to pops::saxpy).
  void axpy(MultiFab& u, Real a, const MultiFab& r) const {
    count_kernel();
    if (const MultiFab* active_cells = active_domain_mask_())
      pops::saxpy_active(u, a, r, *active_cells);
    else
      pops::saxpy(u, a, r);
  }
  void axpy(MultiFab& u, Real a, const MultiFab& r, Real /*dt*/,
            std::initializer_list<ExactCoefficientTerm> /*exact*/) const {
    axpy(u, a, r);
  }

  /// z <- a x + b y over the valid cells (assignment, not accumulation; z may alias x or y).
  /// Forwards to pops::lincomb. The codegen uses it for the committed stage: the block state becomes
  /// z = c_base * z + 1 * acc, where acc holds the non-base terms (self-alias z==x is safe).
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) const {
    count_kernel();
    if (const MultiFab* active_cells = active_domain_mask_())
      pops::lincomb_active(z, a, x, b, y, *active_cells);
    else
      pops::lincomb(z, a, x, b, y);
  }
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y, Real /*dt*/,
               std::initializer_list<ExactCoefficientTerm> /*exact_a*/,
               std::initializer_list<ExactCoefficientTerm> /*exact_b*/) const {
    lincomb(z, a, x, b, y);
  }

  /// Register (idempotent) the history @p name with maximum lag @p lag, allocating the ring buffer
  /// WITHOUT reading it. The codegen emits this ONCE in the artifact-install prelude for each
  /// declared history, so a fresh restart sees the complete ring schema before the first step and
  /// the depth is locked before the first store (the cold-start fill then broadcasts the first
  /// stored value into every -- already allocated -- slot). @p ncomp is the slot component count:
  /// the default -1 resolves to block 0's ncomp (the multistep ring, byte-identical), and an explicit
  /// @p ncomp >= 1 sizes a narrower ring (ADC-427: a 1-component cross-step potential carry).
  /// Forwards to System::register_history. A read-only counterpart of @ref history.
  void register_history(const std::string& name, int lag, int ncomp = -1) const {
    sys_->register_history(name, lag, ncomp);
  }
  void register_history(const std::string& name, int lag, int ncomp, int owner,
                        const std::string& state_identity, const std::string& space_identity,
                        const std::string& clock_identity,
                        const std::string& interpolation_identity) const {
    sys_->register_history(name, lag, ncomp, owner < 0 ? -1 : sys_block(owner), state_identity,
                           space_identity, clock_identity, interpolation_identity);
  }

  /// The history slot @p lag macro-steps back (the SYSTEM-OWNED ring buffer, ADC-406a): lag 1 = the
  /// previous step's stored value (e.g. R_{n-1} for Adams-Bashforth), lag 0 = the current slot. The
  /// codegen emits ``ctx.history("<name>", <lag>)``; the read registers the ring on first use
  /// (idempotent) and forwards to System::read_history, which throws if the history was never stored
  /// (spec error 17). The register uses the DEFAULT ncomp (block 0's ncomp) so a bare read never
  /// changes an already-declared ring's width; a narrower ring (ADC-427) is declared by the prelude
  /// register_history(name, lag, ncomp) the codegen emits before any read. @p lag defaults to 1.
  MultiFab& history(const std::string& name, int lag = 1) const {
    sys_->register_history(name, lag);  // idempotent: allocate the ring on first use
    return sys_->read_history(name, lag);
  }
  /// Owner-qualified mirror used by sources that also contain an AMR entry point.  @p owner is a
  /// Program block index (never a component index); resolving it here preserves the same topology
  /// guard as the AMR context before delegating to the System-owned whole-field ring.
  MultiFab& history(const std::string& name, int lag, int owner) const {
    (void)sys_block(owner);
    return history(name, lag);
  }

  /// ZERO COLD-START history read (ADC-427): like @ref history, but a read BEFORE the first store
  /// returns the zero-filled slot instead of failing loud. A read-first carry (the cross-step
  /// potential: read the previous step's value at the TOP of the step, store the new one at the END)
  /// has no store before its very first read; its declared step-0 value IS zero (the slots are
  /// zero-initialized at registration), so the first read marks the ring initialized and reads it.
  /// The multistep store-first pattern keeps the fail-loud @ref history read unchanged. @p ncomp
  /// mirrors register_history (binds the slot width at the first register; -1 = block 0's ncomp).
  MultiFab& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
    sys_->register_history(name, lag, ncomp);  // idempotent; ncomp binds at the first register
    if (!sys_->history_initialized(name))
      sys_->set_history_initialized(name,
                                    true);  // the zero-filled slots ARE the declared cold start
    return sys_->read_history(name, lag);
  }
  MultiFab& history_zero_start(const std::string& name, int lag, int ncomp, int owner) const {
    (void)sys_block(owner);
    return history_zero_start(name, lag, ncomp);
  }

  /// Store @p value into the CURRENT slot of history @p name (ADC-406a). Registers the ring on first
  /// use (at least a current slot; the lag the program reads via @ref history sets the real depth) and
  /// forwards to System::store_history (which fills every slot on the first store -- the cold start).
  /// The codegen emits ``ctx.store_history("<name>", <value>)`` near the end of the step body. Uses the
  /// default ncomp on register (the width is fixed by the prelude register_history the codegen emits).
  void store_history(const std::string& name, const MultiFab& value) const {
    sys_->register_history(name, 1);  // idempotent: at least a current slot exists before the store
    sys_->store_history(name, value);
  }
  void store_history(const std::string& name, const MultiFab& value, int owner) const {
    (void)sys_block(owner);
    store_history(name, value);
  }

  /// Shift every history ring one macro-step (slot k <- slot k-1). Forwards to
  /// System::rotate_histories. The codegen emits ``ctx.rotate_histories()`` as the LAST statement of
  /// the step body (after the commit), so the next step reads lag k as the value k stores ago.
  void rotate_histories() const { sys_->rotate_histories(); }
  void rotate_histories(const std::string& clock_identity) const {
    sys_->rotate_histories(clock_identity);
  }

  /// @name Reductions (spec op 16)
  /// COLLECTIVE reductions over one explicitly owned Program block.  The block owner selects the
  /// runtime's prepared physical-cell measure: full valid cells on a Cartesian grid, the 0/1 active
  /// mask for staircase geometry, and active cells weighted by kappa for cut-cell geometry.  Integral
  /// reductions (sum/L1/L2/dot) use that relative volume; extrema ignore inactive storage and are not
  /// volume-scaled.  The generated Program always uses these owner-qualified overloads.
  /// @{
  Real sum_component(int owner, const MultiFab& u, int comp) const {
    return pops::reduce_sum(u, comp, relative_cell_measure_(owner));
  }
  Real max_component(int owner, const MultiFab& u, int comp) const {
    return pops::reduce_max(u, comp, relative_cell_measure_(owner));
  }
  Real min_component(int owner, const MultiFab& u, int comp) const {
    return pops::reduce_min(u, comp, relative_cell_measure_(owner));
  }
  Real abs_sum_component(int owner, const MultiFab& u, int comp) const {
    return pops::reduce_abs_sum(u, comp, relative_cell_measure_(owner));
  }
  Real norm2(int owner, const MultiFab& u) const {
    return std::sqrt(pops::dot(u, u, 0, relative_cell_measure_(owner)));
  }
  Real norm_inf(int owner, const MultiFab& u) const {
    return pops::reduce_norm_inf(u, 0, relative_cell_measure_(owner));
  }
  Real dot(int owner, const MultiFab& left, const MultiFab& right) const {
    return pops::dot(left, right, 0, relative_cell_measure_(owner));
  }

  Real sum(int owner, const MultiFab& u) const { return sum_component(owner, u, 0); }
  Real max(int owner, const MultiFab& u) const { return max_component(owner, u, 0); }
  Real min(int owner, const MultiFab& u) const { return min_component(owner, u, 0); }
  Real abs_sum(int owner, const MultiFab& u) const { return abs_sum_component(owner, u, 0); }

  /// Legacy hand-written Cartesian stages may omit the owner.  Under an embedded boundary these
  /// overloads refuse before launching a kernel: silently assuming block 0 would bypass the exact
  /// Program-to-System block map and make a multi-block Program's measure ambiguous.
  Real sum_component(const MultiFab& u, int comp) const {
    require_unqualified_reduction_safe_();
    return pops::reduce_sum(u, comp);
  }
  Real max_component(const MultiFab& u, int comp) const {
    require_unqualified_reduction_safe_();
    return pops::reduce_max(u, comp);
  }
  Real min_component(const MultiFab& u, int comp) const {
    require_unqualified_reduction_safe_();
    return pops::reduce_min(u, comp);
  }
  Real abs_sum_component(const MultiFab& u, int comp) const {
    require_unqualified_reduction_safe_();
    return pops::reduce_abs_sum(u, comp);
  }
  Real sum(const MultiFab& u) const { return sum_component(u, 0); }
  Real max(const MultiFab& u) const { return max_component(u, 0); }
  Real min(const MultiFab& u) const { return min_component(u, 0); }
  Real abs_sum(const MultiFab& u) const { return abs_sum_component(u, 0); }
  /// @}

  /// Fill the ghost cells (halos) of @p x in place: the transport BC (periodic by default), the SAME
  /// exchange laplacian / gradient / divergence run internally before differencing (spec op 22). The
  /// valid cells are untouched; only the halos change. Forwards to the shared pops::fill_ghosts.
  void fill_boundary(MultiFab& x) const {
    const GridContext gc = sys_->grid_context();
    fill_ghosts(x, gc.geom.domain, gc.bc);
  }

  void fill_boundary(MultiFab& x, const ExecutionLane& lane) const {
    const GridContext gc = sys_->grid_context();
    fill_ghosts(x, gc.geom.domain, gc.bc, lane);
  }

  /// Apply block @p b's post-step positivity projection to @p u in place: U <- project(U, aux) over the
  /// valid cells, the SAME Zhang-Shu / floor projection the native per-step path runs (ADC-177, spec
  /// op 21). REUSES the block's own projection closure (set at add_block time); a block WITHOUT a
  /// projection is rejected. Forwards to System::block_project -- it reimplements no positivity.
  void apply_projection(int b, MultiFab& u) const { sys_->block_project(sys_block(b), u); }

  /// Apply every registered coupled-source operator to a complete simultaneous candidate-state
  /// group.  Candidates remain private Program values until the caller projects and commit_many()
  /// publishes them, so a failed coupling cannot expose a partially committed multi-block state.
  void apply_coupling_operators(Real dt,
                                std::initializer_list<CouplingStateOverride> candidates) const {
    if (!std::isfinite(static_cast<double>(dt)) || dt < Real(0))
      throw std::invalid_argument("Program coupling application requires a finite non-negative dt");
    if (coupling_workspace_.in_use)
      throw std::logic_error("Program coupling workspace is already in use");
    prepare_coupling_workspace_(candidates);
    struct WorkspaceUse {
      bool& flag;
      explicit WorkspaceUse(bool& value) : flag(value) { flag = true; }
      ~WorkspaceUse() { flag = false; }
    } use(coupling_workspace_.in_use);
    const std::size_t applied =
        sys_->apply_coupling_operators(dt, coupling_workspace_.system_states);
    count_kernel(static_cast<std::int64_t>(applied));
  }

  /// @name Scheduler value cache (Spec 3 section 17-18, ADC-458)
  /// A held field-solve node recomputes only when DUE (every N macro-steps) and reuses the cached
  /// System aux (phi / grad / E) in between. The cache is owned by the System (one CacheManager per
  /// installed Program, keyed by the Program node id) so the checkpoint can reach it (Spec 3 section
  /// 30); every ProgramContext copy forwards to that single manager via sys_->program_cache(). The
  /// codegen wraps a held solve_fields in
  /// ``if (schedule_decision(id, schedule_is_due(...), true)) {
  ///  solve_fields_from_state(...); cache_store_aux(id); } else { cache_restore_aux(id); }``.
  /// The runtime cadence/checkpoint is exercised in a compiled
  /// .so step loop (validated on ROMEO; not buildable on a host-only Mac).
  /// @{
  /// True if node @p node_id is due to recompute at the current macro step: cold start (never stored),
  /// then every @p every_n macro steps. Wraps CacheManager::is_due with System::macro_step().
  ///
  /// PROFILER scheduler counters (ADC-459, Spec 3 section 29): a DUE step recomputes the node (a cache
  /// "miss" + a "due" scheduled node); a NOT-due step reuses the held value (a cache "hit" + a
  /// "skipped" scheduled node). Counted here at the one decision point every scheduled node routes
  /// through, gated on the profiler (zero cost when off). These move only under the compiled .so step
  /// loop that exercises a held schedule (validated on Kokkos/ROMEO, not buildable host-only).
  bool cache_should_update(int node_id, int every_n) const {
    const bool due = sys_->program_cache().is_due(node_id, sys_->macro_step(), every_n);
    if (due) {
      sys_->profiler().count("cache_misses");
      sys_->profiler().count("nodes_due");
    } else {
      sys_->profiler().count("cache_hits");
      sys_->profiler().count("nodes_skipped");
    }
    return due;
  }
  /// Store a copy of the System aux (the field solve's output) as node @p node_id's cached value,
  /// stamped at the current macro step (resets its accumulated dt).
  void cache_store_aux(int node_id) const {
    sys_->program_cache().store(node_id, *sys_->grid_context().aux, sys_->macro_step());
  }
  /// Restore node @p node_id's cached aux into the System aux (a held step: no elliptic solve).
  void cache_restore_aux(int node_id) const {
    sys_->program_cache().restore_into(node_id, *sys_->grid_context().aux);
  }

  /// Store a copy of a NAMED scratch MultiFab (a held rhs / source / linear_combine output) as node
  /// @p node_id's cached value, stamped at the current macro step. The aux variants cache the System
  /// aux; this caches an arbitrary step-body scratch so ANY schedulable node can hold, not only a
  /// field solve.
  void cache_store_scratch(int node_id, const MultiFab& scratch) const {
    sys_->program_cache().store(node_id, scratch, sys_->macro_step());
  }
  /// Restore node @p node_id's cached scratch into @p scratch (a held step: no recompute).
  void cache_restore_scratch(int node_id, MultiFab& scratch) const {
    sys_->program_cache().restore_into(node_id, scratch);
  }
  /// Add a skipped step's @p dt to node @p node_id's accumulator (accumulate_dt policy): on a NOT-due
  /// step the held node does not recompute but records the dt so the next due step sees the full
  /// skipped interval. Variable step_cfl safe (the actual skipped dt, not N * dt_current).
  void cache_accumulate_dt(int node_id, Real dt) const {
    sys_->program_cache().accumulate_dt(node_id, dt);
  }
  /// The effective dt a due accumulate_dt step applies: @p dt_now plus the summed skipped dt since the
  /// last recompute (resets the accumulator). The codegen feeds this as the step's dt into the held
  /// node's recompute so it advances over the whole skipped interval at once.
  Real cache_effective_dt(int node_id, Real dt_now) const {
    return sys_->program_cache().effective_dt(node_id, dt_now);
  }
  /// @}

 private:
  struct CouplingWorkspace {
    std::vector<int> program_to_system;
    std::vector<MultiFab*> system_states;
    bool in_use = false;
  };

  struct FieldSolveWorkspace {
    std::vector<int> program_to_system;
    std::vector<const MultiFab*> program_stages;
    std::vector<const MultiFab*> system_stages;
    std::vector<int> expected_program_blocks;
    std::string generated_field_identity;
    bool expected_program_blocks_initialized = false;
    bool in_use = false;
  };

  struct FieldPublicationTransaction {
    System* system = nullptr;
    bool active = false;

    void validate_accept() {
      if (!active || system == nullptr)
        throw std::logic_error("Program field publication has no staged candidate");
      system->validate_field_publication_candidate();
    }

    void accept() noexcept {
      if (!active || system == nullptr)
        std::terminate();
      system->accept_field_publication_candidate();
      system = nullptr;
      active = false;
    }

    void rollback() noexcept {
      if (!active)
        return;
      try {
        if (system == nullptr)
          std::terminate();
        system->rollback_field_publication_transaction();
      } catch (...) {
        std::terminate();
      }
      release();
    }

    void release() noexcept {
      system = nullptr;
      active = false;
    }
  };

  struct FieldSolveWorkspaceRegistry {
    FieldSolveWorkspace manual_default;
    std::map<std::string, FieldSolveWorkspace, std::less<>> manual_named;
    std::map<std::int64_t, FieldSolveWorkspace> generated;
    FieldPublicationTransaction publication;
  };

  void prepare_coupling_workspace_(std::initializer_list<CouplingStateOverride> candidates) const {
    const std::vector<int>& block_map = sys_->program_block_map();
    const std::size_t system_blocks = static_cast<std::size_t>(sys_->n_blocks());
    if (block_map.empty())
      throw block_map_error_(
          "ProgramContext::apply_coupling_operators has no explicit program-to-system block map");
    if (block_map.size() != system_blocks || candidates.size() != block_map.size())
      throw std::invalid_argument(
          "Program coupling requires a complete candidate pack for every System block");

    const bool structure_changed = coupling_workspace_.program_to_system != block_map ||
                                   coupling_workspace_.system_states.size() != system_blocks;
    if (structure_changed) {
      coupling_workspace_.system_states.assign(system_blocks, nullptr);
      for (std::size_t program_block = 0; program_block < block_map.size(); ++program_block) {
        const int system_block = sys_block(static_cast<int>(program_block));
        MultiFab*& mapped =
            coupling_workspace_.system_states[static_cast<std::size_t>(system_block)];
        if (mapped != nullptr)
          throw std::invalid_argument(
              "Program coupling block map does not cover each System block exactly once");
        // Non-null sentinel: authenticate uniqueness without a hot-path temporary allocation.
        mapped = &sys_->block_state(system_block);
      }
      coupling_workspace_.program_to_system.assign(block_map.begin(), block_map.end());
    }

    std::fill(coupling_workspace_.system_states.begin(), coupling_workspace_.system_states.end(),
              nullptr);
    std::size_t ordinal = 0;
    for (const CouplingStateOverride& candidate : candidates) {
      if (candidate.program_block != static_cast<int>(ordinal) || candidate.state == nullptr)
        throw std::invalid_argument(
            "Program coupling candidates must be non-null and ordered by Program block");
      require_program_stage_layout_(candidate.program_block, *candidate.state);
      const int system_block =
          coupling_workspace_.program_to_system[static_cast<std::size_t>(candidate.program_block)];
      if (candidate.state == &sys_->block_state(system_block))
        throw std::invalid_argument(
            "Program coupling candidates cannot alias accepted live states");
      coupling_workspace_.system_states[static_cast<std::size_t>(system_block)] = candidate.state;
      ++ordinal;
    }
  }

  void capture_field_publication_(FieldPublicationTransaction& transaction) const {
    // SystemFieldSolver's uniform reductions use MPI_COMM_WORLD, so this transaction must
    // authenticate and release on that exact communicator rather than inventing a private lane.
    if (all_reduce_max(transaction.active ? 1L : 0L) != 0)
      throw std::logic_error(
          "ProgramContext field solves are sequential until their SolveOutcome is consumed");
    if (all_reduce_max(sys_->field_publication_transaction_active_() ? 1L : 0L) != 0)
      throw std::logic_error(
          "System field solves are sequential until their prior SolveOutcome is consumed");

    long capture_failure_local = 0;
    try {
      sys_->begin_field_publication_transaction();
    } catch (...) {
      capture_failure_local = 1;
    }
    if (all_reduce_max(capture_failure_local) != 0) {
      try {
        sys_->rollback_field_publication_transaction();
      } catch (...) {
        std::terminate();
      }
      throw std::runtime_error(
          "ProgramContext field publication snapshot failed on at least one MPI rank");
    }
    transaction.system = sys_;
    transaction.active = true;
  }

  template <class Solve>
  SolveOutcome run_field_solve_transaction_(Solve&& solve) const {
    if (!field_solve_workspace_registry_)
      throw std::logic_error("Program field-solve workspace registry is unavailable");
    const std::shared_ptr<FieldSolveWorkspaceRegistry> registry = field_solve_workspace_registry_;
    FieldPublicationTransaction& transaction = registry->publication;
    capture_field_publication_(transaction);

    SolveReport report;
    std::exception_ptr solve_error;
    long solve_failure_local = 0;
    try {
      report = std::forward<Solve>(solve)();
    } catch (...) {
      solve_error = std::current_exception();
      solve_failure_local = 1;
    }
    if (all_reduce_max(solve_failure_local) != 0) {
      transaction.rollback();
      if (n_ranks() == 1 && solve_error != nullptr)
        std::rethrow_exception(solve_error);
      throw std::runtime_error("ProgramContext field solver failed on at least one MPI rank");
    }
    const bool malformed = !solve_report_is_publishable(report, std::numeric_limits<int>::max());
    if (all_reduce_max(malformed ? 1L : 0L) != 0) {
      transaction.rollback();
      throw std::runtime_error("ProgramContext field solver published a malformed SolveReport");
    }
    ExactSolveReportConsensusScratch consensus;
    if (!consensus.agrees(report)) {
      transaction.rollback();
      throw std::runtime_error("ProgramContext field solver report differs between MPI ranks");
    }
    if (!report.solved_value_available()) {
      // The uniform backends already restore their potential warm start on a failed report. This
      // outer graph transaction additionally restores the complete shared aux channel before the
      // failure can be inspected or acted upon.
      transaction.rollback();
      return SolveOutcome::collective_world(std::move(report));
    }

    long staging_failure_local = 0;
    try {
      sys_->stage_field_publication_candidate();
    } catch (...) {
      staging_failure_local = 1;
    }
    if (all_reduce_max(staging_failure_local) != 0) {
      transaction.rollback();
      throw std::runtime_error(
          "ProgramContext field candidate staging failed on at least one MPI rank");
    }

    // Aux and every backend potential now contain the previous accepted state. The shared registry
    // keeps the callback context alive even if a direct C++ caller moves the outcome beyond this
    // ProgramContext; only collective Accept restores the staged candidate.
    return SolveOutcome::collective_world(
        std::move(report),
        SolveOutcome::PublicationHooks{
            &transaction,
            [](void* context) noexcept {
              static_cast<FieldPublicationTransaction*>(context)->accept();
            },
            nullptr,
            [](void* context) noexcept {
              static_cast<FieldPublicationTransaction*>(context)->rollback();
            },
            std::static_pointer_cast<void>(registry),
            [](void* context) {
              static_cast<FieldPublicationTransaction*>(context)->validate_accept();
            }});
  }

  void prepare_field_solve_structure_(FieldSolveWorkspace& workspace) const {
    const std::vector<int>& block_map = sys_->program_block_map();
    if (block_map.empty())
      throw block_map_error_(
          "ProgramContext::solve_fields_from_blocks: no explicit program-to-system block map is "
          "installed; positional block identity is not supported");
    const std::size_t system_blocks = static_cast<std::size_t>(sys_->n_blocks());
    const bool unchanged = workspace.program_to_system == block_map &&
                           workspace.program_stages.size() == block_map.size() &&
                           workspace.system_stages.size() == system_blocks;
    if (unchanged)
      return;

    // Authenticate the complete map before releasing a previously valid workspace.  A malformed map
    // cannot leave a partially reconfigured context behind.
    for (std::size_t p = 0; p < block_map.size(); ++p) {
      const int mapped = sys_block(static_cast<int>(p));
      for (std::size_t previous = 0; previous < p; ++previous) {
        if (block_map[previous] == mapped)
          throw block_map_error_("ProgramContext::solve_fields_from_blocks: Program blocks " +
                                 std::to_string(previous) + " and " + std::to_string(p) +
                                 " both map to System block " + std::to_string(mapped));
      }
    }
    workspace.program_to_system.assign(block_map.begin(), block_map.end());
    workspace.program_stages.assign(block_map.size(), nullptr);
    workspace.system_stages.assign(system_blocks, nullptr);
    workspace.expected_program_blocks.clear();
    workspace.expected_program_blocks_initialized = false;
  }

  void require_program_stage_layout_(int program_block, const MultiFab& stage) const {
    const MultiFab& live = sys_->block_state(sys_block(program_block));
    if (!scratch_layout_matches_(stage, live, live.ncomp(), live.n_grow()))
      throw std::invalid_argument(
          "Program simultaneous field solve requires each stage state to match its block's exact "
          "distributed layout");
    const int qualified_system_block = sys_block(program_block);
    for (int other = 0; other < sys_->n_blocks(); ++other) {
      if (other == qualified_system_block)
        continue;
      if (&stage == &sys_->block_state(other))
        throw std::invalid_argument(
            "Program stage override cannot borrow another block's live state");
    }
  }

  void fill_manual_field_stages_(FieldSolveWorkspace& workspace,
                                 const std::vector<const MultiFab*>& stages,
                                 bool require_exact_size) const {
    prepare_field_solve_structure_(workspace);
    const std::size_t required = workspace.program_to_system.size();
    if ((require_exact_size && stages.size() != required) ||
        (!require_exact_size && stages.size() < required))
      throw std::runtime_error(
          "ProgramContext::solve_fields_from_blocks: stage vector size mismatch");
    std::fill(workspace.program_stages.begin(), workspace.program_stages.end(), nullptr);
    for (std::size_t p = 0; p < required; ++p) {
      const MultiFab* stage = stages[p];
      if (stage != nullptr)
        require_program_stage_layout_(static_cast<int>(p), *stage);
      workspace.program_stages[p] = stage;
    }
  }

  FieldSolveWorkspace& manual_default_field_solve_workspace_() const {
    if (!field_solve_workspace_registry_)
      throw std::logic_error("Program field-solve workspace registry is unavailable");
    FieldSolveWorkspace& workspace = field_solve_workspace_registry_->manual_default;
    prepare_field_solve_structure_(workspace);
    return workspace;
  }

  FieldSolveWorkspace& manual_named_field_solve_workspace_(const std::string& field) const {
    if (field.empty())
      throw std::invalid_argument(
          "Program named simultaneous field solve requires a field identity");
    if (!field_solve_workspace_registry_)
      throw std::logic_error("Program field-solve workspace registry is unavailable");
    auto& workspaces = field_solve_workspace_registry_->manual_named;
    auto found = workspaces.find(field);
    if (found == workspaces.end())
      found = workspaces.try_emplace(field).first;
    prepare_field_solve_structure_(found->second);
    return found->second;
  }

  FieldSolveWorkspace& generated_field_solve_workspace_(
      std::int64_t value_id, std::string_view field,
      std::initializer_list<FieldStageOverride> overrides) const {
    if (value_id < 0)
      throw std::invalid_argument(
          "generated simultaneous field solve requires a non-negative IR identity");
    if (field.empty())
      throw std::invalid_argument("generated simultaneous field solve requires a field identity");
    if (overrides.size() == 0)
      throw std::invalid_argument(
          "generated simultaneous field solve requires at least one stage override");
    if (!field_solve_workspace_registry_)
      throw std::logic_error("Program field-solve workspace registry is unavailable");

    auto [entry, inserted] = field_solve_workspace_registry_->generated.try_emplace(value_id);
    FieldSolveWorkspace& workspace = entry->second;
    if (inserted)
      workspace.generated_field_identity.assign(field.data(), field.size());
    else if (std::string_view(workspace.generated_field_identity) != field)
      throw std::logic_error(
          "generated simultaneous field solve IR identity was reused for a different field");
    prepare_field_solve_structure_(workspace);

    const bool learn_blocks = !workspace.expected_program_blocks_initialized;
    if (learn_blocks) {
      workspace.expected_program_blocks.clear();
      workspace.expected_program_blocks.reserve(overrides.size());
    } else if (workspace.expected_program_blocks.size() != overrides.size()) {
      throw std::logic_error(
          "generated simultaneous field solve IR identity changed its block pack");
    }
    std::fill(workspace.program_stages.begin(), workspace.program_stages.end(), nullptr);
    std::size_t ordinal = 0;
    for (const FieldStageOverride& override_value : overrides) {
      if (override_value.program_block < 0 ||
          static_cast<std::size_t>(override_value.program_block) >= workspace.program_stages.size())
        throw std::out_of_range("generated simultaneous field solve Program block is out of range");
      if (override_value.state == nullptr)
        throw std::invalid_argument(
            "generated simultaneous field solve stage override cannot be null");
      if (workspace.program_stages[static_cast<std::size_t>(override_value.program_block)] !=
          nullptr)
        throw std::invalid_argument(
            "generated simultaneous field solve contains a duplicate Program block");
      if (learn_blocks)
        workspace.expected_program_blocks.push_back(override_value.program_block);
      else if (workspace.expected_program_blocks[ordinal] != override_value.program_block)
        throw std::logic_error(
            "generated simultaneous field solve IR identity changed its ordered block pack");
      require_program_stage_layout_(override_value.program_block, *override_value.state);
      workspace.program_stages[static_cast<std::size_t>(override_value.program_block)] =
          override_value.state;
      ++ordinal;
    }
    workspace.expected_program_blocks_initialized = true;
    return workspace;
  }

  SolveReport solve_default_field_workspace_(FieldSolveWorkspace& workspace) const {
    std::fill(workspace.system_stages.begin(), workspace.system_stages.end(), nullptr);
    for (std::size_t p = 0; p < workspace.program_to_system.size(); ++p) {
      const int mapped = workspace.program_to_system[p];
      workspace.system_stages[static_cast<std::size_t>(mapped)] = workspace.program_stages[p];
    }
    return sys_->solve_fields_from_blocks_in_place_(workspace.system_stages);
  }

  SolveReport solve_named_field_workspace_(const std::string& field,
                                           FieldSolveWorkspace& workspace) const {
    if (workspace.in_use)
      throw std::logic_error("Program simultaneous field-solve workspace is already in use");
    struct WorkspaceUse {
      bool& flag;
      explicit WorkspaceUse(bool& value) : flag(value) { flag = true; }
      ~WorkspaceUse() { flag = false; }
    } use(workspace.in_use);
    std::fill(workspace.system_stages.begin(), workspace.system_stages.end(), nullptr);
    bool has_override = false;
    for (std::size_t p = 0; p < workspace.program_stages.size(); ++p) {
      if (workspace.program_stages[p] == nullptr)
        continue;
      workspace.system_stages[static_cast<std::size_t>(workspace.program_to_system[p])] =
          workspace.program_stages[p];
      has_override = true;
    }
    if (!has_override)
      throw std::runtime_error(
          "ProgramContext::solve_fields_from_blocks(field): no stage override was supplied");
    return sys_->solve_fields_from_blocks_in_place_(field, workspace.system_stages);
  }

  struct ScratchKey {
    ScratchKind kind = ScratchKind::Rhs;
    std::int64_t value_id = -1;
    int subslot = -1;

    friend bool operator<(const ScratchKey& lhs, const ScratchKey& rhs) noexcept {
      if (lhs.kind != rhs.kind)
        return lhs.kind < rhs.kind;
      if (lhs.value_id != rhs.value_id)
        return lhs.value_id < rhs.value_id;
      return lhs.subslot < rhs.subslot;
    }
  };

  struct ScratchRegistry {
    std::map<ScratchKey, MultiFab> fields;
  };

  static bool scratch_layout_matches_(const MultiFab& field, const MultiFab& prototype, int n_comp,
                                      int n_ghost) {
    return field.box_array().boxes() == prototype.box_array().boxes() &&
           field.dmap().ranks() == prototype.dmap().ranks() && field.ncomp() == n_comp &&
           field.n_grow() == n_ghost;
  }

  MultiFab& program_scratch_for_(ScratchKind kind, std::int64_t value_id, int subslot,
                                 const MultiFab& prototype, int n_comp, int n_ghost) const {
    if (value_id < 0 || subslot < 0)
      throw std::invalid_argument(
          "Program persistent scratch requires non-negative IR value and sub-slot identities");
    if (!scratch_registry_)
      throw std::logic_error("Program persistent scratch registry is unavailable");
    const ScratchKey key{kind, value_id, subslot};
    auto [entry, inserted] = scratch_registry_->fields.try_emplace(key);
    MultiFab& field = entry->second;
    if (inserted || !scratch_layout_matches_(field, prototype, n_comp, n_ghost)) {
      field = MultiFab(prototype.box_array(), prototype.dmap(), n_comp, n_ghost);
      count_scratch(field);
    }
    field.set_val(Real(0));
    return field;
  }

  RelativeCellMeasure relative_cell_measure_(int owner) const {
    const GridContext context = sys_->grid_context(sys_block(owner));
    if (context.embedded_boundary_set == nullptr || !*context.embedded_boundary_set ||
        context.geometry_mode == nullptr || *context.geometry_mode == GeometryMode::None)
      return {};
    if (context.domain_mask == nullptr)
      throw std::runtime_error(
          "ProgramContext physical reduction has no prepared active-cell mask");
    if (*context.geometry_mode == GeometryMode::CutCell) {
      if (context.eb_inverse_volume_fraction == nullptr)
        throw std::runtime_error(
            "ProgramContext cut-cell reduction has no prepared inverse volume fraction");
      return {context.domain_mask, context.eb_inverse_volume_fraction};
    }
    return {context.domain_mask, nullptr};
  }

  void require_unqualified_reduction_safe_() const {
    const GridContext context = sys_->grid_context();
    if (context.embedded_boundary_set != nullptr && *context.embedded_boundary_set &&
        context.geometry_mode != nullptr && *context.geometry_mode != GeometryMode::None)
      throw std::runtime_error(
          "ProgramContext embedded-boundary reduction requires an explicit Program block owner");
  }

  const MultiFab* active_domain_mask_() const {
    const GridContext context = sys_->grid_context();
    if (context.embedded_boundary_set != nullptr && *context.embedded_boundary_set &&
        context.geometry_mode != nullptr && *context.geometry_mode != GeometryMode::None) {
      if (context.domain_mask == nullptr)
        throw std::runtime_error(
            "ProgramContext embedded-boundary algebra has no prepared active-cell mask");
      return context.domain_mask;
    }
    return nullptr;
  }

  runtime::multiblock::BoundaryEvaluationPoint boundary_point_(int stage) const {
    require_rate_identity_(stage);
    if (primary_clock_.empty() || !std::isfinite(current_dt_) || current_dt_ <= 0.0)
      throw std::runtime_error("Program boundary evaluation has no prepared clock/dt");
    const amr::Rational evaluation_stage = logical_phase_begin_ + stage_time_ * logical_phase_span_;
    return {primary_clock_,
            static_cast<std::int64_t>(macro_step()),
            0,
            0,
            stage,
            evaluation_stage,
            current_dt_,
            static_cast<double>(physical_time()) + logical_physical_time_offset_ +
                stage_time_.value() * current_dt_};
  }

  static SolveReport consume_field_outcome_(SolveOutcome outcome) {
    return outcome.consume(outcome.report().solved_value_available()
                               ? SolveConsumption::kAccept
                               : (outcome.report().action == SolveAction::kRejectAttempt
                                      ? SolveConsumption::kRejectAttempt
                                      : SolveConsumption::kFailRun));
  }
  friend class ProgramExecutionServices<ProgramContext>;
  LogicalEvaluationState program_execution_begin_logical_evaluation_(int iteration,
                                                                     int count) const {
    LogicalEvaluationState state;
    state.parent_dt = current_dt_;
    state.stage = stage_time_;
    state.phase_begin = logical_phase_begin_;
    state.phase_span = logical_phase_span_;
    state.physical_time_offset = logical_physical_time_offset_;
    if (!std::isfinite(state.parent_dt) || state.parent_dt <= 0.0)
      throw std::logic_error("Program logical evaluation requires a prepared parent dt");
    state.child_dt = state.parent_dt / static_cast<double>(count);
    const double child_offset =
        state.physical_time_offset + static_cast<double>(iteration) * state.child_dt;
    if (!std::isfinite(state.child_dt) || state.child_dt <= 0.0 || !std::isfinite(child_offset))
      throw std::overflow_error("Program logical evaluation child window is not finite");
    const amr::Rational child_fraction(iteration, count);
    const amr::Rational child_span = state.phase_span * amr::Rational(1, count);
    const amr::Rational child_begin = state.phase_begin + state.phase_span * child_fraction;

    invalidate_active_operator_snapshot_();
    current_dt_ = state.child_dt;
    stage_time_ = amr::Rational(0, 1);
    logical_phase_begin_ = child_begin;
    logical_phase_span_ = child_span;
    logical_physical_time_offset_ = child_offset;
    return state;
  }
  void program_execution_restore_logical_evaluation_(
      const LogicalEvaluationState& state) const noexcept {
    current_dt_ = state.parent_dt;
    stage_time_ = state.stage;
    logical_phase_begin_ = state.phase_begin;
    logical_phase_span_ = state.phase_span;
    logical_physical_time_offset_ = state.physical_time_offset;
    invalidate_active_operator_snapshot_();
  }
  MultiFab& program_execution_scratch_(ScratchKind kind, std::int64_t value_id, int subslot,
                                       const MultiFab& prototype, int n_comp, int n_ghost) const {
    return program_scratch_for_(kind, value_id, subslot, prototype, n_comp, n_ghost);
  }
  const std::vector<int>& program_execution_block_map_() const { return sys_->program_block_map(); }
  int program_execution_block_count_() const { return sys_->n_blocks(); }
  Real program_execution_physical_time_() const { return static_cast<Real>(sys_->time()); }
  void program_execution_record_scalar_(const std::string& name, Real value) const {
    sys_->record_program_diagnostic(name, value);
  }
  RuntimeParams program_execution_params_(int block) const { return sys_->program_params(block); }
  void program_execution_set_field_timepoint_(const std::string& field,
                                              const FieldLogicalTimePoint& point) const {
    sys_->set_field_logical_timepoint(field, point);
  }
  void program_execution_set_field_parameters_(const std::string& field,
                                               const std::vector<double>& parameters) const {
    sys_->set_field_boundary_parameters(field, parameters);
  }
  void program_execution_set_field_kernel_(const std::string& field,
                                           const CompiledFieldBoundaryKernel& kernel) const {
    sys_->set_field_boundary_kernel(field, kernel);
  }
  Profiler& program_execution_profiler_() const { return sys_->profiler(); }
  int program_execution_macro_step_() const { return sys_->macro_step(); }
  int program_execution_active_level_() const { return -1; }

  mutable double current_dt_ = 0.0;
  mutable amr::Rational logical_phase_begin_{0, 1};
  mutable amr::Rational logical_phase_span_{1, 1};
  mutable double logical_physical_time_offset_ = 0.0;
  mutable std::uint64_t operator_snapshot_revision_ = 0;
  mutable std::uint64_t active_operator_snapshot_revision_ = 0;  // zero is never minted
  mutable std::shared_ptr<MultiFab> polar_unit_rr_;
  mutable std::shared_ptr<MultiFab> polar_unit_tt_;
  mutable std::shared_ptr<FieldSolveWorkspaceRegistry> field_solve_workspace_registry_ =
      std::make_shared<FieldSolveWorkspaceRegistry>();
  mutable CouplingWorkspace coupling_workspace_;
  mutable std::shared_ptr<ScratchRegistry> scratch_registry_ = std::make_shared<ScratchRegistry>();
  System* sys_;
};

}  // namespace program
}  // namespace runtime
}  // namespace pops
