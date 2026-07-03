#pragma once

#include <pops/amr/regridding/regrid.hpp>   // tag_cells, grow_tags (per-block tags + phi for the union regrid)
#include <pops/amr/tagging/tag_box.hpp>  // TagBox, tag_union (cell-by-cell OR of the tags of all blocks)
#include <pops/core/state/state.hpp>   // kAuxBaseComps
#include <pops/core/state/variables.hpp>  // VariableSet, VariableRole, role_from_name (role -> component of coupled sources)
#include <pops/coupling/amr/amr_coupler_mp.hpp>  // detail::coupler_inject_aux_mb (aux injection coarse->fine)
#include <pops/coupling/amr/amr_regrid_coupler.hpp>  // regrid_compute_fine_layout + regrid_field_on_layout (split bricks)
#include <pops/coupling/system/amr_system_coupler.hpp>  // detail::same_layout_or_throw (shared-layout guard)
#include <pops/coupling/base/aux_fill.hpp>            // detail::derive_aux_bc (BC of the aux channel)
#include <pops/coupling/source/coupled_source_program.hpp>  // CoupledSourceKernel + CsProgram (flat ABI, P5 bytecode)
#include <pops/coupling/source/coupling_operator.hpp>  // CouplingOperator / CouplingOperatorView (typed contract, ADC-595)
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess, FieldPostProcess
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/time/amr/reflux/amr_reflux_mf.hpp>  // AmrLevelMP, mf_average_down_mb
#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // NewtonReport (OPT-IN IMEX diagnostics, aggregated per block)
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/patch_box.hpp>  // PatchBox: index-space signature of a fine patch (patch_boxes())
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/layout/copy_schedule.hpp>  // copy_schedule_{hit,miss}_count (ADC-607 counters)
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/parallel/comm.hpp>  // n_ranks() / comm_active(): MPI message+reduction counts (Spec 5 criterion 43)
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/runtime/program/profiler.hpp>  // Profiler / ProfileScope: AMR phase timings (Spec 5 criterion 43, ADC-479)
#include <pops/runtime/system/field_problem_registry.hpp>  // FieldProblemRegistry (ADC-596 descriptor)

#include <algorithm>  // std::max (substeps/stride-aware CFL step)
#include <chrono>  // AmrPhaseScope wall-clock timing (Spec 5 criterion 43)
#include <cmath>      // std::isfinite (reject a degenerate dt)
#include <cstddef>
#include <functional>
#include <limits>  // std::numeric_limits (initial dt = +inf, min over the blocks)
#include <map>  // named_aux_: model-named aux fields (comp -> coarse field), re-applied each solve
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @file
/// @brief AMR multi-block engine at RUNTIME (type-erased registry keyed by name).
///
/// Runtime counterpart of System::Impl (python/system.cpp): where System type-erases the species
/// (struct Species) on a SINGLE-LEVEL grid, AmrRuntime type-erases N blocks on a SHARED AMR
/// hierarchy. It FAITHFULLY reproduces the AmrSystemCoupler::solve_fields / step algorithm
/// (include/pops/coupling/amr_system_coupler.hpp), but over type-erased closures (the runtime facade
/// does not know the blocks' Model/Limiter/Flux types at compile time) rather than over a
/// compile-time CoupledSystem<Blocks...>.
///
/// INVARIANTS (multi-block capstone, docs/AMR_MULTIBLOCK_DESIGN.md):
///  - ONE single shared AMR hierarchy (AmrHierarchyLayout, same_layout_or_throw guard): all
///    blocks live on EXACTLY the same BoxArray + DistributionMapping + dx/dy per level;
///  - ALL blocks live on ALL patches (never a local spatial absence of a block);
///  - SYSTEM Poisson with a SUMMED and CO-LOCATED right-hand side: rhs[coarse] = Sum_b
///    elliptic_rhs_b(U_b) read at the SAME cells of the shared coarse;
///  - aux SHARED per level (phi, grad phi); a single coarse Poisson solve then coarse->fine
///    injection (coupler_inject_aux_mb), exactly like AmrSystemCoupler;
///  - PER-BLOCK conservation (reflux + average_down of the AMR engine, in the advance closure).
///
/// SCOPE (capstone). We carry blocks with potentially DIFFERENT spatial schemes over the FROZEN
/// hierarchy (no regrid: AmrSystemCoupler has none), with per-block MULTIRATE: substeps (explicit
/// substeps) and stride (hold-then-catch-up cadence), honored in step() mirroring
/// AmrSystemCoupler::step (#140). The TEMPORAL TREATMENT is PER BLOCK: explicit (forward-Euler
/// source, carried by the AMR step) OR IMEX (stiff source treated IMPLICITLY by
/// backward_euler_source, transport staying explicit; capstone vii), selected in step().
///
/// IMEX SEMANTICS UNDER substeps (integration decision, follow-up review #184). At substeps=1 AND
/// stride=1 the runtime IMEX branch COINCIDES with the IMEX branch of the compile-time engine
/// AmrSystemCoupler::step (a SOURCE-FREE transport + a backward_euler_source over the effective step).
/// FOR substeps>1 the two paths DIVERGE DELIBERATELY:
///   - the COMPILE-TIME engine IGNORES substeps on the IMEX branch: it does ONE single source-free
///     transport then ONE single implicit_advance over the whole effective step bdt (cf.
///     amr_system_coupler.hpp: the substep loop exists only in the Explicit branch);
///   - the RUNTIME SUB-CYCLES the IMEX splitting: it applies imex_advance K=substeps times, each over
///     bdt/K, i.e. K Lie steps [transport(dt/K); implicit source(dt/K)].
/// This choice is INTENTIONAL and SOUND (it is NOT a bug): (a) the source-free explicit transport
/// becomes SAFER in CFL (each substep carries dt/K, so a wave speed K times larger stays
/// admissible); (b) backward-Euler is UNCONDITIONALLY STABLE whatever the step, so sub-cycling never
/// destabilizes the source; (c) refining the backward-Euler step BRINGS the stiff relaxation CLOSER
/// to its continuous trajectory (splitting error and implicit temporal error both O(dt), both
/// reduced). The runtime thus does NOT mirror the compile-time bit-for-bit once substeps>1; it
/// honors substeps CONSISTENTLY with the explicit branch (same split into K equal substeps), which is
/// the behavior expected by a user setting substeps. Non-regression guard:
/// test_amr_multiblock_imex compares a substeps=4 trajectory to substeps=1 and requires them to
/// DIFFER (the sub-cycling is intentional, not accidental).
///
/// The union-tags regrid and the compiled multi-block production DSL remain LATER PRs. The runtime
/// facade (AmrSystem) explicitly REFUSES multi-block + regrid_every > 0 as long as the union regrid
/// does not exist.

namespace pops {

/// Type-erased closures of ONE AMR block, placed on the shared hierarchy. AMR counterpart of the
/// Species struct of System::Impl: a name + its level stack (on the shared layout) + its closures
/// (advance / elliptic-rhs / max_speed / mass / density). The closures capture the CONCRETE
/// Model/Limiter/Flux of the block (resolved at build): the kernel stays COMPILED, only the block
/// list is type-erased. Produced by detail::build_amr_block (amr_dsl_block.hpp).
struct AmrRuntimeBlock {
  std::string name;
  int ncomp = 1;
  double gamma = static_cast<double>(kPhysicalDefaultGamma);
  /// EXPLICIT substeps of the block within ITS effective macro-step: the effective step (stride * dt)
  /// is split into substeps equal pieces and each piece is advanced by ONE advance_amr (cf.
  /// AmrRuntime::step). substeps=1 => a single advance_amr over the whole effective step (bit-identical).
  int substeps = 1;
  /// HOLD-THEN-CATCH-UP cadence of the block (multirate). stride=1 (default): the block advances at
  /// EVERY macro-step (bit-identical). stride=M>1: the block is HELD at macro-steps 0..M-2 (not
  /// advanced) then CATCHES UP at macro-step M-1, where (macro_step+1)%M==0, by an effective step
  /// M*dt. Same semantics as block_stride_v / AmrSystemCoupler::step (#140). The INVARIANT of the
  /// end-of-window catch-up: at macro-step k the system time is (k+1)*dt and the block that catches
  /// up has then accumulated (k+1)*dt, so it stays temporally CONSISTENT with the fast blocks (never
  /// "in the future"), which keeps the Poisson coupling (summed RHS) meaningful: a held block
  /// contributes with its FROZEN state (its last advance), not with an anticipated state that would
  /// falsify q_b n_b in the sum.
  int stride = 1;
  /// Width of the aux channel READ by the block model (aux_comps<Model>(); >= kAuxBaseComps). The aux
  /// channel SHARED per level is sized to the MAX of this width over all blocks, so that a block
  /// reading an extra field (B_z, T_e; n_aux > 3) never reads out of bounds.
  int aux_ncomp = kAuxBaseComps;

  /// Descriptor of the model CONSERVATIVE variables (names + physical ROLES, Model::conservative_vars()).
  /// Single source of truth to resolve a role (Density, MomentumX, ...) -> component index in
  /// add_coupled_source, like System::add_coupled_source reads Species::cons_vars. The resolution is
  /// STRICT (#181): if the block does NOT expose the requested canonical role (index_of < 0),
  /// add_coupled_source THROWS instead of falling back to component 0 (a silent fallback would apply
  /// the source to the wrong field).
  VariableSet cons_vars;

  /// Level stack of the block (level 0 = coarse, > 0 = fine patches), ON the shared layout. The aux
  /// pointer of each AmrLevelMP is (re)wired by AmrRuntime to the SHARED aux of the level. shared_ptr:
  /// AmrRuntimeBlock stays MOVABLE (a std::vector<AmrLevelMP> is heavy to move into a std::function,
  /// and the engine ctor needs a stable address for the closures).
  std::shared_ptr<std::vector<AmrLevelMP>> levels;

  /// Advances the block by ONE substep of size dt: AMR transport (Berger-Oliger + conservative reflux
  /// + average_down) over the block level stack, with ITS spatial scheme (Limiter, Flux). Captures
  /// advance_amr<Limiter, Flux> on the concrete Model. The substep loop and the stride cadence are
  /// carried by AmrRuntime::step (runtime counterpart of AmrSystemCoupler::step): the closure does ONE
  /// advance_amr, the engine calls it substeps times (dt = effective step/substeps). The signature
  /// passes the base domain + periodicity + coarse ownership policy, rewired by the engine.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool)> advance;

  /// TEMPORAL TREATMENT of the block: false (default) = EXPLICIT (forward-Euler source, in advance);
  /// true = IMEX (stiff source treated IMPLICITLY by backward_euler_source). The facade (AmrSystem)
  /// freezes it from time="imex". Selected EXPLICITLY in AmrRuntime::step (runtime counterpart of the
  /// constexpr block_time_treatment_v dispatch of AmrSystemCoupler::step): an explicit block goes
  /// through advance, an IMEX block through imex_advance. false everywhere -> bit-identical trajectory
  /// to the historical one.
  bool imex = false;

  /// IMEX advance of the block by ONE substep of size dt: (1) EXPLICIT TRANSPORT on the SOURCE-FREE
  /// model (-div F only, SourceFreeModel<Model>) by the AMR engine (Berger-Oliger + conservative
  /// reflux + average_down), then (2) IMPLICIT STIFF SOURCE backward_euler_source AT EACH LEVEL (local
  /// Newton, finite-difference jacobian; implicit mask CARRIED BY THE BLOCK for partial IMEX),
  /// followed by a fine -> coarse cascade (mf_average_down_mb). ONE call = ONE Lie step [transport;
  /// implicit source] over dt. The SEMANTICS of this splitting (source-free transport then
  /// backward-Euler) mirror the IMEX branch of AmrSystemCoupler::step (SourceFreeModel +
  /// AmrImplicitSourceStepper); at substeps=1 it is IDENTICAL to it. But step() calls THIS closure
  /// substeps times (over dt = effective step / substeps), so for substeps>1 the runtime SUB-CYCLES the
  /// IMEX splitting where the compile-time applies it once over the whole effective step: DIVERGENCE
  /// INTENTIONAL (cf. IMEX SEMANTICS UNDER substeps, file header). Captures the CONCRETE
  /// Model/Limiter/Flux + the mask (build_amr_block); the kernel stays COMPILED, only the block
  /// registry is type-erased. CONSERVATION INVARIANT (LOCAL source): the source is cell-local (outside
  /// face fluxes), so OUTSIDE the reflux registers -> conservation at coarse-fine interfaces stays
  /// intact; a COVERED coarse cell becomes again the 2x2 average of its children through the final
  /// cascade (otherwise the mass diagnostic, sum of the coarse only, would count a phantom source).
  /// Empty for an explicit block (imex == false): step() never calls it.
  std::function<void(std::vector<AmrLevelMP>&, const Box2D&, Real, Periodicity, bool)> imex_advance;

  /// POINTWISE PROJECTION post-pas (ADC-177) : U <- project(U, aux) appliquee PAR NIVEAU a la FIN
  /// de l'avance complete du bloc (substeps + reflux/cascade faits). Vide -> aucune projection
  /// (modele sans HasPointwiseProjection : trajectoire bit-identique). Locale par niveau (aucun
  /// collectif MPI). Cf. detail::apply_pointwise_project_amr, cable par build_amr_block.
  std::function<void(std::vector<AmrLevelMP>&)> project_per_level;

  /// NEWTON DIAGNOSTICS (AMR counterpart of System::newton_report). false (default) -> imex_advance
  /// passes report=nullptr to backward_euler_source: FAST bit-identical path, no extra allocation or
  /// reduction. true -> imex_advance passes @c newton_report.get() (STABLE address since shared_ptr)
  /// to backward_euler_source of EACH level; the report is AGGREGATED (max residual, max iterations,
  /// sum of failed cells, MPI all_reduce, structured fail_policy events) over all levels AND all
  /// substeps of a macro-step. AmrRuntime::step RESETS the report at the head of the block advance
  /// (parity with System::AdvanceImex which resets at the head of operator()). MULTI-BLOCK native only
  /// (the single-block coupler and the .so loaders reject it at build / at the facade). STABLE address
  /// (shared_ptr): captured by the imex_advance closure AND read by AmrRuntime::newton_report.
  bool newton_diagnostics = false;
  std::shared_ptr<NewtonReport> newton_report;

  /// Contribution of the block to the Poisson right-hand side: rhs += elliptic_rhs_b(U_b) on the
  /// coarse. CO-LOCATED: the loop reads U_b and writes rhs AT THE SAME cells (same shared coarse
  /// BoxArray). The SUM of the contributions of all blocks forms the system Poisson RHS.
  std::function<void(const MultiFab&, MultiFab&)> add_elliptic_rhs;

  /// Per-NAMED-field elliptic right-hand-side contributions of the block (ADC-428): field name ->
  /// closure rhs += elliptic_field_rhs_b(U_b) on the coarse, exactly like @c add_elliptic_rhs but for a
  /// SECOND (user-named) elliptic field declared by the block's model (m.elliptic_field). The native
  /// loader attaches one closure here per declared field (set_block_elliptic_field). AmrRuntime sums
  /// them over the blocks into the named field's dedicated solver RHS (solve_named_fields). Empty for a
  /// block that declares no named field -> the named-field solve loop never reads it (bit-identical).
  std::map<std::string, std::function<void(const MultiFab&, MultiFab&)>> named_elliptic_rhs;

  /// SEMI-DISCRETE residual of the block on ONE level (epic ADC-508, compiled-Program AMR driver):
  /// R <- -div F(U) + S(U, aux) over the level's grid, the per-level counterpart of System's
  /// Species::rhs_into. Signature (U, aux, geom, R): @c U the level state, @c aux the SHARED per-level
  /// aux (phi / grad / B_z, filled by solve_fields + coarse->fine injection), @c geom the level metric
  /// (dx/dy >> k, domain << k), @c R the output residual. Captures BlockRhsEval<Limiter, Flux, Model>
  /// (the SAME evaluator System uses, device-clean named functor) on the concrete scheme; the level
  /// geometry / domain are passed in so the SAME closure serves every level. EMPTY for a block built
  /// before the seam (the host .so prototype loader): AmrRuntime::level_rhs_into fails loud then. Used
  /// ONLY by an installed compiled time Program (AmrProgramContext); the native AMR step never calls it.
  std::function<void(MultiFab&, const MultiFab&, const Geometry&, MultiFab&)> level_rhs;
  /// FLUX-ONLY per-level residual R <- -div F(U) (NO default source), the SourceFreeModel<Model> path
  /// (Lie/Strang split, ADC-425). Same signature / device contract as @ref level_rhs.
  std::function<void(MultiFab&, const MultiFab&, const Geometry&, MultiFab&)> level_neg_div_flux;
  /// SOURCE-ONLY per-level residual R <- S(U, aux) (NO flux divergence), the exact MIRROR of @ref
  /// level_neg_div_flux (ADC-430). Same signature / device contract as @ref level_rhs.
  std::function<void(MultiFab&, const MultiFab&, const Geometry&, MultiFab&)> level_source;

  /// Speed driving the block CFL on the coarse. By default max_wave_speed (historical); when the
  /// model declares the HasStabilitySpeed trait, it is lambda* (stability_speed) that the closure
  /// reduces -- SAME policy as System (make_max_speed), cf. build_amr_block.
  std::function<Real(const MultiFab&, const MultiFab&)> max_speed;

  /// OPTIONAL STEP BOUNDS of the block (AMR StabilityPolicy, audit 2026-06): evaluated on the COARSE
  /// (level 0, where the AMR CFL lives -- cf. step_cfl: h = dx_coarse). EMPTY (default) -> step_cfl
  /// keeps the transport bound only, bit-identical. Filled by build_amr_block / build_amr_compiled when
  /// the model declares HasSourceFrequency / HasStabilityDt (same semantics as System: mu in 1/s ->
  /// dt <= cfl*substeps/(stride*mu), without h; direct admissible step -> dt <=
  /// dt_adm*substeps/stride, without cfl).
  std::function<Real(const MultiFab&, const MultiFab&)> source_frequency;
  std::function<Real(const MultiFab&, const MultiFab&)> stability_dt;

  /// Mass of component 0 of the block coarse (sum u*dV; cross-rank reduced if distributed).
  std::function<Real()> mass;

  /// Coarse density (component 0) of the block as a global n*n row-major field (diagnostic).
  std::function<std::vector<double>()> density;

  /// Coarse potential read from the shared aux (component 0) as an n*n row-major field (diagnostic).
  /// Identical for all blocks (shared aux); carried per block for API symmetry.
  std::function<std::vector<double>(const MultiFab&)> potential;
};

/// AMR multi-block engine at runtime. Owns the SHARED aux per level, the coarse Poisson
/// (GeometricMG), the geometry + BC, and the type-erased block REGISTRY. Reproduces the
/// AmrSystemCoupler algorithm (solve_fields + step) over closures rather than a CoupledSystem.
class AmrRuntime {
 public:
  /// @param geom        geometry of the coarse level (domain + physical extents).
  /// @param ba_coarse   BoxArray of the coarse (the coarse Poisson lives on it).
  /// @param bcPhi       BC of the coarse Poisson.
  /// @param blocks      block registry (>= 1), all on the SAME layout (guarded at the ctor).
  /// @param base_per    periodicity of the base domain (transport).
  /// @param replicated_coarse  ownership of level 0 (replicated single-box, or distributed multi-box).
  /// @param active      conductive-wall predicate (passed to MG; empty = none).
  AmrRuntime(const Geometry& geom, const BoxArray& ba_coarse, const BCRec& bcPhi,
             std::vector<AmrRuntimeBlock> blocks, Periodicity base_per = Periodicity{true, true},
             bool replicated_coarse = true, std::function<bool(Real, Real)> active = {})
      : geom_(geom),
        dom_(geom.domain),
        base_per_(base_per),
        bcPhi_(bcPhi),
        aux_bc_(detail::derive_aux_bc(bcPhi)),
        replicated_coarse_(replicated_coarse),
        mg_(geom, ba_coarse, bcPhi, active, replicated_coarse),
        ba_coarse_(ba_coarse),
        wall_active_(std::move(active)),  // copy already consumed by mg_ (earlier in decl order)
        blocks_(std::move(blocks)) {
    if (blocks_.empty())
      throw std::runtime_error("AmrRuntime : at least one block required");
    for (const auto& b : blocks_)
      if (!b.levels || b.levels->empty())
        throw std::runtime_error(
            "AmrRuntime : each block must carry at least one level "
            "(coarse) on the shared layout");
    nlev_ = static_cast<int>(blocks_[0].levels->size());

    // EXACT layout consistency between blocks (the aux is shared per level): same number of levels,
    // and per level same BoxArray (boxes AND order), same DistributionMapping, same dx/dy. SAME guard
    // as AmrSystemCoupler (detail::same_layout_or_throw): all blocks live on ALL patches of the
    // UNIQUE shared hierarchy. A single block matches itself trivially (the loop over the other blocks
    // is empty).
    {
      std::vector<std::vector<AmrLevelMP>> ref;
      ref.reserve(blocks_.size());
      for (const auto& b : blocks_)
        ref.push_back(*b.levels);
      detail::same_layout_or_throw(ref);
    }

    // Width of the SHARED aux channel: max of the blocks' aux_comps (>= kAuxBaseComps). Counterpart of
    // AmrSystemCoupler::system_aux_comps: a block reading an extra field (B_z, T_e) has the room at
    // each level, a base block ignores the extra components. PR1 does not POPULATE multi-block B_z (no
    // bz_ here), but we size the channel to the widest anyway so that load_aux<aux_comps<Model>> never
    // reads out of bounds. Without an extra-field block -> kAuxBaseComps (3) -> allocation strictly
    // identical to the base case.
    aux_ncomp_ = kAuxBaseComps;
    for (const auto& b : blocks_)
      if (b.aux_ncomp > aux_ncomp_)
        aux_ncomp_ = b.aux_ncomp;

    // SHARED aux: one MultiFab (phi, grad phi) per level, on the common grid. Sized once -> stable
    // addresses for the blocks' aux pointers. The shared layout is that of block 0
    // (same_layout_or_throw guard: identical for all).
    aux_.resize(nlev_);
    const auto& L0 = *blocks_[0].levels;
    for (int k = 0; k < nlev_; ++k)
      aux_[k] = MultiFab(L0[k].U.box_array(), L0[k].U.dmap(), aux_ncomp_, 1);
    for (auto& b : blocks_)
      for (int k = 0; k < nlev_; ++k)
        (*b.levels)[k].aux = &aux_[k];

    // Tag predicates of the union regrid: one empty slot per block (set_block_tag_predicate fills
    // them). Empty by default -> no tag -> frozen hierarchy (regrid is not called anyway as long as
    // set_regrid has not activated regrid_every_ > 0).
    block_tag_.resize(blocks_.size());
  }

  int nlev() const { return nlev_; }
  std::size_t n_blocks() const { return blocks_.size(); }
  /// Conservative VariableSet (names + physical roles, Model::conservative_vars()) of block @p b. The
  /// SAME cons_vars that add_coupled_source resolves (block, role) against; exposed read-only so the
  /// facade can resolve a name/role-selected regrid variable into a component per block (ADC-296).
  /// @throws if @p b is out of bounds.
  const VariableSet& block_cons_vars(std::size_t b) const {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::block_cons_vars : block index out of bounds");
    return blocks_[b].cons_vars;
  }
  std::size_t n_coupled_sources() const { return coupled_sources_.size(); }
  /// Read-only view of the registered coupling operators (ADC-595, parity with System): label plus the
  /// declared conservation / frequency contracts, in registration order, so a Program or a runtime
  /// report enumerates the AMR couplings as typed operators. A raw add_coupled_source registers an
  /// "unchecked" entry (empty contract); add_coupling_operator records the declared contract.
  const std::vector<CouplingOperatorView>& coupled_operators() const { return coupled_operators_; }
  MultiFab& phi() { return mg_.phi(); }
  // System Poisson right-hand side after the last solve_fields: f = Sum_b elliptic_rhs_b(U_b) on the
  // shared coarse. Exposed to check the CO-LOCATED SUM (PR1 test); same grid as the coarse (the
  // blocks' contributions are accumulated there at the same cells).
  MultiFab& poisson_rhs() { return mg_.rhs(); }
  const MultiFab& aux(int k) const { return aux_[k]; }
  std::vector<AmrLevelMP>& levels(std::size_t b) { return *blocks_[b].levels; }
  Real mass(std::size_t b) const { return blocks_[b].mass(); }
  std::vector<double> density(std::size_t b) const { return blocks_[b].density(); }
  int solve_count() const { return solve_count_; }
  int regrid_count() const { return regrid_count_; }

  /// @name Compiled time-Program AMR driver seam (epic ADC-508): per-level primitives exposing the
  /// engine internals an AmrProgramContext composes into a per-level macro-step. APPEND-ONLY: these
  /// surface existing storage / closures; none reimplement numerics. All are NO-OP-safe under MPI
  /// (loops over local_size()). The native AMR step does not call any of them.
  /// @{
  /// The live state MultiFab of block @p b at level @p k (zero-copy; same address an AmrProgramContext
  /// reads each macro-step). @c b is the AMR block index (sys_block-resolved by the caller).
  MultiFab& level_state(std::size_t b, int k) { return (*blocks_[b].levels)[k].U; }
  const MultiFab& level_state(std::size_t b, int k) const { return (*blocks_[b].levels)[k].U; }
  /// Geometry of level @p k: the coarse metric refined k times (dx/dy >> k, domain << k). The metric
  /// the per-level Laplacian / gradient / RHS read (parity with System's grid_context().geom).
  Geometry level_geom(int k) const { return geom_.refine(1 << k); }
  /// Transport BCRec derived from the base periodicity (periodic where periodic, else Foextrap) -- the
  /// SAME convention System::make_bc uses, so a Program's per-level ghost fill matches the System path.
  BCRec transport_bc() const {
    BCRec b;  // periodic by default
    if (!base_per_.x)
      b.xlo = b.xhi = BCType::Foextrap;
    if (!base_per_.y)
      b.ylo = b.yhi = BCType::Foextrap;
    return b;
  }
  /// BC of the coarse Poisson (for a matrix-free Krylov preconditioner's GeometricMG).
  const BCRec& poisson_bc() const { return bcPhi_; }
  /// A fresh scalar field co-distributed with level @p k's grid (its ba/dm), @p n_comp components,
  /// @p n_ghost ghosts, zero-initialized -- the Krylov scratch (r/p/Ap) of a per-level field solve.
  /// Counterpart of System::alloc_scalar_field, but at the level's layout (read from block 0).
  MultiFab level_scalar_field(int k, int n_comp, int n_ghost) const {
    const MultiFab& U = (*blocks_[0].levels)[k].U;
    MultiFab f(U.box_array(), U.dmap(), n_comp, n_ghost);
    f.set_val(Real(0));
    return f;
  }
  /// R <- -div F(U) + S(U, aux_[k]) for block @p b on level @p k (the per-level analogue of
  /// System::block_rhs_into). Forwards to the block's level_rhs closure with the level metric + shared
  /// aux; fails loud if the block built no such closure (a host .so prototype).
  void level_rhs_into(std::size_t b, int k, MultiFab& U, MultiFab& R) {
    if (!blocks_[b].level_rhs)
      throw std::runtime_error(
          "AmrRuntime::level_rhs_into: block '" + blocks_[b].name +
          "' has no per-level residual closure (rebuild the AMR block via the production DSL "
          "target='amr_system')");
    blocks_[b].level_rhs(U, aux_[k], level_geom(k), R);
  }
  /// R <- -div F(U) only (NO default source) for block @p b on level @p k (SourceFreeModel path).
  void level_neg_div_flux_into(std::size_t b, int k, MultiFab& U, MultiFab& R) {
    if (!blocks_[b].level_neg_div_flux)
      throw std::runtime_error("AmrRuntime::level_neg_div_flux_into: block '" + blocks_[b].name +
                               "' has no flux-only per-level residual closure");
    blocks_[b].level_neg_div_flux(U, aux_[k], level_geom(k), R);
  }
  /// R <- S(U, aux_[k]) only (NO flux) for block @p b on level @p k (the source half of level_rhs).
  void level_source_into(std::size_t b, int k, MultiFab& U, MultiFab& R) {
    if (!blocks_[b].level_source)
      throw std::runtime_error("AmrRuntime::level_source_into: block '" + blocks_[b].name +
                               "' has no source-only per-level residual closure");
    blocks_[b].level_source(U, aux_[k], level_geom(k), R);
  }
  /// Max |wave speed| of block @p b on @p U (the SAME closure step_cfl reads). Evaluated on the aux of
  /// level @p k. A Program dt bound reads it as cfl*hmin/max_wave_speed.
  Real level_max_speed(std::size_t b, int k, const MultiFab& U) const {
    return blocks_[b].max_speed(U, aux_[k]);
  }
  /// MIN physical cell size of level @p k (min(dx, dy) >> k): the per-level hmin a Program dt bound reads.
  Real level_hmin(int k) const {
    const Real r = static_cast<Real>(1 << k);
    return std::min(geom_.dx(), geom_.dy()) / r;
  }
  /// fine -> coarse restriction of block @p b between levels @p k and @p k-1 (covered coarse cell <-
  /// 2x2 fine average): the SAME mf_average_down_mb solve_fields / the native step run. Exposed for the
  /// synchronous Program driver's inter-level coupling. No-op when k < 1.
  void average_down_level(std::size_t b, int k) {
    if (k < 1)
      return;
    auto& L = *blocks_[b].levels;
    mf_average_down_mb(L[k].U, L[k - 1].U);
  }
  /// Head-of-step union-tags regrid at the Program driver's cadence (the SAME regrid() the native step
  /// runs at its head). @p macro_step gates it like AmrRuntime::step (skip step 0; honor regrid_every_).
  void regrid_if_due(int macro_step) {
    if (regrid_every_ > 0 && macro_step > 0 && macro_step % regrid_every_ == 0)
      regrid();
  }
  /// @}

  /// Tag predicate of the union regrid: (ConstArray4 of the read field, i, j) -> should we refine ?
  /// HOST type (evaluated in the host loop of tag_cells, never on device): a std::function capturing a
  /// concrete functor is licit (nvcc-safe -- the predicate does not enter a kernel). We use it for the
  /// PER-BLOCK criterion (read on the block density/U, component 0) and for the phi criterion (read on
  /// the shared aux). docs/AMR_REGRID_UNION_TAGS_DESIGN.md (D1, D4).
  using TagPredicate = std::function<bool(const ConstArray4&, int, int)>;

  /// Activates the UNION-TAGS REGRID at the cadence @p every (in macro-steps): every @p every
  /// macro-steps, BEFORE the macro-step's step(dt) (D2, consistent with the single-block
  /// amr_dsl_block.hpp:104), the shared hierarchy is re-gridded from the UNION of the tags of all
  /// blocks + phi. @p every == 0 (DEFAULT) -> FROZEN hierarchy, regrid never called -> BIT-IDENTICAL
  /// trajectory to the historical one (the feature is opt-in). @p grow: tag dilation (nesting +
  /// anticipation); @p margin: nesting (clamp the patches to the boundaries). Must be called BEFORE
  /// the first step.
  void set_regrid(int every, int grow = 2, int margin = 2) {
    if (every < 0)
      throw std::runtime_error("AmrRuntime::set_regrid : regrid_every >= 0");
    regrid_every_ = every;
    regrid_grow_ = grow;
    regrid_margin_ = margin;
  }

  /// Registers the TAG PREDICATE of block @p b (D1: PER-BLOCK union criterion). The predicate is
  /// evaluated on the block U (component 0 = density, or a discrete gradient at the caller's charge) at
  /// the PARENT level during the regrid; the UNION (OR) of the predicates of all blocks + the phi
  /// criterion drives the clustering. A block WITHOUT a registered predicate tags nothing on ITS side
  /// (it stays re-gridded as background, present everywhere, by the union of the other criteria).
  /// @throws if @p b is out of bounds.
  void set_block_tag_predicate(std::size_t b, TagPredicate crit) {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::set_block_tag_predicate : block index out of bounds");
    block_tag_[b] = std::move(crit);
  }

  /// Registers the PHI TAG PREDICATE (D4: SEPARATE phi criterion, on |grad phi|). The predicate is
  /// evaluated on the shared aux of the parent level (components 1,2 = grad phi in x,y) during the
  /// regrid; it adds to the union of the blocks' tags. Not registered -> phi does not contribute to
  /// the union.
  void set_phi_tag_predicate(TagPredicate crit) { phi_tag_ = std::move(crit); }

  /// Registers a model-NAMED aux field (ADC-291) at shared-channel component @p comp (= kAuxNamedBase
  /// + k for the k-th named field of a block), as a coarse base-level field @p field (n*n row-major,
  /// global cell index j*nx+i). The field is STATIC (external to the elliptic): solve_fields re-applies
  /// it onto the coarse aux every macro-step AFTER field_postprocess (which only writes phi/grad,
  /// comps 0..2) and BEFORE the coarse->fine injection, so it reaches every level and SURVIVES a regrid
  /// (regrid re-solves). AMR counterpart of System::set_aux_field_component. No-op default: without a
  /// named field the map is empty and the path is bit-identical. @p comp must be >= kAuxNamedBase and
  /// within the channel (the facade validates and resolves the name).
  void set_named_aux(int comp, std::vector<Real> field) {
    named_aux_[comp] = std::move(field);
    if (!aux_.empty())
      apply_named_aux();  // reflect immediately if the hierarchy already exists
  }

  /// Registers a per-field aux HALO policy (ADC-369) for the named component @p comp: solve_fields
  /// applies it onto the COARSE aux AFTER the shared fill_ghosts, overriding only that component's
  /// physical-face ghosts (periodic faces stay periodic). Coarse-level scope (fine patches touching the
  /// domain boundary inherit the shared BC). No-op default. AMR counterpart of
  /// System::set_aux_field_halo_component.
  void set_named_aux_bc(int comp, AuxHaloPolicy policy) { named_aux_bc_[comp] = policy; }

  /// @name Named multi-elliptic fields (ADC-428)
  /// A SECOND elliptic solve (beyond the default coarse Poisson) for a user-named field
  /// (m.elliptic_field("psi", rhs=..., aux=[...])) on the AMR hierarchy. AMR counterpart of
  /// SystemFieldSolver::register_named_field / solve_named_field_from_state. Each named field owns a
  /// DEDICATED coarse GeometricMG solver (built lazily, REUSING the native solver -- the operator is
  /// never reimplemented), its RHS = sum over blocks of @c named_elliptic_rhs[field], and its own aux
  /// output components (the model's named aux slots, >= kAuxNamedBase). solve_fields() solves every
  /// registered named field right after the default Poisson and injects its aux to the fine levels, so a
  /// bare run() leaves the field SOLVED (readable via named_field_values). The default Poisson path
  /// (mg_) is untouched / bit-identical. Empty default -> the named-field loop is a no-op.
  /// @{
  /// Registers named @c field's aux output components: @p phi_comp where the solved potential lands, @p
  /// gx_comp / @p gy_comp where its centered gradient lands. @p gx_comp / @p gy_comp < 0 => only phi is
  /// written (the field declared fewer than 3 aux slots). Idempotent (re-register overwrites the
  /// components and drops the lazily-built solver so the next solve rebuilds it). The dedicated solver is
  /// built on the first solve, never here.
  void register_named_field(const std::string& field, int phi_comp, int gx_comp, int gy_comp) {
    NamedField nf;
    nf.phi_comp = phi_comp;
    nf.gx_comp = gx_comp;
    nf.gy_comp = gy_comp;
    named_fields_[field] = std::move(nf);  // solver built lazily by ensure_named_elliptic
    // ADC-596: mirror the field into the unified descriptor registry (a named GeometricMG field on
    // the AMR route -- AMR always uses GeometricMG, never FFT). Purely descriptive: the lazy solver
    // build (ensure_named_elliptic) and RHS assembly are untouched.
    if (field_problems_.find("phi") < 0)
      field_problems_.register_problem(default_poisson_entry());
    field_problems_.register_problem(
        named_field_entry(field, phi_comp, gx_comp, gy_comp, EllipticSolverKind::GeometricMG));
  }

  /// The unified field-problem registry (ADC-596), seeding the default "phi" entry on first access so
  /// the single-field case is described like a named one. Both Uniform and AMR expose this SAME type,
  /// and both validate their entries for their route before bind. Descriptive only (no numerics).
  const FieldProblemRegistry& field_problem_registry() {
    if (field_problems_.find("phi") < 0)
      field_problems_.register_problem(default_poisson_entry());
    return field_problems_;
  }
  /// Attaches named @p field's RHS contribution closure (rhs += elliptic_field_rhs(U_b)) to block @p b.
  /// Called per declared field once the runtime owns the blocks. @throws if @p b is out of bounds.
  void set_block_named_elliptic_rhs(std::size_t b, const std::string& field,
                                    std::function<void(const MultiFab&, MultiFab&)> rhs) {
    if (b >= blocks_.size())
      throw std::runtime_error(
          "AmrRuntime::set_block_named_elliptic_rhs : block index out of bounds");
    blocks_[b].named_elliptic_rhs[field] = std::move(rhs);
  }
  /// Number of registered named elliptic fields (diagnostic / test).
  std::size_t n_named_fields() const { return named_fields_.size(); }
  /// True if @p field is a registered named elliptic field.
  bool has_named_field(const std::string& field) const {
    return named_fields_.find(field) != named_fields_.end();
  }
  /// Solved potential of named @p field as a COARSE n*n row-major field (diagnostic / read-back). Solves
  /// the fields if needed (counterpart of potential() for the default phi), then reads the field's
  /// phi_comp on the coarse aux. @throws if @p field is unregistered. AMR counterpart of
  /// System::aux_field_component for a named elliptic field.
  std::vector<double> named_field_values(const std::string& field) {
    auto it = named_fields_.find(field);
    if (it == named_fields_.end())
      throw std::runtime_error("AmrRuntime::named_field_values : unknown named elliptic field '" +
                               field + "' (register it via m.elliptic_field + the compiled block)");
    solve_fields();  // up-to-date phi (counterpart of potential()); named solve runs inside
    return coarse_aux_component(it->second.phi_comp);
  }
  /// @}

  /// Registers an inter-species COUPLED SOURCE (DSL CoupledSource, P5 bytecode) on the runtime facade,
  /// counterpart of System::add_coupled_source. The ABI is FLAT (postfix bytecode): we resolve each
  /// (block, role) into (block index, component) then store a closure that, at each macro-step AFTER
  /// the transport, applies the source by additive forward-Euler splitting via coupled_source_step. The
  /// coupling is ENTIRELY baked into a stack machine (device-clean functor CoupledSourceKernel): NO
  /// per-cell Python callback in the hot path.
  ///
  /// CONSERVATION (conservative exchange): with an add_pair construction (one +expr term on one block,
  /// -expr exactly on the other, SAME cell), the two per-cell contributions are opposite up to sign, so
  /// n_a + n_b is conserved PER CELL (and globally) to machine precision, independent of dt and of the
  /// state. The engine does not enforce it (an ionization creating a pair is licit): conservation is a
  /// property of the constructed coupling, checked test-side.
  ///
  /// @param in_blocks/in_roles  READ fields (one register per (block, role)), in register order.
  /// @param consts              constants (parameters), loaded into the registers after the inputs.
  /// @param out_blocks/out_roles target (block, role) of each source term.
  /// @param prog_ops/prog_args  CONCATENATED postfix bytecode of all the terms (split by prog_lens).
  /// @param prog_lens           program length of each term (size == out_blocks).
  /// @throws std::runtime_error on an inconsistent form, an unknown role, an unknown block, an opcode
  ///         or register out of bounds, or a program too long (same guards as System).
  void add_coupled_source(const std::vector<std::string>& in_blocks,
                          const std::vector<std::string>& in_roles,
                          const std::vector<double>& consts,
                          const std::vector<std::string>& out_blocks,
                          const std::vector<std::string>& out_roles,
                          const std::vector<int>& prog_ops, const std::vector<int>& prog_args,
                          const std::vector<int>& prog_lens) {
    const int n_in = static_cast<int>(in_blocks.size());
    const int n_const = static_cast<int>(consts.size());
    const int n_terms = static_cast<int>(out_blocks.size());
    // --- form validation (before any step, EXPLICIT errors); mirror of System::add_coupled_source.
    if (n_terms == 0)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : no source term (out_blocks empty)");
    if (static_cast<int>(in_roles.size()) != n_in)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : in_blocks / in_roles of different sizes");
    if (static_cast<int>(out_roles.size()) != n_terms ||
        static_cast<int>(prog_lens.size()) != n_terms)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : out_blocks / out_roles / prog_lens of "
          "different sizes");
    if (prog_ops.size() != prog_args.size())
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : prog_ops / prog_args of different sizes");
    if (n_in + n_const > kCsMaxReg)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_source : too many registers (inputs + constants > " +
          std::to_string(kCsMaxReg) + ")");
    if (n_terms > kCsMaxTerms)
      throw std::runtime_error("AmrRuntime::add_coupled_source : too many source terms (> " +
                               std::to_string(kCsMaxTerms) + ")");
    // Resolves (block, role) -> (block index, component) by the block CONSERVATIVE descriptor, like
    // System (#181). An unknown block throws immediately; an unknown (non-canonical) role too.
    auto resolve = [&](const std::string& block, const std::string& role) -> std::pair<int, int> {
      const int b = block_index(block);
      if (b < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_source : no block named '" + block + "'");
      // STRICT (no silent fallback; mirror of System::add_coupled_source #181): a DSL coupled source
      // targets a (block, role) EXPLICITLY requested by the user. The role is addressed BY NAME: a
      // canonical role name OR a user-defined role label (index_of(string), ADC-292). If the block does
      // NOT expose this role, a fallback to component 0 would apply the source to the wrong field
      // SILENTLY (the false-positive identified at the Lot E review). We throw, listing what the block
      // exposes.
      const VariableSet& vs = blocks_[static_cast<std::size_t>(b)].cons_vars;
      const int comp = vs.index_of(role);
      if (comp < 0)
        throw std::runtime_error(
            "AmrRuntime::add_coupled_source : block '" + block + "' does not expose role '" + role +
            "' (roles: " + (vs.roles.empty() ? std::string("<none>") : roles_csv(vs)) +
            ", no silent fallback to component 0)");
      return {b, comp};
    };
    // Inputs: (block, component) read per cell. Captured by INDEX -> we rebuild the Array4 at EACH
    // application (the fabs live in the level stack, repointed per level in the splitting).
    std::vector<CsRef> ins(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c) {
      auto [b, comp] =
          resolve(in_blocks[static_cast<std::size_t>(c)], in_roles[static_cast<std::size_t>(c)]);
      ins[static_cast<std::size_t>(c)] = {b, comp, CsProgram{}};
    }
    std::vector<CsRef> outs(static_cast<std::size_t>(n_terms));
    int off = 0;
    for (int t = 0; t < n_terms; ++t) {
      auto [b, comp] =
          resolve(out_blocks[static_cast<std::size_t>(t)], out_roles[static_cast<std::size_t>(t)]);
      const int len = prog_lens[static_cast<std::size_t>(t)];
      if (len < 0 || len > kCsMaxProg)
        throw std::runtime_error("AmrRuntime::add_coupled_source : program of term " +
                                 std::to_string(t) + " too long (> " + std::to_string(kCsMaxProg) +
                                 ")");
      if (off + len > static_cast<int>(prog_ops.size()))
        throw std::runtime_error(
            "AmrRuntime::add_coupled_source : prog_lens inconsistent with prog_ops");
      CsProgram pg;
      pg.len = len;
      for (int k = 0; k < len; ++k) {
        const int opc = prog_ops[static_cast<std::size_t>(off + k)];
        const int a = prog_args[static_cast<std::size_t>(off + k)];
        if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
          throw std::runtime_error("AmrRuntime::add_coupled_source : invalid opcode");
        if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
          throw std::runtime_error(
              "AmrRuntime::add_coupled_source : register out of bounds in the program");
        pg.op[k] = opc;
        pg.arg[k] = a;
      }
      validate_cs_program_stack(pg, "AmrRuntime::add_coupled_source term " + std::to_string(t));
      outs[static_cast<std::size_t>(t)] = {b, comp, pg};
      off += len;
    }
    std::vector<Real> kconsts(consts.begin(), consts.end());
    coupled_sources_.push_back(CoupledSourceSpec{std::move(ins), std::move(outs),
                                                 std::move(kconsts), n_in, n_const, n_terms});
  }

  /// Registers a TYPED coupling operator on the AMR runtime (ADC-595, parity with
  /// System::add_coupling_operator): validates the DECLARED conservation contract against the terms
  /// (host, fail-loud) BEFORE storing, lowers the program through the SAME add_coupled_source path
  /// (bit-identical), then records the declared contract for coupled_operators(). @p frequency /
  /// @p label name the operator's declared frequency bound in the inspect view (the AMR frequency
  /// bound itself is registered separately via add_coupled_freq / add_coupled_freq_expr).
  void add_coupling_operator(const CouplingOperator& op, double frequency, const std::string& label) {
    validate_coupling_contract(op, "AmrRuntime::add_coupling_operator");
    const CoupledSourceProgram& p = op.program;
    add_coupled_source(p.in_blocks, p.in_roles, p.consts, p.out_blocks, p.out_roles, p.prog_ops,
                       p.prog_args, p.prog_lens);
    CouplingOperatorView view;
    view.label = label;
    view.conservation = op.conservation;
    view.frequency.constant_mu = frequency;
    view.frequency.per_cell = !p.freq_prog_ops.empty() || !p.freq_prog_args.empty();
    coupled_operators_.push_back(std::move(view));
  }

  /// Applies ALL the registered coupled sources of a step dt, by forward-Euler splitting. Runtime
  /// counterpart of AmrSystemCoupler::coupled_source_step: we refresh the fields (aux per level) then,
  /// source by source, we apply the bytecode INDEPENDENTLY AT EACH LEVEL of the shared hierarchy (the
  /// blocks live on ALL levels), followed by a fine -> coarse cascade.
  ///
  /// COVERAGE INVARIANT (#169): the source was applied independently on EACH level, so a coarse cell
  /// COVERED by a fine patch would otherwise carry its own coarse source, unrelated to the source seen
  /// by its fine children. A covered coarse cell MUST be the 2x2 average of its children (it does not
  /// represent matter on its own). We restore this consistency by the SAME fine -> coarse cascade
  /// (mf_average_down_mb) as solve_fields and the compile-time engine: without it, the mass diagnostic
  /// (sum of the coarse only) would count a phantom coarse source under the patch. Single-level
  /// hierarchy: no covered cell, the cascade loops do not run -> bit-identical to the no-patch case.
  ///
  /// PER-CELL CONSERVATION: at a given level, each term writes out(i,j,comp) += dt * S(reg(i,j)) on
  /// the SAME cell (i,j) read by the inputs; an add_pair exchange lays +S on one block and -S on the
  /// other AT THE SAME (i,j), so the sum of the two blocks is unchanged cell by cell. Without a
  /// registered source (coupled_sources_ empty): total no-op -> bit-identical trajectory to the
  /// historical one.
  void coupled_source_step(Real dt) {
    if (coupled_sources_.empty())
      return;        // opt-in: no source -> bit-identical path
    solve_fields();  // aux per level up to date (a term may read phi/grad via a future input)
    for (const auto& cs : coupled_sources_) {
      // PER-LEVEL application: at each level k, the blocks share EXACTLY the same layout
      // (same_layout_or_throw guard), so same local_size() and same local indexing -> we iterate in
      // parallel over the local fabs. local_size()==0 on a rank without a box -> empty loop (MPI-safe).
      for (int k = 0; k < nlev_; ++k) {
        const int sref = cs.n_in > 0 ? cs.ins[0].block : cs.outs[0].block;
        MultiFab& Uref = (*blocks_[static_cast<std::size_t>(sref)].levels)[k].U;
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledSourceKernel kern;
          kern.dt = dt;
          kern.n_in = cs.n_in;
          kern.n_const = cs.n_const;
          kern.n_terms = cs.n_terms;
          for (int c = 0; c < cs.n_in; ++c) {
            kern.in[c] =
                (*blocks_[static_cast<std::size_t>(cs.ins[static_cast<std::size_t>(c)].block)]
                      .levels)[k]
                    .U.fab(li)
                    .array();
            kern.in_comp[c] = cs.ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < cs.n_const; ++c)
            kern.consts[c] = cs.kconsts[static_cast<std::size_t>(c)];
          for (int t = 0; t < cs.n_terms; ++t) {
            kern.out[t] =
                (*blocks_[static_cast<std::size_t>(cs.outs[static_cast<std::size_t>(t)].block)]
                      .levels)[k]
                    .U.fab(li)
                    .array();
            kern.out_comp[t] = cs.outs[static_cast<std::size_t>(t)].comp;
            kern.prog[t] = cs.outs[static_cast<std::size_t>(t)].prog;
          }
          for_each_cell(Uref.box(li),
                        kern);  // NAMED functor (device-clean), additive forward-Euler
        }
      }
      // Restore the consistency of the covered coarse cells (cf. COVERAGE INVARIANT above).
      for (auto& b : blocks_)
        for (int k = nlev_ - 1; k >= 1; --k)
          mf_average_down_mb((*b.levels)[k].U, (*b.levels)[k - 1].U);
    }
  }

  /// sync_down (per block) + system coarse Poisson (CO-LOCATED SUMMED RHS) + coarse aux + fine
  /// injection. Reproduces AmrSystemCoupler::solve_fields identically, but the system RHS is assembled
  /// by the blocks' add_elliptic_rhs closures (Sum_b elliptic_rhs_b(U_b)) instead of a compile-time
  /// RhsAssembler.
  void solve_fields() {
    ++solve_count_;
    // 1. average_down per block (fine -> coarse) over the whole hierarchy. AMR PROFILING (Spec 5
    // criterion 43): time the restriction cascade into the "average_down" scope + bump its per-solve
    // count. The scope is per-solve_fields (NOT per-cell), so a profiled run pays one clock pair here;
    // an unprofiled run constructs nothing (profiler_ null or disabled). See profile_amr_scope below.
    {
      auto _ad = profile_amr_scope("average_down");
      if (profiler_ != nullptr)
        profiler_->count("average_down");
      for (auto& b : blocks_) {
        auto& L = *b.levels;
        for (int k = nlev_ - 1; k >= 1; --k)
          mf_average_down_mb(L[k].U, L[k - 1].U);
      }
    }

    // 2. SUMMED and CO-LOCATED system RHS: f = Sum_b elliptic_rhs_b(U_b) on the coarse. We reset to
    // zero then each block ACCUMULATES (+=) its contribution on the SAME cells of the shared coarse
    // (mg_.rhs() shares the coarse layout).
    mg_.rhs().set_val(Real(0));
    for (auto& b : blocks_)
      b.add_elliptic_rhs((*b.levels)[0].U, mg_.rhs());
    mg_.solve();

    // 3. coarse aux = (phi, grad phi) via the SAME clean path as AmrSystemCoupler: fill the ghosts of
    // phi according to bcPhi_, field_postprocess (phi + grad), fill the ghosts of aux according to
    // aux_bc_ (derived from bcPhi_). Handles the non-periodic case (Foextrap).
    fill_ghosts_profiled(mg_.phi(), dom_, bcPhi_);
    const Real cx = Real(1) / (2 * geom_.dx()), cy = Real(1) / (2 * geom_.dy());
    field_postprocess(mg_.phi(), aux_[0], cx, cy,
                      FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
    // 3b. model-NAMED aux (ADC-291): re-apply the static named fields onto the coarse valid cells
    // BEFORE fill_ghosts (so their ghosts are filled) and the injection (so they reach every level).
    // No-op when no named field was set; field_postprocess wrote only comps 0..2, so this never clobbers
    // phi/grad. This is what makes named aux survive a regrid (regrid re-solves -> re-applies).
    apply_named_aux();
    fill_ghosts_profiled(aux_[0], dom_, aux_bc_);
    apply_named_aux_bc();  // ADC-369: per-field halo override on the coarse physical ghosts (after the
                           // shared fill, before injection); no-op when no policy declared.
    // 4. coarse->fine injection of the aux (parent replicated only at level 1 if coarse replicated).
    for (int k = 1; k < nlev_; ++k)
      detail::coupler_inject_aux_mb(aux_[k - 1], aux_[k],
                                    /*replicated_parent=*/(k == 1) && replicated_coarse_);

    // 5. NAMED multi-elliptic fields (ADC-428): a SECOND elliptic solve per user-named field, written to
    // the field's OWN aux components and injected to the fine levels. No-op when none is registered ->
    // the default-Poisson trajectory above is strictly bit-identical.
    solve_named_fields();
  }

  /// Solves every registered NAMED elliptic field (ADC-428) on the coarse, writes phi (+ centered grad)
  /// into the field's own aux components, ghost-fills them and injects coarse->fine. Mirror of the
  /// default Poisson block above (steps 2-4) but per named field, reusing a DEDICATED GeometricMG. The
  /// default phi/grad (comps 0..2) are never touched. No-op (early return) without a named field, so the
  /// default-only path stays bit-identical.
  void solve_named_fields() {
    if (named_fields_.empty())
      return;
    const Real dx = geom_.dx(), dy = geom_.dy();
    for (auto& [field, nf] : named_fields_) {
      if (nf.phi_comp < 0 || nf.phi_comp >= aux_ncomp_)
        throw std::runtime_error("AmrRuntime : named elliptic field '" + field +
                                 "' aux component out of the channel width (add the block that "
                                 "declares its aux fields)");
      ensure_named_elliptic(nf);
      // SUMMED + CO-LOCATED RHS on the coarse: f = Sum_b named_elliptic_rhs_b[field](U_b), exactly like
      // the default Poisson (mg_.rhs()), but reading the per-field block closures. A field with no
      // contributing block solves a zero RHS -> reject loud (mirror of the System named path).
      MultiFab& rhs = nf.mg->rhs();
      rhs.set_val(Real(0));
      bool any = false;
      for (auto& b : blocks_) {
        auto it = b.named_elliptic_rhs.find(field);
        if (it == b.named_elliptic_rhs.end() || !it->second)
          continue;
        it->second((*b.levels)[0].U, rhs);
        any = true;
      }
      if (!any)
        throw std::runtime_error(
            "AmrRuntime : named elliptic field '" + field +
            "' has no contributing block (declare m.elliptic_field on the block model)");
      nf.mg->solve();
      device_fence();  // CRITICAL: the V-cycle must finish before phi is read (same invariant as mg_)
      // Write phi (+ centered grad) into the field's OWN aux components on the coarse valid cells. The
      // default field_postprocess hardcodes comps 0..2, so we write the named comps with a dedicated
      // loop (mirror of SystemFieldSolver::solve_named_field_from_state). Per-local-fab (MPI-safe).
      MultiFab& phi_mf = nf.mg->phi();
      const int cphi = nf.phi_comp, cgx = nf.gx_comp, cgy = nf.gy_comp;
      const bool grad = (cgx >= 0 && cgx < aux_ncomp_ && cgy >= 0 && cgy < aux_ncomp_);
      for (int li = 0; li < aux_[0].local_size(); ++li) {
        const ConstArray4 p = phi_mf.fab(li).const_array();
        Array4 a = aux_[0].fab(li).array();
        const Box2D v = aux_[0].box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i) {
            a(i, j, cphi) = p(i, j);
            if (grad) {
              a(i, j, cgx) = (p(i + 1, j) - p(i - 1, j)) / (2 * dx);
              a(i, j, cgy) = (p(i, j + 1) - p(i, j - 1)) / (2 * dy);
            }
          }
      }
    }
    // Ghost-fill the named components (shared aux fill: same routing as the default) + per-field halo
    // override (ADC-369), then inject coarse->fine so the named field reaches every level. We re-fill the
    // WHOLE aux: the default comps 0..2 were just written by the Poisson block, so their valid cells are
    // unchanged -- only ghosts are refreshed (idempotent). Cheap (one extra fill per solve_fields when a
    // named field exists; none otherwise).
    fill_ghosts_profiled(aux_[0], dom_, aux_bc_);
    apply_named_aux_bc();
    for (int k = 1; k < nlev_; ++k)
      detail::coupler_inject_aux_mb(aux_[k - 1], aux_[k],
                                    /*replicated_parent=*/(k == 1) && replicated_coarse_);
  }

  /// UNION-TAGS REGRID (capstone Phase 2, C.6; docs/AMR_REGRID_UNION_TAGS_DESIGN.md, steps R0-R8).
  /// Re-grids the SHARED hierarchy from the UNION (cell-by-cell OR) of the tags of ALL blocks (per-block
  /// predicate, D1) + the phi tags (on |grad phi|, D4), followed by ONE SINGLE Berger-Rigoutsos
  /// clustering -> ONE SINGLE new fine layout applied to ALL blocks (including those held by their
  /// stride, D3) AND to the shared aux. Maintains the shared-layout PRECONDITION (same_layout_or_throw)
  /// after the regrid. v1 with 2 LEVELS (coarse + 1 fine, D5): no-op if nlev < 2. No-op (grid
  /// unchanged) if the union of the tags is empty (nothing to refine).
  void regrid() {
    if (nlev_ < 2)
      return;  // 2 levels required (D5): nothing to re-grid in single-level
    const int fk = nlev_ - 1, pk = fk - 1;  // fine + its parent (pk == 0 in v1 with 2 levels)

    // AMR PROFILING (Spec 5 criterion 43): time the WHOLE regrid attempt (tag + cluster + prolong +
    // re-solve) into the "regrid" scope. RAII -> the scope covers EVERY early-return path below (empty
    // tags / nothing to refine), so the timing reflects the real regrid cost. The per-run "regrid"
    // COUNT is bumped at the tail (++regrid_count_) only when a regrid actually completed -- a no-op
    // attempt times itself but does not inflate the count. Null/disabled profiler -> no scope object.
    auto _rg = profile_amr_scope("regrid");

    // AMR PROFILING (ADC-607): baseline the process-wide parallel_copy schedule counters so we can
    // attribute this regrid's BoxHash rebuilds + copy-cache hits/misses (the R6 prolong/restrict and
    // R8 re-solve replay parallel_copy). A miss builds a fresh schedule (== one BoxHash rebuild); a
    // hit reuses a memoized plan. Sampled only when profiling; zero cost otherwise.
    const std::int64_t copy_miss_before = copy_schedule_miss_count();
    const std::int64_t copy_hit_before = copy_schedule_hit_count();

    // (R0) PRECONDITION: fields up to date (aux per level, for the |grad phi| criterion). The per-block
    // mass snapshot is NOT needed by the engine (conservation is checked test-side V1).
    solve_fields();

    // (R1)+(R2) PER-BLOCK TAGS (on the block U at the parent level) + PHI TAGS (on the shared aux).
    const int PNX = dom_.nx() << pk, PNY = dom_.ny() << pk;
    const Box2D pdom = Box2D::from_extents(PNX, PNY);
    std::vector<TagBox> parts;
    parts.reserve(blocks_.size() + 1);
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
      const TagPredicate& crit = block_tag_[b];
      if (!crit)
        continue;  // block without a criterion: tags nothing on its side (re-gridded as background)
      parts.push_back(tag_cells((*blocks_[b].levels)[pk].U, pdom, crit));
    }
    if (phi_tag_)
      parts.push_back(tag_cells(aux_[pk], pdom, phi_tag_));
    if (parts.empty())
      return;  // no active criterion -> no tagged cell -> grid unchanged

    // (R3) UNION (OR) of the tags + dilation (nesting + anticipation of the structures moving).
    TagBox grown = grow_tags(tag_union(parts), regrid_grow_, pdom);

    // AMR PROFILING (ADC-607): tag density = tagged cells / total parent cells (x1000, integer
    // permille). Records how full the DENSE TagBox is -- the decision to keep TagBox dense (a sparse
    // grid would degrade Berger-Rigoutsos on a high-density front) is measured, not assumed. count()
    // and box.num_cells() are the same dense buffer this regrid already walks; no extra sweep.
    if (profiler_ != nullptr) {
      const std::int64_t total = grown.box.num_cells();
      if (total > 0)
        profiler_->count("tag_density", (grown.count() * 1000) / total);
    }

    // (R4)+(R5) cross-rank collective reduction (if coarse distributed) + UNIQUE clustering -> SHARED
    // fine layout. all_reduce_or_inplace is called INSIDE regrid_compute_fine_layout for distributed
    // pk==0: all ranks start from the SAME tag grid -> IDENTICAL fb/dmap per rank (otherwise MPI
    // desync).
    auto [fb, dmap] =
        regrid_compute_fine_layout(std::move(grown), pdom, pk, regrid_margin_, replicated_coarse_);
#ifdef POPS_HAS_MPI
    // MPI COLLECTIVE COUNT (Spec 5 criterion 43): regrid_compute_fine_layout issues ONE
    // all_reduce_or_inplace over the tag grid when the coarse is distributed (multi-rank) -- every rank
    // must cluster from the SAME gathered tags. Count it as one reduction (np>1 only; serial / single
    // rank issues no collective). np==1 is bit-identical with no count.
    if (profiler_ != nullptr && n_ranks() > 1)
      profiler_->count("mpi_reductions");
#endif
    if (fb.size() == 0)
      return;  // nothing to refine: we keep the current grid (no-op)

    // (R6) COHERENT PROLONG / RESTRICT of ALL blocks on the SAME fb/dmap (including the blocks held by
    // their stride: their frozen state is present everywhere and contributes to the Poisson, D3). The
    // ghost width is INHERITED per block (a MUSCL order-2 block carries 2 ghosts; a Minmod block and a
    // VanLeer one may differ), so the scheme does not read out of bounds at the next step (V2 / risk
    // X4).
    for (auto& b : blocks_) {
      auto& L = *b.levels;
      const int ngf = L[fk].U.n_grow();
      L[fk].U = regrid_field_on_layout(fb, dmap, L[pk].U, L[fk].U, pk, ngf, replicated_coarse_);
    }

    // (R7) REBUILD OF THE SHARED AUX (one only, width aux_ncomp_) on the new layout + RE-WIRING of the
    // aux pointer of EACH block. The address &aux_[fk] stays stable (in-place reallocation of the
    // MultiFab in the existing std::vector) -> the pointers of the other levels do not move.
    aux_[fk] = MultiFab(fb, dmap, aux_ncomp_, 1);
    for (auto& b : blocks_)
      (*b.levels)[fk].aux = &aux_[fk];

    // (V3) SHARED-LAYOUT INVARIANT: all blocks MUST live on EXACTLY the same fb/dmap (boxes, order,
    // rank per box) after the regrid. Collective guard (cross-block); catches any inconsistent
    // reconstruction before it corrupts the shared aux / the summed Poisson.
    {
      std::vector<std::vector<AmrLevelMP>> ref;
      ref.reserve(blocks_.size());
      for (const auto& b : blocks_)
        ref.push_back(*b.levels);
      detail::same_layout_or_throw(ref);
    }

    // (R8) RESTORATION OF THE COVERAGE INVARIANT: re-solve so that phi / grad phi are consistent with
    // the new grid AND to trigger the fine -> coarse cascade (mf_average_down_mb, in solve_fields) that
    // restores the covered coarse cells (otherwise a mass diagnostic, sum of the coarse only, would
    // count a phantom coarse value under the new patch, X5).
    solve_fields();
    ++regrid_count_;
    // AMR PROFILING (Spec 5 criterion 43): a regrid COMPLETED -> bump the per-run "regrid" counter
    // (parity with regrid_count_). The "regrid" TIMING scope (_rg above) already covered the whole
    // attempt; this counts only the regrids that actually rebuilt the hierarchy.
    if (profiler_ != nullptr)
      profiler_->count("regrid");
    // AMR PROFILING (ADC-607): attribute this regrid's parallel_copy schedule work. A miss is one
    // BoxHash rebuild (a fresh schedule enumeration); box_hash_rebuilds should stay small (one per
    // distinct layout pair the R6/R8 copies touch), so a growing count flags a cache that is not
    // engaging. Deltas over the whole regrid body (baseline sampled at the head).
    if (profiler_ != nullptr) {
      const std::int64_t misses = copy_schedule_miss_count() - copy_miss_before;
      const std::int64_t hits = copy_schedule_hit_count() - copy_hit_before;
      profiler_->count("box_hash_rebuilds", misses);
      profiler_->count("copy_cache_misses", misses);
      profiler_->count("copy_cache_hits", hits);
    }
  }

  /// Advances the system by one macro-step dt. We first solve the fields (co-located summed Poisson,
  /// ONCE per macro-step: OncePerStep cadence), then each block advances over ITS level stack with ITS
  /// scheme, honoring its stride cadence and its substeps, and ITS temporal treatment. Runtime
  /// counterpart of AmrSystemCoupler::step (OncePerStep): the compile-time version carries
  /// substeps/stride in block_substeps_v / block_stride_v and chooses the treatment by the constexpr
  /// block_time_treatment_v; here the engine carries the substep loop, the stride filter AND the
  /// IMEX-vs-explicit selection.
  ///
  /// TREATMENT SELECTION (capstone vii):
  ///  - EXPLICIT block (b.imex == false): the advance closure does ONE advance_amr (transport +
  ///    forward-Euler source), called substeps times;
  ///  - IMEX block (b.imex == true): the imex_advance closure does ONE SOURCE-FREE advance_amr then the
  ///    IMPLICIT stiff source backward_euler_source per level + cascade (cf.
  ///    AmrRuntimeBlock::imex_advance), called substeps times. Unconditionally stable on a stiff
  ///    relaxation (where the explicit, of factor |1 - dt/eps|, DIVERGES as soon as dt > 2 eps).
  /// The substep loop is COMMON to both treatments (substeps applications of h = bdt/substeps), so the
  /// runtime also SUB-CYCLES the IMEX splitting. At substeps=1 this sub-cycling is a no-op and the IMEX
  /// path coincides with the IMEX branch of the compile-time engine AmrSystemCoupler::step; for
  /// substeps>1 it DIVERGES deliberately from that engine (which itself ignores substeps on its IMEX
  /// branch): see IMEX SEMANTICS UNDER substeps in the header (CFL-safe on the transport,
  /// backward-Euler stable at any step, stiff relaxation more accurate). imex == false everywhere ->
  /// advance path only -> bit-identical trajectory to the historical one (the IMEX is opt-in).
  void step(Real dt) {
    solve_count_ = 0;
    // UNION-TAGS REGRID (capstone Phase 2, C.6; D2: BEFORE the macro-step's step, consistent with the
    // single-block amr_dsl_block.hpp:108). regrid_every_ cadence in MACRO-STEPS, OUTSIDE the substep
    // loops and the stride windows (macro-step granularity ONLY, D3). regrid_every_ == 0 -> FROZEN
    // hierarchy, regrid never called -> BIT-IDENTICAL trajectory to the historical one. The guard
    // macro_step_ > 0 (like the single-block) avoids a regrid at the very first step (the initial grid
    // is already the build one). The regrid sits BEFORE solve_fields below: it does its own
    // solve_fields (R0/R8), then the step's solve_fields recomputes phi on the re-gridded grid.
    if (regrid_every_ > 0 && macro_step_ > 0 && macro_step_ % regrid_every_ == 0)
      regrid();
    // System Poisson solved ONCE on the current state (OncePerStep cadence). A HELD block (stride > 1,
    // outside end-of-window) contributed with its FROZEN state since its last advance: loose coupling
    // assumed by the multirate, exactly like System::step / AmrSystemCoupler in OncePerStep. phi stays
    // frozen during the blocks' advance (no per-substep re-solve here). When reached from step_cfl this
    // re-solves an unchanged state (a second solve), kept on purpose; see the ADC-318 note in step_cfl.
    solve_fields();
    for (auto& b : blocks_) {
      // HOLD-THEN-CATCH-UP cadence (cf. AmrRuntimeBlock::stride, #140): the block is HELD as long as
      // (macro_step_+1) % stride != 0, then CATCHES UP at end-of-window by an effective step stride*dt.
      // The end-of-window catch-up keeps the block temporally consistent with the fast ones at the
      // coupling point (never in the future). stride=1: always true -> every step, bit-identical.
      if ((macro_step_ + 1) % b.stride != 0)
        continue;
      // NEWTON DIAGNOSTICS (OPT-IN): RESET of the report at the HEAD of the block advance (parity with
      // System::AdvanceImex::operator() which resets nreport before its substep loop). The report then
      // AGGREGATES over all the levels AND substeps of THIS advance (imex_advance accumulates per level
      // via backward_euler_source; step() calls imex_advance substeps times without re-resetting).
      // Placed AFTER the stride skip: a HELD block keeps the report of its LAST advance ("last advance"
      // semantics of System). No-op for a block without diagnostics (newton_report null).
      if (b.newton_diagnostics && b.newton_report)
        b.newton_report->reset();
      const Real bdt = dt * static_cast<Real>(b.stride);  // catch-up: effective step stride*dt
      // substeps equal substeps of bdt/substeps. The chosen closure does ONE advance per call;
      // substeps=1 -> a single advance of bdt (bit-identical to the single-substep case). Per-block
      // treatment SELECTION: IMEX (source-free transport + implicit stiff source, mirrors the IMEX
      // branch of AmrSystemCoupler::step) if b.imex, otherwise EXPLICIT (transport + forward-Euler
      // source). The test is PER BLOCK and stable: a single IMEX block changes nothing for the
      // neighboring explicit blocks.
      // NOTE substeps>1: the loop below calls step_block substeps times for BOTH treatments, so the
      // IMEX splitting is SUB-CYCLED (K Lie steps over bdt/K). The compile-time, for its part, applies
      // its IMEX only once over bdt (it ignores substeps on its IMEX branch): divergence INTENTIONAL
      // and sound for substeps>1 (cf. IMEX SEMANTICS UNDER substeps in the file header).
      const Real h = bdt / static_cast<Real>(b.substeps);
      auto& step_block = b.imex ? b.imex_advance : b.advance;
      for (int s = 0; s < b.substeps; ++s)
        step_block(*b.levels, dom_, h, base_per_, replicated_coarse_);
      // PROJECTION PONCTUELLE post-pas (ADC-177) : par niveau, APRES substeps + reflux/cascade.
      // Cell-local + idempotente -> conservation preservee (flux-registres deja regles). No-op si vide.
      if (b.project_per_level)
        b.project_per_level(*b.levels);
    }
    // Inter-species coupled sources AFTER the transport (same order as AmrSystemCoupler: transport then
    // coupled_source_step), by forward-Euler splitting. No-op if no source registered -> bit-identical
    // trajectory to the historical one (the feature is opt-in).
    coupled_source_step(dt);
    ++macro_step_;
  }

  /// substeps/stride-aware CFL step (runtime counterpart of System::step_cfl, EXACT mirror of its
  /// formula). A block of stride cadence advances by an effective step stride*dt in substeps substeps,
  /// so each substep is worth stride*dt/substeps; the per-substep stability condition
  /// stride*dt/substeps <= cfl*h/w_b gives dt <= cfl*h*substeps_b/(stride_b*w_b). The GLOBAL dt is the
  /// min over the blocks (the most constraining). We first solve the fields (per-block max_speed
  /// requires the aux up to date), compute dt, then advance by one step(dt). @p h = coarse mesh spacing
  /// (dx_coarse). Returns the dt used. Single-block (a single block, stride=1): if w_b is the only
  /// constraining one, dt = cfl*h*substeps/w (identical to System::step_cfl single-block).
  Real step_cfl(Real cfl, Real h) {
    const Real dt = cfl_dt(cfl, h);
    step(dt);
    return dt;
  }

  /// The CFL dt computation of @ref step_cfl WITHOUT the trailing advance (no step(dt)): solves the
  /// fields (max_speed needs the aux), scans the per-block transport / source / stability bounds + the
  /// coupled-frequency + global bounds, and returns the macro-step dt (records last_dt_reason_). Split
  /// out so an installed compiled Program can take the SAME CFL dt and drive the macro-step itself
  /// (AmrSystem::step_cfl's Program route, parity SystemStepper::step_cfl) instead of the native step.
  /// The native @ref step_cfl path is byte-identical (it is this body + step(dt)).
  Real cfl_dt(Real cfl, Real h) {
    // NOTE (ADC-318): this pre-solve plus step(dt)'s own head solve below is a DOUBLE Poisson solve on
    // the SAME unchanged state (regrid_every=0 freezes the grid in between). It looks redundant but is
    // NOT, and is INTENTIONALLY kept. GeometricMG::solve() is warm-started and iterates to a RELATIVE
    // tolerance (rel_tol 1e-8; abs_tol 0 by default, so its off-step early-exit never fires here), so the
    // second solve does not recompute identical phi: starting from the first solve's iterate it
    // over-converges it by ~rel_tol. Skipping the second solve would therefore NOT be bit-identical; it
    // drifts the trajectory by ~3e-10 over 20 steps (below the solver tolerance and far below the O(dt^2)
    // scheme error, but nonzero). The de-dup was declined to preserve the exact historical bit-stream
    // (SystemStepper::step_cfl avoids the double solve by INLINING its advance, not by skipping a solve).
    solve_fields();  // aux up to date: each block's max_speed reads it on the current coarse
    Real dt = std::numeric_limits<Real>::infinity();
    last_dt_reason_ = "degenerate";
    for (auto& b : blocks_) {
      const Real w = std::max(b.max_speed((*b.levels)[0].U, aux_[0]), kCflSpeedFloor);
      Real dt_b = cfl * h * static_cast<Real>(b.substeps) / (static_cast<Real>(b.stride) * w);
      const char* why = "transport";
      // OPTIONAL block BOUNDS (AMR StabilityPolicy, audit 2026-06): same substeps/stride formulas as
      // SystemStepper::step_cfl, evaluated on the COARSE. Empty closures (model without the trait) ->
      // not queried, transport bound only (bit-identical).
      if (b.source_frequency) {
        const Real mu = b.source_frequency((*b.levels)[0].U, aux_[0]);
        if (mu > Real(0)) {
          const Real dt_src =
              cfl * static_cast<Real>(b.substeps) / (static_cast<Real>(b.stride) * mu);
          if (dt_src < dt_b) {
            dt_b = dt_src;
            why = "source_frequency";
          }
        }
      }
      if (b.stability_dt) {
        const Real db = b.stability_dt((*b.levels)[0].U, aux_[0]);
        if (db > Real(0)) {
          const Real dt_adm = db * static_cast<Real>(b.substeps) / static_cast<Real>(b.stride);
          if (dt_adm < dt_b) {
            dt_b = dt_adm;
            why = "stability_dt";
          }
        }
      }
      if (dt_b < dt) {
        dt = dt_b;
        last_dt_reason_ = std::string(why) + ":" + b.name;
      }
    }
    // Declared frequencies of the coupled sources (CoupledSource.frequency): bound on the MACRO-step
    // (the couplings apply once per macro-step), dt <= cfl / mu, without substeps/stride.
    for (const auto& cs : coupled_freqs_) {
      const Real dt_cs = cfl / cs.mu;
      if (dt_cs < dt) {
        dt = dt_cs;
        last_dt_reason_ = "coupled_source:" + cs.label;
      }
    }
    // PER-CELL frequencies (CoupledSource.frequency with an Expr): mu(U) reduced (MAX) on the COARSE
    // level of the input blocks (where the AMR CFL lives), GLOBAL all_reduce_max (ALL ranks, neutral
    // without a local box), bound dt <= cfl / max(mu). Same reason "coupled_source:<label>" as the
    // constant. No per-cell source -> empty loop (bit-identical). The Array4 are rebuilt at EACH step
    // (the hierarchy fabs are repointed by the regrid), like coupled_source_step.
    for (const auto& ce : coupled_freq_exprs_) {
      Real m = 0;
      if (ce.n_in > 0) {
        auto& Uref =
            (*blocks_[static_cast<std::size_t>(ce.ins[0].block)].levels)[0].U;  // coarse (lev 0)
        for (int li = 0; li < Uref.local_size(); ++li) {
          CoupledFreqKernel kern;
          kern.n_in = ce.n_in;
          kern.n_const = ce.n_const;
          for (int c = 0; c < ce.n_in; ++c) {
            kern.in[c] =
                (*blocks_[static_cast<std::size_t>(ce.ins[static_cast<std::size_t>(c)].block)]
                      .levels)[0]
                    .U.fab(li)
                    .array();
            kern.in_comp[c] = ce.ins[static_cast<std::size_t>(c)].comp;
          }
          for (int c = 0; c < ce.n_const; ++c)
            kern.consts[c] = ce.kconsts[static_cast<std::size_t>(c)];
          kern.prog = ce.prog;
          m = std::max(m, reduce_max_cell(Uref.box(li), kern));
        }
      } else {
        // Program WITHOUT an input field (constant in bytecode): evaluated once on the constants.
        Real reg[kCsMaxReg];
        for (int c = 0; c < ce.n_const; ++c)
          reg[c] = ce.kconsts[static_cast<std::size_t>(c)];
        const Real mu0 = ce.prog.eval(reg);
        if (mu0 > Real(0))
          m = mu0;
      }
      const double mu = all_reduce_max(static_cast<double>(m));  // ALL ranks (collective symmetry)
#ifdef POPS_HAS_MPI
      // MPI COLLECTIVE COUNT (Spec 5 criterion 43): one all_reduce_max per per-cell coupled-frequency
      // bound, multi-rank only (serial all_reduce_max is an identity, no collective).
      if (profiler_ != nullptr && n_ranks() > 1)
        profiler_->count("mpi_reductions");
#endif
      if (mu > 0.0) {
        const Real dt_cs = cfl / static_cast<Real>(mu);
        if (dt_cs < dt) {
          dt = dt_cs;
          last_dt_reason_ = "coupled_source:" + ce.label;
        }
      }
    }
    // GLOBAL bounds (AmrRuntime::add_dt_bound, parity with System::add_dt_bound): evaluated PER RANK
    // then reduced all_reduce_min (dt identical on all ranks; <= 0/non-finite = inert).
    for (const auto& g : dt_bounds_) {
      if (!g.fn)
        continue;
      double v = g.fn();
      if (!(v > 0.0) || !std::isfinite(v))
        v = std::numeric_limits<double>::infinity();
      v = all_reduce_min(v);
#ifdef POPS_HAS_MPI
      // MPI COLLECTIVE COUNT (Spec 5 criterion 43): one all_reduce_min per registered global dt bound,
      // multi-rank only (the global min keeps dt identical on all ranks). Serial -> identity, no count.
      if (profiler_ != nullptr && n_ranks() > 1)
        profiler_->count("mpi_reductions");
#endif
      if (static_cast<Real>(v) < dt) {
        dt = static_cast<Real>(v);
        last_dt_reason_ = "global:" + g.label;
      }
    }
    if (!std::isfinite(dt)) {
      dt = cfl * h / kCflSpeedFloor;  // guard (no block: impossible here)
      last_dt_reason_ = "degenerate";
    }
    return dt;
  }

  /// MACRO-STEP counter of the engine (regrid + hold-then-catch-up stride cadence: regrid when
  /// macro_step_ % regrid_every == 0, stride catch-up when (macro_step_+1) % stride == 0).
  int macro_step() const { return macro_step_; }
  /// RESTORES the macro-step counter (IO v1, reserved for restart via AmrSystem::set_clock): without
  /// it the regrid/stride cadence would restart from phase 0 after a resume. No effect on the level
  /// state; only sets the cadence phase.
  void set_macro_step(int s) { macro_step_ = s; }

  /// AMR / MPI PROFILING SEAM (Spec 5 sec.12.5, ADC-479 criterion 43). The AmrSystem owns the
  /// runtime::program::Profiler (parity with System::profiler_) and wires it in here AFTER build, so
  /// the engine times its non-numeric AMR phases -- regrid, fill_boundary (the cross-rank ghost
  /// exchange), average_down (fine -> coarse restriction) -- into the SAME table profile_report()
  /// renders, alongside the coarse step / field_solve phases. The pointer is null by default (the
  /// engine never touches it), and every scope/count is guarded by profiler_->enabled(), so a run
  /// WITHOUT profiling pays ZERO cost (no scope object, no clock read) -- the granularity is
  /// per-regrid / per-solve, NOT per-cell. Passing nullptr detaches the profiler (no-op timing).
  void set_profiler(runtime::program::Profiler* prof) { profiler_ = prof; }

  /// GLOBAL step bound (AMR counterpart of System::add_dt_bound): fn() evaluated once per step_cfl,
  /// all_reduce_min, <= 0/non-finite = inert. For user coupling/scheduler/policies.
  void add_dt_bound(const std::string& label, std::function<double()> fn) {
    dt_bounds_.push_back(GlobalDtBound{label, std::move(fn)});
  }

  /// DECLARED frequency of a coupled source (CoupledSource.frequency, wave-3 audit): step bound
  /// dt <= cfl / mu on the MACRO-step (the couplings apply once per macro-step). mu <= 0 = inert (no
  /// bound).
  void add_coupled_frequency(const std::string& label, Real mu) {
    if (mu > Real(0))
      coupled_freqs_.push_back(CoupledFreqDecl{label, mu});
  }

  /// PER-CELL COUPLED frequency (CoupledSource.frequency with an Expr, refinement of the CONSTANT
  /// frequency above): a bytecode program mu(U) on the SAME register table as the source (inputs
  /// in_blocks/in_roles then constants consts). Evaluated at each step_cfl on the COARSE level of the
  /// input blocks (where the AMR CFL lives: h = dx_coarse), MAX reduction + global all_reduce_max,
  /// bound dt <= cfl / max(mu) on the macro-step. The bound is thus evaluated on the COARSE (not on the
  /// fine patches): consistent with the AMR transport CFL, but a local under-estimate of mu under a
  /// fine patch is not seen (assumed choice, documented). Empty program -> ignored (no bound). Form
  /// validation (opcodes / register bounds) and STRICT role resolution, like add_coupled_source.
  void add_coupled_frequency_expr(const std::string& label,
                                  const std::vector<std::string>& in_blocks,
                                  const std::vector<std::string>& in_roles,
                                  const std::vector<double>& consts,
                                  const std::vector<int>& freq_prog_ops,
                                  const std::vector<int>& freq_prog_args) {
    if (freq_prog_ops.empty() && freq_prog_args.empty())
      return;  // no per-cell frequency
    const int n_in = static_cast<int>(in_blocks.size());
    const int n_const = static_cast<int>(consts.size());
    if (static_cast<int>(in_roles.size()) != n_in)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : in_blocks / in_roles of different sizes");
    if (n_in + n_const > kCsMaxReg)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : too many registers (inputs + constants > " +
          std::to_string(kCsMaxReg) + ")");
    if (freq_prog_ops.size() != freq_prog_args.size())
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : freq_prog_ops / freq_prog_args of different "
          "sizes");
    if (static_cast<int>(freq_prog_ops.size()) > kCsMaxProg)
      throw std::runtime_error(
          "AmrRuntime::add_coupled_frequency_expr : frequency program too long (> " +
          std::to_string(kCsMaxProg) + ")");
    // Resolves (block, role) -> (block index, component), STRICT (mirror of add_coupled_source).
    std::vector<CsRef> ins(static_cast<std::size_t>(n_in));
    for (int c = 0; c < n_in; ++c) {
      const std::string& block = in_blocks[static_cast<std::size_t>(c)];
      const std::string& role = in_roles[static_cast<std::size_t>(c)];
      const int b = block_index(block);
      if (b < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_frequency_expr : no block named '" +
                                 block + "'");
      // Role addressed BY NAME: a canonical role name OR a user-defined role label (ADC-292), STRICT.
      const VariableSet& vs = blocks_[static_cast<std::size_t>(b)].cons_vars;
      const int comp = vs.index_of(role);
      if (comp < 0)
        throw std::runtime_error("AmrRuntime::add_coupled_frequency_expr : block '" + block +
                                 "' does not expose role '" + role + "' (roles: " +
                                 (vs.roles.empty() ? std::string("<none>") : roles_csv(vs)) +
                                 ", no silent fallback to component 0)");
      ins[static_cast<std::size_t>(c)] = {b, comp, CsProgram{}};
    }
    CsProgram pg;
    pg.len = static_cast<int>(freq_prog_ops.size());
    for (int k = 0; k < pg.len; ++k) {
      const int opc = freq_prog_ops[static_cast<std::size_t>(k)];
      const int a = freq_prog_args[static_cast<std::size_t>(k)];
      if (opc < 0 || opc > static_cast<int>(CsOp::Sqrt))
        throw std::runtime_error(
            "AmrRuntime::add_coupled_frequency_expr : invalid opcode in the frequency");
      if (opc == static_cast<int>(CsOp::PushReg) && (a < 0 || a >= n_in + n_const))
        throw std::runtime_error(
            "AmrRuntime::add_coupled_frequency_expr : register out of bounds in the frequency");
      pg.op[k] = opc;
      pg.arg[k] = a;
    }
    validate_cs_program_stack(pg, "AmrRuntime::add_coupled_frequency_expr");
    std::vector<Real> kconsts(consts.begin(), consts.end());
    coupled_freq_exprs_.push_back(
        CoupledFreqExprDecl{label, std::move(ins), pg, n_in, n_const, std::move(kconsts)});
  }

  /// ACTIVE bound of the last step_cfl ("transport:<block>" / "source_frequency:<block>" /
  /// "stability_dt:<block>" / "global:<label>" / "degenerate" / "" before the first step).
  const std::string& last_dt_bound() const { return last_dt_reason_; }

  /// NEWTON REPORT (OPT-IN IMEX diagnostics) of block @p name, AGGREGATED over the levels and substeps
  /// of its LAST advance (cf. AmrRuntimeBlock::newton_report). AMR counterpart of System::newton_report.
  /// @throws std::runtime_error if the block is unknown, or if it was not added with
  ///         newton_diagnostics=true / newton_fail_policy warn|throw (no silently empty report).
  const NewtonReport& newton_report(const std::string& name) const {
    const int b = block_index(name);
    if (b < 0)
      throw std::runtime_error("AmrRuntime::newton_report : no block named '" + name + "'");
    const AmrRuntimeBlock& blk = blocks_[static_cast<std::size_t>(b)];
    if (!blk.newton_diagnostics || !blk.newton_report)
      throw std::runtime_error(
          "AmrRuntime::newton_report : Newton diagnostics not enabled for block '" + name +
          "' ; add the block with newton_diagnostics=True "
          "(pops.IMEX(newton_diagnostics=True)) or newton_fail_policy='warn'/'throw'");
    return *blk.newton_report;
  }

  /// Coarse potential (component 0 of the shared aux) as an n*n row-major field. Solves the fields if
  /// needed (counterpart of AmrSystem::potential), then reads aux(0). Identical for all blocks.
  std::vector<double> potential() {
    solve_fields();
    return blocks_[0].potential(aux_[0]);
  }

  /// Max SYSTEM wave speed (max over the blocks) on the current coarse. Requires the aux up to date.
  Real max_speed() {
    solve_fields();
    Real w = Real(1e-12);
    for (auto& b : blocks_) {
      const Real wb = b.max_speed((*b.levels)[0].U, aux_[0]);
      if (wb > w)
        w = wb;
    }
    return w;
  }

  int n_patches() const {
    const auto& L = *blocks_[0].levels;
    return L.size() >= 2 ? static_cast<int>(L[1].U.box_array().size()) : 0;
  }

  // Index-space signatures of the fine patches (level + inclusive lo/hi corners), for ALL fine levels.
  // Read-only of the GLOBAL BoxArray (all boxes/all ranks) already stored -> rank-independent, zero
  // communication, NO hot-path cost (query between steps). Mirror of n_patches(): the same box_array()
  // that gives the COUNT gives the BOXES. Block 0 representative (SHARED layout, same_layout_or_throw
  // guard). Loop k = 1..nlev-1: a single fine level today (ratio 2), correct if a future adds levels
  // (the level field disambiguates the spacing dx = L / (n << level) Python-side).
  std::vector<PatchBox> patch_boxes() const {
    const auto& L = *blocks_[0].levels;
    std::vector<PatchBox> out;
    for (int k = 1; k < static_cast<int>(L.size()); ++k) {
      const auto& bxs = L[k].U.box_array().boxes();
      for (const Box2D& b : bxs)
        out.push_back(PatchBox{k, b.lo[0], b.lo[1], b.hi[0], b.hi[1]});
    }
    return out;
  }

  // COARSE-level (base) box counts (ADC-319, MPI ownership diagnostic). Block 0 is the SHARED layout
  // (same_layout_or_throw), so its level-0 MultiFab carries the base BoxArray + DistributionMapping
  // common to all blocks. local_size() = base boxes OWNED by this rank; box_array().size() = total base
  // boxes (all ranks). Mirror of n_patches(): a query between steps, no communication, no hot-path cost.
  int coarse_local_boxes() const { return (*blocks_[0].levels)[0].U.local_size(); }
  int coarse_total_boxes() const { return (*blocks_[0].levels)[0].U.box_array().size(); }

  // ----------------------------------------------------------------------------------------------
  // MULTI-BLOCK AMR CHECKPOINT / RESTART (ADC-509). PER-BLOCK PER-LEVEL state accessors + the
  // level-0 phi (multigrid warm-start), counterpart of AmrCouplerMP::level_state on the SHARED
  // hierarchy. The shared layout is FROZEN at build (make_shared_amr_layout: a deterministic central
  // fine patch, regrid_every==0): replaying the SAME composition reproduces the SAME hierarchy, so a
  // restart only needs to restore each block's valid cells + phi (no set_hierarchy on the runtime).
  // The _global variants all_reduce_sum the per-rank LOCAL fabs into the complete field (np>1
  // gather), MIRROR of System::state_global / gather_global; mono-rank they are the identity. @p b:
  // block index, @p k: level (0 = coarse, >= 1 = fine).
  // ----------------------------------------------------------------------------------------------

  // Conserved components of block @p b (Model::n_vars, carried by the AmrRuntimeBlock).
  int block_n_vars(std::size_t b) const {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::block_n_vars : block index out of bounds");
    return blocks_[b].ncomp;
  }

  // FULL conservative state (all components) of block @p b at level @p k, flat component-major
  // c*nf*nf + j*nf + i (nf = nx << k); zeros outside the patches at the fine level. LOCAL fabs only
  // (no gather): the facade calls this mono-rank. Mirror of AmrCouplerMP::level_state.
  std::vector<double> block_level_state(std::size_t b, int k) const {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::block_level_state : block index out of bounds");
    const std::vector<AmrLevelMP>& L = *blocks_[b].levels;
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrRuntime::block_level_state : level out of bounds");
    const MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const std::size_t nf = static_cast<std::size_t>(dom_.nx()) << k;
    std::vector<double> out(static_cast<std::size_t>(nc) * nf * nf, 0.0);
    device_fence();
    fill_level_state(U, nc, nf, out);
    return out;
  }

  // Same as block_level_state but all_reduce_sum the per-rank contributions -> every rank holds the
  // complete field (np>1 gather, AMR reflux pattern). COLLECTIVE: all ranks MUST call it.
  std::vector<double> block_level_state_global(std::size_t b, int k) const {
    std::vector<double> out = block_level_state(b, k);
    all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
    return out;
  }

  // Restores block @p b at level @p k from @p s (same layout as block_level_state). Writes ONLY the
  // VALID cells of the local fabs (the ghosts are redone at the next solve_fields/step, like after a
  // regrid). NO re-prolongation: restored AS-IS. Mirror of AmrCouplerMP::set_level_state.
  void set_block_level_state(std::size_t b, int k, const std::vector<double>& s) {
    if (b >= blocks_.size())
      throw std::runtime_error("AmrRuntime::set_block_level_state : block index out of bounds");
    std::vector<AmrLevelMP>& L = *blocks_[b].levels;
    if (k < 0 || k >= static_cast<int>(L.size()))
      throw std::runtime_error("AmrRuntime::set_block_level_state : level out of bounds");
    MultiFab& U = L[k].U;
    const int nc = U.ncomp();
    const std::size_t nf = static_cast<std::size_t>(dom_.nx()) << k;
    if (s.size() != static_cast<std::size_t>(nc) * nf * nf)
      throw std::runtime_error("AmrRuntime::set_block_level_state : state size != ncomp*nf*nf");
    device_fence();
    for (int li = 0; li < U.local_size(); ++li) {
      Array4 u = U.fab(li).array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            u(i, j, c) = s[static_cast<std::size_t>(c) * nf * nf +
                           static_cast<std::size_t>(j) * nf + static_cast<std::size_t>(i)];
    }
  }

  // Potential phi of level @p k, flat nf*nf row-major, zeros outside patches. Level 0: the multigrid
  // WARM-START mg_.phi() (the state reused by the next solve -> bit-identical restart). Level >= 1:
  // shared aux comp 0 (recomputed at solve_fields). Mirror of AmrCouplerMP::level_potential; the phi
  // is SHARED by all blocks (single aux), so it carries no block index. NON-const like
  // AmrRuntime::potential() (GeometricMG::phi() returns a mutable warm-start reference).
  std::vector<double> level_potential(int k) {
    if (k < 0 || k >= nlev_)
      throw std::runtime_error("AmrRuntime::level_potential : level out of bounds");
    const std::size_t nf = static_cast<std::size_t>(dom_.nx()) << k;
    std::vector<double> out(nf * nf, 0.0);
    device_fence();
    const MultiFab& P = (k == 0) ? mg_.phi() : aux_[k];
    fill_level_phi(P, nf, out);
    return out;
  }

  // Same as level_potential but all_reduce_sum (np>1 gather). COLLECTIVE: all ranks MUST call it.
  std::vector<double> level_potential_global(int k) {
    std::vector<double> out = level_potential(k);
    all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
    return out;
  }

  // Restores phi of level @p k. Level 0: warm-start mg_.phi() -> bit-identical restart (1st
  // post-restart solve starts from the same guess). Level >= 1: shared aux comp 0 (idempotent,
  // recomputed at solve_fields). Mirror of AmrCouplerMP::set_level_potential.
  void set_level_potential(int k, const std::vector<double>& p) {
    if (k < 0 || k >= nlev_)
      throw std::runtime_error("AmrRuntime::set_level_potential : level out of bounds");
    const std::size_t nf = static_cast<std::size_t>(dom_.nx()) << k;
    if (p.size() != nf * nf)
      throw std::runtime_error("AmrRuntime::set_level_potential : phi size != nf*nf");
    device_fence();
    MultiFab& P = (k == 0) ? mg_.phi() : aux_[k];
    for (int li = 0; li < P.local_size(); ++li) {
      Array4 q = P.fab(li).array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          q(i, j, 0) = p[static_cast<std::size_t>(j) * nf + static_cast<std::size_t>(i)];
    }
  }

 private:
  // Fills @p out (zero-initialized, size nc*nf*nf) from the LOCAL valid cells of @p U at GLOBAL
  // component-major indices c*nf*nf + j*nf + i. Shared by block_level_state and its _global gather
  // variant (the loop is verbatim with AmrCouplerMP::level_state -> bit-identical layout).
  static void fill_level_state(const MultiFab& U, int nc, std::size_t nf,
                               std::vector<double>& out) {
    for (int li = 0; li < U.local_size(); ++li) {
      const ConstArray4 u = U.fab(li).const_array();
      const Box2D v = U.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          for (int c = 0; c < nc; ++c)
            out[static_cast<std::size_t>(c) * nf * nf + static_cast<std::size_t>(j) * nf +
                static_cast<std::size_t>(i)] = u(i, j, c);
    }
  }

  // Fills @p out (zero-initialized, size nf*nf) from the LOCAL valid cells of @p P (comp 0) at GLOBAL
  // row-major indices j*nf + i. Shared by level_potential and its _global gather variant.
  static void fill_level_phi(const MultiFab& P, std::size_t nf, std::vector<double>& out) {
    for (int li = 0; li < P.local_size(); ++li) {
      const ConstArray4 p = P.fab(li).const_array();
      const Box2D v = P.box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[static_cast<std::size_t>(j) * nf + static_cast<std::size_t>(i)] = p(i, j, 0);
    }
  }

  // Re-applies the model-NAMED aux fields (ADC-291) onto the COARSE shared aux valid cells. Mirror of
  // SystemFieldSolver::apply_named_aux_one (cartesian System): per LOCAL fab (MPI-safe), valid cells
  // only, global flat index j*nx+i. The coarse layout is frozen across regrid (only fine levels are
  // rebuilt), so the stored coarse field stays valid; solve_fields runs the coarse->fine injection
  // right after, carrying the named comps to every level. No-op without a named field.
  void apply_named_aux() {
    if (named_aux_.empty() || aux_.empty())
      return;
    const int row = dom_.nx();
    for (const auto& [comp, field] : named_aux_) {
      if (field.empty() || comp >= aux_ncomp_)
        continue;
      for (int li = 0; li < aux_[0].local_size(); ++li) {
        Array4 a = aux_[0].fab(li).array();
        const Box2D v = aux_[0].box(li);
        for (int j = v.lo[1]; j <= v.hi[1]; ++j)
          for (int i = v.lo[0]; i <= v.hi[0]; ++i)
            a(i, j, comp) = field[static_cast<std::size_t>(j) * row + i];
      }
    }
  }

  // NAMED multi-elliptic field (ADC-428): a field name's aux output components + a DEDICATED coarse
  // GeometricMG (built lazily, REUSING the native solver; the operator is never reimplemented).
  // shared_ptr<GeometricMG>: GeometricMG owns Fabs (non-copyable/non-movable), and named_fields_ is a
  // std::map (stable nodes), so a heap GeometricMG gives a stable address without making NamedField
  // movable. Defined here (before ensure_named_elliptic, whose parameter type must be visible).
  struct NamedField {
    int phi_comp = -1;
    int gx_comp = -1;
    int gy_comp = -1;
    std::shared_ptr<GeometricMG>
        mg;  // dedicated coarse solver, built lazily by ensure_named_elliptic
  };

  // Builds a NAMED elliptic field's dedicated coarse GeometricMG (ADC-428), lazily, IDENTICAL to the
  // default mg_ (same coarse geometry / BoxArray / Poisson BC / wall predicate / replication). REUSES the
  // native solver -- no operator is reimplemented. The variable / anisotropic permittivity of the default
  // Poisson is NOT carried onto a named field (a named field is a plain Laplacian, like the System named
  // path). No-op if already built.
  void ensure_named_elliptic(NamedField& nf) {
    if (nf.mg)
      return;
    nf.mg =
        std::make_shared<GeometricMG>(geom_, ba_coarse_, bcPhi_, wall_active_, replicated_coarse_);
  }

  // Reads aux component @p comp of the COARSE level as a GLOBAL n*n row-major field (diagnostic /
  // read-back). Same marshaling as detail::coupler_read_coarse_phi (the default potential read-back) but
  // for an arbitrary component: local n*n buffer, all_reduce_sum_inplace when the coarse is DISTRIBUTED
  // (disjoint boxes -> exact recompose; serial / replicated is identity). Used by named_field_values.
  std::vector<double> coarse_aux_component(int comp) const {
    device_fence();
    const int nx = dom_.nx(), ny = dom_.ny();
    std::vector<double> out(static_cast<std::size_t>(nx) * ny, 0.0);
    for (int li = 0; li < aux_[0].local_size(); ++li) {
      const ConstArray4 a = aux_[0].fab(li).const_array();
      const Box2D v = aux_[0].box(li);
      for (int j = v.lo[1]; j <= v.hi[1]; ++j)
        for (int i = v.lo[0]; i <= v.hi[0]; ++i)
          out[static_cast<std::size_t>(j) * nx + i] = static_cast<double>(a(i, j, comp));
    }
    if (!replicated_coarse_)
      all_reduce_sum_inplace(out.data(), static_cast<int>(out.size()));
    return out;
  }

  // Per-field aux HALO override (ADC-369) on the COARSE aux, AFTER the shared fill_ghosts. Overrides
  // only each declared component's physical-face ghosts (aux_halo_override keeps periodic faces
  // periodic). No-op without a policy. Mirror of SystemFieldSolver::apply_named_aux_bc.
  void apply_named_aux_bc() {
    if (named_aux_bc_.empty() || aux_.empty())
      return;
    for (const auto& [comp, policy] : named_aux_bc_) {
      if (comp >= aux_ncomp_)
        continue;
      fill_physical_bc(aux_[0], dom_, aux_halo_override(aux_bc_, policy), comp);
    }
  }

  // Index of the block named @p name in the registry (-1 if absent). Counterpart of
  // AmrSystem::Impl::block_index (the facade names the blocks; the coupled sources target them by name,
  // resolved once at registration).
  int block_index(const std::string& name) const {
    for (std::size_t i = 0; i < blocks_.size(); ++i)
      if (blocks_[i].name == name)
        return static_cast<int>(i);
    return -1;
  }

  // Resolved reference of a coupled-source field: (block index, component) + the term bytecode program
  // (empty for an input). Inputs carry only block/comp; outputs carry in addition the postfix program
  // evaluated per cell. We capture the block INDEX (not a fab pointer): the Array4 are rebuilt at each
  // application, per level.
  struct CsRef {
    int block;
    int comp;
    CsProgram prog;  // outputs: term program; inputs: unused (CsProgram{})
  };
  // A registered coupled source: its inputs, its output terms and its constants, ready to be marshaled
  // into a CoupledSourceKernel per level / per fab at application.
  struct CoupledSourceSpec {
    std::vector<CsRef> ins;
    std::vector<CsRef> outs;
    std::vector<Real> kconsts;
    int n_in = 0;
    int n_const = 0;
    int n_terms = 0;
  };

  Geometry geom_;
  Box2D dom_;
  Periodicity base_per_;
  BCRec bcPhi_, aux_bc_;
  bool replicated_coarse_;
  GeometricMG mg_;
  // Coarse BoxArray + conductive-wall predicate stashed at the ctor so a NAMED elliptic field (ADC-428)
  // can build its own coarse GeometricMG identical to mg_ (ensure_named_elliptic). mg_ consumes them at
  // its construction but does not expose them, so we keep a copy here (cheap; the coarse layout is small).
  BoxArray ba_coarse_;
  std::function<bool(Real, Real)> wall_active_;
  std::vector<AmrRuntimeBlock> blocks_;
  // GLOBAL step bounds (add_dt_bound, parity with System) + ACTIVE bound of the last step_cfl.
  struct GlobalDtBound {
    std::string label;
    std::function<double()> fn;
  };
  std::vector<GlobalDtBound> dt_bounds_;
  // Declared frequencies of the coupled sources (bound dt <= cfl/mu on the macro-step, wave 3).
  struct CoupledFreqDecl {
    std::string label;
    Real mu;
  };
  std::vector<CoupledFreqDecl> coupled_freqs_;
  // PER-CELL frequencies of the coupled sources (CoupledSource.frequency with an Expr): bytecode
  // program mu(U) evaluated on the coarse at each step_cfl (MAX + all_reduce_max -> dt <= cfl/max(mu)).
  // ins = (block, comp) of the inputs (prog unused); kconsts = constants (same as the source).
  struct CoupledFreqExprDecl {
    std::string label;
    std::vector<CsRef> ins;
    CsProgram prog;
    int n_in = 0;
    int n_const = 0;
    std::vector<Real> kconsts;
  };
  std::vector<CoupledFreqExprDecl> coupled_freq_exprs_;
  std::string last_dt_reason_;
  std::vector<MultiFab> aux_;  // [level], shared by all blocks
  // Model-NAMED aux fields (ADC-291): component (>= kAuxNamedBase) -> coarse base-level field
  // (n*n row-major). STATIC user fields re-applied by solve_fields each macro-step (so they persist
  // across regrid). Empty by default -> bit-identical. cf. set_named_aux / apply_named_aux.
  std::map<int, std::vector<Real>> named_aux_;
  // Per-field aux HALO policy (ADC-369): component -> uniform boundary policy, applied to the coarse aux
  // after the shared fill (apply_named_aux_bc). Empty by default -> bit-identical.
  std::map<int, AuxHaloPolicy> named_aux_bc_;
  // NAMED multi-elliptic fields (ADC-428): field name -> its aux output components + a DEDICATED coarse
  // GeometricMG. The NamedField struct itself is defined higher up (before ensure_named_elliptic, which
  // takes it by reference: a parameter type must be visible at the function declaration, unlike a member
  // body). Empty default -> bit-identical (the solve_named_fields loop early-returns).
  std::map<std::string, NamedField> named_fields_;
  /// Unified DESCRIPTOR registry of the field problems this AMR runtime realizes (ADC-596): the
  /// default shared Poisson ("phi") plus every named field, the SAME abstraction the Uniform
  /// SystemFieldSolver uses. Owns no solver, changes no numerics; populated by register_named_field.
  FieldProblemRegistry field_problems_;
  std::vector<CoupledSourceSpec>
      coupled_sources_;  // registered coupled sources (applied after transport)
  // TYPED coupling operator inspect metadata (ADC-595, parity with System::Impl::coupled_operators_):
  // one read-only view (label + declared contracts) per registered coupled source, in registration
  // order. Populated by add_coupled_source (unchecked) / add_coupling_operator (declared). Metadata
  // only; the step never reads it.
  std::vector<CouplingOperatorView> coupled_operators_;
  // UNION-TAGS REGRID (capstone Phase 2, C.6). regrid_every_ == 0 -> FROZEN hierarchy (default,
  // bit-identical). block_tag_: PER-BLOCK tag predicate (D1; same size as blocks_, empty = this block
  // tags nothing on its side). phi_tag_: phi tag predicate on |grad phi| (D4; empty = phi does not
  // contribute to the union).
  std::vector<TagPredicate> block_tag_;
  TagPredicate phi_tag_;
  int regrid_every_ = 0;
  int regrid_grow_ = 2;
  int regrid_margin_ = 2;
  int aux_ncomp_ = kAuxBaseComps;
  int nlev_ = 0;
  int macro_step_ = 0;
  mutable int solve_count_ = 0;
  int regrid_count_ = 0;
  // AMR / MPI PROFILING (Spec 5 criterion 43, ADC-479): non-owning pointer to the AmrSystem-owned
  // Profiler (lifetime guaranteed by the facade). Null by default -> the engine never profiles
  // (zero overhead). Set via set_profiler after build (parity with System::profiler_).
  runtime::program::Profiler* profiler_ = nullptr;

  // RAII phase-timing scope for an AMR phase (regrid / average_down / fill_boundary). Mirrors
  // runtime::program::ProfileScope but over a NULLABLE profiler pointer: it reads the clock and
  // records only when profiler_ is non-null AND enabled. A null/disabled run constructs a cheap inert
  // scope (one pointer copy + one clock read) -- the granularity is per-phase, not per-cell, so this
  // is off the hot path. Returned BY VALUE from profile_amr_scope (movable: only POD members).
  class AmrPhaseScope {
   public:
    AmrPhaseScope(runtime::program::Profiler* prof, const char* name)
        : prof_(prof != nullptr && prof->enabled() ? prof : nullptr),
          name_(name),
          t0_(std::chrono::steady_clock::now()) {}
    AmrPhaseScope(AmrPhaseScope&& o) noexcept : prof_(o.prof_), name_(o.name_), t0_(o.t0_) {
      o.prof_ = nullptr;  // the moved-from scope must not record
    }
    AmrPhaseScope(const AmrPhaseScope&) = delete;
    AmrPhaseScope& operator=(const AmrPhaseScope&) = delete;
    AmrPhaseScope& operator=(AmrPhaseScope&&) = delete;
    ~AmrPhaseScope() {
      if (prof_ == nullptr)
        return;
      const auto t1 = std::chrono::steady_clock::now();
      try {
        prof_->record(name_, std::chrono::duration<double>(t1 - t0_).count());
      } catch (...) {  // NOLINT(bugprone-empty-catch) -- a profiler never throws out of a scope
      }
    }

   private:
    runtime::program::Profiler* prof_;
    const char* name_;
    std::chrono::steady_clock::time_point t0_;
  };

  // Build an AMR phase scope (no-op when profiling is off). Used at the head of regrid /
  // average_down / fill_boundary.
  AmrPhaseScope profile_amr_scope(const char* name) { return AmrPhaseScope(profiler_, name); }

  // fill_ghosts wrapped in the "fill_boundary" timing scope + per-call count (Spec 5 criterion 43).
  // Under MPI np>1 the ghost fill is a cross-rank halo exchange -> also count one "mpi_messages" (a
  // point-to-point round, distinct from the all_reduce collectives counted as "mpi_reductions").
  // Serial / single rank: no message count (the fill is a local copy). Off-profiling: inert.
  void fill_ghosts_profiled(MultiFab& mf, const Box2D& dom, const BCRec& bc) {
    auto _fb = profile_amr_scope("fill_boundary");
    if (profiler_ != nullptr) {
      profiler_->count("fill_boundary");
#ifdef POPS_HAS_MPI
      if (n_ranks() > 1)
        profiler_->count("mpi_messages");
#endif
    }
    fill_ghosts(mf, dom, bc);
  }
};

}  // namespace pops
