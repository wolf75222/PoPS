#pragma once

#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/mesh/layout/patch_box.hpp>  // PatchBox: index-space signature of a fine patch (patch_boxes())
#include <pops/mesh/boundary/physical_bc.hpp>                // BCRec
#include <pops/numerics/time/integrators/implicit_stepper.hpp>  // NewtonOptions (Newton options of the IMEX source)
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/coupling/source/coupling_operator.hpp>  // CouplingOperator / CouplingOperatorView (typed contract, ADC-595)
#include <pops/runtime/export.hpp>  // POPS_EXPORT: set_compiled_block resolved by the native AMR loader
#include <pops/runtime/facade_options.hpp>  // CoupledSourceProgram (facade POD, ADC-214)
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams (compiled-Program runtime params on AMR, ADC-508)
#include <pops/runtime/numerical_defaults.hpp>

#include <functional>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

/// @file
/// @brief Multi-species composition on AMR at runtime: the refined counterpart of System.
///
/// One or SEVERAL blocks (species, described by ModelSpec of generic bricks) carried on an
/// AMR hierarchy. Like System but on an adaptive mesh.
///
/// MONO-BLOCK (1 add_block): a single-model AmrCouplerMP<Model> (coarse + one fine level tracked by
/// regrid, conservative reflux). Historical path, UNTOUCHED -> bit-identical.
///
/// MULTI-BLOCK (>= 2 add_block, capstone, docs/AMR_MULTIBLOCK_DESIGN.md): N blocks co-located on
/// ONE SHARED AMR hierarchy (same BoxArray + DistributionMapping + dx/dy per level, guarded by
/// same_layout_or_throw). All blocks live on ALL patches. A single aux per level (phi,
/// grad phi) and a single coarse Poisson whose right-hand side is the CO-LOCATED SUM of the blocks'
/// elliptic bricks (f = Sum_b q_b n_b read at the same cells). Conservation PER BLOCK (reflux +
/// average_down). AmrRuntime runtime engine (type-erased registry by name). Blocks with potentially
/// DIFFERENT spatial schemes and with PER-BLOCK TEMPORAL TREATMENT (explicit or IMEX, local stiff
/// implicit source; capstone vii) with union-of-tags regrid (multi-block + regrid_every > 0 is NOW
/// SUPPORTED: the mesh re-grids from the union of the tags; regrid_every == 0 = frozen hierarchy). Multirate (substeps/stride), inter-species
/// coupled sources: already wired. Multiple COMPILED blocks (add_compiled_model) and a MIX of
/// compiled + native: wired (capstone v, multi-block production DSL). Union regrid: later PR.
///
/// @note Resolved explicit bootstrap supports N ratio-2 levels; the legacy implicit config remains
/// two-level. Temporal treatment is explicit or IMEX per block.

namespace pops {

// Forward declarations of the runtime multi-block engine (definitions in amr_runtime.hpp /
// amr_dsl_block.hpp). set_compiled_block stores a DEFERRED runtime-block BUILDER which, given the
// SHARED layout materialized at lazy build, returns the type-erased AmrRuntimeBlock of the compiled
// block: this is what lets SEVERAL compiled blocks (multi-block production DSL) co-exist on the
// SAME AMR hierarchy, exactly like add_block in native multi-block. We forward-declare so as NOT
// to weigh down this public header (read by bindings.cpp and the loaders) with amr_runtime.hpp: only
// the TUs that BUILD/CALL the builder (amr_dsl_block.hpp and python/amr_system.cpp) include the
// complete definitions; a std::function with incomplete-type signature is legal as long as it is
// not instantiated with a concrete callable outside those TUs (PIMPL std::function recipe).
struct AmrRuntimeBlock;
// Forward-declared for the named-elliptic-field RHS closure signature (ADC-428,
// set_block_elliptic_field): a std::function with an incomplete-type parameter is legal as long as it is
// only INSTANTIATED with a concrete callable in the TUs that include the full definition (the native AMR
// loader / python/bindings/amr/amr_system.cpp), per the PIMPL std::function recipe noted above.
class MultiFab;
class AmrRuntime;  // the multi-block engine (engine() exposes it to the AmrProgramContext driver, ADC-508)
namespace detail {
struct SharedAmrLayout;
}
namespace runtime {
namespace program {
class Profiler;  // forward-declared so engine()/profiler_handle() do not pull profiler.hpp into this header
}  // namespace program
}  // namespace runtime

/// AMR mesh and cadence (per-block physical parameters live in the ModelSpec).
struct AmrSystemConfig {
  int n = 128;            ///< coarse-level cells per direction
  double L = 1.0;         ///< size of the square domain [0,L]^2
  int regrid_every = 20;  ///< re-refinement every N steps (0 = never after init)
  int level_count = 2;    ///< exact materialized hierarchy depth (>= 1)
  int regrid_grow = 2;    ///< tag lookahead/dilation from the resolved hierarchy
  int regrid_margin = 2;  ///< proper-nesting buffer from the resolved hierarchy
  bool explicit_bootstrap = false;  ///< coarse-only start; BootstrapPlan creates fine levels
  bool periodic = true;   ///< periodic domain
  /// OWNERSHIP POLICY of the coarse level (cf. AmrCouplerMP::replicated_coarse).
  /// false (DEFAULT, historical): coarse mono-box REPLICATED on all ranks. The coarse Poisson
  ///   and the coarse transport are REDUNDANT on each GPU (zero communication,
  ///   better geometric MG) but DO NOT SCALE: only the fine patches are distributed.
  /// true (strong-scaling mode): coarse MULTI-BOX (BoxArray::from_domain, tile size
  ///   coarse_max_grid) distributed round-robin across the ranks. The coarse Poisson and the coarse
  ///   transport distribute (each rank carries only its tiles), which removes the redundancy
  ///   and enables AMR strong-scaling. The geometric MG then operates on a multi-box coarse
  ///   (cf. geometric_mg.hpp): convergence to be measured (may require more cycles).
  bool distribute_coarse = false;
  /// Coarse tile size when distribute_coarse=true (BoxArray::from_domain). 0 => n/2
  /// (minimal 2x2 split, least aggressive for the MG). Ignored if distribute_coarse=false.
  int coarse_max_grid = 0;
  /// ADC-616: Berger-Rigoutsos clustering params of the regrid layout. <= 0 (default) keeps the
  /// historical ClusterParams {0.7, 1, 32}, bit-identical. min_efficiency in (0,1], sizes >= 1,
  /// min_box_size <= max_box_size (validated at set_clustering / the facade descriptor).
  double cluster_min_efficiency = 0.0;
  int cluster_min_box_size = 0;
  int cluster_max_box_size = 0;
};

/// Frozen parameters passed to the deferred build of the compiled path (add_compiled_model). Materialized
/// by AmrSystem at ensure_built time: the geometry + the refine/poisson/density choices known
/// at that moment. The amr_dsl_block header consumes them to instantiate AmrCouplerMP<Model>.
///
/// STRUCTURE (ADC-610). The 31 settings are grouped into NAMED sub-structs by ownership/role
/// (mesh, physics, regrid, poisson, initial data, Schur source stage, named aux) instead of one flat
/// append-only bag. A new setting goes INTO its semantic group -- the historical "add at the tail so an
/// older .so loader falls back silently" idiom is RETIRED because it no longer describes how this struct
/// evolves. The ABI story is now the VERSIONED KEY, not tail-only placement: this struct crosses the
/// dlopen .so boundary BY VALUE, and any layout change (a new field, a regroup) shifts POPS_HEADER_SIG
/// (a sha256 over include/, cf. abi_key.hpp / python/CMakeLists.txt), which re-keys pops_native_abi_key.
/// A .so generated before the change then diverges from the module key and add_native_block REJECTS it
/// with a clear "regenerate" error (never silent UB). So an older .so is refused, not silently truncated.
struct AmrBuildParams {
  /// Coarse mesh geometry + coarse ownership policy (AMR strong-scaling).
  struct Mesh {
    int n = 128;                     ///< coarse cells per direction
    double L = 1.0;                  ///< size of the square domain [0, L]^2
    int regrid_every = 20;           ///< re-refinement cadence (0 = never after init)
    bool distribute_coarse = false;  ///< distributed multi-box coarse (AMR strong-scaling)
    int coarse_max_grid = 0;         ///< tile size of the distributed coarse (0 => n/2)
  } mesh;
  /// Physical + temporal treatment of the block (gamma, substeps, recon, IMEX, time method, Newton,
  /// positivity floor). newton_options serves the IMEX source; pos_floor serves the transport.
  struct Physics {
    double gamma = static_cast<double>(kPhysicalDefaultGamma);
    int substeps = 1;
    bool recon_prim = false;  ///< recon == "primitive" (frozen by add_compiled_model)
    bool imex = false;        ///< time == "imex": stiff implicit source (backward_euler)
    int time_method = 0;      ///< pops::AmrTimeMethod: 0 kEuler (default), 1 kSsprk3
    // NEWTON OPTIONS of the IMEX source on the MONO-BLOCK path (wave 3: mono-block AMR options wired).
    // DEFAULT {} = historical constants (2 / 0 / 0 / 1e-7 / 1.0 / none) -> backward_euler_source path
    // (2a) bit-identical. Consumed by build_amr_compiled (the mono-block closure passes it to cpl->step).
    NewtonOptions newton_options{};
    // Zhang-Shu positivity floor (ADC-259): Density-role face-state + C/F-ghost-mean floor on the AMR
    // transport. 0 (default) -> inactive, bit-identical. Consumed by build_amr_compiled (mono-block ->
    // cpl->step / advance_transport). The COMPILED .so path carries it too (ADC-322): the loader marshals
    // it (pops_install_native_amr) into add_compiled_model -> set_compiled_block, which stores it here.
    double pos_floor = 0.0;
  } physics;
  /// Refinement criterion frozen at build.
  struct Regrid {
    double threshold =
        static_cast<double>(kAmrRefinementDisabledThreshold);  ///< no refinement (sentinel)
  } regrid;
  /// Coarse Poisson boundary condition + conductive wall (resolved by set_poisson).
  struct Poisson {
    BCRec bc;                              ///< coarse Poisson BC
    std::function<bool(Real, Real)> wall;  ///< conductive wall predicate (empty = none)
    // ADC-645: opt-in COMPOSITE FAC field solve (the fine patch refines the elliptic;
    // AmrCouplerMP::set_composite_poisson). false (default) = the historical Option A coarse solve +
    // gradient injection, bit-identical. The fac_* knobs mirror the SchurStage block below (<= 0 =
    // the kFAC* default, same convention).
    bool composite = false;           ///< true: composite FAC field solve (single-block coupler)
    int fac_max_iters = 0;            ///< FAC outer iterations (<= 0 = default kFACDefaultMaxIters)
    int fac_fine_sweeps = 0;          ///< SOR sweeps per fine solve (<= 0 = kFACDefaultFineSweeps)
    double fac_tol = 0.0;             ///< composite-residual stop (<= 0 = kFACDefaultTol)
    double fac_coarse_rel_tol = 0.0;  ///< internal coarse rel_tol (<= 0 = kFACInitialCoarseRelTol)
    int fac_coarse_cycles = 0;        ///< internal coarse cycles (<= 0 = kFACInitialCoarseMaxCycles)
    bool fac_verbose = false;         ///< record the FAC per-iteration residual trace
  } poisson;
  /// Initial coarse seed: density only (historical) OR the FULL conservative state (priority).
  struct InitialData {
    bool has_density = false;
    std::vector<double> density;  ///< initial coarse density (component 0), n*n
    // FULL initial conservative state (all components), takes priority over `density` when has_state.
    bool has_state = false;
    std::vector<double>
        state;  ///< ncomp*n*n, component-major c*n*n + j*n + i; ncomp == Model::n_vars
  } initial;
  /// Model-NAMED aux fields (ADC-291) + their per-field HALO policies (ADC-369). Seeded onto the coupler's
  /// shared aux at build (build_amr_compiled), like bz_field; re-applied each update (persist across
  /// regrid). Both empty -> bit-identical.
  struct NamedAux {
    std::map<int, std::vector<double>> fields;   ///< component (>= kAuxNamedBase) -> coarse field (n*n)
    std::map<int, AuxHaloPolicy> halo_policies;  ///< component -> uniform boundary policy
  } named_aux;
  /// NATIVE per-block RUNTIME parameters (ADC-514): the SHARED value vector the block's bricks read
  /// through apply_runtime_params, re-injected each macro-step so AmrSystem::set_block_params changes
  /// the trajectory WITHOUT recompiling the .so. GATED ON count > 0: a model declaring no runtime param
  /// (count == 0, values empty) leaves the historical build byte-identical (no injection emitted). The
  /// shared_ptr is allocated MODULE-SIDE (add_compiled_model) and crosses the .so boundary by value; a
  /// header change re-keys POPS_HEADER_SIG so a stale .so is rejected with the regenerate error.
  struct Runtime {
    std::shared_ptr<std::vector<double>> values;  ///< shared current values (empty when count == 0)
    int count = 0;                                ///< number of runtime params (0 = inactive, historical)
  } runtime;
};

/// Type-erased closures of a compiled AMR block, produced by amr_dsl_block::build_amr_compiled and
/// installed via AmrSystem::set_compiled_block. Symmetric with the std::function hooks of AmrSystem::Impl.
///
/// STRUCTURE (ADC-610). The 22 closures are grouped into the FIVE named tiers documented in the design
/// (lifetime, base, stability, checkpoint, MPI gather) instead of one flat append-only list. A new
/// closure goes INTO its tier -- the historical "add at the tail" idiom is retired (a regroup or add
/// shifts POPS_HEADER_SIG anyway, which re-keys pops_native_abi_key, so a stale .so is REJECTED at
/// add_native_block with a clear regenerate error, never silently truncated). The builder always
/// populates every closure (none are optional except stability, empty without the trait).
struct AmrCompiledHooks {
  /// LIFETIME tier: keeps the concrete coupler alive (every other closure captures it).
  std::shared_ptr<void> coupler_holder;  ///< keeps the AmrCouplerMP<Model> alive
  /// BASE tier: the macro-step + the primary observables.
  struct Base {
    std::function<void(double)> step;                ///< one macro-step (periodic regrid included)
    std::function<double()> max_speed;               ///< max wave speed (CFL step)
    std::function<double()> mass;                    ///< coarse mass
    std::function<int()> n_patches;                  ///< number of fine patches
    std::function<std::vector<double>()> density;    ///< coarse density, n*n row-major
    std::function<std::vector<double>()> potential;  ///< coarse-level phi, n*n row-major
  } base;
  /// STABILITY tier: patch signatures + OPTIONAL step bounds (empty without the HasSourceFrequency /
  /// HasStabilityDt traits, bit-identical). patch_boxes mirrors base.n_patches (count becomes boxes).
  struct Stability {
    std::function<std::vector<PatchBox>()>
        patch_boxes;                           ///< index-space signatures of the fine patches
    std::function<double()> source_frequency;  ///< coarse max of mu [1/s] (0 = does not constrain)
    std::function<double()>
        stability_dt;  ///< coarse min of the admissible step (0 = does not constrain)
  } stability;
  /// CHECKPOINT tier (ADC-65 mono-rank restart): cadence phase + per-level state/phi + hierarchy.
  struct Checkpoint {
    std::function<void(int)>
        set_macro_step;  ///< restores the cadence (regrid) phase of the mono-block
    std::function<int()> n_levels;                        ///< number of levels (>= 1)
    std::function<int()> n_vars;                          ///< conserved components of the block
    std::function<std::vector<double>(int)> level_state;  ///< full state of level k (c*nf*nf+j*nf+i)
    std::function<void(int, const std::vector<double>&)>
        set_level_state;                                      ///< restores the state of level k
    std::function<std::vector<double>(int)> level_potential;  ///< phi of level k (nf*nf row-major)
    std::function<void(int, const std::vector<double>&)>
        set_level_potential;  ///< restores phi of level k
    std::function<void(const std::vector<PatchBox>&)>
        set_hierarchy;  ///< imposes the saved fine patches
  } checkpoint;
  /// MPI-GATHER tier: coarse ownership diagnostic (ADC-319) + the all_reduce_sum gather counterparts of
  /// the checkpoint state/phi (ADC-509), so a bit-identical np>1 checkpoint gathers onto rank 0.
  struct MpiGather {
    std::function<int()> coarse_local_boxes;  ///< per-rank owned coarse (level-0) fab count
    std::function<int()> coarse_total_boxes;  ///< global coarse box count (identical on all ranks)
    std::function<std::vector<double>(int)> level_state_global;      ///< level k state, gathered
    std::function<std::vector<double>(int)> level_potential_global;  ///< level k phi, gathered
  } mpi_gather;
};

/// DEFERRED builder of a COMPILED block on the multi-block hierarchy: receives the SHARED layout (created
/// ONCE at lazy build, common to all blocks) plus the block parameters frozen at
/// add time (name, initial density, gamma, substeps/stride, recon/imex, partial IMEX mask resolved into
/// component indices), and returns the type-erased AmrRuntimeBlock of the block (captures the CONCRETE
/// Model/Limiter/Flux via detail::dispatch_amr_block, the kernel stays COMPILED). Symmetric with the
/// native add_block path: the (sole) difference is only that the types are known at add time (compiled
/// model) rather than resolved from a ModelSpec at build. The SIGNATURE mentions FORWARD-DECLARED types:
/// it is instantiated with a concrete callable only in add_compiled_model(AmrSystem&) (header
/// amr_dsl_block.hpp) where those types are complete, and invoked only in python/amr_system.cpp.
/// The trailing pos_floor (ADC-322) is the Zhang-Shu positivity floor of the block (0 = inactive),
/// forwarded to dispatch_amr_block -> build_amr_block exactly like a native multi-block.
/// The trailing runtime_params (ADC-514) is the SHARED per-block runtime-param vector (empty for a
/// model with no runtime param, bit-identical); the multi-block builder captures it and re-injects it
/// into the block's model each macro-step (parity with the mono-block build_amr_compiled path).
using AmrCompiledBlockBuilder = std::function<AmrRuntimeBlock(
    const detail::SharedAmrLayout& layout, const std::string& name,
    const std::vector<double>& density, bool has_density, double gamma, int substeps,
    bool recon_prim, bool imex, int stride, const std::vector<std::string>& implicit_vars,
    const std::vector<std::string>& implicit_roles, double pos_floor,
    std::shared_ptr<std::vector<double>> runtime_params)>;

/// Single block carried on an AMR hierarchy, composed at runtime.
///
/// @code{.cpp}
/// pops::AmrSystemConfig cfg;                // base level: n x n on [0, L]^2
/// cfg.n = 64;
/// pops::AmrSystem amr(cfg);
///
/// pops::ModelSpec ne;
/// ne.transport = "exb"; ne.source = "none"; ne.elliptic = "charge";
/// amr.add_block("ne", ne, "minmod", "rusanov", "conservative", "explicit");
/// amr.set_poisson("charge_density", "geometric_mg");
/// amr.set_refinement(0.1);                 // refine where any block's field exceeds the threshold
///
/// amr.set_density("ne", rho0);             // rho0: initial density on the base level
/// amr.step_cfl(0.4);                       // conservative refluxed step + composite FAC Poisson
/// @endcode
class AmrSystem {
 public:
  explicit AmrSystem(const AmrSystemConfig& cfg);
  ~AmrSystem();
  // RULE OF FIVE (C.21): move-only (PIMPL unique_ptr). The copy was already IMPLICITLY deleted
  // (move ctor declared); we make it EXPLICIT for intent. No API change (the copy was
  // already unusable).
  AmrSystem(const AmrSystem&) = delete;
  AmrSystem& operator=(const AmrSystem&) = delete;
  AmrSystem(AmrSystem&&) noexcept;
  AmrSystem& operator=(AmrSystem&&) noexcept;

  /// GLOBAL time-step bound (AMR counterpart of System::add_dt_bound): fn() evaluated ONCE
  /// per step_cfl (host), all_reduce_min (identical dt on all ranks), <= 0 / non-finite =
  /// inert this step. Hook for non-local constraints (coupling, scheduler, user ramp).
  void add_dt_bound(const std::string& label, std::function<double()> fn);

  /// ACTIVE bound of the last step_cfl: "transport:<block>" | "source_frequency:<block>" |
  /// "stability_dt:<block>" | "global:<label>" | "degenerate" | "" (no CFL step yet).
  std::string last_dt_bound() const;

  /// Adds a block carried on the AMR. Same spatial-scheme parameters as System
  /// (limiter x riemann x recon), applied to each level/patch of the hierarchy. The FIRST
  /// add_block defines the block; a 2nd (or more) switches to the multi-block engine (shared
  /// hierarchy, co-located sum Poisson). Blocks can have DIFFERENT SPATIAL SCHEMES.
  /// @param name    block name: INDEXES the block (set_density(name), mass(name), density(name)). In
  ///                multi-block the name must be unique; mono-block an empty name targets the single block.
  /// @param model   composition of bricks (transport/source/elliptic + parameters)
  /// @param limiter "none" | "minmod" | "vanleer" | "weno5" (weno5 = WENO5-Z, 3 ghosts; rusanov)
  /// @param riemann "rusanov" | "hll" (generic signed-wave, requires model.wave_speeds) | "hllc"
  ///                | "roe" (generic when the model supplies the Riemann capability
  ///                HasHLLCStructure / HasRoeDissipation; else the canonical Euler 2D layout,
  ///                4 variables + pressure)
  /// @param time    "explicit" (SSPRK2, forward-Euler source carried by the AMR step) | "ssprk3"
  ///                (SSPRK3, order 3, reflux per stage; explicit transport, EXCLUSIVE of imex) |
  ///                "imex" (stiff source handled IMPLICITLY by backward_euler_source; the transport
  ///                stays explicit, carried by the conservative reflux; cf. capstone vii). Any other
  ///                treatment is refused.
  /// @param substeps explicit substeps of the block (>= 1): the effective step is split into substeps
  ///                equal pieces (MULTI-BLOCK only; in mono-block, carried by AmrCouplerMP).
  /// @param stride  HOLD-THEN-CATCH-UP cadence of the block (>= 1; default 1 = each macro-step). stride=M
  ///                holds the block M-1 macro-steps then catches it up by an effective step M*dt (multirate).
  ///                MULTI-BLOCK only (a single block always advances every step). step_cfl honors
  ///                the cadence: dt = cfl*h*min_b(substeps_b/(stride_b*w_b)), mirror of System::step_cfl.
  /// @param implicit_vars / implicit_roles  partial IMEX mask CARRIED BY THE BLOCK (cf. System::add_block):
  ///                conserved components handled IMPLICITLY, by NAME (implicit_vars) or by physical
  ///                ROLE (implicit_roles). EMPTY (default) -> full backward-Euler (all
  ///                components implicit). Only meaningful with time="imex": requesting them in explicit
  ///                is an ERROR (no silent ignore). MULTI-BLOCK only (the mono-block
  ///                AmrCouplerMP carries its IMEX without a mask; a mask there is therefore refused).
  /// @throws std::runtime_error if a block is already defined, if substeps < 1, if stride < 1, if time
  ///         is not in {explicit, ssprk3, imex}, if recon is not in {conservative,
  ///         primitive}, or if an implicit mask is requested outside IMEX / with a name-role absent from the block.
  /// @param newton  options of the IMEX source Newton grouped in a POD (ADC-214; cf.
  ///                 NewtonOptions; parity with System::add_block): max_iters / rel_tol / abs_tol /
  ///                 fd_eps / damping / fail_policy. Default {} = historical constants, bit-identical.
  ///                 SUPPORT (wave 3, settled): these OPTIONS are wired in MONO-BLOCK (coupler
  ///                 AmrCouplerMP) AND in MULTI-BLOCK (AmrRuntime engine); the .so loaders
  ///                 reject them (flat ABI). fail_policy='throw' works everywhere. fail_policy='warn'
  ///                 requires the structured Newton report, therefore native multi-block.
  /// @param newton_diagnostics  aggregated Newton report (newton_report): wired in NATIVE MULTI-BLOCK
  ///                 only (the mono-block rejects it at build, the .so loaders at the facade). Stays
  ///                 flat (a separate bool, outside the homogeneous family of convergence options).
  /// @param positivity_floor  Zhang-Shu positivity floor (ADC-259): if > 0, the AMR transport floors
  ///                 the Density-role face states (reconstruct_pp / zhang_shu_scale) AND the C/F fine
  ///                 ghost means to >= floor. Default 0 = inactive, bit-identical. Guarantee = face /
  ///                 ghost-state Density positivity only (order-1 fallback), NOT updated-mean nor
  ///                 pressure positivity (parity with System::add_block). A model without a Density
  ///                 role rejects floor > 0. The COMPILED .so path carries it too (ADC-322): a loader
  ///                 regenerated against this header marshals the floor (pops_install_native_amr).
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod", const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative", const std::string& time = "explicit",
                 int substeps = 1, int stride = 1,
                 const std::vector<std::string>& implicit_vars = {},
                 const std::vector<std::string>& implicit_roles = {},
                 const NewtonOptions& newton = {}, bool newton_diagnostics = false,
                 double positivity_floor = 0.0);

  /// Report of the implicit (IMEX) source Newton of a block, AGGREGATED over the levels and substeps of
  /// the block's LAST advance. Exists only if the block was added with newton_diagnostics=true IN
  /// NATIVE MULTI-BLOCK (explicit error otherwise: mono-block, .so loader, or block without diagnostics).
  /// Flat copy (no dependence on the numerics header on the caller side), parity with System::SourceNewtonReport.
  struct SourceNewtonReport {
    bool enabled;           ///< a report was computed (at least one IMEX advance played)
    bool converged;         ///< no cell failed on the last advance
    double max_residual;    ///< max over cells/levels/substeps of ||F||_inf at the Newton exit
    double max_iters_used;  ///< max over cells/levels/substeps of the iterations consumed
    double
        n_failed;  ///< count (cells x levels x substeps) failed (non-finite / pivot / non-convergence)
    double failed_i;     ///< i of ONE faulty cell (-1 if none; max index encoded)
    double failed_j;     ///< j of the same cell (-1 if none)
    double failed_comp;  ///< conserved component of the worst residual of that cell (-1 unknown)
    std::vector<RuntimeDiagnosticEvent> diagnostics;  ///< structured policy/solver events
  };
  /// @throws std::runtime_error if the block is unknown, in mono-block, on a .so loader, or if the block
  ///         did not enable newton_diagnostics. Forces the lazy build (ensure_built).
  SourceNewtonReport newton_report(const std::string& name);

  /// Registers a COMPILED block (add_compiled_model path, header amr_dsl_block.hpp). TWO type-erased
  /// builders are frozen here, for the TWO routings of the facade:
  ///  - @p mono_builder: given the AmrBuildParams frozen at lazy build, returns the
  ///    AmrCompiledHooks of a concrete AmrCouplerMP<Model>. Used IN MONO-BLOCK (1 single compiled block)
  ///    -> historical AmrCouplerMP path, UNTOUCHED, bit-identical.
  ///  - @p multi_builder: given the SHARED layout materialized at lazy build (common to all
  ///    blocks), returns the type-erased AmrRuntimeBlock of the block. Used IN MULTI-BLOCK (>= 2 blocks,
  ///    compiled and/or native mixed) -> AmrRuntime runtime engine, exactly like add_block.
  /// @p recon_prim / @p imex / @p stride / @p implicit_vars / @p implicit_roles: metadata of the block
  /// (temporal scheme, multirate, partial IMEX mask) frozen at add time, consumed by the
  /// multi-block routing (the mono-block already carries them in the AmrBuildParams via mono_builder).
  /// DO NOT call directly: go through the free function add_compiled_model(AmrSystem&, ...).
  /// @throws std::runtime_error if the system is already built.
  /// DEFAULT VISIBILITY (POPS_EXPORT): the ONLY method called by the header template
  /// add_compiled_model(AmrSystem&) (cf. amr_dsl_block.hpp). A generated .so loader (DSL
  /// "production" path on the AMR side, emit_cpp_native_loader(target="amr_system") / add_native_block) inlines this
  /// template and must resolve this symbol from the already-loaded _pops module; compiled with
  /// -fvisibility=hidden (pybind11), the module would not export it without this annotation and the dlopen
  /// of the loader would fail. Symmetric with the POPS_EXPORT methods of System (grid_context/install_block).
  /// @param runtime_params  NATIVE per-block runtime-param vector (ADC-514): the SHARED current values
  ///                the block's bricks read via apply_runtime_params, re-injected each macro-step so
  ///                set_block_params changes the trajectory WITHOUT recompiling. Empty / nullptr for a
  ///                model with no runtime param -> the historical build is byte-identical (no injection).
  ///                Allocated module-side by add_compiled_model and also registered under @p name via
  ///                register_block_params so set_block_params resolves it by name before the lazy build.
  POPS_EXPORT void set_compiled_block(
      int ncomp, double gamma, int substeps,
      std::function<AmrCompiledHooks(const AmrBuildParams&)> mono_builder,
      AmrCompiledBlockBuilder multi_builder = {}, const std::string& name = std::string(),
      bool recon_prim = false, bool imex = false, int stride = 1,
      const std::vector<std::string>& implicit_vars = {},
      const std::vector<std::string>& implicit_roles = {}, double pos_floor = 0.0,
      std::shared_ptr<std::vector<double>> runtime_params = {});

  /// Wires a NATIVE AMR block from a .so loader generated by the DSL (backend "production", target
  /// "amr_system": dsl.compile_native(target="amr_system") / compile(backend="production",
  /// target="amr_system")). AMR counterpart of System::add_native_block: the .so inlines the header template
  /// add_compiled_model(AmrSystem&, ...), which materializes a concrete AmrCouplerMP<Model> at lazy
  /// build and installs its hooks via set_compiled_block -- NATIVE path, SAME AMR hierarchy as
  /// add_block (conservative reflux, regrid), no flat-array marshaling.
  ///
  /// The _pops module is PROMOTED to global scope (RTLD_NOLOAD) then the loader is dlopen-ed in
  /// RTLD_GLOBAL to resolve set_compiled_block; the ABI key baked in the loader
  /// (pops_native_abi_key) is compared to the module's (abi_key()) -- mismatch => clear error (no
  /// silent UB at the C++ boundary). Same scheme guard-rails as System (upstream validation).
  ///
  /// MULTI-BLOCK (capstone v): add_native_block CAN now be called several times (or mixed
  /// with native add_block) -> the compiled blocks co-exist on the shared hierarchy via AmrRuntime
  /// (the loader recompiled against this header provides the runtime builder; cf. set_compiled_block). The
  /// name then INDEXES the block (set_density/mass/density), like add_block.
  /// time is wired to {explicit, imex} (imex = stiff implicit source via backward_euler_source; any
  /// other treatment is rejected by add_compiled_model). The multirate (stride) and the partial IMEX
  /// mask do NOT transit through the flat ABI of the loader (ABI unchanged): this .so path now REJECTS
  /// them at the Python facade level (AmrSystem.add_equation raises ValueError on stride>1 or a
  /// non-empty IMEX mask, rather than ignoring them silently). For these parameters, use
  /// native add_block (ModelSpec) or add_compiled_model(AmrSystem&) DIRECTLY (header), which expose
  /// stride and the mask. recon "primitive" and flux "roe"/"hllc" are WIRED at parity (#113:
  /// dispatch_amr_compiled accepts them; the Python facade applies a pressure guard for hllc/roe).
  /// limiter "weno5" (WENO5-Z, 3 ghosts) is WIRED on rusanov (#105: the coupler levels are
  /// allocated to Limiter::n_ghost and the regrid inherits n_grow(): no out-of-bounds read).
  /// @throws std::runtime_error if the ABI diverges, if a symbol is missing, or substeps < 1.
  /// @param name block name: cosmetic in mono-block, INDEXES the block in multi-block (set_density/
  ///             mass/density; must be unique and non-empty from the 2nd block on, like add_block).
  /// @param positivity_floor  Zhang-Shu positivity floor of the block (ADC-322): the .so flat ABI now
  ///             carries it (pops_install_native_amr -> add_compiled_model -> set_compiled_block), so a
  ///             loader regenerated against this header floors the Density-role face states like a
  ///             native add_block. 0 (default) = inactive, bit-identical.
  void add_native_block(const std::string& name, const std::string& so_path,
                        const std::string& limiter = "minmod",
                        const std::string& riemann = "rusanov",
                        const std::string& recon = "conservative",
                        const std::string& time = "explicit",
                        double gamma = static_cast<double>(kPhysicalDefaultGamma),
                        int substeps = 1,
                        double positivity_floor = 0.0);

  /// Refines the cells where the SELECTED conserved variable exceeds @p threshold. By default the
  /// variable is component 0 (historically the density), preserving the bit-identical @c 1e30 no-op.
  /// Optionally the variable is selected PER BLOCK by NAME (@p variable, e.g. "E") or by physical ROLE
  /// (@p role, e.g. "energy"): each block resolves it against its OWN conservative VariableSet, so a
  /// model whose refinement variable is NOT at component 0 refines correctly (ADC-296). Resolution is
  /// STRICT -- a block lacking the requested name/role raises an explicit error at build, never a silent
  /// fallback to component 0 (mirror of add_coupled_source). At most one of @p variable / @p role may be
  /// set. MULTI-BLOCK only for a non-default selector (the AmrRuntime union-of-tags regrid carries the
  /// per-block predicates); the mono-block AmrCouplerMP path and the compiled .so loader refine on
  /// component 0 only and reject a non-empty selector, mirroring how set_phi_refinement is multi-block
  /// only. @param threshold refinement threshold (@c 1e30 default elsewhere => no refinement, frozen).
  /// @param variable conserved-variable NAME to threshold; empty (default) => component 0.
  /// @param role conserved-variable physical ROLE to threshold; empty (default) => component 0.
  void set_refinement(double threshold, const std::string& variable = std::string(),
                      const std::string& role = std::string());
  void set_bootstrap_refinement(const std::string& block, const std::string& variable,
                                double threshold, const std::string& provider_identity);
  void set_bootstrap_tagging(
      const std::vector<std::string>& leaf_blocks,
      const std::vector<std::string>& leaf_variables,
      const std::vector<int>& leaf_ops,
      const std::vector<double>& leaf_thresholds,
      const std::vector<int>& refine_ops, const std::vector<int>& refine_args,
      const std::vector<int>& coarsen_ops, const std::vector<int>& coarsen_args,
      int min_cycles, const std::string& equality_policy,
      const std::string& conflict_policy, const std::string& provider_identity);

  /// Adds to the regrid criterion the PHI tag on |grad phi| (D4 of the design
  /// docs/AMR_REGRID_UNION_TAGS_DESIGN.md): also refines the cells where the norm of the gradient of the
  /// electrostatic potential |grad phi| (components 1,2 of the shared aux) exceeds @p grad_threshold.
  /// MULTI-BLOCK only (the AmrRuntime runtime engine carries the union-of-tags regrid; the mono-block
  /// path AmrCouplerMP has no separate phi predicate). The phi tag is ADDED to the union of the density
  /// tags per block (set_refinement): the mesh refines where ANY block exceeds its
  /// density threshold OR |grad phi| exceeds @p grad_threshold. PHYSICAL criterion: a sharp ring/edge
  /// feature follows the gradient of the potential, not the density alone.
  /// @param grad_threshold threshold of |grad phi|. <= 0 (DEFAULT) -> the phi tag is DISABLED (phi does not
  ///        contribute to the union; bit-identical to before this call). Without regrid_every > 0, no
  ///        effect (the regrid is never called). To be called BEFORE the first step.
  void set_phi_refinement(double grad_threshold);

  /// Configures the coarse Poisson (cf. System::set_poisson). On AMR the elliptic solver is
  /// ALWAYS GeometricMG and the right-hand side ALWAYS f = sum of the block's elliptic bricks.
  /// @param rhs    "charge_density" | "composite" (same composed right-hand side as System)
  /// @param solver "geometric_mg" only (the only one wired on the hierarchy; no FFT)
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param wall   "none" | "circle" (circular conductive wall, requires wall_radius > 0)
  /// @param composite ADC-645: true opts the FIELD solve into the composite FAC path (the fine patch
  ///                  refines the elliptic; AmrCouplerMP::set_composite_poisson). Supported scope =
  ///                  the coupler's: single block, 2 levels, ONE mono-box fine patch, replicated
  ///                  coarse -- an out-of-scope hierarchy REFUSES at build (never a silent fallback).
  ///                  false (default) = the historical Option A solve, bit-identical.
  /// @param fac_max_iters / fac_fine_sweeps / fac_tol / fac_coarse_rel_tol / fac_coarse_cycles /
  ///        fac_verbose the composite-FAC knobs (<= 0 = the kFAC* default); inert when composite is false.
  /// @throws std::runtime_error if rhs, solver, bc, wall or a FAC knob is outside the supported domain.
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg", const std::string& bc = "auto",
                   const std::string& wall = "none", double wall_radius = 0.0,
                   bool composite = false, int fac_max_iters = 0, int fac_fine_sweeps = 0,
                   double fac_tol = 0.0, double fac_coarse_rel_tol = 0.0,
                   int fac_coarse_cycles = 0, bool fac_verbose = false);

  /// Install one fully resolved AMR field route.  The native registry key is the digest of the
  /// complete block-qualified provider identity; the canonical identity is retained for collision
  /// detection, manifests and restart validation.
  void set_field_solver_plan(const std::string& provider_slot,
                             const std::string& provider_identity,
                             const std::string& output_owner_identity,
                             const std::string& output_block,
                             const std::string& output_key,
                             const std::vector<std::string>& provider_identities,
                             const std::vector<std::string>& provider_blocks,
                             const std::vector<std::string>& provider_keys,
                             const std::vector<double>& provider_coefficients,
                             const std::string& solver,
                             const std::string& hierarchy, double abs_tol, double rel_tol,
                             int max_cycles, int min_coarse, int pre_smooth,
                             int post_smooth, int bottom_sweeps, int coarse_threshold);
  void set_field_boundary_plan(const std::string& provider_slot,
                               const std::vector<std::string>& kind,
                               const std::vector<double>& alpha,
                               const std::vector<double>& beta,
                               const std::vector<double>& value);
  void set_field_boundary_dependencies(
      const std::string& provider_slot,
      const std::vector<std::string>& state_blocks,
      const std::vector<int>& state_components,
      const std::vector<std::string>& field_blocks,
      const std::vector<std::string>& field_keys,
      const std::vector<int>& field_components);
  POPS_EXPORT void set_field_boundary_kernel(
      const std::string& provider_slot, const CompiledFieldBoundaryKernel& kernel);
  POPS_EXPORT void set_field_logical_timepoint(
      const std::string& provider_slot, const FieldLogicalTimePoint& point);
  POPS_EXPORT void set_field_boundary_parameters(
      const std::string& provider_slot, const std::vector<double>& parameters);
  void set_field_newton_plan(const std::string& provider_slot, double tolerance,
                             int max_iterations, double linear_tolerance,
                             int linear_max_iterations, int restart, double armijo,
                             double minimum_step);
  void set_field_nullspace(const std::string& provider_slot, bool constant_kernel,
                           bool mean_zero_gauge);

  /// Sets the initial density on the coarse level (component 0), n*n row-major.
  /// @param name cosmetic label (mono-block AMR: the density targets the single block).
  void set_density(const std::string& name, const std::vector<double>& rho);

  /// Sets the FULL INITIAL CONSERVATIVE STATE (all components) on the coarse level, then
  /// prolongs it to the fine levels at build (constant injection, like the density). @p U is flat
  /// component-major (c*n*n + j*n + i) of size ncomp*n*n; ncomp == n_vars of the model (checked at
  /// build, where only Model::n_vars is known). Takes priority over set_density: allows starting the AMR
  /// from a full drift state (rho, rho*u, rho*v) instead of m=0. The conversion
  /// primitive -> conservative (rho_u = rho*u) is done on the Python side (the caller already supplies the
  /// conservative). Wired on the NATIVE blocks (mono-block as well as multi-block: threaded to the native builder,
  /// seed the coarse then inject to the fine); in multi-block @p name indexes the target block. A
  /// COMPILED (.so) block carrying a state raises at build in multi-block (the .so loader does not transport
  /// the state): use a native block pops.Model(...) or set_density.
  /// @throws std::runtime_error if the system is already built, if U is empty, or if its size
  ///         is not a multiple of n*n.
  void set_conservative_state(const std::string& name, const std::vector<double>& U);
  void begin_bootstrap_plan();
  void bootstrap_next_level(int refinement_ratio);  ///< execute one resolved transition
  void commit_bootstrap_level();
  void rollback_bootstrap_level();
  void register_bootstrap_transfer_route(
      const std::string& identity, const std::vector<std::string>& subjects,
      const std::string& provider_identity, const std::string& space,
      const std::string& centering, const std::string& representation,
      const std::string& storage, const std::string& operation,
      const std::string& kernel, int order, const std::vector<int>& ghost_depth,
      int dimension, int refinement_ratio);
  void register_bootstrap_array(const std::string& subject, const std::string& centering,
                                int ncomp, int ny, int nx,
                                const std::vector<double>& values);
  void register_bootstrap_face_vector(const std::vector<std::string>& subjects);
  void bind_bootstrap_block_subject(const std::string& subject, const std::string& block);
  void register_analytic_constant(const std::string& subject, const std::string& block,
                                  const std::string& space, const std::string& centering,
                                  const std::vector<double>& components);
  void register_analytic_gaussian(const std::string& subject, const std::string& block,
                                  double center_x, double center_y, double background,
                                  double amplitude, double inverse_width);
  std::int64_t bootstrap_analytic_reproject(const std::string& subject, int level);
  int apply_bootstrap_component_floor(const std::string& subject, int level,
                                      int component, double floor);
  std::int64_t recompute_bootstrap_field(const std::string& subject,
                                         const std::string& field_name);
  std::int64_t bootstrap_prolong_array(const std::string& subject, int level);
  void synchronize_bootstrap_state(const std::string& subject, int fine_level);
  std::vector<double> bootstrap_array_level(const std::string& subject, int level) const;
  void invalidate_bootstrap_cache(const std::string& subject, int level);
  std::vector<PatchBox> rebuild_bootstrap_topology_cache(const std::string& subject, int level);
  std::uint64_t bootstrap_cache_epoch(const std::string& subject) const;

  /// Sets the magnetic field B_z(x, y) of the coarse level (n*n row-major), required by the Schur-condensed
  /// source stage (Lorentz term Omega = B_z). AMR counterpart of System::set_magnetic_field.
  /// MONO-BLOCK only (the condensed AMR stage is wired on the mono-block coupler AmrCouplerMP).
  /// @throws std::runtime_error if the system is already built or if bz is not of size n*n.
  void set_magnetic_field(const std::vector<double>& bz);

  /// Sets a model-NAMED aux field (ADC-291) at shared-channel component @p comp (>= kAuxNamedBase) from
  /// a coarse base-level field @p field (n*n row-major). AMR counterpart of
  /// System::set_aux_field_component: the field is STATIC (re-applied by the engine each update, so it
  /// survives a regrid) and reaches every level via the coarse->fine aux injection. Works on the
  /// single-block (AmrCouplerMP) AND multi-block (AmrRuntime) paths. The Python facade resolves the name
  /// to @p comp and reshapes the array. Mono-rank facade (same as set_density). @throws if the system is
  /// already built, if @p comp is reserved (< kAuxNamedBase), or if @p field is not of size n*n.
  void set_aux_field_component(int comp, const std::vector<double>& field);

  /// Declares a per-field aux HALO policy (ADC-369) for the NAMED component @p comp (>= kAuxNamedBase):
  /// @p bc_type is pops::BCType (Foextrap=1 / Dirichlet=2), @p value the Dirichlet boundary value. Seeded
  /// onto the engine at build and applied after the shared coarse aux fill (overriding only that
  /// component's physical-face ghosts; periodic faces keep their wrap). AMR counterpart of
  /// System::set_aux_field_halo_component. @throws on a reserved component or an unsupported type.
  void set_aux_field_halo_component(int comp, int bc_type, double value);

  /// @name Named multi-elliptic fields (ADC-428)
  /// A SECOND elliptic solve (beyond the default coarse Poisson) for a user-named field
  /// (m.elliptic_field("psi", rhs=..., aux=[...])) on the AMR hierarchy. AMR counterpart of
  /// System::register_elliptic_field / set_block_elliptic_field. The named field owns its RHS (a per-block
  /// brick), a DEDICATED coarse GeometricMG, and its OWN aux output components; AmrRuntime solves it each
  /// solve_fields and injects it to the fine levels, so a bare run() leaves it SOLVED. The default Poisson
  /// path is untouched / bit-identical. Registering a named field forces the MULTI-BLOCK runtime engine
  /// (AmrRuntime) even for a single block (the named-field solve lives there, not on the single-block
  /// AmrCouplerMP coupler). POPS_EXPORT: resolved by the generated AMR .so / native loader across the
  /// dlopen boundary, like set_compiled_block.
  /// @{
  /// Registers named @p field's aux output components (where its solved phi / centered grad land). Called
  /// by the native AMR loader for each m.elliptic_field. @p gx_comp / @p gy_comp < 0 => only phi is
  /// written (the field declared fewer than 3 aux slots). @throws if the system is already built.
  POPS_EXPORT void register_elliptic_field(const std::string& block_name,
                                           const std::string& provider_key, int phi_comp,
                                           int gx_comp, int gy_comp);
  /// Attaches named @p field's RHS closure (rhs += elliptic_field_rhs(U)) to block @p block_name. Called
  /// by the native AMR loader (make_poisson_rhs of the per-field brick). @throws if the system is already
  /// built or the block is unknown.
  POPS_EXPORT void set_block_elliptic_field(const std::string& block_name, const std::string& field,
                                            std::function<void(const MultiFab&, MultiFab&)> rhs);
  /// Solved potential of named @p field on the COARSE level, n*n row-major (read-back). Solves the
  /// hierarchy fields if needed (so it is current even before any step), then reads the field's phi
  /// component. AMR counterpart of System::aux_field_component for a named elliptic field. @throws if the
  /// field is unregistered (or in the single-block AmrCouplerMP path, which carries no named field).
  std::vector<double> named_field_values(const std::string& field);
  std::vector<std::string> field_provider_slots() const;
  int field_provider_levels(const std::string& provider_slot);
  void set_field_potential(const std::string& provider_slot, const std::vector<double>& phi);
  void set_field_potential_level(const std::string& provider_slot, int level,
                                 const std::vector<double>& phi);
  std::vector<double> field_potential_global(const std::string& provider_slot);
  std::vector<double> field_potential_level_global(const std::string& provider_slot, int level);
  /// Transaction bracket used by the v3 reader after complete payload preflight.  Every hierarchy,
  /// state, aux, field warm-start, history and clock mutation is rolled back if any restore step fails.
  void begin_restart_transaction();
  void commit_restart_transaction();
  void rollback_restart_transaction();
  int checkpoint_regrid_count() const;
  std::uint64_t checkpoint_topology_epoch() const;
  void restore_checkpoint_counters(int regrid_count, std::uint64_t topology_epoch);
  std::vector<int> checkpoint_temporal_ratios() const;
  /// Canonical rows for every required bootstrap transfer route: subject, operation, route identity,
  /// provider, kernel, descriptor fields.  The sealed checkpoint compares these rows byte-for-byte.
  std::vector<std::vector<std::string>> checkpoint_transfer_routes() const;
  /// @}

  /// Registers an inter-species COUPLED SOURCE (compiled pops.dsl.CoupledSource, flat bytecode ABI
  /// P5), refined counterpart of System::add_coupled_source but on the SHARED AMR hierarchy. The source
  /// is applied at EACH macro-step AFTER the transport, by forward-Euler splitting, level by
  /// level, followed by a fine -> coarse cascade (consistency of the covered coarse cells, #169).
  /// The coupling is baked into a device-clean stack machine (CoupledSourceKernel): NO per-cell Python
  /// callback in the hot path. MULTI-BLOCK only (>= 2 add_block: the coupling reads/writes
  /// SEVERAL named blocks). Must be called BEFORE the first step (the runtime engine is built
  /// at lazy build; the source is injected into it).
  ///
  /// CONSERVATION: an add_pair construction (a term +expr on a block, -expr exactly on the other,
  /// SAME cell) makes the sum of the two blocks conserved PER CELL (and globally) to machine
  /// precision. The engine does NOT IMPOSE it (an ionization creating an e/i pair is legal): it is a
  /// property of the constructed coupling (verify_conservation on the DSL side checks it symbolically).
  ///
  /// @throws std::runtime_error if called in mono-block, if the system is already built, or if the
  ///         shape of the bytecode / a role / a block is invalid (same guards as System).
  /// @param prog      bytecode description of the coupling grouped in a POD (ADC-214; cf.
  ///                  CoupledSourceProgram; parity with System::add_coupled_source): in_blocks /
  ///                  in_roles / consts / out_blocks / out_roles + prog_ops / prog_args / prog_lens
  ///                  (stack machine) + freq_prog_ops / freq_prog_args (PER-CELL frequency mu(U)
  ///                  optional; EMPTY = constant frequency only, bit-identical; non-empty:
  ///                  evaluated on the COARSE LEVEL of the input blocks at each step_cfl, MAX +
  ///                  all_reduce_max, bound dt <= cfl / max(mu) on the coarse, not the patches).
  /// @param frequency CONSTANT declared frequency mu [1/s] of the coupling (wave 3): bound
  ///                  dt <= cfl/mu on the macro-step of step_cfl; <= 0 (default) = no bound.
  /// @param label     name of the coupling (reason "coupled_source:<label>" of last_dt_bound).
  void add_coupled_source(const CoupledSourceProgram& prog, double frequency = 0.0,
                          const std::string& label = "coupled_source");

  /// Registers a TYPED coupling operator (ADC-595, parity with System::add_coupling_operator): the
  /// same coupled-source program PLUS its declared conservation contract and frequency bound. The
  /// declared contract is VALIDATED at registration (host, fail-loud) against the actual output terms,
  /// then the program is lowered through the SAME add_coupled_source path (bit-identical numerics), and
  /// the declared contract is recorded for coupled_operators(). An empty (unchecked) contract is
  /// equivalent to add_coupled_source.
  void add_coupling_operator(const CouplingOperator& op);

  /// Read-only view of the registered coupling operators (ADC-595, parity with System): label plus the
  /// declared conservation / frequency contracts, in registration order, so a Program or a runtime
  /// report enumerates the AMR couplings as typed operators. A raw add_coupled_source registers an
  /// "unchecked" entry (empty contract). Empty until the first coupling is added.
  const std::vector<CouplingOperatorView>& coupled_operators() const;

  void step(double dt);  ///< one AMR macro-step (periodic regrid included)
  void advance(double dt, int nsteps);
  /// Advances at dt = cfl * coarse_dx / max wave speed. @return the dt used.
  double step_cfl(double cfl, double speed_floor = static_cast<double>(kCflSpeedFloor));

  /// @name Compiled time-program install seam on the AMR hierarchy (epic ADC-511 / ADC-508, Spec 6)
  /// AMR counterpart of System::install_program: load a generated problem.so and install its compiled
  /// time Program over the AMR hierarchy. Mirrors the System seam (install_program_step registers the
  /// macro-step body; the cadence + per-block RuntimeParams stores live HERE on the Impl, NOT in the
  /// .so closure, so a value change reaches the captured context and a later checkpoint can reach
  /// them). A generated AMR Program .so resolves these across the dlopen boundary (RTLD_GLOBAL,
  /// POPS_EXPORT), exactly like set_compiled_block on the native AMR loader.
  /// @{
  /// Install the macro-step body. When set, AmrSystem::step calls it instead of the historical
  /// AmrRuntime::step path (keeping t / macro_step coherent). Pass an empty std::function to clear it.
  /// POPS_EXPORT: the generated AMR Program .so resolves it across the dlopen boundary, like
  /// set_compiled_block. The closure drives the per-level Lie/Strang macro-step through an
  /// AmrProgramContext (the AMR counterpart of ProgramContext).
  POPS_EXPORT void install_program_step(std::function<void(double)> step);
  /// Set the compiled-Program macro-step cadence (parity with System::set_program_cadence, ADC-411):
  /// GLOBAL @p substeps and @p stride around the installed program closure. @p substeps subdivides each
  /// effective step into @p substeps program closure calls; @p stride runs the program once per @p
  /// stride macro-steps (hold-then-catch-up). Both must be >= 1 (throws std::invalid_argument).
  /// Default 1/1 -> a single program closure call per macro-step. Kept SEPARATE from install_program so
  /// the generated .so ABI is untouched (the cadence is runtime metadata).
  POPS_EXPORT void set_program_cadence(int substeps, int stride);
  /// Installed GLOBAL macro-step cadence (ADC-594, parity System): the current @c substeps / @c stride
  /// the compiled Program runs at (default 1/1 with no cadence set). Const, side-effect-free -- the
  /// structured ProgramRuntimeReport reads them; there was no Python-visible getter before.
  POPS_EXPORT int program_substeps() const;
  POPS_EXPORT int program_stride() const;
  /// Install the program-index -> AMR-block-index map (entry p = the AMR block index of Program block
  /// p), built by install_program after matching the .so's block names to the instantiated AMR blocks
  /// BY NAME (Spec 3 criterion 23, ADC-457). Empty clears it (identity: a single-block or order-matching
  /// Program). Read by the AmrProgramContext to resolve a Program block index to the name-matched block.
  POPS_EXPORT void set_program_block_map(const std::vector<int>& prog_to_sys);
  /// The installed program-index -> AMR-block-index map (empty = identity). Read by the AmrProgramContext.
  POPS_EXPORT const std::vector<int>& program_block_map() const;
  /// Load a generated problem.so and install its compiled time Program on the AMR hierarchy. dlopens
  /// @p so_path, checks its ABI key against this module (fail-loud on mismatch), runs the section-24
  /// install-time requirement validation (aux / solver / block instance, verbatim spec messages), binds
  /// the Program's blocks to the AMR blocks BY NAME, seeds each block's RuntimeParams from the .so
  /// pops_program_param_* metadata, then calls the .so's pops_install_program_amr(this), which wraps the
  /// AmrSystem in an AmrProgramContext and installs the macro-step closure. Mirrors add_native_block and
  /// System::install_program; the .so stays loaded for the process lifetime.
  POPS_EXPORT void install_program(const std::string& so_path);
  /// IR hash of the installed compiled Program (the string returned by the .so's pops_program_hash), or
  /// "" if no program is installed. Parity with System::installed_program_hash (checkpoint guard).
  POPS_EXPORT std::string installed_program_hash() const;
  /// The last macro-step dt handed to the installed Program (ADC-631): the AmrProgramContext reads it
  /// so a history ring's store_history tags the per-slot dt (variable-dt replay). POPS_EXPORT for the
  /// dlopen boundary (the generated AMR Program .so reads it via the AmrProgramContext).
  POPS_EXPORT double program_last_dt() const;
  /// Authenticated accepted-state image owned by the compiled AMR Program context.  This is distinct
  /// from the dense field/history arrays: it preserves exact level clocks, qualified history-slot
  /// identities and lagged effective-flux publications required for conservative multistep restart.
  POPS_EXPORT std::vector<std::uint8_t> program_accepted_state() const;
  /// Replace the accepted image during strict restart.  Each replacement advances a revision observed
  /// by the persistent AmrProgramContext before its next attempt; no stale context state is reused.
  POPS_EXPORT void restore_program_accepted_state(const std::vector<std::uint8_t>& state);
  POPS_EXPORT std::uint64_t program_accepted_state_revision() const;
  /// Human/audit-readable qualification rows decoded from the same accepted image persisted as bytes.
  POPS_EXPORT std::vector<std::vector<std::string>> program_accepted_state_manifest() const;

  /// @name Runtime freeze lifecycle (ADC-592, parity with System)
  /// Assembly mutable BEFORE bind, composition FROZEN once pops.bind completes. mark_bound() is
  /// called LAST by the Python bind flow (after every install call), so the install sequence itself
  /// never trips the structural-setter guards. NOTE: 'bound' (bind completed) is DISTINCT from the
  /// lazy 'built' materialization (bind runs BEFORE the first step/mass/density triggers ensure_built),
  /// so the existing 'already built' messages of the lazy path are untouched; the structural guards
  /// now also refuse a call once bound_ is set, with the bind-vocabulary message.
  /// @{
  /// Mark the composition as bound (frozen): every structural setter then rejects with a precise
  /// error. Runtime-data setters (set_density / set_conservative_state on the base level BEFORE a step
  /// / set_program_params / set_clock) that DATA-write stay allowed. A second mark_bound() throws.
  void mark_bound();
  /// The runtime lifecycle state: "assembling" (not bound), "bound" (mark_bound() ran, no macro-step),
  /// "running" (bound AND macro_step() > 0). Parity with System::lifecycle_state.
  std::string lifecycle_state() const;
  /// @}
  /// @name Compiled-Program RUNTIME parameters on AMR (epic ADC-511 / ADC-508, parity with ADC-510)
  /// Per-PROGRAM-block RuntimeParams of a compiled time Program whose physics reads a
  /// dsl.Param(..., kind="runtime"), owned HERE so set_program_params changes it at run time WITHOUT
  /// recompiling (the same no-recompile contract as System). install_program seeds each block's defaults
  /// from the .so pops_program_param_* metadata. The lowered kernels read the CURRENT value via the
  /// AmrProgramContext.
  /// @{
  /// Overwrite block @p prog_block's RuntimeParams with @p values (the COMPLETE block, sorted-name order
  /// matching the .so pops_program_param_* metadata). @p prog_block is the PROGRAM block index. @throws
  /// std::out_of_range if the block was not seeded by a runtime-param Program, std::runtime_error on a
  /// size mismatch. Effect on the next step.
  POPS_EXPORT void set_program_params(int prog_block, const std::vector<double>& values);
  /// Block @p prog_block's CURRENT RuntimeParams (a device-clean by-value copy). An UNSEEDED block
  /// returns a default-constructed RuntimeParams (count 0). Read by the AmrProgramContext.
  POPS_EXPORT RuntimeParams program_params(int prog_block) const;
  /// Seed block @p prog_block's RuntimeParams to its DECLARATION defaults (@p count values, the .so
  /// pops_program_param_default metadata). Called by install_program once per runtime-param Program
  /// block; a later set_program_params overwrites only the supplied values. Idempotent.
  POPS_EXPORT void seed_program_params(int prog_block, const std::vector<double>& defaults);
  /// @}
  /// @name NATIVE per-block RUNTIME parameters on AMR (ADC-514, parity with System::set_block_params)
  /// The production AMR path (add_native_block / add_compiled_model) carries per-block runtime params:
  /// each block whose model declares dsl.Param(kind="runtime") owns a SHARED value vector re-injected
  /// each macro-step, so set_block_params changes the trajectory WITHOUT recompiling. Distinct from the
  /// compiled-Program params above (those key the time Program; these key a NAMED block).
  /// @{
  /// Overwrite the runtime-param values of block @p name (ADC-514). @p values is the COMPLETE vector
  /// (order = the block model's runtime_param_names). The block must have been added on the production
  /// path AND declare at least one runtime parameter; otherwise an explicit error (a silent set on a
  /// block without a runtime param would mask a bug). Effect at the next step (the block closures read
  /// the SHARED vector). @throws std::runtime_error if the block is unknown, has no runtime params, or
  /// if @p values has the wrong length. VERBATIM mirror of System::set_block_params.
  POPS_EXPORT void set_block_params(const std::string& name, const std::vector<double>& values);
  /// Register block @p name's SHARED runtime-param vector (ADC-514) so set_block_params finds it by name
  /// even before the lazy build. Called by add_compiled_model(AmrSystem&) when model_nparams<Model>() >
  /// 0 (never for a param-free model, so the store stays empty and the build byte-identical). POPS_EXPORT:
  /// the generated AMR .so loader inlines add_compiled_model and resolves this across the dlopen boundary.
  POPS_EXPORT void register_block_params(const std::string& name,
                                         std::shared_ptr<std::vector<double>> values);
  /// The built multi-block AMR engine (the AmrRuntime the AmrProgramContext driver wraps), or nullptr
  /// before the lazy build. install_program forces the build so the .so's pops_install_program_amr gets
  /// a live engine. POPS_EXPORT: the generated AMR Program .so resolves it across the dlopen boundary.
  POPS_EXPORT AmrRuntime* engine() const;
  /// True when the built system runs on the multi-block AmrRuntime engine (shared layout + per-block
  /// level stacks), as opposed to the single-block AmrCouplerMP coupler. A compiled Program forces the
  /// runtime engine even for ONE block, so `n_blocks() == 1` does NOT imply the coupler: the v3
  /// checkpoint routes state I/O on THIS predicate (per-block accessors on the runtime engine, the
  /// mono-block level_state path on the coupler). POPS_EXPORT for the dlopen boundary parity.
  POPS_EXPORT bool uses_runtime_engine() const;
  /// The facade-owned Profiler (the AmrProgramContext forwards count_kernel / profile_record to it).
  /// POPS_EXPORT for the dlopen boundary. Disabled by default -> zero hot-path cost.
  POPS_EXPORT runtime::program::Profiler& profiler_handle();
  /// Record a runtime Scalar diagnostic under @p name (the AmrProgramContext's record_scalar seam),
  /// retrievable via program_diagnostic / program_diagnostics. A pure side effect (inspection / logging).
  POPS_EXPORT void record_program_diagnostic(const std::string& name, double value);
  /// The recorded diagnostic @p name (0 if absent) / the whole map. Exposed to Python for inspection.
  POPS_EXPORT double program_diagnostic(const std::string& name) const;
  POPS_EXPORT std::map<std::string, double> program_diagnostics() const;
  /// LEVEL-COMPOSITE collective reduction over a named block, the AMR counterpart of
  /// System::reduce_component the diagnostics driver drives (ADC-542). @p kind is per-component
  /// "sum" / "min" / "max" / "abs_sum" / "sum_sq" / "abs_max", or the full-state "*_all" variants.
  /// Volume-weighted sums exclude covered coarse cells; extrema fold all levels unmasked (a covered
  /// coarse cell is the average of its children, within their extrema). Multi-block routes to the
  /// AmrRuntime; single-block composes the native per-level reductions. Unknown block / kind throws.
  POPS_EXPORT double composite_reduce(const std::string& block, const std::string& kind,
                                      int comp) const;
  /// @}
  /// @}

  int nx() const;
  POPS_EXPORT double time() const;
  /// MACRO-STEP counter (0-indexed; incremented by step / advance / step_cfl), parity with
  /// System::macro_step. Required for checkpoint/restart (the stride / regrid cadence depends on
  /// macro_step % stride|regrid_every, not only on t). Prerequisite IO PR-IO-3 (audit 2026-06).
  /// POPS_EXPORT: the AmrProgramContext (a generated AMR Program .so) reads it across the dlopen
  /// boundary for the head-of-step regrid cadence, like the other program seam accessors (ADC-508).
  POPS_EXPORT int macro_step() const;
  /// RESTORES the AMR clock (t, macro_step) -- parity with System::set_clock. Sets the time AND the
  /// macro-step counter (propagated to the regrid/stride cadence of the engine, mono-block as well as multi-block). Useful
  /// alone (stride cadence + clock resumption). @throws if macro_step < 0.
  POPS_EXPORT void set_clock(double t, int macro_step);

  /// @name AMR / MPI profiling (Spec 5 sec.12.5, ADC-479 criterion 43)
  /// Per-phase wall-clock timing of the AMR runtime: the engine times its non-numeric phases --
  /// "regrid" (rebuild the patch hierarchy), "fill_boundary" (the cross-rank ghost halo exchange),
  /// "average_down" (restrict fine onto coarse) -- plus integer counters ("regrid" / "fill_boundary"
  /// per-run counts, and under MPI np>1 "mpi_reductions" / "mpi_messages"). Disabled by default (no
  /// hot-path cost when off, parity with System). enable_profiling() then step()/step_cfl() over a
  /// run where a regrid fires (regrid_every set) then profile_report() returns the table; the typed
  /// PerformanceSummary.by_amr_mpi() view surfaces it. Per-rank (no cross-rank reduction of the
  /// report). Multi-block engine only (the runtime owns the union regrid + shared-aux halo).
  /// @{
  void enable_profiling();
  void disable_profiling();
  bool is_profiling() const;
  void reset_profiling();
  std::string profile_report() const;
  /// @}
  int n_blocks() const;  ///< number of blocks (1 = mono-block AmrCouplerMP; >= 2 = AmrRuntime)
  /// Names of the blocks in add order (parity with System::block_names): the IO facade iterates over them
  /// to write EACH block by its name (an empty name -> block 0, historical mono-block compat).
  std::vector<std::string> block_names() const;
  /// Structured report of effective numerical, solver and physical options currently configured.
  EffectiveOptionsReport effective_options_report() const;
  int n_patches();  ///< number of current fine patches (of the shared hierarchy)
  /// Index-space signatures of the current fine patches: one PatchBox (level, ilo, jlo, ihi, jhi) per
  /// fine box, for ALL fine levels (level >= 1). INCLUSIVE corners in the index space of the
  /// level (n << level cells/direction, ratio 2). SAME source as n_patches() (the GLOBAL fine
  /// BoxArray, all boxes/all ranks -> rank-independent, MPI-safe, zero communication). It is a
  /// QUERY (between steps): read-only of the already-stored boxes, NO hot-path cost. The
  /// conversion to [0, L]^2 is done on the Python side (which knows n via nx() and L). Forces the lazy
  /// build (ensure_built) like n_patches()/mass()/density().
  std::vector<PatchBox> patch_boxes();
  /// COARSE-level (base) box counts, MPI ownership diagnostic (ADC-319). coarse_local_boxes() = number
  /// of base boxes OWNED by this rank (level-0 MultiFab local_size()); coarse_total_boxes() = total base
  /// boxes across all ranks (BoxArray size, identical on every rank). With distribute_coarse=true the
  /// base is split into several boxes spread round-robin, so local < total per rank and the coarse
  /// transport distributes (MPI strong-scaling); a single-box or replicated base gives local == total on
  /// every rank. coarse_local_boxes() is rank-dependent, coarse_total_boxes() is rank-independent.
  /// Forces the lazy build (ensure_built) like n_patches()/mass()/density().
  int coarse_local_boxes();
  int coarse_total_boxes();

  /// AMR CHECKPOINT / RESTART (ADC-65 single-block single-rank; ADC-509 multi-block + np>1):
  /// per-level STATE accessors + hierarchy imposition for a BIT-IDENTICAL resumption (cf.
  /// AmrSystem.checkpoint/restart on the Python side). MONO-BLOCK uses these (level_state /
  /// level_potential / set_hierarchy on the AmrCouplerMP coupler); MULTI-BLOCK (AmrRuntime engine,
  /// SHARED layout + aux) uses the per-BLOCK variants below (block_level_state ...), plus the SHARED
  /// level_potential / n_levels (no set_hierarchy: the shared hierarchy is the deterministic frozen
  /// central patch, reproduced by replaying the same composition). The _global variants all_reduce_sum
  /// the per-rank fabs so a np>1 checkpoint gathers onto rank 0 (mono-rank: identity, bit-identical).
  /// Force the lazy build (ensure_built) like patch_boxes()/mass(). @p k: level (0 = coarse, >= 1 = fine).
  int n_levels();  ///< number of levels of the hierarchy (>= 1; mono OR multi-block)
  int n_vars();    ///< number of conserved components (MONO-BLOCK; multi-block: block_n_vars)
  /// FULL conservative state of level @p k, flat component-major c*nf*nf + j*nf + i (nf = n << k;
  /// zeros outside the patches at the fine level -- only the patch interior is defined). MONO-BLOCK.
  std::vector<double> level_state(int k);
  std::vector<double> level_state_global(int k);  ///< MONO-BLOCK, np>1 gather (all ranks call)
  void set_level_state(int k,
                       const std::vector<double>& s);  ///< restores the state of level @p k (as is)
  /// Potential phi of level @p k, flat nf*nf row-major. Level 0 = warm-start of the multigrid
  /// (bit-identical resumption); level >= 1 = aux comp 0 (recomputed at update). SHARED -> works in
  /// MONO-BLOCK as well as MULTI-BLOCK (single aux). The _global variant gathers under np>1.
  std::vector<double> level_potential(int k);
  std::vector<double> level_potential_global(int k);             ///< np>1 gather (all ranks call)
  void set_level_potential(int k, const std::vector<double>& p);  ///< restores phi of level @p k
  /// Imposes the SAVED fine hierarchy (at restart) instead of Berger-Rigoutsos clustering: @p boxes
  /// are the patch_boxes() signatures of the checkpoint (filtered to level 1 in mono-block). MONO-BLOCK.
  void set_hierarchy(const std::vector<PatchBox>& boxes);

  /// Impose a mid-run MULTI-BLOCK hierarchy from a v3 checkpoint (ADC-542): @p boxes are ALL the
  /// checkpoint patch boxes (level tagged, level 0 implicit), @p owner_ranks the per-box owner rank
  /// aligned with @p boxes. Routes to AmrRuntime::rebuild_hierarchy (all levels rebuilt, reusing regrid
  /// R6/R7). MULTI-BLOCK / runtime engine only; @throws on the single-block coupler path (use
  /// set_hierarchy). The v3 restart calls this so restartable=True works under ACTIVE regridding.
  void rebuild_hierarchy(const std::vector<PatchBox>& boxes, const std::vector<int>& owner_ranks);

  /// MULTI-BLOCK per-BLOCK per-level checkpoint accessors (ADC-509). The AmrRuntime engine shares the
  /// layout AND the aux across blocks, so the per-level STATE is read/restored PER BLOCK (by NAME)
  /// while phi stays shared (level_potential above). @p name indexes the block (block_names()); @p k:
  /// level. The _global variant all_reduce_sum the per-rank fabs (np>1 gather, all ranks call); the
  /// shared hierarchy is the deterministic frozen central patch (regrid_every==0), reproduced at
  /// restart by replaying the same composition -> no set_hierarchy needed. @throws in MONO-BLOCK (use
  /// the level_state path) or if @p name / @p k is out of bounds.
  int block_n_vars(const std::string& name);  ///< conserved components of the named block
  std::vector<double> block_level_state(const std::string& name, int k);
  std::vector<double> block_level_state_global(const std::string& name,
                                               int k);  ///< np>1 gather (all ranks call)
  void set_block_level_state(const std::string& name, int k, const std::vector<double>& s);
  /// Owner rank per box of level @p k (the shared layout's DistributionMapping), aligned with the
  /// level-@p k rows of patch_boxes(). The v3 checkpoint (ADC-542) serializes it so a restart
  /// reproduces the LOCAL-fab iteration order. MULTI-BLOCK / runtime engine; empty on the coupler path.
  std::vector<int> level_owner_ranks(int k);
  /// FULL shared aux of level @p k (ALL components, flat c*nf*nf+j*nf+i; _global = np>1 gather,
  /// COLLECTIVE) + the owner-rank restore -- the v3 checkpoint aux payload (ADC-542). MULTI-BLOCK /
  /// runtime engine; the read returns EMPTY on the single-block coupler path (its aux is derived +
  /// static-reapplied each solve, phi_<k> suffices there) and the write throws on it.
  std::vector<double> level_aux_flat(int k);
  std::vector<double> level_aux_flat_global(int k);
  void set_level_aux_flat(int k, const std::vector<double>& v);

  /// @name Multistep history-ring checkpoint / replay (ADC-631, Uniform System seam names)
  /// The compiled-Program AMR route carries per-level `keep_history` / `T.prev` ring slots on the
  /// AmrRuntime engine (remapped through regrid). These wrappers expose the SAME seam names as System
  /// so the shared _system_io_history.py serialize/restore is reused verbatim: history_global returns
  /// the per-level slices concatenated into ONE flat buffer (level axis hidden, parity level_aux_flat),
  /// restore_history scatters it back per level, rebuild_history_slots replays the policy-recomputed
  /// slots by re-stepping the installed Program. Engine-less coupler -> history_names() is empty.
  /// @{
  std::vector<std::string> history_names() const;
  int history_depth(const std::string& name) const;
  int history_ncomp(const std::string& name) const;
  bool history_initialized(const std::string& name) const;
  void set_history_initialized(const std::string& name, bool initialized);
  std::vector<double> history_global(const std::string& name, int slot) const;
  void restore_history(const std::string& name, int slot, const std::vector<double>& values);
  double history_slot_dt(const std::string& name, int slot) const;
  void restore_history_slot_dt(const std::string& name, int slot, double dt);
  int rebuild_history_slots(const std::string& name, const std::vector<int>& stored_slots);
  /// The sorted macro-step cursors at which the LAST rebuild_history_slots fired an in-window regrid
  /// (ADC-635). The v3 reader asserts it against the checkpoint's recorded schedule fingerprint; empty
  /// after a Dense / clean-window / no-regrid replay.
  std::vector<int> last_replay_regrid_steps() const;
  /// @}

  double mass();  ///< mass of the 1st block on the coarse (conserved at reflux)
  double mass(
      const std::string& name);   ///< mass of the named block on the coarse (conserved PER BLOCK)
  std::vector<double> density();  ///< coarse density of the 1st block (component 0), n*n row-major
  std::vector<double> density(const std::string& name);  ///< coarse density of the named block, n*n
  /// Electrostatic potential phi of the COARSE LEVEL (base), n*n row-major. Level 0 covers
  /// the whole domain: enough to sample a median circle (azimuthal FFT), SAME
  /// observable as System::potential() on a single-level mesh. Solves the coarse Poisson if
  /// needed (cf. System::potential / ensure_elliptic), so current value even before any step.
  /// MULTI-BLOCK: phi results from the SYSTEM Poisson (Sum_b q_b n_b co-located); shared by all
  /// the blocks (single aux). The block name therefore does not intervene.
  std::vector<double> potential();

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace pops
