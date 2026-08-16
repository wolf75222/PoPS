#pragma once

#include <limits>

#include <pops/mesh/boundary/prepared_hyperbolic_boundary.hpp>
#include <pops/mesh/boundary/prepared_boundary_component.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>
#include <pops/numerics/nonlinear/newton_options.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <pops/coupling/source/coupling_operator.hpp>  // CouplingOperator / CouplingOperatorView (typed contract, ADC-595)
#include <pops/runtime/export.hpp>  // POPS_EXPORT: exact package seams resolved by native loaders
#include <pops/runtime/facade_options.hpp>  // CoupledSourceProgram (facade POD, ADC-214)
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams (compiled-Program runtime params on AMR, ADC-508)
#include <pops/runtime/config/spatial_domain.hpp>
#include <pops/runtime/amr_patch.hpp>
#include <pops/runtime/analytic/initial_materialization.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/numerical_defaults.hpp>
#include <pops/runtime/amr/prepared_tagging_execution.hpp>
#include <pops/runtime/amr/prepared_multiblock_hierarchy.hpp>
#include <pops/runtime/amr/exact_field_solver_provider.hpp>
#include <pops/runtime/amr/field_solver_options.hpp>
#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>
#include <pops/runtime/amr/hierarchy_policy_authority.hpp>
#include <pops/parallel/prepared_load_balance.hpp>
#include <pops/runtime/output_piece.hpp>
#include <pops/runtime/system/system_poisson_options.hpp>
#include <pops/runtime/system/auxiliary_checkpoint.hpp>
#include <pops/runtime/system/exact_aux_registry.hpp>

#include <array>
#include <functional>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

/// @file
/// @brief Multi-species composition on AMR at runtime: the refined counterpart of System.
///
/// One or SEVERAL blocks (species, described by ModelSpec of generic bricks) carried on an
/// AMR hierarchy. Like System but on an adaptive mesh.
///
/// One and many-block systems use the same AmrRuntime engine. Every block is co-located on ONE
/// SHARED AMR hierarchy (same ranked patches, ownership and spacing per level, guarded by
/// same_layout_or_throw). All blocks live on ALL patches. A single aux per level (phi,
/// grad phi) and a single coarse Poisson whose right-hand side is the CO-LOCATED SUM of the blocks'
/// elliptic bricks (f = Sum_b q_b n_b read at the same cells). Conservation PER BLOCK (reflux +
/// average_down). AmrRuntime runtime engine (type-erased registry by name). Blocks with potentially
/// DIFFERENT spatial schemes. Per-block temporal descriptors are immutable authoring metadata for
/// Program normalization; this spatial runtime does not decode or execute them. Union-of-tags
/// regridding is supported (multi-block + regrid_every > 0 regrids from the union; zero freezes the
/// hierarchy). Multirate cadence and inter-species coupled rates execute only where the installed
/// ProgramGraph places their typed operations. Multiple COMPILED blocks (add_compiled_model) and a
/// MIX of compiled + native blocks share the same hierarchy (capstone v, multi-block production DSL).
///
/// @note Resolved explicit Programs support N levels with hierarchy-authenticated exact-rank
/// transition ratios. An implicit/IMEX AMR composition
/// without a typed Program primitive fails closed; there is no private Newton or time-step fallback.

namespace pops::runtime::program {
struct AmrProgramHistoryRemapDescriptor;
}

namespace pops {

template <int Dim>
class FieldNullspaceProvider;
struct FieldLogicalTimePoint;
struct AuxHaloPolicy;
template <int Dim>
struct CompiledFieldBoundaryKernel;
class SolveOutcome;

namespace component {
class LoadedComponent;
class PreparedExecutionContextV1;
}  // namespace component

class ObserverMpiLane;
class ExecutionLane;
namespace runtime::program {
template <int Dim, class MemorySpace>
class AmrProgramContext;
class AcceptedProgramContextSnapshot;
}  // namespace runtime::program

namespace runtime::field {
struct PreparedFieldSolverSpec;
struct FieldTopologyReportRow;
}  // namespace runtime::field

/// Exact read-only backend configuration retained for one resolved AMR field solver.
struct AmrFieldSolverConfiguration {
  std::string plan_identity;
  std::string provider_identity;
  std::string solver;
  AmrFieldHierarchyPolicyAuthority hierarchy_policy;
  AmrFieldSolverOptions options;
};

// Forward declarations of the exact-ranked runtime package types. The generated-package header
// materializes PreparedAmrSystemBlock<Dim>; this public facade retains it without importing the
// implementation-heavy AMR/operator headers into every binding and native loader translation unit.
template <int Dim>
struct AmrRuntimeBlock;
template <int Dim, class MemorySpace>
struct PreparedAmrSystemBlock;
template <int Dim, class MemorySpace>
struct PreparedNativeAmrPackage;
template <int Dim, class MemorySpace>
struct PreparedAmrLevelEvaluation;
namespace runtime::amr {
template <int Dim, class MemorySpace>
class AmrRuntime;
}
namespace detail {
template <int Dim>
struct SharedAmrLayout;
}
namespace runtime {
namespace program {
class
    Profiler;  // forward-declared so engine()/profiler_handle() do not pull profiler.hpp into this header
template <int Dim>
struct ProgramRuntimeState;
}  // namespace program
namespace multiblock {
template <int Dim>
struct AxisAlignedInterface;
template <int Dim>
struct PreparedInterfaceFluxSpec;
struct BoundaryEvaluationPoint;
}  // namespace multiblock
}  // namespace runtime

/// Exact ranked AMR mesh and cadence (per-block physical parameters live in the ModelSpec).
template <int Dim>
struct AmrSystemConfig : RuntimeSpatialDomain<Dim> {
  static_assert(Dim >= 1 && Dim <= 3, "AmrSystemConfig only supports dimensions 1, 2, and 3");

  AmrSystemConfig() { this->shape = runtime_config_detail::filled_extent<Dim>(128); }

  int regrid_every = 20;  ///< re-refinement every N steps (0 = never after init)
  int level_count = 2;    ///< maximum active hierarchy depth (>= 1)
  /// Exact level-to-level hierarchy graph. Each table contains one ranked row per transition;
  /// ratio axes are positive and each row refines at least one axis; buffers/lookaheads are >= 0.
  std::vector<Extent<Dim>> transition_ratios{runtime_config_detail::filled_extent<Dim>(2)};
  std::vector<Extent<Dim>> transition_buffers{runtime_config_detail::filled_extent<Dim>(2)};
  std::vector<Extent<Dim>> transition_lookaheads{runtime_config_detail::filled_extent<Dim>(2)};
  bool explicit_bootstrap = false;  ///< coarse-only start; BootstrapPlan creates fine levels
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
  /// Per-axis coarse tile limits when distribute_coarse=true. Zero selects the hierarchy policy's
  /// deterministic ranked default on that axis. Ignored otherwise.
  Extent<Dim> coarse_max_grid{};
  /// ADC-616: Berger-Rigoutsos clustering params of the regrid layout. <= 0 (default) keeps the
  /// historical ClusterParams {0.7, 1, 32}, bit-identical. min_efficiency in (0,1], sizes >= 1,
  /// min_box_size <= max_box_size (validated at set_clustering / the facade descriptor).
  double cluster_min_efficiency = 0.0;
  int cluster_min_box_size = 0;
  int cluster_max_box_size = 0;
  /// Prepared ownership provider selected by the resolved adaptive-layout authority.  The semantic
  /// identity covers the public route, exact provider options and its weight capability.
  std::string load_balance_route = "space_filling_curve";
  std::string load_balance_identity = "pops.amr.default.space-filling-curve@1";
  PreparedProviderOptions load_balance_options{"pops.amr.load-balance.space-filling-curve@1", {}};
};

/// Frozen parameters passed to the deferred build of the compiled path (add_compiled_model). Materialized
/// by AmrSystem at ensure_built time: the geometry, Poisson and initial-state choices known
/// at that moment. The amr_dsl_block header consumes them to instantiate AmrCouplerMP<Model>.
///
/// STRUCTURE (ADC-610). Settings are grouped into NAMED sub-structs by ownership/role
/// (mesh, poisson, initial data, named aux) instead of one flat
/// append-only bag. A new setting goes INTO its semantic group -- the historical "add at the tail so an
/// older .so loader falls back silently" idiom is RETIRED because it no longer describes how this struct
/// evolves. The ABI story is now the VERSIONED KEY, not tail-only placement: this struct crosses the
/// dlopen .so boundary BY VALUE, and any layout change (a new field, a regroup) shifts POPS_HEADER_SIG
/// (a sha256 over include/, cf. abi_key.hpp / python/CMakeLists.txt), which re-keys pops_native_abi_key.
/// A .so generated before the change then diverges from the module key and add_native_block REJECTS it
/// with a clear "regenerate" error (never silent UB). So an older .so is refused, not silently truncated.
template <int Dim>
struct AmrBuildParams {
  static_assert(Dim >= 1 && Dim <= 3, "AmrBuildParams only supports dimensions 1, 2, and 3");

  /// Coarse mesh geometry + coarse ownership policy (AMR strong-scaling).
  struct Mesh : RuntimeSpatialDomain<Dim> {
    Mesh() { this->shape = runtime_config_detail::filled_extent<Dim>(128); }

    int regrid_every = 20;           ///< re-refinement cadence (0 = never after init)
    bool distribute_coarse = false;  ///< distributed multi-box coarse (AMR strong-scaling)
    Extent<Dim> coarse_max_grid{};   ///< exact per-axis tile cap
    std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>>
        load_balance;  ///< one prepared authority reused by coarse, fine seeds and every regrid
  } mesh;
  /// Exact prepared field provider slot. Legacy 2D Poisson/embedded-boundary objects are installed
  /// through their capability-qualified provider and never become part of this ranked build POD.
  std::string field_provider_slot;
  /// Initial coarse seed: density only (historical) OR the FULL conservative state (priority).
  struct InitialData {
    bool has_density = false;
    std::vector<double> density;  ///< initial coarse density (component 0), one native cell product
    // FULL initial conservative state (all components), takes priority over `density` when has_state.
    bool has_state = false;
    std::vector<double> state;  ///< ncomp*cell_count, component-major in native index order
  } initial;
};

/// Frozen argument type of the rejected pre-final loader ABI. It remains only so an old shared
/// object can resolve set_compiled_block and receive a deterministic fail-closed error; the final
/// runtime never stores or invokes this callable.
template <int Dim>
using AmrCompiledBlockBuilder = std::function<AmrRuntimeBlock<Dim>(
    const detail::SharedAmrLayout<Dim>& layout, const std::string& name,
    const std::vector<double>& density, bool has_density, const std::vector<double>& state,
    bool has_state, double gamma, int substeps, bool recon_prim, int stride,
    // Compatibility slots in the frozen builder ABI. Registration rejects every non-empty selector
    // before this callable is stored; the spatial builder never resolves or executes an IMEX mask.
    const std::vector<std::string>& implicit_vars, const std::vector<std::string>& implicit_roles,
    double pos_floor, double weno_epsilon, bool wave_speed_cache)>;

/// Single block carried on an AMR hierarchy, composed at runtime.
///
/// @code{.cpp}
/// pops::AmrSystemConfig cfg;                // base level: n x ny on independent x/y bounds
/// cfg.n = 64;
/// pops::AmrSystem amr(cfg);
///
/// pops::ModelSpec ne;
/// ne.transport = "exb"; ne.source = "none"; ne.elliptic = "charge";
/// amr.add_block("ne", ne, "minmod", "rusanov", "conservative", "explicit");
/// amr.set_poisson("charge_density", "geometric_mg");
/// amr.set_density("ne", rho0);             // rho0: initial density on the base level
/// amr.step_cfl(0.4);                       // conservative refluxed step + composite FAC Poisson
/// @endcode
template <int Dim>
class AmrSystem {
  static_assert(Dim >= 1 && Dim <= 3, "AmrSystem only supports dimensions 1, 2, and 3");

 public:
  using HyperbolicBoundary = PreparedHyperbolicBoundary<Dim>;
  using memory_space = typename MultiFab<Dim>::memory_space;
  using PreparedBlock = PreparedAmrSystemBlock<Dim, memory_space>;
  using PreparedNativePackage = PreparedNativeAmrPackage<Dim, memory_space>;
  using PreparedLevelEvaluation = PreparedAmrLevelEvaluation<Dim, memory_space>;
  using PreparedMultiBlockHierarchy =
      runtime::amr::PreparedMultiBlockAmrHierarchy<Dim, memory_space>;
  using ProgramBlockMap = typename PreparedMultiBlockHierarchy::ProgramBlockMap;
  using PreparedCouplingOperator = runtime::system::PreparedCouplingOperator<Dim>;
  struct PreparedAmrProgramFluxExpressionBlockBudget {
    std::size_t rhs_basis_bound = 0;
    std::size_t coefficient_term_bound = 0;
  };
  struct PreparedAmrProgramFluxExpressionBudget {
    std::string program_hash;
    std::uint64_t generation = 0;
    std::size_t interface_coupling_application_bound = 0;
    std::size_t interface_coupling_identity_character_bound = 0;
    ProgramBlockMap program_block_map;
    std::vector<PreparedAmrProgramFluxExpressionBlockBudget> blocks;
    std::string exact_contract;
  };
  static constexpr int dimension = Dim;

  explicit AmrSystem(const AmrSystemConfig<Dim>& cfg);
  ~AmrSystem();
  // RULE OF FIVE (C.21): move-only (PIMPL unique_ptr). The copy was already IMPLICITLY deleted
  // (move ctor declared); we make it EXPLICIT for intent. No API change (the copy was
  // already unusable).
  AmrSystem(const AmrSystem&) = delete;
  AmrSystem& operator=(const AmrSystem&) = delete;
  AmrSystem(AmrSystem&&);
  AmrSystem& operator=(AmrSystem&&);

  /// GLOBAL time-step bound (AMR counterpart of System::add_dt_bound): fn() evaluated ONCE
  /// per step_cfl (host), all_reduce_min (identical dt on all ranks), <= 0 / non-finite =
  /// inert this step. Hook for non-local constraints (coupling, scheduler, user ramp).
  void add_dt_bound(const std::string& label, std::function<double()> fn);

  /// ACTIVE bound of the last step_cfl: "transport:<block>" | "parabolic_frequency:<block>" |
  /// "source_frequency:<block>" | "stability_dt:<block>" | "global:<label>" | "degenerate" | ""
  /// (no CFL step yet).
  std::string last_dt_bound() const;

  /// Adds a block carried on the AMR. Same spatial-scheme parameters as System
  /// (limiter x riemann x recon), applied to each level/patch of the hierarchy. The FIRST
  /// add_block defines the block; a 2nd (or more) switches to the multi-block engine (shared
  /// hierarchy, co-located sum Poisson). Blocks can have DIFFERENT SPATIAL SCHEMES.
  /// @param name    block name: INDEXES the block (set_density(name), mass(name), density(name)). In
  ///                multi-block the name must be unique; mono-block an empty name targets the single block.
  /// @param model   composition of bricks (transport/source/elliptic + parameters)
  /// @param limiter "none" | "minmod" | "vanleer" | "weno5" | "mc" | "superbee"
  ///                (weno5 = WENO5-Z, 3 ghosts; native low-level stencil route). The resolved
  ///                Case route derives its
  ///                coarse/fine order and halo requirements from this spatial descriptor and
  ///                selects the minimum sufficient conservative provider.
  /// @param riemann "rusanov" | "hll" (generic signed-wave, requires model.wave_speeds) | "hllc"
  ///                | "roe" (requires the model's exact HasHLLCStructure /
  ///                HasRoeDissipation capability; no layout inference or fallback)
  /// @param time    "explicit" (SSPRK2) | "euler" (forward Euler) | "ssprk3"
  ///                (SSPRK3, order 3, reflux per stage; explicit transport, EXCLUSIVE of imex) |
  ///                "imex". This is immutable authoring metadata consumed by Program normalization;
  ///                the spatial block never executes a time method. A composition without a typed
  ///                Program primitive fails closed instead of reaching a private fallback.
  /// @param substeps declared Program substeps of the block (>= 1): the normalized ProgramGraph
  ///                partitions the effective step into equal pieces.
  /// @param stride  HOLD-THEN-CATCH-UP cadence of the block (>= 1; default 1 = each macro-step). stride=M
  ///                holds the block M-1 macro-steps then catches it up by an effective step M*dt (multirate).
  ///                Program-owned step_cfl honors the cadence: dt =
  ///                cfl*h*min_b(substeps_b/(stride_b*w_b)), mirror of System::step_cfl.
  /// @param implicit_vars / implicit_roles  reserved partial-IMEX selectors. The current AMR target
  ///                has no typed implicit-source Program primitive, so every non-empty selector
  ///                fails closed, including with time="imex"; it is never stored or ignored.
  /// @param newton  compatibility input validated like System::add_block. The spatial AMR block does
  ///                 not retain or execute source-Newton configuration, so every non-default value
  ///                 fails closed until the AMR target provides an executable typed local
  ///                 nonlinear/Newton Program primitive.
  /// @param newton_diagnostics  request for an aggregated Newton report owned by the typed implicit
  ///                 Program solve. The spatial runtime has no such report carrier, so true fails
  ///                 closed instead of exposing a permanently empty diagnostic.
  /// @param positivity_floor  Zhang-Shu positivity floor (ADC-259): if > 0, the AMR transport floors
  ///                 the Density-role face states (reconstruct_pp / zhang_shu_scale) AND the C/F fine
  ///                 ghost means to >= floor. Default 0 = inactive, bit-identical. Guarantee = face /
  ///                 ghost-state Density positivity only (order-1 fallback), NOT updated-mean nor
  ///                 pressure positivity (parity with System::add_block). A model without a Density
  ///                 role rejects floor > 0. The COMPILED .so path carries it too (ADC-322): a loader
  ///                 regenerated against this header marshals the floor (pops_install_native_amr).
  /// @throws std::runtime_error if a block is already defined, if substeps < 1, if stride < 1, if time
  ///         is not in {explicit, euler, ssprk3, imex}, if recon is not in {conservative,
  ///         primitive}, if a non-empty implicit selector is requested, or if source-Newton options
  ///         or diagnostics are requested on the spatial AMR block.
  void add_block(const std::string& name, const ModelSpec& model,
                 const std::string& limiter = "minmod", const std::string& riemann = "rusanov",
                 const std::string& recon = "conservative", const std::string& time = "explicit",
                 int substeps = 1, int stride = 1,
                 const std::vector<std::string>& implicit_vars = {},
                 const std::vector<std::string>& implicit_roles = {},
                 const NewtonOptions& newton = {}, bool newton_diagnostics = false,
                 double positivity_floor = 0.0,
                 double weno_epsilon = static_cast<double>(kWenoEpsilon),
                 bool wave_speed_cache = false);

  /// Retired deferred-builder ABI fence. It always fails before mutation; generated loaders must
  /// publish the complete PreparedAmrSystemBlock below so no metadata-only runtime can survive.
  POPS_EXPORT void set_compiled_block(
      int ncomp, double gamma, int substeps, AmrCompiledBlockBuilder<Dim> runtime_builder,
      const std::string& name = std::string(), bool recon_prim = false,
      const std::string& time = "euler", int stride = 1,
      const std::vector<std::string>& implicit_vars = {},
      const std::vector<std::string>& implicit_roles = {}, double pos_floor = 0.0,
      double weno_epsilon = static_cast<double>(kWenoEpsilon), bool wave_speed_cache = false);

  /// Atomically retain one complete generated spatial package. No deferred builder, legacy
  /// runtime block, or dimension-erased fallback is published by this route.
  POPS_EXPORT void install_prepared_amr_block(PreparedBlock block);
  /// Commit the one block plus every generated elliptic attachment as a single rank-local native
  /// package candidate. This seam is accepted only while the host owns installer_local; the outer
  /// RuntimeInstance transaction performs exact agreement before retaining the package lifetime.
  POPS_EXPORT void install_prepared_native_amr_package(PreparedNativePackage package);

  /// Borrow one accepted block/level carrier through its authenticated runtime identity.
  POPS_EXPORT const MultiFab<Dim>& prepared_amr_block_state(int runtime_block, int level) const;
  POPS_EXPORT MultiFab<Dim>& prepared_amr_block_state(int runtime_block, int level);
  /// Borrow the exact prepared embedded-boundary active mask for one block/level, or null when
  /// that level has no active embedded-boundary authority.
  [[nodiscard]] POPS_EXPORT const MultiFab<Dim>* prepared_amr_block_level_active_mask(
      int runtime_block, int level) const;
  POPS_EXPORT void install_prepared_amr_coupling_operator(std::string provider_contract,
                                                          CouplingOperatorView view,
                                                          PreparedCouplingOperator operation);
  POPS_EXPORT void install_prepared_amr_interface_flux_provider(
      std::string provider_contract,
      std::function<void(runtime::multiblock::InterfaceFluxScheduler<Dim>&)> installer);
  POPS_EXPORT const ProgramBlockMap& prepared_amr_program_block_map() const;
  POPS_EXPORT void install_prepared_amr_program_flux_expression_budget(
      std::string program_hash, std::vector<PreparedAmrProgramFluxExpressionBlockBudget> blocks,
      std::size_t interface_coupling_application_bound,
      std::size_t interface_coupling_identity_character_bound);
  POPS_EXPORT const PreparedAmrProgramFluxExpressionBudget&
  prepared_amr_program_flux_expression_budget() const;
  POPS_EXPORT ::pops::amr::InterfaceFluxLedgerBudget prepared_amr_interface_flux_ledger_budget()
      const;
  POPS_EXPORT std::size_t apply_prepared_amr_program_candidates(
      int level, Real dt, std::span<MultiFab<Dim>* const> program_candidates,
      const runtime::multiblock::BoundaryEvaluationPoint& point,
      runtime::multiblock::InterfaceFluxFragmentPublication* interface_publication);
  POPS_EXPORT void publish_prepared_amr_program_candidates(
      int level, std::span<MultiFab<Dim>* const> program_candidates);

  /// Materialize every level-bound operator, auxiliary owner, halo provider and flux ledger for
  /// the current exact hierarchy generation. A topology mutation invalidates the prior graph and
  /// this operation prepares a complete replacement before publication.
  POPS_EXPORT void refresh_prepared_amr_levels();

  /// Evaluate the installed generated operator on one level and atomically publish its residual
  /// together with the exact face-integrated fluxes used to assemble it.
  POPS_EXPORT const PreparedLevelEvaluation& evaluate_prepared_amr_level(
      const runtime::multiblock::BoundaryEvaluationPoint& point);
  /// Prepare/evaluate an exact stage candidate without replacing the hierarchy's accepted state.
  /// The candidate must retain the active level's complete layout/distribution/component contract.
  POPS_EXPORT void prepare_generated_amr_level_state(
      const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state);
  POPS_EXPORT const PreparedLevelEvaluation& evaluate_prepared_amr_level_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state);
  POPS_EXPORT void prepare_generated_amr_block_level_state(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state);
  POPS_EXPORT const PreparedLevelEvaluation& evaluate_prepared_amr_block_level_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state);
  POPS_EXPORT const PreparedLevelEvaluation& evaluate_prepared_amr_block_level_flux_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state);
  [[nodiscard]] POPS_EXPORT bool requires_prepared_amr_block_boundary_session(
      int runtime_block) const;
  [[nodiscard]] POPS_EXPORT bool has_prepared_amr_block_boundary_linearization(
      int runtime_block) const;
  POPS_EXPORT void prepared_amr_block_level_rhs_core_into_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, MultiFab<Dim>& result, bool flux_only);
  POPS_EXPORT void prepared_amr_block_level_boundary_residual_into_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, MultiFab<Dim>& result);
  POPS_EXPORT void prepared_amr_block_level_boundary_jvp_into_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, const MultiFab<Dim>& direction, MultiFab<Dim>& result);
  POPS_EXPORT void prepared_amr_block_level_source_into_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, MultiFab<Dim>& rhs);
  [[nodiscard]] POPS_EXPORT SolveOutcome solve_prepared_amr_block_level_source_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, Real dt, const NewtonOptions& options);
  POPS_EXPORT const PreparedLevelEvaluation& prepared_amr_level_evaluation(int level) const;
  POPS_EXPORT const PreparedLevelEvaluation* prepared_amr_level_evaluation_if_present(
      int level) const noexcept;
  POPS_EXPORT void clear_prepared_amr_level_evaluations() const noexcept;
  POPS_EXPORT void bind_program_hierarchy_candidates(
      const std::vector<MultiFab<Dim>>* candidates) const;
  POPS_EXPORT void unbind_program_hierarchy_candidates(
      const std::vector<MultiFab<Dim>>* candidates) const noexcept;
  POPS_EXPORT void bind_program_block_hierarchy_candidates(
      int runtime_block, const std::vector<MultiFab<Dim>>* candidates) const;
  POPS_EXPORT void unbind_program_block_hierarchy_candidates(
      int runtime_block, const std::vector<MultiFab<Dim>>* candidates) const noexcept;

  /// Exact level geometry/topology and model speed retained by the prepared hierarchy graph.
  POPS_EXPORT Geometry<Dim> prepared_amr_level_geometry(int level) const;
  POPS_EXPORT BoundaryTopology<Dim> prepared_amr_boundary_topology() const;
  POPS_EXPORT Real prepared_amr_level_maximum_speed(int level, const MultiFab<Dim>& state) const;
  POPS_EXPORT Real prepared_amr_block_level_maximum_speed(int runtime_block, int level,
                                                          const MultiFab<Dim>& state) const;
  POPS_EXPORT void validate_prepared_amr_state_publication_candidate(
      int runtime_block, int level, const MultiFab<Dim>& candidate) const;
  /// Apply one generated block's prepared pointwise projection to an owner-qualified detached
  /// Program candidate.
  POPS_EXPORT void project_prepared_amr_block_level_state(int runtime_block, int level,
                                                          int candidate_runtime_block,
                                                          MultiFab<Dim>& detached_candidate);

  /// Accumulate the generated block's exact elliptic right-hand side on one live level.
  POPS_EXPORT void add_prepared_amr_poisson_rhs(int level, MultiFab<Dim>& rhs);
  POPS_EXPORT void add_prepared_amr_block_poisson_rhs(int runtime_block, int level,
                                                      const MultiFab<Dim>& state,
                                                      MultiFab<Dim>& rhs);

  /// Install the same exact-ranked hyperbolic authority as System. Same-level halo exchange and
  /// coarse/fine transfer remain separate hierarchy operations; physical laws are evaluated only
  /// by this retained model-qualified object.
  POPS_EXPORT void install_hyperbolic_boundary(
      const std::string& name, const std::string& identity, int required_depth,
      const std::vector<std::string>& face_types, const std::vector<double>& face_values,
      const std::vector<std::string>& face_identities,
      const std::vector<std::string>& component_roles, const std::string& state_identity,
      const std::vector<std::string>& face_representations = {},
      const std::vector<std::string>& face_converter_identities = {},
      const std::vector<std::vector<std::string>>& face_analytic_opcodes = {},
      const std::vector<std::vector<double>>& face_analytic_literals = {},
      const std::vector<std::string>& face_analytic_clocks = {});
  POPS_EXPORT void install_prepared_hyperbolic_boundary(
      const std::string& name, const std::string& identity, int required_depth,
      const std::string& state_identity, std::shared_ptr<const HyperbolicBoundary> boundary);
  /// Register the exact state Handle independently from physical-boundary ownership.
  POPS_EXPORT void install_block_state_route(const std::string& name,
                                             const std::string& state_identity);
  /// Bind one exact solved-field Handle identity to its authenticated provider storage slot.
  /// Boundary components and AMR tagging consume this common prepared route.
  POPS_EXPORT void install_field_storage_route(const std::string& field_identity,
                                               const std::string& provider_slot);
  /// Retain the already duplicated RuntimeInstance package lane and its lane-qualified execution
  /// descriptor. AMR never reconstructs or retains the embedding-owned parent communicator.
  POPS_EXPORT void install_prepared_boundary_execution_context(
      std::shared_ptr<ExecutionLane> package_assembly_lane,
      std::shared_ptr<const component::PreparedExecutionContextV1> execution);
  POPS_EXPORT void stage_prepared_ghost_boundary_component(
      const std::string& block, std::shared_ptr<PreparedGhostBoundaryComponent> component);
  POPS_EXPORT void stage_prepared_boundary_flux_component(
      const std::string& block, std::shared_ptr<PreparedBoundaryFluxComponent> component);
  /// Atomically retain one complete operation-qualified FieldBoundary residual/JVP pair.
  POPS_EXPORT void stage_prepared_field_boundary_component_pair(
      const std::string& block, std::shared_ptr<PreparedFieldBoundaryResidualComponent> residual,
      std::shared_ptr<PreparedFieldBoundaryJvpComponent> jvp);
  /// Roll back a failed pre-build runtime-authority transaction.  Internal bind seam only.
  POPS_EXPORT void discard_hyperbolic_boundaries();

  /// Internal installation seam for a compiled AMR production package. The .so prepares one
  /// complete PreparedNativeAmrPackage<Dim> containing its PreparedAmrSystemBlock<Dim> and every
  /// elliptic attachment, then submits that inert candidate through the installer-local package
  /// seam. The host witnesses and publishes the complete package transaction atomically. No
  /// deferred runtime builder, flat-array numerical fallback, or alternate temporal engine is
  /// involved.
  ///
  /// The _pops host module is PROMOTED to global scope (RTLD_NOLOAD), then an authenticated private
  /// image of the generated package is opened RTLD_LOCAL: it can resolve the exact package
  /// installation symbol without exporting its generated templates to later semantic artifacts.
  /// The ABI key baked in the package
  /// (pops_native_abi_key) is compared to the module's (abi_key()) -- mismatch => clear error (no
  /// silent UB at the C++ boundary). Same scheme guard-rails as System (upstream validation).
  ///
  /// The exact generated core currently accepts one complete package; a second package fails before
  /// mutation rather than constructing a metadata-only or dimension-erased multi-block route.
  /// time accepts canonical Program-authoring tokens {explicit, euler, ssprk3, imex}; the spatial
  /// loader never executes one of those methods. In particular, ``imex`` requires a typed implicit
  /// Program primitive and has no backward-Euler/Newton fallback. The multirate stride and partial
  /// IMEX mask do not transit through the flat block-loader ABI: the Python facade rejects them
  /// rather than ignoring them silently. They must be expressed by a compiled typed Program whose
  /// target provides the corresponding primitive. recon "primitive" and flux "roe"/"hllc" are WIRED at parity (#113);
  /// the Python facade applies a pressure guard for hllc/roe.
  /// The low-level loader contains a WENO5-Z stencil route and allocates its three-cell halo. The
  /// resolved Case route authenticates the matching order-five conservative coarse/fine provider;
  /// neither the facade nor AmrRuntime substitutes the order-two route.
  /// @throws std::runtime_error if the ABI diverges, if a symbol is missing, or substeps < 1.
  /// @param name block name: cosmetic in mono-block, INDEXES the block in multi-block (set_density/
  ///             mass/density; must be unique and non-empty from the 2nd block on, like add_block).
  /// @param positivity_floor  Zhang-Shu positivity floor of the block (ADC-322): the .so flat ABI now
  ///             carries it (pops_install_native_amr -> add_compiled_model -> prepared package), so
  ///             a loader regenerated against this header floors the Density-role face states like
  ///             a native add_block. 0 (default) = inactive, bit-identical.
  void add_native_block(
      const std::string& name, const std::string& so_path,
      const std::string& expected_model_identity, const std::string& expected_binary_identity,
      const std::string& limiter = "minmod", const std::string& riemann = "rusanov",
      const std::string& recon = "conservative", const std::string& time = "explicit",
      double gamma = static_cast<double>(kPhysicalDefaultGamma), int substeps = 1,
      const std::vector<double>& params = {}, double positivity_floor = 0.0,
      double weno_epsilon = static_cast<double>(kWenoEpsilon), bool wave_speed_cache = false);

  /// Authenticate and install one exact-ranked external Riemann package through the ordinary
  /// prepared AMR block path. Both canonical System/AMR provider hooks are mandatory, including
  /// explicit empty hooks for a zero-provider brick. AMR routes are registered before hierarchy
  /// materialization; the authenticated DSO remains alive until every package closure is destroyed.
  void register_external_riemann_package(
      const std::string& name, const std::string& so_path, const std::string& brick_id,
      const std::string& expected_sha256, int expected_nvars, int expected_provider_count,
      const std::string& expected_model_identity, const std::string& provider_consumer_qid,
      const std::string& limiter = "minmod", const std::string& recon = "conservative",
      const std::string& time = "explicit",
      double gamma = static_cast<double>(kPhysicalDefaultGamma), int substeps = 1, int stride = 1,
      double positivity_floor = 0.0, double weno_epsilon = static_cast<double>(kWenoEpsilon));

  /// Install the exact prepared AMRTagging program resolved from the layout authority.
  /// This is the only tagging installation seam: the runtime never synthesizes a scalar
  /// threshold, component-zero default, or shared-potential fallback. The native tagger owns
  /// `min_cycles > 0` as accepted sparse state and persists it through AMR checkpoint/restart.
  void set_bootstrap_tagging(
      const std::vector<std::string>& leaf_subject_kinds,
      const std::vector<std::string>& leaf_subject_identities,
      const std::vector<std::string>& leaf_blocks, const std::vector<std::string>& leaf_variables,
      const std::vector<int>& leaf_field_component_indices, const std::vector<int>& leaf_ops,
      const std::vector<double>& leaf_thresholds, const std::vector<int>& leaf_stencil_indices,
      const std::vector<typename runtime::amr::PreparedTaggingProgram<Dim>::Stencil>& stencils,
      const std::vector<std::int32_t>& refine_ops, const std::vector<std::int32_t>& refine_args,
      const std::vector<std::int32_t>& coarsen_ops, const std::vector<std::int32_t>& coarsen_args,
      int min_cycles, const std::string& equality_policy, const std::string& conflict_policy,
      const std::string& clock_identity, const std::string& provider_identity);
  /// Install one authenticated native Tagger component. This private runtime seam selects the
  /// candidate evaluator only; all policy and hierarchy publication remain in AmrSystem.
  void install_tagger_component(
      std::shared_ptr<component::LoadedComponent> component, const std::string& component_id,
      const std::string& manifest_identity, std::uint32_t interface_version,
      const std::string& provider_identity, const std::string& tagging_graph_identity,
      const std::string& layout_identity, const std::string& clock_identity,
      const std::string& execution_mode, const std::string& parameters_json,
      const std::string& target_json,
      std::shared_ptr<const component::PreparedExecutionContextV1> execution);
  /// Execute the immutable exact-ranked tagging program against one live parent level. The
  /// returned masks are owner-local candidates; no clustering, hysteresis, or topology mutation is
  /// performed by this inspection route.
  POPS_EXPORT runtime::amr::PreparedTaggerCandidates<Dim> execute_prepared_tagging(
      int parent_level);
  /// Consume the prepared candidates for one parent, cluster their exact global union, transfer
  /// accepted state into the candidate child, and publish the regrid atomically. Returns whether a
  /// child remains active after publication.
  POPS_EXPORT bool regrid_from_prepared_tagging(int parent_level);
  /// Freeze every accepted history ring/level/slot before a multi-parent restart regrid.  Every
  /// transition then uses this immutable same-level overlap image while prolongating new coverage
  /// from the successively remapped parent.
  POPS_EXPORT void begin_restart_regrid_history_sequence();
  POPS_EXPORT void end_restart_regrid_history_sequence() noexcept;
  /// Install one exact parent/child temporal relation per AMR transition.  These ratios are an
  /// independent execution authority and are never inferred from spatial refinement.
  void set_temporal_relations(const std::vector<std::int64_t>& numerators,
                              const std::vector<std::int64_t>& denominators,
                              const std::vector<std::string>& remainder_policies);

  /// Configures the default coarse elliptic field. This convenience uses the same provider registry
  /// and exact option contract as resolved named fields; it is not a second solver path. The
  /// right-hand side is always the sum of the blocks' elliptic bricks.
  /// @param rhs    "charge_density" | "composite" (same composed right-hand side as System)
  /// @param solver exact registered provider identity
  /// @param bc     "auto" | "periodic" | "dirichlet" | "neumann"
  /// @param solver_options provider-owned typed options. An empty carrier requests the provider's
  ///        exact defaults; the AMR facade never interprets option keys.
  /// @throws std::runtime_error if rhs, provider, options, or bc violates the provider contract.
  void set_poisson(const std::string& rhs = "charge_density",
                   const std::string& solver = "geometric_mg", const std::string& bc = "auto",
                   const AmrFieldSolverOptions& solver_options = {});
  /// Attach the default field's exact-ranked provider outputs before hierarchy materialization.
  /// ``output_keys`` contains either the potential alone or the potential followed by one gradient
  /// component per native axis. Every key must be unique and owned by the sealed auxiliary
  /// registry's ``field_output`` provider class. An exact repeat is idempotent; every differing
  /// repeat is refused. This is the explicit direct-C++ publication seam for the default field;
  /// typed Python authoring publishes outputs through its named ``Case.field`` plan instead, while
  /// native ``fields_from_state`` attachments remain RHS-only by contract.
  POPS_EXPORT void register_default_elliptic_field_output(
      const std::vector<runtime::system::AuxiliaryComponentKey>& output_keys, int gradient_sign);

  /// Install one fully resolved AMR field route. The registry key is the digest of its
  /// block-qualified provider identity. ``plan_identity`` independently commits the complete
  /// resolved semantics. Before lazy runtime materialization, the canonical ordered
  /// (slot, plan_identity) registry must agree exactly on every MPI rank. Duplicate slots are
  /// refused, including exact repeats.
  void set_field_solver_plan(const std::string& provider_slot, const std::string& plan_identity,
                             const std::string& provider_identity,
                             const std::string& output_owner_identity,
                             const std::string& output_block, const std::string& output_key,
                             const std::vector<runtime::system::AuxiliaryComponentKey>& output_keys,
                             int output_gradient_sign,
                             const std::vector<std::string>& provider_identities,
                             const std::vector<std::string>& provider_blocks,
                             const std::vector<std::string>& provider_keys,
                             const std::vector<double>& provider_coefficients,
                             const std::string& solver,
                             const AmrFieldHierarchyPolicyAuthority& hierarchy_policy,
                             const AmrFieldSolverOptions& solver_options);
  /// Adds one native AMR field solver provider before binding. Builtins and extensions are resolved
  /// through the same per-system registry and must expose exact collective contracts.
  void register_field_solver_provider(
      std::shared_ptr<const runtime::amr::ExactAmrFieldSolverProvider<Dim>> provider);
  /// Installs one authenticated external FieldTopology@2 + FieldSolver@2 pair as an AMR provider.
  /// The returned route is exactly ``provider_slot`` and is suitable for set_field_solver_plan.
  POPS_EXPORT std::string register_field_solver_provider(
      const std::string& provider_slot, runtime::field::PreparedFieldSolverSpec spec,
      std::shared_ptr<component::LoadedComponent> topology,
      std::shared_ptr<component::LoadedComponent> solver);
  /// Adds one native field-nullspace provider before binding. The selected route is resolved only
  /// after operator, boundary, topology and distribution facts have materialized.
  void register_field_nullspace_provider(
      std::shared_ptr<const FieldNullspaceProvider<Dim>> provider);
  /// Select the provider for the principal field configured by set_poisson. The AMR facade retains
  /// only the opaque provider identity and its exact typed options; it never interprets a nullspace
  /// family or gauge name.
  void set_default_field_nullspace(const std::string& nullspace_provider_identity,
                                   const PreparedProviderOptions& options);
  /// Adds one hierarchy tensor-solver provider before binding. Compiled Programs resolve their
  /// opaque provider identity through this per-system registry; builtins and extensions use the
  /// same preparation protocol.
  void register_hierarchy_tensor_solver_provider(
      std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<Dim>> provider);
  /// Collective generated-Program extension seam. Unlike the pre-build authoring registration
  /// above, this may run after materialization: it mutates only the provider registry, authenticates
  /// the exact declaration on every rank, and is idempotent for the same component declaration.
  POPS_EXPORT void register_program_hierarchy_tensor_solver_provider(
      std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<Dim>> provider);
  /// Internal generated-Program seam. The returned immutable registry outlives every installed
  /// Program context because it is owned by this facade.
  POPS_EXPORT std::shared_ptr<const runtime::program::HierarchyTensorSolverProviderRegistry<Dim>>
  hierarchy_tensor_solver_provider_registry() const;
  /// Exact read-only backend configuration retained by one resolved field plan.
  AmrFieldSolverConfiguration field_solver_configuration(const std::string& provider_slot) const;
  /// Install the resolved scalar reaction coefficient of one named screened field.
  void set_field_reaction(const std::string& provider_slot, double reaction);
  void set_field_topology_authority(const std::string& provider_slot,
                                    const std::string& provider_kind, const std::string& provenance,
                                    const std::string& topology_digest);
  std::vector<runtime::field::FieldTopologyReportRow> field_topology_report(
      const std::string& provider_slot) const;
  void set_field_boundary_plan(const std::string& provider_slot,
                               const std::vector<std::string>& kind,
                               const std::vector<double>& alpha, const std::vector<double>& beta,
                               const std::vector<double>& value);
  void set_field_boundary_dependencies(const std::string& provider_slot,
                                       const std::vector<std::string>& state_blocks,
                                       const std::vector<int>& state_components,
                                       const std::vector<std::string>& field_blocks,
                                       const std::vector<std::string>& field_keys,
                                       const std::vector<int>& field_components);
  POPS_EXPORT void set_field_boundary_kernel(const std::string& provider_slot,
                                             const CompiledFieldBoundaryKernel<Dim>& kernel);
  POPS_EXPORT void set_field_logical_timepoint(const std::string& provider_slot,
                                               const FieldLogicalTimePoint& point);
  POPS_EXPORT void set_field_boundary_parameters(const std::string& provider_slot,
                                                 const std::vector<double>& parameters);
  void set_field_newton_plan(const std::string& provider_slot, double tolerance, int max_iterations,
                             double linear_tolerance, int linear_max_iterations, int restart,
                             double armijo, double minimum_step);

  /// Install one immutable analytic embedded-boundary definition.  The expression is sampled
  /// independently on every live AMR level whenever the hierarchy is materialized or regridded.
  /// Staircase and cut-cell transport use the same exact-ranked provider in dimensions 1, 2, and 3.
  void set_analytic_level_set(const std::vector<std::string>& opcodes,
                              const std::vector<double>& literals, const std::string& mode = "none",
                              double kappa_min = 0.0, double face_open_eps = 0.0,
                              double cut_theta_min = 0.0);
  /// Change only the numerical EB mode while retaining the accepted analytic definition.
  void set_geometry_mode(const std::string& mode);
  void set_field_nullspace(const std::string& provider_slot,
                           const std::string& nullspace_provider_identity,
                           const PreparedProviderOptions& options);

  /// Sets the initial density on the coarse level (component 0), flattened in native index order.
  /// @param name cosmetic label (mono-block AMR: the density targets the single block).
  void set_density(const std::string& name, const std::vector<double>& rho);

  /// Sets the FULL INITIAL CONSERVATIVE STATE (all components) on the coarse level, then
  /// prolongs it to the fine levels at build (constant injection, like the density). @p U is flat
  /// component-major with one exact-ranked cell product per component; ncomp == n_vars of the model (checked at
  /// build, where only Model::n_vars is known). Takes priority over set_density: allows starting the AMR
  /// from a full drift state (rho, rho*u, rho*v) instead of m=0. The conversion
  /// primitive -> conservative (rho_u = rho*u) is done on the Python side (the caller already supplies the
  /// conservative). Wired on native and compiled blocks, mono-block as well as multi-block: the full
  /// state is threaded to the deferred concrete builder, seeds the coarse, then is injected to the
  /// fine levels. In multi-block @p name indexes the target block.
  /// @throws std::runtime_error if the system is already built, if U is empty, or if its size
  ///         is not a multiple of the exact-ranked coarse cell count.
  void set_conservative_state(const std::string& name, const std::vector<double>& U);
  /// Authenticate one resolved initial-condition subject against its exact runtime block before
  /// staging its payload.  The subject must be the already-installed state route for that block.
  void bind_bootstrap_subject(const std::string& subject_id, const std::string& runtime_block,
                              const std::string& source_route);
  /// Stage one canonical analytic initial state.  Native code validates and compiles one postfix
  /// program per conservative component; the accepted programs are materialized by the existing
  /// hierarchy bootstrap path before Tagger execution.
  void stage_bootstrap_analytic_state(const std::string& subject_id,
                                      const std::string& runtime_block, const std::string& space,
                                      const std::string& centering, const std::string& projection,
                                      const analytic::AnalyticOpcodeRows& opcodes,
                                      const analytic::AnalyticLiteralRows& literals);
  /// Stage one exact-rank conservative array.  @p spatial_shape is the native spatial extent and
  /// @p components is its leading conservative-component count, both checked before publication.
  void stage_bootstrap_array(const std::string& subject_id, const std::string& runtime_block,
                             const std::string& space, const std::string& centering, int components,
                             const Extent<Dim>& spatial_shape, const std::vector<double>& values);
  /// Materialize one authenticated InitialConditionPlan action into the live hierarchy.  This is
  /// the only bootstrap source execution seam: level zero consumes the staged source, analytic
  /// fine levels re-evaluate that same source, and array fine levels consume their registered
  /// conservative transfer provider.  The active bootstrap snapshot owns rollback of every write.
  std::size_t materialize_bootstrap_action(const std::string& subject_id, const std::string& action,
                                           const std::string& action_route, int level);
  /// Recompute one authenticated elliptic field during an active bootstrap transaction.
  /// @p subject_id is the derived-field bootstrap subject; @p provider_slot is the exact
  /// field-plan identity used by the generated Program. Every staged conservative source
  /// must already be materialized on every live hierarchy level, and the slot must own an
  /// elliptic solver route. After create_level the prepared solvers are discarded; this
  /// seam rematerializes them on the complete prolonged hierarchy through the existing
  /// topology-field authority before CompositeFAC runs. Publication stays on the prepared
  /// hierarchy lane and is rolled back by the enclosing bootstrap snapshot.
  std::size_t recompute_bootstrap_field(const std::string& subject_id,
                                        const std::string& provider_slot);
  /// Restrict one already-materialized fine state onto its live parent during an explicit
  /// bootstrap transaction.  The subject must own the exact cell-conservative volume-average
  /// restriction authority for this ranked transition; publication is collective on the prepared
  /// hierarchy lane and changes only parent cells covered by the fine layout.
  void synchronize_bootstrap_state(const std::string& subject_id, int fine_level);
  void begin_bootstrap_plan();
  bool bootstrap_next_level();  ///< execute the next exact ranked transition if tagged
  void commit_bootstrap_level();
  void rollback_bootstrap_level();
  void register_bootstrap_transfer_route(
      const std::string& identity, const std::vector<std::string>& subjects,
      const std::string& provider_identity, const std::string& space, const std::string& centering,
      const std::string& representation, const std::string& storage, const std::string& operation,
      const std::string& kernel, int order, const Extent<Dim>& ghost_depth,
      const Extent<Dim>& refinement_ratio);
  void register_bootstrap_oriented_face_subjects(const std::vector<std::string>& oriented_subjects);

  /// Register immutable owner-qualified auxiliary producers and native consumer views.  The graph
  /// is sealed before hierarchy materialization; physical aliases and raw component indices are
  /// intentionally absent from this interface.
  POPS_EXPORT void install_prepared_auxiliary_provider(
      runtime::system::PreparedAuxiliaryProvider<Dim> provider);
  POPS_EXPORT void install_auxiliary_consumer_plan(
      runtime::system::AuxiliaryConsumerProviderPlan<Dim> plan);
  POPS_EXPORT void seal_auxiliary_providers();
  POPS_EXPORT void stage_auxiliary_input(const runtime::system::AuxiliaryComponentKey& key,
                                         const std::vector<double>& values);
  POPS_EXPORT void refresh_auxiliary(const runtime::system::AuxiliaryEvaluationPoint& point);
  [[nodiscard]] POPS_EXPORT runtime::system::AuxiliaryStorageAddress<Dim> auxiliary_address(
      const runtime::system::AuxiliaryComponentKey& key) const;
  [[nodiscard]] POPS_EXPORT std::vector<double> auxiliary_component(
      const runtime::system::AuxiliaryComponentKey& key, int level = 0) const;
  [[nodiscard]] POPS_EXPORT std::string auxiliary_registry_contract() const;
  [[nodiscard]] POPS_EXPORT const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>&
  prepared_auxiliary_consumer_plan(const std::string& consumer_qid) const;
  /// Level-qualified group and plan access for generated AMR Program contexts.  Every consumer
  /// binds its compact local view against the active hierarchy level; no shared auxiliary slab is
  /// exposed at this seam.  These are rank-local hot-path lookups: callers must first perform one
  /// collective ``refresh_prepared_amr_levels()`` for the enclosing Program resource traversal.
  [[nodiscard]] POPS_EXPORT const runtime::system::AuxiliaryStorageGroups<Dim>*
  prepared_amr_provider_storage_groups(int level) const;
  [[nodiscard]] POPS_EXPORT const runtime::system::ResolvedAuxiliaryConsumerPlan<Dim>&
  prepared_amr_auxiliary_consumer_plan(const std::string& consumer_qid, int level) const;

  /// Durable accepted metadata for each AMR hierarchy level.  The native checkpoint backend owns
  /// rank-local group payload staging; this image authenticates its exact group identities,
  /// owner-qualified ComponentKeys, shapes and accepted provider generations before publication.
  [[nodiscard]] POPS_EXPORT std::vector<runtime::system::AuxiliaryCheckpointAcceptedState<Dim>>
  capture_auxiliary_checkpoint_accepted_state() const;
  /// Rank-local, collective-free rollback witness for accepted state, auxiliary and elliptic
  /// carriers plus each level-qualified auxiliary registry. Values include full ghost storage.
  [[nodiscard]] POPS_EXPORT std::vector<std::vector<std::string>>
  checkpoint_rank_local_carrier_manifest() const;
  /// Exact pending provider identities retained by accepted rollback snapshots.
  [[nodiscard]] POPS_EXPORT std::vector<std::string> dirty_auxiliary_provider_identities() const;
  /// Rank-local capacity derived from the qualified provider registries. The pair is the largest
  /// payload-free POPSAUX2 image and scalar width of one full-domain level.
  [[nodiscard]] POPS_EXPORT std::pair<std::size_t, std::size_t>
  checkpoint_auxiliary_level_capacity() const;
  /// Restore only after the caller has staged compatible rank-local group payloads privately.  A
  /// communicator preflight and a full registry rollback image prevent a rejected level from
  /// exposing a partial accepted generation.
  POPS_EXPORT void restore_auxiliary_checkpoint_accepted_state(
      const std::vector<runtime::system::AuxiliaryCheckpointAcceptedState<Dim>>& state);
  /// Decode every sealed level POPSAUX2 image inside the prepared hierarchy lane while one native
  /// restart transaction owns rollback authority. The complete decoded vector is consensus-closed
  /// before the typed restore enters finite/registry phases; this route never lazily builds an
  /// engine or selects a process-global communicator.
  using AuxiliaryCheckpointByteViewProvider =
      std::function<std::span<const std::uint8_t>(std::size_t)>;
  using AuxiliaryCheckpointByteCountProvider = std::function<std::size_t()>;
  POPS_EXPORT void restore_restart_auxiliary_checkpoint_accepted_state_bytes(
      const AuxiliaryCheckpointByteCountProvider& payload_count,
      const AuxiliaryCheckpointByteViewProvider& payload_at);

  /// @name Named multi-elliptic fields (ADC-428)
  /// Exact-ranked API for a SECOND elliptic solve on the AMR hierarchy. Installation is accepted
  /// only when the selected native specialization owns a dimension-qualified hierarchy field-solver
  /// provider; the facade never stores an unused contract or falls back to the historical 2-D engine.
  /// @{
  /// Registers named @p field's exact-ranked provider outputs. ``output_keys`` contains either the
  /// potential alone or the potential followed by one gradient component per native axis. Every key
  /// must be owned by a sealed ``field_output`` provider; no raw carrier component crosses this API.
  /// @throws if the system is already built, the output contract is malformed, or no exact-ranked
  /// hierarchy field-solver provider is installed. Provider-unavailable failure happens before mutation.
  POPS_EXPORT void register_elliptic_field(
      const std::string& block_name, const std::string& provider_key,
      const std::vector<runtime::system::AuxiliaryComponentKey>& output_keys, int gradient_sign);
  /// Attaches named @p field's RHS closure (rhs += elliptic_field_rhs(U)) to block @p block_name.
  /// ``rhs_provider_identity`` is the stable executable identity authenticated before initial
  /// hierarchy materialization; callable bytes are never serialized. Called by the native AMR
  /// loader (make_poisson_rhs of the per-field brick). @throws before mutation if the identity is
  /// empty, the system is already built, the block is unknown, or no exact-ranked hierarchy
  /// provider exists.
  POPS_EXPORT void set_block_elliptic_field(
      const std::string& block_name, const std::string& field,
      const std::string& rhs_provider_identity,
      std::function<void(const MultiFab<Dim>&, MultiFab<Dim>&)> rhs);
  /// Solved potential of named @p field on the coarse level, flattened in native index order. Solves the
  /// hierarchy fields if needed (so it is current even before any step), then reads the field's phi
  /// component. AMR counterpart of System::aux_field_component for a named elliptic field. @throws if the
  /// field is unregistered.
  std::vector<double> named_field_values(const std::string& field);
  std::vector<std::string> field_provider_slots() const;
  /// Collective-free identity of the exact provider aliased by the historical ``phi`` checkpoint
  /// member. The configured default wins; a generic-only registry uses its deterministic first
  /// slot. Every provider remains independently present in the all-provider image.
  std::string checkpoint_phi_provider_slot() const;
  /// Immutable, collective-free field-provider checkpoint manifest. Each canonical row records
  /// the slot, active depth, resolved provider/plan/configuration identities, field and auxiliary
  /// dependencies, output ownership, live topology/materialization generations and whether the
  /// solver is currently materialized. This accessor never builds a solver or enters a collective.
  std::vector<std::vector<std::string>> field_provider_checkpoint_manifest() const;
  int field_provider_levels(const std::string& provider_slot) const;
  /// Restore the complete all-provider warm-start image as one detached candidate. Every slot and
  /// level is shape/finite/consensus validated before any accepted solver storage is replaced.
  void restore_field_potentials(const std::vector<std::string>& provider_slots,
                                const std::vector<std::vector<std::vector<double>>>& potentials);
  /// Recompute every typed derived field after a restart regrid in the authenticated combined
  /// auxiliary/field dependency order. Returns a canonical witness row per published field.
  std::vector<std::vector<std::string>> recompute_fields_after_restart_regrid();
  void set_field_potential(const std::string& provider_slot, const std::vector<double>& phi);
  void set_field_potential_level(const std::string& provider_slot, int level,
                                 const std::vector<double>& phi);
  std::vector<double> field_potential_global(const std::string& provider_slot);
  std::vector<double> field_potential_level_global(const std::string& provider_slot, int level);
  /// Exact rank-local valid-cell pieces for one qualified field provider.  The returned metadata
  /// explicitly marks replicated level-zero ownership so output modes never infer it from box counts.
  std::vector<OutputPiece<Dim>> output_field_local_pieces(const std::string& provider_slot,
                                                          int level);
  std::vector<OutputPiece<Dim>> output_field_root_pieces(const ObserverMpiLane& lane,
                                                         const std::string& provider_slot,
                                                         int level);
  /// Transaction bracket used by the accepted-state reader after complete payload preflight. Every
  /// hierarchy,
  /// state, aux, field warm-start, history and clock mutation is rolled back if any restore step fails.
  void begin_restart_transaction();
  /// Collectively authenticate that every rank retains one rollback-capable accepted snapshot.
  /// This seals the transaction for commit without releasing rollback authority.
  void commit_restart_transaction();
  /// Release a collectively committed restart snapshot.  The commit precondition is established by
  /// commit_restart_transaction(), so this phase performs only no-throw ownership release.
  void finalize_restart_transaction() noexcept;
  void rollback_restart_transaction();
  /// Force exactly one artifact-owned scientific regrid inside an active restart transaction.
  /// The exact recorded accepted state must already have been restored and authenticated.
  void preflight_regrid_on_restart();
  void regrid_on_restart();
  int checkpoint_regrid_count() const;
  std::uint64_t checkpoint_topology_epoch() const;
  void restore_checkpoint_counters(int regrid_count, std::uint64_t topology_epoch);
  std::vector<std::vector<std::string>> checkpoint_temporal_relations() const;
  /// Canonical rows for every required bootstrap transfer route: subject, operation, route identity,
  /// provider, kernel, descriptor fields.  The sealed checkpoint compares these rows byte-for-byte.
  std::vector<std::vector<std::string>> checkpoint_transfer_routes() const;
  /// @}

  void step(double dt);  ///< one AMR macro-step (periodic regrid included)
  void advance(double dt, int nsteps);
  void begin_step_transaction();
  void commit_step_transaction();
  void finalize_step_transaction();
  void rollback_step_transaction();
  /// Internal rollback authority used by the installed Program context.  A Program nested in a
  /// public AmrSystem step borrows the facade's accepted image instead of taking a second full engine
  /// snapshot; a direct C++ Program context remains autonomous.
  POPS_EXPORT bool has_active_step_transaction() const noexcept;
  POPS_EXPORT void restore_active_step_transaction_for_program();
  /// Volume-weighted L2 norm of each block's accepted AMR macro-step change. Collective and valid
  /// while the retained outer transaction snapshot still owns U^n.
  POPS_EXPORT std::map<std::string, double> step_change_l2() const;
  /// Advances using the smallest exact-ranked level/block bound.  Each explicit Cartesian
  /// diffusive candidate uses cfl * substeps / (stride * (max(speed, speed_floor) / h_min +
  /// 2 nu sum_a h_a^-2)); source, model, global, Program and strategy bounds may reduce it further.
  /// The request, prepared schedule and selected decision are authenticated on the hierarchy lane.
  /// @return the dt used.
  double step_cfl(double cfl, double speed_floor = static_cast<double>(kCflSpeedFloor),
                  double max_dt = std::numeric_limits<double>::infinity(), double min_dt = 0.0);

  /// @name Compiled time-program install seam on the AMR hierarchy (epic ADC-511 / ADC-508, Spec 6)
  /// AMR counterpart of System::install_program: load a generated problem.so and install its compiled
  /// time Program over the AMR hierarchy. Mirrors the System seam (install_program_step registers the
  /// macro-step body; the cadence + per-block RuntimeParams stores live HERE on the Impl, NOT in the
  /// .so closure, so a value change reaches the captured context and a later checkpoint can reach
  /// them). A generated AMR Program .so resolves these POPS_EXPORT seams from the globally promoted
  /// host while the generated package remains RTLD_LOCAL, exactly like the exact spatial-package
  /// installation seam on the native AMR loader.
  /// @{
  /// Install the mandatory macro-step body. AmrSystem::step, advance and step_cfl reject before lazy
  /// hierarchy construction or any other mutation while it is absent. An empty std::function is
  /// rejected: there is no public temporal route that silently clears the whole-system Program.
  /// POPS_EXPORT: the generated AMR Program .so resolves it across the dlopen boundary. The closure
  /// executes the normalized ProgramGraph on the hierarchy through
  /// an AmrProgramContext (the AMR counterpart of ProgramContext).
  POPS_EXPORT void install_program_step(std::function<void(double)> step);
  /// Install the companion callback that republishes Program-owned accepted clocks/history whenever
  /// explicit bootstrap commits a hierarchy level. Generated artifacts own this seam; direct
  /// low-level steps may omit it because they have no authenticated checkpoint context.
  POPS_EXPORT void install_program_hierarchy_refresh(std::function<void()> refresh);
  /// Install the artifact-owned restart preflight, transform, forced resynchronization and
  /// phase-safe accepted-context snapshot hooks.
  POPS_EXPORT void install_program_restart_hooks(
      std::function<void()> preflight, std::function<void()> regrid, std::function<void()> resync,
      std::function<std::unique_ptr<runtime::program::AcceptedProgramContextSnapshot>()>
          accepted_context_snapshot);
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
  /// Exact duration, accepted public-step count and physical start currently held by the GLOBAL
  /// Program stride window. All are zero at a stride boundary and form mandatory checkpoint state.
  POPS_EXPORT double program_cadence_window_dt() const;
  POPS_EXPORT int program_cadence_window_steps() const;
  POPS_EXPORT double program_cadence_window_start_time() const;
  /// Exact accepted Program interval provenance. Zero means no Program invocation has been accepted.
  POPS_EXPORT double program_last_dt() const;
  /// Stage the exact held-window image before set_clock during strict restart. The image must match
  /// the exact accepted (@p accepted_time, @p macro_step) cursor, @p accepted_last_dt and installed
  /// stride; malformed, missing or mismatched state is rejected without mutating accepted state.
  POPS_EXPORT void restore_program_cadence_window(double accumulated_dt, int held_steps,
                                                  double window_start_time, double accepted_last_dt,
                                                  double accepted_time, int macro_step);
  /// Install the program-index -> AMR-block-index map (entry p = the AMR block index of Program block
  /// p), built by install_program after matching the .so's block names to the instantiated AMR blocks
  /// BY NAME (Spec 3 criterion 23, ADC-457). Empty clears it and is never an implicit positional
  /// identity. Read by the AmrProgramContext to resolve a Program block index to the name-matched block.
  POPS_EXPORT void set_program_block_map(const std::vector<int>& prog_to_sys);
  /// The installed program-index -> AMR-block-index map. Empty means no authenticated mapping and is
  /// rejected by AmrProgramContext.
  POPS_EXPORT const std::vector<int>& program_block_map() const;
  /// Load a generated problem.so and install its compiled time Program on the AMR hierarchy. dlopens
  /// @p so_path, checks its ABI key against this module (fail-loud on mismatch), runs the section-24
  /// install-time requirement validation (aux / solver / block instance, verbatim spec messages), binds
  /// the Program's blocks to the AMR blocks BY NAME, seeds each block's RuntimeParams from the .so
  /// pops_program_param_* metadata, then calls the .so's pops_install_program_amr(this), whose shared
  /// facade factory selects the hierarchy provider and installs the macro-step closure. Mirrors
  /// add_native_block and System::install_program; the .so stays loaded for the process lifetime.
  POPS_EXPORT void install_program(const std::string& so_path);
  /// IR hash of the installed compiled Program (the string returned by the .so's pops_program_hash), or
  /// "" if no program is installed. Parity with System::installed_program_hash (checkpoint guard).
  POPS_EXPORT std::string installed_program_hash() const;
  /// The last macro-step dt handed to the installed Program (ADC-631): the AmrProgramContext reads it
  /// so a pre-commit history sample records its outgoing interval (variable-dt replay). POPS_EXPORT
  /// for the dlopen boundary (the generated AMR Program .so reads it via the AmrProgramContext).
  /// Authenticated accepted-state image owned by the compiled AMR Program context.  This is distinct
  /// from the dense field/history arrays: it preserves exact level clocks, qualified history-slot
  /// identities and lagged effective-flux publications required for conservative multistep restart.
  POPS_EXPORT std::vector<std::uint8_t> program_accepted_state() const;
  /// Artifact-authenticated upper bounds for the complete POPSAND4 image and its fixed-size
  /// source-rematerialization digest. The bound covers every configured hierarchy level, temporal
  /// execution, history slot, tagging cell and accepted flux publication.
  POPS_EXPORT std::pair<std::size_t, std::size_t> checkpoint_program_state_capacity() const;
  /// Copy the same authenticated image into caller-owned storage.  The Program attempt transaction
  /// keeps this storage resident so a stable retry snapshots bytes without allocating a temporary
  /// vector on every macro-step; the returned-by-value accessor remains the public convenience API.
  POPS_EXPORT void copy_program_accepted_state_into(std::vector<std::uint8_t>& state) const;
  /// Replace the accepted image during strict restart.  Each replacement advances a revision observed
  /// by the persistent AmrProgramContext before its next attempt; no stale context state is reused.
  POPS_EXPORT void restore_program_accepted_state(const std::vector<std::uint8_t>& state);
  /// Strict checkpoint counterpart: authenticate the complete accepted image and its runtime-owned
  /// tagging payload before atomically publishing either. A rejected payload changes neither bytes,
  /// revision nor the live AMR hysteresis state.
  POPS_EXPORT void restore_checkpoint_accepted_state(const std::vector<std::uint8_t>& state);
  /// Validate the exact history registry encoded by @p state and materialize its native per-level
  /// rings on the already rebuilt restart hierarchy. This is a transactional restart seam: it never
  /// advances the Program and refuses any name/depth/component/owner mismatch before allocation.
  POPS_EXPORT void materialize_program_restart_histories(const std::vector<std::uint8_t>& state,
                                                         const std::vector<std::string>& names,
                                                         const std::vector<int>& depths,
                                                         const std::vector<int>& ncomps);
  POPS_EXPORT std::uint64_t program_accepted_state_revision() const;
  /// Human/audit-readable qualification rows decoded from the same accepted image persisted as bytes.
  POPS_EXPORT std::vector<std::vector<std::string>> program_accepted_state_manifest() const;
  POPS_EXPORT std::vector<std::vector<std::string>> program_clock_manifest() const;
  /// Accepted temporal-partition provider, synchronization tick and per-rung cell counts. The rows
  /// are decoded from the same opaque image used by strict restart, never a capability ledger.
  POPS_EXPORT std::vector<std::vector<std::string>> program_temporal_partition_manifest() const;
  POPS_EXPORT std::vector<std::vector<std::string>> program_flux_ledger_manifest() const;
  POPS_EXPORT std::vector<std::vector<std::string>> program_interface_flux_ledger_manifest() const;
  POPS_EXPORT std::vector<std::vector<std::string>> program_sync_manifest() const;
  POPS_EXPORT std::vector<::pops::amr::ParentChildClockRelation>
  prepared_program_temporal_relations() const;

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
  /// The built AMR spatial engine (the AmrRuntime the AmrProgramContext driver wraps), or nullptr
  /// before the lazy build. install_program forces the build so the .so's pops_install_program_amr
  /// receives a live engine. POPS_EXPORT: the generated AMR Program .so resolves it across the dlopen
  /// boundary.
  POPS_EXPORT runtime::amr::AmrRuntime<Dim, typename Kokkos::DefaultExecutionSpace::memory_space>*
  engine() const;
  /// Compatibility inspection seam. Once built, every resolved AMR system uses AmrRuntime, so this
  /// returns true; before build engine() remains null. POPS_EXPORT for dlopen-boundary parity.
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
  /// Five current-attempt scalars for one typed balance route. RuntimeInstance calls this only
  /// inside its active outer accepted-step transaction; missing/stale/non-finite evidence fails.
  POPS_EXPORT std::map<std::string, double> accepted_balance_terms(const std::string& route) const;
  /// The same accepted route with selected attempt-local native reflux/projection producers.
  POPS_EXPORT std::map<std::string, double> selected_accepted_balance_terms(
      const std::string& route, const std::string& block, int component,
      const std::vector<int>& levels, const std::vector<std::string>& automatic_terms) const;
  POPS_EXPORT void begin_step_projection_report();
  POPS_EXPORT void note_step_projection(const std::string& name);
  POPS_EXPORT std::vector<std::string> consume_step_projections();
  /// LEVEL-COMPOSITE collective reduction over a named block, the AMR counterpart of
  /// System::reduce_component the diagnostics driver drives (ADC-542). @p kind is per-component
  /// "sum" / "min" / "max" / "abs_sum" / "sum_sq" / "abs_max", or the full-state "*_all" variants.
  /// @p levels is the exact strictly-increasing level selection; empty is the low-level C++
  /// all-level convention. Volume-weighted sums and extrema mask coarser selected cells covered by
  /// the next selected finer level. Multi-block and single-block both remain native/Kokkos.
  POPS_EXPORT double composite_reduce(const std::string& block, const std::string& kind, int comp,
                                      const std::vector<int>& levels = {}) const;
  /// Composite integral/reduction of one qualified native field provider.  This route is available
  /// only after the AMR runtime has been built because that engine owns the typed provider hierarchy.
  POPS_EXPORT double composite_reduce_field(const std::string& provider_slot,
                                            const std::string& kind, int comp,
                                            const std::vector<int>& levels = {});
  /// @}
  /// @}

  Extent<Dim> spatial_shape() const;
  /// Generated Program shared libraries read the accepted clock through the flat loader ABI.
  POPS_EXPORT double time() const;
  /// ACCEPTED macro-step counter (0-indexed; incremented by step / advance / step_cfl), parity with
  /// System::macro_step. Required for checkpoint/restart because Program schedules and regrid cadence
  /// depend on accepted-step phase, not only on physical time. Persisted by accepted-state restart.
  /// POPS_EXPORT: the AmrProgramContext (a generated AMR Program .so) reads it across the dlopen
  /// boundary for the head-of-step regrid cadence, like the other program seam accessors (ADC-508).
  POPS_EXPORT int macro_step() const;
  /// RESTORES the AMR clock (t, macro_step) -- parity with System::set_clock. Sets the time AND the
  /// macro-step counter propagated to Program schedules and regrid cadence. A mid-window cursor also
  /// requires an immediately preceding restore_program_cadence_window; neither the start nor
  /// accumulated variable-dt duration can be inferred from the clock. @throws on invalid state.
  POPS_EXPORT void set_clock(double t, int macro_step);

  /// @name AMR / MPI profiling (Spec 5 sec.12.5, ADC-479 criterion 43)
  /// Per-phase wall-clock timing of the AMR runtime: the engine times its non-numeric phases --
  /// "regrid" (rebuild the patch hierarchy), "fill_boundary" (the cross-rank ghost halo exchange),
  /// "average_down" (restrict fine onto coarse) -- plus integer counters ("regrid" / "fill_boundary"
  /// per-run counts, and under MPI np>1 "mpi_reductions" / "mpi_messages"). Disabled by default (no
  /// hot-path cost when off, parity with System). enable_profiling() then step()/step_cfl() over a
  /// run where a regrid fires (regrid_every set) then profile_report() returns the table; the typed
  /// PerformanceSummary.by_amr_mpi() view surfaces it. Per-rank (no cross-rank reduction of the
  /// report). The same runtime owns these phases for every block count.
  /// @{
  void enable_profiling();
  void disable_profiling();
  bool is_profiling() const;
  void reset_profiling();
  std::string profile_report() const;
  /// @}
  POPS_EXPORT int n_blocks() const;  ///< number of blocks on the shared AmrRuntime engine
  /// Names of the blocks in add order (parity with System::block_names): the IO facade iterates over them
  /// to write EACH block by its name (an empty name -> block 0, historical mono-block compat).
  std::vector<std::string> block_names() const;
  /// Structured report of effective numerical, solver and physical options currently configured.
  EffectiveOptionsReport effective_options_report() const;
  int n_patches();  ///< number of current fine patches (of the shared hierarchy)
  /// Index-space signatures of the current fine patches: one AmrPatch<Dim>
  /// (level, inclusive lower/upper Index<Dim> corners) per fine box, for ALL
  /// fine levels (level >= 1). Inclusive corners in the index space of the
  /// level (each base-axis count refined by the authenticated hierarchy
  /// transitions). SAME source as n_patches() (the GLOBAL fine BoxArray, all
  /// boxes/all ranks -> rank-independent, MPI-safe, zero communication). It is a
  /// QUERY (between steps): read-only of the already-stored boxes, NO hot-path
  /// cost. The conversion to exact physical bounds is done on the Python side.
  /// Forces the lazy build (ensure_built) like n_patches()/mass()/density().
  std::vector<AmrPatch<Dim>> patch_boxes();
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
  /// AmrSystem.checkpoint/restart on the Python side). Unqualified level_state is a one-block
  /// convenience over AmrRuntime; multi-block callers use the qualified block_level_state variants.
  /// Potential, hierarchy and aux remain shared. The _global variants all_reduce_sum
  /// the per-rank fabs so a np>1 checkpoint gathers onto rank 0 (mono-rank: identity, bit-identical).
  /// Force the lazy build (ensure_built) like patch_boxes()/mass(). @p k: level (0 = coarse, >= 1 = fine).
  int n_levels();    ///< number of levels of the hierarchy (>= 1; mono OR multi-block)
  int max_levels();  ///< resolved maximum active hierarchy depth
  POPS_EXPORT int configured_n_levels();  ///< immutable resolved hierarchy capacity
  int n_vars();  ///< number of conserved components (MONO-BLOCK; multi-block: block_n_vars)
  /// FULL conservative state of level @p k, flat component-major c*nf*nf + j*nf + i (nf = n << k;
  /// zeros outside the patches at the fine level -- only the patch interior is defined). MONO-BLOCK.
  std::vector<double> level_state(int k);
  std::vector<double> level_state_global(int k);  ///< one-block, np>1 gather (all ranks call)
  void set_level_state(int k,
                       const std::vector<double>& s);  ///< restores the state of level @p k (as is)
  /// Potential phi of level @p k, flat nf*nf row-major. Level 0 = warm-start of the multigrid
  /// (bit-identical resumption); level >= 1 = aux comp 0 (recomputed at update). Shared across blocks;
  /// the _global variant gathers under np>1.
  std::vector<double> level_potential(int k);
  std::vector<double> level_potential_global(int k);              ///< np>1 gather (all ranks call)
  void set_level_potential(int k, const std::vector<double>& p);  ///< restores phi of level @p k
  /// Imposes the SAVED fine hierarchy (at restart) instead of Berger-Rigoutsos clustering: @p boxes
  /// are the patch_boxes() signatures of the checkpoint. Serial convenience over rebuild_hierarchy.
  void set_hierarchy(const std::vector<AmrPatch<Dim>>& boxes);

  /// Impose a mid-run hierarchy from a v3 checkpoint (ADC-542): @p boxes are ALL the
  /// checkpoint patch boxes (level tagged, level 0 implicit), @p owner_ranks the per-box owner rank
  /// aligned with @p boxes.  ``-1`` is the private rebuild-only replicated-level witness: all patches
  /// of one fine level must use it, while concrete ranks select partitioned ownership. Routes to
  /// AmrRuntime::rebuild_hierarchy (all levels rebuilt, reusing regrid R6/R7).
  void rebuild_hierarchy(const std::vector<AmrPatch<Dim>>& boxes,
                         const std::vector<int>& owner_ranks);

  /// Re-evaluate ownership for the exact recorded fine-level boxes under the load-balance authority
  /// prepared at bind. This is a collective, non-mutating restart seam: geometry and box ordering
  /// stay unchanged while the returned owner list is aligned with @p boxes for the current
  /// communicator size.
  std::vector<int> rematerialize_hierarchy_ownership(const std::vector<AmrPatch<Dim>>& boxes,
                                                     const std::vector<std::string>& level_modes);

  /// Merge exact source-rank Program images and return this rank's image under the current
  /// communicator ownership. Both ownership tables are indexed [level][global patch].  The
  /// source authority is an opaque contract emitted only while the source image is the live
  /// accepted state; it binds that image to its artifact, spatial epoch/generation and owner map.
  std::vector<std::uint8_t> program_accepted_state_source_authority(
      const std::vector<std::vector<int>>& source_level_owners,
      const std::vector<std::string>& source_level_modes, int source_rank_count) const;
  std::vector<std::uint8_t> rematerialize_program_accepted_state(
      const std::vector<std::uint8_t>& source_state, int source_rank_count,
      const std::vector<std::vector<int>>& source_level_owners,
      const std::vector<std::vector<int>>& target_level_owners,
      const std::vector<std::string>& source_level_modes,
      const std::vector<std::string>& target_level_modes,
      const std::vector<std::uint8_t>& source_authority);

  /// Per-block per-level checkpoint accessors (ADC-509). The AmrRuntime engine shares the
  /// layout AND the aux across blocks, so the per-level STATE is read/restored PER BLOCK (by NAME)
  /// while phi stays shared (level_potential above). @p name indexes the block (block_names()); @p k:
  /// level. The _global variant all_reduce_sum the per-rank fabs (np>1 gather, all ranks call); the
  /// shared hierarchy is the deterministic frozen central patch (regrid_every==0), reproduced at
  /// restart by replaying the same composition. @throws if @p name / @p k is out of bounds.
  /// Installed variable names of one authenticated block. @p kind is "conservative" or
  /// "primitive"; names are read from the prepared native block image, never reconstructed by a
  /// facade-side convention.
  std::vector<std::string> variable_names(const std::string& name,
                                          const std::string& kind = "conservative") const;
  int block_n_vars(const std::string& name);  ///< conserved components of the named block
  std::vector<double> block_level_state(const std::string& name, int k);
  std::vector<double> block_level_state_global(const std::string& name,
                                               int k);  ///< np>1 gather (all ranks call)
  void set_block_level_state(const std::string& name, int k, const std::vector<double>& s);
  /// Unified scientific-output state accessor. Unlike the checkpoint names above, this routes an
  /// exactly named block through the shared runtime and returns compact native valid-cell pieces
  /// without allocating a global level buffer.
  std::vector<OutputPiece<Dim>> output_state_local_pieces(const std::string& name, int k);
  /// Exact per-level EB sidecars: pops_active, pops_phi, or pops_kappa.
  std::vector<OutputPiece<Dim>> output_embedded_boundary_local_pieces(const std::string& name,
                                                                      int k);
  std::vector<AmrPatch<Dim>> output_geometry_boxes();
  std::vector<OutputPiece<Dim>> output_state_root_pieces(const ObserverMpiLane& lane,
                                                         const std::string& name, int k);
  std::vector<OutputPiece<Dim>> output_embedded_boundary_root_pieces(const ObserverMpiLane& lane,
                                                                     const std::string& name,
                                                                     int k);
  /// Owner rank per box of level @p k (the shared ranked ownership plan), aligned with the
  /// level-@p k rows of patch_boxes(). The v3 checkpoint (ADC-542) serializes it so a restart
  /// reproduces the LOCAL-fab iteration order.
  std::vector<int> level_owner_ranks(int k);
  /// Exact active-level distribution mode: ``replicated`` or ``partitioned``.
  std::string level_distribution_mode(int k) const;
  /// @name Multistep history-ring checkpoint / replay (ADC-631, Uniform System seam names)
  /// The compiled-Program AMR route carries per-level `keep_history` / `T.prev` ring slots on the
  /// AmrRuntime engine (remapped through regrid).  Every value and provenance accessor is qualified
  /// by the exact hierarchy level so subcycled levels retain distinct dt/fill metadata.
  /// @{
  std::vector<std::string> history_names() const;
  std::vector<int> history_levels(const std::string& name) const;
  int history_depth(const std::string& name) const;
  int history_ncomp(const std::string& name) const;
  bool history_initialized(const std::string& name, int level) const;
  int history_fill_count(const std::string& name, int level) const;
  void set_history_initialized(const std::string& name, int level, bool initialized);
  void restore_history_fill_count(const std::string& name, int level, int fill_count);
  void restore_history_metadata(const std::string& name, int level, bool initialized,
                                int fill_count);
  void restore_history_provenance(const std::string& name, int level,
                                  const std::vector<double>& slot_dt, bool initialized,
                                  int fill_count);
  std::vector<double> history_global(const std::string& name, int level, int slot) const;
  void restore_history(const std::string& name, int level, int slot,
                       const std::vector<double>& values);
  double history_slot_dt(const std::string& name, int level, int slot) const;
  void restore_history_slot_dt(const std::string& name, int level, int slot, double dt);
  int rebuild_history_slots(const std::string& name, const std::vector<int>& stored_slots);
  /// The sorted macro-step cursors at which the LAST rebuild_history_slots fired an in-window regrid
  /// (ADC-635). The accepted-state reader asserts it against the checkpoint's recorded schedule
  /// fingerprint; empty after a Dense / clean-window / no-regrid replay.
  std::vector<int> last_replay_regrid_steps() const;
  /// @}

  double mass();  ///< mass of the 1st block on the coarse (conserved at reflux)
  double mass(
      const std::string& name);   ///< mass of the named block on the coarse (conserved PER BLOCK)
  std::vector<double> density();  ///< coarse density of the first block, native index order
  std::vector<double> density(
      const std::string& name);  ///< named-block density, native index order
  /// Electrostatic potential phi of the coarse level, flattened in native index order. Level 0
  /// covers the whole domain and is the same observable as System::potential() on a single-level
  /// mesh. Solves the coarse Poisson if
  /// needed (cf. System::potential / ensure_elliptic), so current value even before any step.
  /// MULTI-BLOCK: phi results from the SYSTEM Poisson (Sum_b q_b n_b co-located); shared by all
  /// the blocks through one qualified field output. The block name therefore does not intervene.
  std::vector<double> potential();

 private:
  template <int ContextDim, class MemorySpace>
  friend class runtime::program::AmrProgramContext;
  /// Private DSO seam: only the generated AmrProgramContext may install the post-publication
  /// prepared-history remap boundary. It is intentionally absent from the public facade surface.
  POPS_EXPORT void install_program_history_remap_accepted(
      std::function<void(const runtime::program::AmrProgramHistoryRemapDescriptor&)> refresh);
  std::vector<std::string> prepare_topology_field_order(
      std::string_view reason, const runtime::multiblock::BoundaryEvaluationPoint& accepted_point);
  std::vector<std::vector<std::string>> rematerialize_fields_after_topology_change(
      std::string_view reason, const runtime::multiblock::BoundaryEvaluationPoint& accepted_point);
  POPS_EXPORT PreparedMultiBlockHierarchy& prepared_amr_multiblock_hierarchy_();
  POPS_EXPORT const PreparedMultiBlockHierarchy& prepared_amr_multiblock_hierarchy_() const;
  POPS_EXPORT void prepare_generated_amr_block_level_state(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, int parent_level, const MultiFab<Dim>* staged_parent);
  POPS_EXPORT const PreparedLevelEvaluation& evaluate_prepared_amr_block_level_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, int parent_level, const MultiFab<Dim>* staged_parent);
  POPS_EXPORT const PreparedLevelEvaluation& evaluate_prepared_amr_block_level_flux_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, int parent_level, const MultiFab<Dim>* staged_parent);
  /// Friend-only transaction seam.  These evaluate into the hierarchy-owned candidate workspace
  /// and deliberately leave the published evaluation ledger untouched.
  POPS_EXPORT const PreparedLevelEvaluation& prepare_prepared_amr_block_level_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state);
  POPS_EXPORT const PreparedLevelEvaluation& prepare_prepared_amr_block_level_flux_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state);
  POPS_EXPORT const PreparedLevelEvaluation& prepare_prepared_amr_block_level_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, int parent_level, const MultiFab<Dim>* staged_parent);
  POPS_EXPORT const PreparedLevelEvaluation& prepare_prepared_amr_block_level_flux_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, int parent_level, const MultiFab<Dim>* staged_parent);
  /// The validation phase is collective and must complete before any caller publishes another
  /// transaction member.  The companion publication only performs proven-noexcept swaps/stores.
  POPS_EXPORT void validate_prepared_amr_block_level_batch(
      std::span<const std::pair<int, int>> targets) const;
  POPS_EXPORT void publish_prepared_amr_block_level_batch(
      std::span<const std::pair<int, int>> targets) noexcept;
  POPS_EXPORT void prepared_amr_block_level_source_into_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, MultiFab<Dim>& rhs, int parent_level,
      const MultiFab<Dim>* staged_parent);
  [[nodiscard]] POPS_EXPORT SolveOutcome solve_prepared_amr_block_level_source_at(
      int runtime_block, const runtime::multiblock::BoundaryEvaluationPoint& point,
      MultiFab<Dim>& state, Real dt, const NewtonOptions& options, int parent_level,
      const MultiFab<Dim>* staged_parent);
  /// Dedicated generated-Program sink for one validated, attempt-local balance term. It remains
  /// private to AmrProgramContext and is deliberately absent from Python bindings.
  POPS_EXPORT void record_program_balance_term(const std::string& route, const std::string& term,
                                               double value);
  POPS_EXPORT bool program_balance_consumer_is_due(const std::string& contract,
                                                   const std::string& route, int every_n) const;
  POPS_EXPORT runtime::program::ProgramRuntimeState<Dim>& program_runtime_state_();
  /// Read-only compiled-artifact capability check; artifact authority installation is private to
  /// AmrSystem::install_program and cannot be injected through the public facade.
  POPS_EXPORT bool program_owns_operator_authority(
      const std::array<std::uint64_t, 4>& authority) const noexcept;
  POPS_EXPORT SolveOutcome solve_program_field_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int active_level, const MultiFab<Dim>* stage_override);
  POPS_EXPORT void with_program_field_candidate_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int active_level, const MultiFab<Dim>& stage_override, const std::function<void()>& evaluate);
  POPS_EXPORT SolveOutcome solve_program_field_from_blocks_at(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int active_level, const std::vector<const MultiFab<Dim>*>& stage_overrides);
  POPS_EXPORT SolveOutcome solve_program_field_from_blocks_on_prepared_lane(
      const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
      int active_level, const std::vector<const MultiFab<Dim>*>& stage_overrides);
  POPS_EXPORT void refresh_auxiliary_on_prepared_lane(
      const runtime::system::AuxiliaryEvaluationPoint& point);
  void install_prepared_amr_block_candidate_(PreparedBlock block, bool native_package_candidate);
  POPS_EXPORT void restore_auxiliary_checkpoint_accepted_state_on_prepared_lane(
      const std::vector<runtime::system::AuxiliaryCheckpointAcceptedState<Dim>>& state,
      const ExecutionLane& lane);
  POPS_EXPORT SolveOutcome solve_program_default_field(int active_level);
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace pops
